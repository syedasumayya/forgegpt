import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from forgegpt.config import GPTConfig


class RMSNorm(nn.Module):
    """Keeps numbers at a healthy size (like a volume knob)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def make_norm(cfg):
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd)   # the older alternative, used for the ablation


def precompute_rope(head_dim, max_len, base=10000.0):
    """Prepare the rotation angles for every position."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, inv_freq)       # (max_len, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs of numbers by an angle that depends on the token's position."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.use_rope = (cfg.pos_type == "rope")
        self.dropout = cfg.dropout

    def forward(self, x, cos=None, sin=None, kv_cache=None):
        B, T, C = x.shape                                  # batch, tokens, embedding size

        q, k, v = self.qkv(x).split(C, dim=2)              # make Query, Key, Value
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None:                           # used only during fast generation
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None),                  # block peeking at future tokens
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # join the heads back together
        return self.proj(y), new_cache


class MLP(nn.Module):
    """The 'thinking' sub-network inside each block."""
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.mlp_type
        if self.kind == "swiglu":
            hidden = int(8 * cfg.n_embd / 3)
            hidden = 64 * ((hidden + 63) // 64)            # round to a GPU-friendly size
            self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        else:                                              # classic GELU version
            hidden = 4 * cfg.n_embd
            self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return self.w_down(F.gelu(self.w_up(x)))


class Block(nn.Module):
    """One transformer block = attention + feed-forward, each with a safety shortcut."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cos=None, sin=None, kv_cache=None):
        a, new_cache = self.attn(self.norm1(x), cos, sin, kv_cache)
        x = x + a                                          # residual connection
        x = x + self.mlp(self.norm2(x))                    # residual connection
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_type == "learned" else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying (see below)

        if cfg.pos_type == "rope":
            cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():            # gentler start for output layers
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, kv_caches=None, last_only=False):
        B, T = idx.shape
        start = 0 if kv_caches is None else kv_caches[0][0].size(2)
        assert start + T <= self.cfg.block_size, "Sequence longer than block_size"

        x = self.tok_emb(idx)
        cos = sin = None
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start, start + T, device=idx.device))
        else:
            cos = self.rope_cos[start:start + T]
            sin = self.rope_sin[start:start + T]

        new_caches = []
        for i, block in enumerate(self.blocks):
            past = None if kv_caches is None else kv_caches[i]
            x, cache = block(x, cos, sin, past)
            new_caches.append(cache)

        x = self.norm_f(x)
        logits = self.lm_head(x[:, -1:, :] if last_only else x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches
