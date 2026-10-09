import torch
import torch.nn.functional as F


def sample_next(logits, temperature=0.8, top_k=None, top_p=None):
    """Pick the next token from the model's scores."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = (probs.cumsum(dim=-1) - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate(model, idx, max_new_tokens=200, temperature=0.8, top_k=40,
             top_p=0.95, use_cache=True, eos_id=None):
    """Write tokens one at a time. use_cache=True is the fast KV-cache path."""
    model.eval()
    max_new_tokens = min(max_new_tokens, model.cfg.block_size - idx.size(1))
    caches = None
    for _ in range(max_new_tokens):
        if use_cache:
            inp = idx if caches is None else idx[:, -1:]    # first pass: whole prompt, then 1 token
            logits, _, caches = model(inp, kv_caches=caches, last_only=True)
        else:
            logits, _, _ = model(idx, last_only=True)       # re-reads everything every time
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    return idx
