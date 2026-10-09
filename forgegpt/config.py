from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 4096      # number of tokens the tokenizer knows
    block_size: int = 256       # how many tokens the model can look at once (context length)
    n_layer: int = 6            # number of blocks stacked
    n_head: int = 6             # attention "heads" (see below)
    n_embd: int = 384           # size of each meaning vector
    dropout: float = 0.0        # randomly switching off parts during training (0 = off)

    # Switches for our experiments later (ablations):
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    pos_type: str = "rope"      # "rope" or "learned"
    mlp_type: str = "swiglu"    # "swiglu" or "gelu"
