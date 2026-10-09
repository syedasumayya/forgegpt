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


@dataclass
class TrainConfig:
    run_name: str = "base"          # each experiment gets its own name/folder
    batch_size: int = 64            # snippets per micro-batch
    grad_accum: int = 2             # micro-batches per update (effective batch = 128)
    max_steps: int = 5000           # total updates
    warmup_steps: int = 200
    lr: float = 1e-3                # peak learning rate
    min_lr: float = 1e-4            # learning rate at the very end
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250        # measure train/val loss every N steps
    eval_iters: int = 50            # batches averaged for each measurement
    ckpt_interval: int = 500        # save a resumable checkpoint every N steps
    log_interval: int = 50          # print a progress line every N steps
    data_dir: str = "/content/data"
    out_dir: str = "/content/drive/MyDrive/forgegpt_data/checkpoints"
    use_wandb: bool = False         # set True later if you want W&B graphs
