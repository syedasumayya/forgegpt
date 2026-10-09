import os, math, time, json
from dataclasses import asdict
import torch
from forgegpt.config import GPTConfig, TrainConfig
from forgegpt.model import GPT
from forgegpt.data import get_batch


def get_lr(step, t):
    """Warmup, then cosine decay from lr down to min_lr."""
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    if step >= t.max_steps:
        return t.min_lr
    progress = (step - t.warmup_steps) / (t.max_steps - t.warmup_steps)
    return t.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (t.lr - t.min_lr)


def safe_save(obj, path):
    """Write to a temp file first, so a disconnect can't leave a half-written checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def estimate_loss(model, cfg, t, device):
    """Average loss on train and val data (honest progress check)."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(t.eval_iters)
        for k in range(t.eval_iters):
            x, y = get_batch(split, t.data_dir, cfg.block_size, t.batch_size, device)
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: GPTConfig, t: TrainConfig):
    device = "cuda"
    run_dir = os.path.join(t.out_dir, t.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")      # resumable snapshot
    best_path = os.path.join(run_dir, "best.pt")      # best model by validation loss
    log_path = os.path.join(run_dir, "log.jsonl")     # history for graphs

    model = GPT(cfg).to(device)
    print(f"Run '{t.run_name}': {model.num_params()/1e6:.1f}M parameters")

    # Weight decay only on big weight matrices, not on norms/biases
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.lr, betas=(0.9, 0.95), fused=True,
    )
    scaler = torch.amp.GradScaler("cuda")

    # Resume if a checkpoint exists
    step, best_val = 0, float("inf")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"Resumed from checkpoint at step {step}")

    if t.use_wandb:
        import wandb
        wandb.init(project="forgegpt", name=t.run_name,
                   config={**asdict(cfg), **asdict(t)}, resume="allow")

    torch.manual_seed(1337 + step)
    tokens_per_step = t.batch_size * t.grad_accum * cfg.block_size
    running, t_last = 0.0, time.time()
    model.train()

    def evaluate_and_log(step):
        nonlocal best_val
        l = estimate_loss(model, cfg, t, device)
        lr = get_lr(step, t)
        print(f"[EVAL] step {step:5d} | train {l['train']:.3f} | val {l['val']:.3f} "
              f"| val perplexity {math.exp(l['val']):.2f} | lr {lr:.2e}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "train_loss": l["train"],
                                "val_loss": l["val"], "lr": lr}) + "\n")
        if t.use_wandb:
            wandb.log({"train_loss": l["train"], "val_loss": l["val"], "lr": lr}, step=step)
        if l["val"] < best_val:
            best_val = l["val"]
            safe_save({"model": model.state_dict(), "cfg": asdict(cfg),
                       "step": step, "val_loss": best_val}, best_path)

    while step < t.max_steps:
        lr = get_lr(step, t)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % t.eval_interval == 0:
            evaluate_and_log(step)

        # ---- one update step ----
        optimizer.zero_grad(set_to_none=True)
        for _ in range(t.grad_accum):
            x, y = get_batch("train", t.data_dir, cfg.block_size, t.batch_size, device)
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss, _ = model(x, y)
            running += loss.item() / t.grad_accum
            scaler.scale(loss / t.grad_accum).backward()    # guess -> blame
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        scaler.step(optimizer)                              # nudge the numbers
        scaler.update()
        step += 1

        if step % t.log_interval == 0:
            now = time.time()
            tok_s = tokens_per_step * t.log_interval / (now - t_last)
            print(f"  step {step:5d} | loss {running / t.log_interval:.3f} | {tok_s:,.0f} tokens/sec")
            running, t_last = 0.0, now

        if step % t.ckpt_interval == 0 or step == t.max_steps:
            safe_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict(), "step": step,
                       "best_val": best_val, "cfg": asdict(cfg)}, ckpt_path)

    evaluate_and_log(step)                                  # final measurement
    safe_save({"model": model.state_dict(), "cfg": asdict(cfg), "step": step}, 
              os.path.join(run_dir, "final.pt"))
    print("Training finished. Best validation loss:", round(best_val, 3))
    return model
