from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from radiance.config import Config, load_config, resolve_device, resolve_dtype
from radiance.data import build_dataloaders, build_tokenizer
from radiance.model import DenseTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> LambdaLR:
    warmup_steps = cfg.train.warmup_steps
    max_steps = cfg.train.max_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def estimate_batch_size(raw_model: DenseTransformer, cfg: Config, device: str, device_type: str) -> tuple[int, int]:
    """Conservative analytical batch_size/grad_accum_steps for cfg.train.auto_batch_size, derived
    from free VRAM and model size rather than an expensive live probe. CUDA-only — callers must
    check device_type == "cuda" before calling this."""
    assert cfg.train.target_effective_batch_size is not None, (
        "train.target_effective_batch_size must be set when train.auto_batch_size is True"
    )
    assert device_type == "cuda", "estimate_batch_size requires CUDA"

    # Parameters/grad/optimizer state stay fp32 regardless of train.dtype (see TODO-DTYPE-MODE.md).
    # grad and AdamW's exp_avg/exp_avg_sq are lazily allocated (on first backward()/step()
    # respectively), so mem_get_info below doesn't yet reflect them — add them analytically.
    param_dtype_bytes = 4
    torch.cuda.synchronize(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    num_params = raw_model.num_parameters()
    not_yet_allocated_bytes = 3 * num_params * param_dtype_bytes  # grad + 2 Adam buffers
    usable_bytes = max(0.0, free_bytes - not_yet_allocated_bytes) * cfg.train.vram_safety_margin

    activation_dtype_bytes = 4 if cfg.train.dtype == "fp32" else 2
    bytes_per_token = raw_model.activation_bytes_per_token(activation_dtype_bytes)
    max_tokens = usable_bytes / bytes_per_token
    batch_size = max(1, int(max_tokens // cfg.data.seq_len))
    grad_accum_steps = max(1, math.ceil(cfg.train.target_effective_batch_size / batch_size))

    print(
        f"[radiance] auto_batch_size: {free_bytes / 1e9:.2f} GB free, {num_params:,} params, "
        f"vram_safety_margin={cfg.train.vram_safety_margin} -> batch_size={batch_size}, "
        f"grad_accum_steps={grad_accum_steps} (effective_batch_size={batch_size * grad_accum_steps}, "
        f"target={cfg.train.target_effective_batch_size})"
    )
    return batch_size, grad_accum_steps


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


@torch.no_grad()
def evaluate(model: DenseTransformer, val_loader, device: str, device_type: str, dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32):
            logits, _, _ = model(input_ids)
            loss = compute_loss(logits, input_ids)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    # TF32 matmuls run at full tensor-core throughput on Ampere/Hopper/Blackwell with a
    # small precision tradeoff; PyTorch defaults this off, so opt in explicitly.
    torch.set_float32_matmul_precision("high")
    device = resolve_device(cfg.train.device)
    device_type = device.split(":")[0]
    dtype = resolve_dtype(cfg.train.dtype)
    # Only fp16 needs loss scaling (its exponent range is narrow enough to underflow small
    # gradients); bf16 has fp32's exponent range so it trains fine unscaled.
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == torch.float16))

    tokenizer = build_tokenizer(cfg)

    raw_model = DenseTransformer(cfg.model, vocab_size=len(tokenizer)).to(device)

    if cfg.train.auto_batch_size:
        if device_type == "cuda":
            cfg.train.batch_size, cfg.train.grad_accum_steps = estimate_batch_size(
                raw_model, cfg, device, device_type
            )
        else:
            print(
                f"[radiance] auto_batch_size requires CUDA (device_type={device_type!r}); "
                "using configured batch_size/grad_accum_steps."
            )

    model = torch.compile(raw_model, mode="reduce-overhead") if cfg.train.compile else raw_model

    # batch_size must be finalized (auto_batch_size, if any, already ran) before the DataLoader is built.
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    if cfg.train.tokens_per_param is not None:
        tokens_per_step = cfg.train.effective_batch_size * cfg.data.seq_len
        target_tokens = cfg.train.tokens_per_param * raw_model.num_parameters()
        cfg.train.max_steps = max(1, round(target_tokens / tokens_per_step))
        print(
            f"[radiance] tokens_per_param={cfg.train.tokens_per_param} over {raw_model.num_parameters():,} params "
            f"-> max_steps={cfg.train.max_steps:,} ({target_tokens:,.0f} tokens at {tokens_per_step:,} tokens/step, "
            f"batch_size={cfg.train.batch_size} x grad_accum_steps={cfg.train.grad_accum_steps} "
            f"= effective_batch_size={cfg.train.effective_batch_size})"
        )

    optimizer = AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, fused=(device_type == "cuda")
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    if wandb.run is None:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=cfg.run_name,
            config={
                "data": vars(cfg.data),
                "model": vars(cfg.model),
                "train": vars(cfg.train),
            },
        )
    wandb.log(
        {
            "num_parameters": raw_model.num_parameters(),
            "train/auto_batch_size": cfg.train.auto_batch_size,
            "train/batch_size": cfg.train.batch_size,
            "train/grad_accum_steps": cfg.train.grad_accum_steps,
        },
        step=0,
    )

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    data_iter = iter(train_loader)

    grad_accum_steps = cfg.train.grad_accum_steps
    # Physical per-forward-pass chunk size: starts equal to batch_size (i.e. one chunk per
    # micro-batch, a no-op split) and only ever shrinks, via OOM backoff below. Splitting an
    # already-fetched micro-batch into smaller chunks — rather than rebuilding the DataLoader at a
    # smaller batch_size — keeps batch_size/grad_accum_steps/effective_batch_size (and therefore
    # tokens_per_param accounting) completely unaffected by backoff; it only changes how many
    # forward/backward calls it takes to process the same data.
    micro_chunk_size = cfg.train.batch_size
    give_up = False

    while step < cfg.train.max_steps and not give_up:
        will_log = (step + 1) % cfg.train.log_every == 0
        step_done = False

        while not step_done and not give_up:
            if will_log:
                accum_loss = torch.zeros((), device=device)
                accum_lm_loss = torch.zeros((), device=device)
                accum_ponder_cost = torch.zeros((), device=device)
                accum_mean_loop_depth = torch.zeros((), device=device)

            try:
                optimizer.zero_grad(set_to_none=True)
                for _ in range(grad_accum_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        batch = next(data_iter)

                    input_ids = batch["input_ids"].to(device)
                    for chunk in input_ids.split(micro_chunk_size, dim=0):
                        # chunk_weight reconstructs the same overall mean-of-token-losses as one
                        # micro_loss / grad_accum_steps backward would, regardless of how many
                        # (possibly uneven) chunks a micro-batch got split into.
                        chunk_weight = chunk.size(0) / cfg.train.batch_size / grad_accum_steps
                        with torch.autocast(
                            device_type=device_type, dtype=dtype, enabled=dtype != torch.float32
                        ):
                            logits, ponder_cost, mean_loop_depth = model(chunk)
                            lm_loss = compute_loss(logits, chunk)
                            chunk_loss = lm_loss + cfg.model.ponder_weight * ponder_cost

                        scaler.scale(chunk_loss * chunk_weight).backward()

                        if will_log:
                            accum_loss += chunk_loss.detach() * chunk_weight
                            accum_lm_loss += lm_loss.detach() * chunk_weight
                            accum_ponder_cost += ponder_cost.detach() * chunk_weight
                            accum_mean_loop_depth += mean_loop_depth.detach() * chunk_weight

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                step_done = True

                if will_log:
                    wandb.log(
                        {
                            "train/loss": accum_loss.item(),
                            "train/lm_loss": accum_lm_loss.item(),
                            "train/ponder_cost": accum_ponder_cost.item(),
                            "train/mean_loop_depth": accum_mean_loop_depth.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/micro_chunk_size": micro_chunk_size,
                        },
                        step=step,
                    )

                if val_loader is not None and step % cfg.train.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device, device_type, dtype)
                    wandb.log({"val/loss": val_loss}, step=step)

                if step % cfg.train.save_every == 0:
                    ckpt_path = output_dir / f"step_{step}.pt"
                    torch.save({"model": raw_model.state_dict(), "step": step, "config": cfg}, ckpt_path)
            except torch.cuda.OutOfMemoryError:
                if cfg.train.auto_batch_size and micro_chunk_size > 1:
                    torch.cuda.empty_cache()
                    micro_chunk_size = max(1, micro_chunk_size // 2)
                    print(
                        f"[radiance] CUDA OOM at step {step}, backing off micro_chunk_size to "
                        f"{micro_chunk_size} and retrying."
                    )
                    wandb.log({"train/oom_backoff": micro_chunk_size}, step=step)
                else:
                    # Exit cleanly instead of raising, so a W&B sweep records this run as
                    # finished (e.g. loss/mem too high for this config) rather than crashed.
                    print(f"[radiance] CUDA OOM at step {step}, ending run early.")
                    wandb.log({"train/oom": True}, step=step)
                    give_up = True

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
