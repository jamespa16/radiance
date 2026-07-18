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
            logits = model(input_ids)
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
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    raw_model = DenseTransformer(cfg.model, vocab_size=len(tokenizer)).to(device)
    model = torch.compile(raw_model, mode="reduce-overhead") if cfg.train.compile else raw_model

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
    wandb.log({"num_parameters": raw_model.num_parameters()}, step=0)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    data_iter = iter(train_loader)

    while step < cfg.train.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        try:
            input_ids = batch["input_ids"].to(device)
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(input_ids)
                loss = compute_loss(logits, input_ids)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step % cfg.train.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
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
            # Exit cleanly instead of raising, so a W&B sweep records this run as
            # finished (e.g. loss/mem too high for this config) rather than crashed.
            print(f"[radiance] CUDA OOM at step {step}, ending run early.")
            wandb.log({"train/oom": True}, step=step)
            break

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
