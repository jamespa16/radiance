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
from radiance.model import DenseTransformer, padded_vocab_size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> LambdaLR:
    warmup_steps = cfg.train.warmup_steps
    max_steps = cfg.train.max_steps
    min_lr_ratio = cfg.train.min_lr_ratio

    def lr_lambda(step: int) -> float:
        # step + 1: LambdaLR evaluates the lambda at step 0 to set the LR for the *first*
        # optimizer.step(), so a plain step/warmup_steps ramp spends that entire first step at
        # lr=0 — a wasted update. Ramping over (step + 1) makes the first step take a real,
        # nonzero LR and still reach the full LR exactly at the end of warmup.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        # Decay to min_lr_ratio * lr rather than all the way to 0: the last steps of a run at a
        # ~0 LR contribute nothing, and a small floor is standard (Llama/Chinchilla both keep one).
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decayed / non-decayed groups.

    AdamW over model.parameters() applies weight decay uniformly, which decays RMSNorm gains and
    every bias. Those are 1-D scale/shift parameters with no "shrink toward zero is a useful
    prior" interpretation — decaying them just fights the norm layers. Standard practice
    (GPT-2/Llama/nanoGPT) is to decay only the >=2-D weight matrices, which is what this does.
    The tied token_emb/lm_head weight is 2-D and stays in the decayed group, matching those
    references.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW with exp_avg/exp_avg_sq kept in pinned CPU memory instead of on `device`, freeing the
    ~2x num_params fp32 moment-buffer VRAM cost for the rest of the run. Params/grads stay
    GPU-resident the whole time — only a grad copy (down) and the resulting update (up) cross PCIe,
    once per step() call rather than once per forward/backward — so this only touches optimizer
    bookkeeping, never the forward path or torch.compile's captured graph. Used as auto_batch_size's
    second OOM-recovery tier once chunk-size backoff (micro_chunk_size == 1) is exhausted — see
    migrate_optimizer_to_cpu_offload and the OOM handler in train()."""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        device: str,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))
        self._device = device
        self._is_cuda = device.split(":")[0] == "cuda"

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, weight_decay = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            params, grads_cpu, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, device="cpu", dtype=torch.float32).pin_memory()
                    state["exp_avg_sq"] = torch.zeros_like(p, device="cpu", dtype=torch.float32).pin_memory()
                    state["grad_cpu"] = torch.empty_like(p, device="cpu", dtype=torch.float32).pin_memory()
                    state["step"] = 0
                state["grad_cpu"].copy_(p.grad, non_blocking=True)
                state["step"] += 1
                params.append(p)
                grads_cpu.append(state["grad_cpu"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])

            if not params:
                continue

            # Every grad copy above was issued non_blocking against a pinned buffer; sync once here
            # (rather than per-param) before touching them on the CPU side.
            if self._is_cuda:
                torch.cuda.synchronize(self._device)

            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads_cpu, alpha=1 - beta1)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_cpu, grads_cpu, value=1 - beta2)

            for p, exp_avg, exp_avg_sq, step in zip(params, exp_avgs, exp_avg_sqs, steps):
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                update = (exp_avg / bias_correction1) / denom
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update.to(p.device, non_blocking=True), alpha=-lr)

            if self._is_cuda:
                torch.cuda.synchronize(self._device)

        return loss


def migrate_optimizer_to_cpu_offload(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module, cfg: Config, device: str
) -> CPUOffloadAdamW:
    """Swap a live AdamW for CPUOffloadAdamW, migrating any existing exp_avg/exp_avg_sq/step per
    param onto pinned CPU tensors instead of resetting momentum. Params with no prior state (e.g.
    training OOM'd before its first successful step) are left to lazy-init on first step(), matching
    fresh-AdamW behavior."""
    new_optimizer = CPUOffloadAdamW(
        build_param_groups(model, cfg.train.weight_decay),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        device=device,
    )
    # LambdaLR.__init__ requires 'initial_lr' already present in param_groups whenever it's
    # constructed with last_epoch != -1 (its "resuming a schedule" path) — copy it over from the old
    # optimizer so the caller can rebuild the scheduler against new_optimizer at the current step.
    for old_group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
        if "initial_lr" in old_group:
            new_group["initial_lr"] = old_group["initial_lr"]
    for p, old_state in optimizer.state.items():
        if "exp_avg" not in old_state:
            continue
        step = old_state["step"]
        new_optimizer.state[p] = {
            "exp_avg": old_state["exp_avg"].detach().to("cpu", dtype=torch.float32).pin_memory(),
            "exp_avg_sq": old_state["exp_avg_sq"].detach().to("cpu", dtype=torch.float32).pin_memory(),
            "grad_cpu": torch.empty_like(p, device="cpu", dtype=torch.float32).pin_memory(),
            "step": step.item() if torch.is_tensor(step) else step,
        }
    return new_optimizer


def save_checkpoint(
    path: Path,
    raw_model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    step: int,
    cfg: Config,
) -> None:
    """Write a resumable checkpoint.

    Everything needed to continue the run goes in, not just the weights: without the optimizer's
    moment buffers a resumed run restarts AdamW from zero momentum, which shows up as a visible
    loss spike, and without the scheduler/step the LR trajectory restarts from warmup. `config` is
    the full Config object (pickled), which is what generate.py reads back.

    Not captured: the DataLoader's position, and RNG state. Those two trade off against each other,
    and the resolution here favours data freshness — train() re-seeds off the resumed step, so the
    loader draws a *different* shuffle order rather than replaying batches the run already trained
    on (which is what restoring RNG state verbatim would cause). The cost is that dropout draws a
    different mask stream, so a resumed run is statistically equivalent to an uninterrupted one
    rather than bit-identical. With dropout disabled, resume reproduces an uninterrupted run
    exactly — model weights, AdamW moments, and LR schedule all match to the bit.
    """
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def find_resume_checkpoint(cfg: Config) -> Path | None:
    """Resolve cfg.train.resume_from to an actual checkpoint path.

    "auto" picks the highest-numbered step_*.pt in output_dir, so a config can be re-launched
    unchanged after an interruption and simply continue. Returns None when there's nothing to
    resume from (including "auto" against an empty/missing output_dir, which is a fresh run, not
    an error) — an explicit path that doesn't exist *is* an error, since silently starting a
    50-hour run from scratch is far worse than failing loudly.
    """
    if not cfg.train.resume_from:
        return None
    if cfg.train.resume_from != "auto":
        path = Path(cfg.train.resume_from)
        if not path.exists():
            raise FileNotFoundError(f"train.resume_from={cfg.train.resume_from!r} does not exist")
        return path
    candidates = sorted(
        Path(cfg.train.output_dir).glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return candidates[-1] if candidates else None


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard causal-LM loss: token t's logits predict token t+1.

    The shift is done on the *labels* (cheap, one int64 row) rather than by slicing the logits.
    Slicing logits[:, :-1] yields a non-contiguous view whose .contiguous()/.view() forces a full
    copy of the (batch, seq, vocab_size) tensor — the single largest activation in the model — on
    every forward. Padding the labels with ignore_index instead lets logits.view(-1, vocab_size)
    be a free reshape of the already-contiguous tensor, and cross_entropy skips the padded
    positions, giving a numerically identical mean over exactly the same (seq - 1) targets.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(
    model: DenseTransformer,
    val_loader,
    device: str,
    device_type: str,
    dtype: torch.dtype,
    max_batches: int | None = None,
) -> float:
    """Mean LM loss over the validation loader, capped at max_batches batches.

    The cap matters because this runs every eval_every steps: uncapped, a large validation split
    (or a streaming one, which has no length at all) makes each eval cost a meaningful fraction of
    the run. A fixed batch count also keeps val/loss comparable across configs whose val split
    sizes differ. max_batches=None keeps the original full-pass behavior.
    """
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32):
            logits, _, _, _ = model(input_ids)
            loss = compute_loss(logits, input_ids)
        total += loss.item()
        count += 1
    model.train()
    return total / count if count else float("nan")


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

    vocab_size = padded_vocab_size(len(tokenizer), cfg.model.vocab_pad_multiple)
    if vocab_size != len(tokenizer):
        print(
            f"[radiance] padding vocab {len(tokenizer):,} -> {vocab_size:,} "
            f"(multiple of {cfg.model.vocab_pad_multiple}) for tensor-core-aligned lm_head matmuls"
        )
    raw_model = DenseTransformer(cfg.model, vocab_size=vocab_size).to(device)

    if cfg.train.auto_batch_size:
        if device_type == "cuda":
            if cfg.train.target_effective_batch_size is None:
                cfg.train.target_effective_batch_size = cfg.train.effective_batch_size
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
        # Active (not flat) param count: Chinchilla-style scaling assumes every parameter multiplies
        # against every token, which is false for MoE — most expert params aren't touched by most
        # tokens. num_active_parameters() equals num_parameters() when no MoE layers exist.
        target_tokens = cfg.train.tokens_per_param * raw_model.num_active_parameters()
        cfg.train.max_steps = max(1, round(target_tokens / tokens_per_step))
        print(
            f"[radiance] tokens_per_param={cfg.train.tokens_per_param} over "
            f"{raw_model.num_active_parameters():,} active params ({raw_model.num_parameters():,} total) "
            f"-> max_steps={cfg.train.max_steps:,} ({target_tokens:,.0f} tokens at {tokens_per_step:,} tokens/step, "
            f"batch_size={cfg.train.batch_size} x grad_accum_steps={cfg.train.grad_accum_steps} "
            f"= effective_batch_size={cfg.train.effective_batch_size})"
        )

    optimizer = AdamW(
        build_param_groups(model, cfg.train.weight_decay),
        lr=cfg.train.lr,
        fused=(device_type == "cuda"),
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    if cfg.train.compile:
        # mode="reduce-overhead" captures the backward pass as a CUDA graph. If a param's .grad is
        # still None going into the first compiled backward (optimizer.zero_grad(set_to_none=True)'s
        # default), that backward allocates the .grad tensor *inside* the graph's own memory pool
        # during capture; any later replay within the same accumulated step (grad_accum_steps > 1, or
        # OOM-driven micro_chunk splitting) then legally reuses/overwrites that pool before the prior
        # chunk's contribution has been consumed, corrupting the accumulated gradient. Preallocating
        # stable, zeroed .grad buffers here — before any compiled backward runs — keeps them outside
        # the graph's pool for the rest of the run; zero_grad below is switched to zero them in place
        # instead of freeing them back to None, so they stay stable across every step too.
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)

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
            "num_active_parameters": raw_model.num_active_parameters(),
            "train/auto_batch_size": cfg.train.auto_batch_size,
            "train/batch_size": cfg.train.batch_size,
            "train/grad_accum_steps": cfg.train.grad_accum_steps,
        },
        step=0,
    )

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    resume_path = find_resume_checkpoint(cfg)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        # Re-seed off the resumed step before the train iterator is created below. The DataLoader's
        # shuffle order is drawn from the global RNG at iteration time, so without this a resumed
        # run replays exactly the batches the original run already trained on — measurably slower
        # progress than an uninterrupted run. Offsetting the seed draws a different order instead.
        # (This is a mitigation, not exact resumption: the loader's true position isn't recoverable,
        # least of all for a streaming dataset.)
        set_seed(cfg.train.seed + step)
        print(f"[radiance] resuming from {resume_path} at step {step:,}/{cfg.train.max_steps:,}")
        if step >= cfg.train.max_steps:
            print("[radiance] checkpoint is already at/past max_steps; nothing to do.")
            wandb.finish()
            return

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
    # Sticky, like micro_chunk_size: once an OOM forces the optimizer's moment buffers to pinned CPU
    # memory (see the OOM handler below), it stays there for the rest of the run.
    cpu_offload_active = False

    while step < cfg.train.max_steps and not give_up:
        will_log = (step + 1) % cfg.train.log_every == 0
        step_done = False

        while not step_done and not give_up:
            if will_log:
                accum_loss = torch.zeros((), device=device)
                accum_lm_loss = torch.zeros((), device=device)
                accum_ponder_cost = torch.zeros((), device=device)
                accum_mean_loop_depth = torch.zeros((), device=device)
                accum_moe_aux_loss = torch.zeros((), device=device)

            try:
                # set_to_none=False when compiled: keeps .grad buffers stable/preallocated (see the
                # warmup above) rather than freeing them back to None, which would reintroduce the
                # CUDA-graph-pool allocation bug on the next step's first backward.
                optimizer.zero_grad(set_to_none=not cfg.train.compile)
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
                            logits, ponder_cost, mean_loop_depth, moe_aux_loss = model(chunk)
                            lm_loss = compute_loss(logits, chunk)
                            chunk_loss = (
                                lm_loss
                                + cfg.model.ponder_weight * ponder_cost
                                + cfg.model.moe_aux_loss_weight * moe_aux_loss
                            )

                        scaler.scale(chunk_loss * chunk_weight).backward()

                        if will_log:
                            accum_loss += chunk_loss.detach() * chunk_weight
                            accum_lm_loss += lm_loss.detach() * chunk_weight
                            accum_ponder_cost += ponder_cost.detach() * chunk_weight
                            accum_mean_loop_depth += mean_loop_depth.detach() * chunk_weight
                            accum_moe_aux_loss += moe_aux_loss.detach() * chunk_weight

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
                            "train/moe_aux_loss": accum_moe_aux_loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/micro_chunk_size": micro_chunk_size,
                        },
                        step=step,
                    )

                if val_loader is not None and step % cfg.train.eval_every == 0:
                    val_loss = evaluate(
                        model, val_loader, device, device_type, dtype, cfg.train.eval_max_batches
                    )
                    wandb.log({"val/loss": val_loss}, step=step)

                if step % cfg.train.save_every == 0:
                    save_checkpoint(
                        output_dir / f"step_{step}.pt", raw_model, optimizer, scheduler, scaler, step, cfg
                    )
            except torch.cuda.OutOfMemoryError:
                # An OOM anywhere at or after scaler.unscale_() below leaves this optimizer marked
                # as already-unscaled for the current scaler generation, so the retry's own
                # unscale_ would raise "unscale_() has already been called on this optimizer since
                # the last update()" — an uncaught crash exactly when the backoff is supposed to be
                # saving the run. update(get_scale()) resets that per-optimizer bookkeeping without
                # disturbing the scale factor (a bare update() would treat the aborted step as a
                # successful one and grow it). No-op unless dtype is fp16, since bf16/fp32 run with
                # the scaler disabled.
                if scaler.is_enabled():
                    scaler.update(scaler.get_scale())
                if cfg.train.auto_batch_size and micro_chunk_size > 1:
                    torch.cuda.empty_cache()
                    micro_chunk_size = max(1, micro_chunk_size // 2)
                    print(
                        f"[radiance] CUDA OOM at step {step}, backing off micro_chunk_size to "
                        f"{micro_chunk_size} and retrying."
                    )
                    wandb.log({"train/oom_backoff": micro_chunk_size}, step=step)
                elif cfg.train.auto_batch_size and not cpu_offload_active:
                    # Chunk-size backoff alone can't help once it's already at its floor: that tier
                    # only attacks activation memory, which scales with batch size, not the fixed
                    # cost of AdamW's exp_avg/exp_avg_sq. Move those to pinned CPU memory instead.
                    optimizer = migrate_optimizer_to_cpu_offload(optimizer, model, cfg, device)
                    # Reconstructed (not resumed) against the new optimizer so the LR trajectory
                    # continues exactly as if training had never paused - LambdaLR only needs the
                    # lambda(s) it was built with and the step to resume from.
                    scheduler = LambdaLR(optimizer, scheduler.lr_lambdas[0], last_epoch=step - 1)
                    # Only reclaims anything once the old optimizer (holding the GPU-resident
                    # exp_avg/exp_avg_sq we just migrated off of) has lost its last reference above.
                    torch.cuda.empty_cache()
                    cpu_offload_active = True
                    print(f"[radiance] CUDA OOM at step {step}, offloading optimizer state to CPU and retrying.")
                    wandb.log({"train/cpu_offload": True}, step=step)
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
