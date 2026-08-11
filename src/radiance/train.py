from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import LambdaLR

from radiance.config import Config, load_config, resolve_device, resolve_dtype
from radiance.data import build_dataloaders, build_tokenizer
from radiance.model import DenseTransformer, padded_vocab_size
from radiance.optim import build_optimizer, migrate_optimizer_to_cpu_offload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> LambdaLR:
    warmup_steps = cfg.train.warmup_steps
    max_steps = cfg.train.max_steps
    min_lr_ratio = cfg.train.min_lr_ratio
    schedule = cfg.train.lr_schedule
    if schedule not in ("cosine", "wsd"):
        raise ValueError(f"Unknown train.lr_schedule {schedule!r}, expected 'cosine' or 'wsd'")
    # Precomputed rather than derived per call: lr_lambda runs once per optimizer step, and this
    # is also the value the "wsd" branch needs to know where the stable phase ends.
    decay_steps = max(1, round(max_steps * cfg.train.wsd_decay_ratio))
    decay_start = max(warmup_steps, max_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        # step + 1: LambdaLR evaluates the lambda at step 0 to set the LR for the *first*
        # optimizer.step(), so a plain step/warmup_steps ramp spends that entire first step at
        # lr=0 — a wasted update. Ramping over (step + 1) makes the first step take a real,
        # nonzero LR and still reach the full LR exactly at the end of warmup.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if schedule == "wsd":
            # Warmup-stable-decay: hold full LR, then decay only over the final wsd_decay_ratio.
            # The stable phase's LR doesn't depend on max_steps, which is what lets a run be
            # extended or branched from a mid-training checkpoint without the earlier steps having
            # been trained on the "wrong" schedule — cosine's shape is a function of max_steps, so
            # changing it invalidates everything before the change.
            if step < decay_start:
                return 1.0
            progress = min(1.0, (step - decay_start) / max(1, max_steps - decay_start))
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(progress))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        # Decay to min_lr_ratio * lr rather than all the way to 0: the last steps of a run at a
        # ~0 LR contribute nothing, and a small floor is standard (Llama/Chinchilla both keep one).
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

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
            # Which algorithm produced that state. Muon's per-param state is a single momentum
            # buffer where AdamW's is two moments, so loading one into the other silently produces
            # a wrong (or shape-mismatched) resume — see train()'s resume block, which resets
            # optimizer state with a warning rather than crashing when these disagree.
            "optimizer_type": cfg.train.optimizer,
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


def compute_mtp_loss(
    model: DenseTransformer, mtp_hidden: tuple[torch.Tensor, ...] | None, input_ids: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy over the auxiliary multi-token-prediction heads.

    Head `d` (1-indexed) predicts the token d+1 positions ahead, so its labels are input_ids shifted
    left by d+1 with the tail padded to ignore_index — the same label-shifting trick compute_loss
    uses, for the same reason (never slice the logits).

    Heads are projected and reduced one at a time so only one (batch, seq, vocab_size) tensor is
    live in the loop, rather than materialising all of them and then reducing.
    """
    if not mtp_hidden:
        return input_ids.new_zeros((), dtype=torch.float32)

    total = None
    for depth, hidden in enumerate(mtp_hidden, start=1):
        shift = depth + 1
        labels = torch.cat(
            [input_ids[:, shift:], input_ids.new_full((input_ids.size(0), shift), -100)], dim=1
        )
        logits = model._project_logits(hidden)
        head_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        total = head_loss if total is None else total + head_loss
    return total / len(mtp_hidden)


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard causal-LM loss: token t's logits predict token t+1. Returns (lm_loss, z_loss).

    The shift is done on the *labels* (cheap, one int64 row) rather than by slicing the logits.
    Slicing logits[:, :-1] yields a non-contiguous view whose .contiguous()/.view() forces a full
    copy of the (batch, seq, vocab_size) tensor — the single largest activation in the model — on
    every forward. Padding the labels with ignore_index instead lets logits.view(-1, vocab_size)
    be a free reshape of the already-contiguous tensor, and cross_entropy skips the padded
    positions, giving a numerically identical mean over exactly the same (seq - 1) targets.

    z_loss is the log-Z regulariser mean(logsumexp(logits)^2) (PaLM/Chinchilla), returned
    separately so callers decide whether to apply it: train() adds cfg.model.z_loss_weight * z_loss
    to the training objective, while evaluate() discards it so val/loss stays a pure LM number and
    remains comparable across configurations — exactly how ponder_cost and moe_aux_loss are already
    handled.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    lm_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    # Reduced over the vocab first, then masked — rather than flat_logits[mask], which would
    # materialise a near-full copy of the largest activation in the model just to drop one row per
    # sequence. logsumexp is internally max-subtracted so it's stable in the autocast dtype; only
    # the (n_tokens,) result is upcast for the square/mean.
    keep = (flat_labels != -100).to(flat_logits.dtype)
    z = torch.logsumexp(flat_logits, dim=-1).float()
    z_loss = (z.square() * keep).sum() / keep.sum().clamp(min=1)
    return lm_loss, z_loss


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
            out = model(input_ids)
            loss, _ = compute_loss(out.logits, input_ids)  # z_loss discarded: val/loss stays pure LM
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
    # eos_id is what recovers packed document boundaries for doc_attention_mask (see
    # model.document_ids) — data.py joins documents with exactly this token.
    raw_model = DenseTransformer(cfg.model, vocab_size=vocab_size, eos_id=tokenizer.eos_token_id).to(device)

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

    stochastic_depth = cfg.model.loop_count_min is not None
    compile_mode = "reduce-overhead"
    if cfg.train.compile and stochastic_depth:
        # Stochastic loop depth means the Python `for` over the loop body runs a different number
        # of times per step, so dynamo traces one graph per distinct count. The default cache limit
        # (8) would be exceeded by a wide range and silently fall back to eager for the rest of the
        # run. Raise it to cover the range with headroom for the separate eval/generate traces.
        span = (cfg.model.loop_count_max or cfg.model.loop_count) - cfg.model.loop_count_min + 1
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 4 * span + 8)

    if cfg.train.compile and (stochastic_depth or cfg.model.grad_checkpoint):
        # mode="reduce-overhead" captures each graph as a CUDA graph sharing one memory pool, which
        # assumes a *static* execution graph. Two independent things give that up:
        #
        # 1. Stochastic loop depth: replaying a different loop count overwrites the previous
        #    graph's gradient tensors ("accessing gradient tensor output of CUDAGraphs that has
        #    been overwritten by a subsequent run").
        # 2. grad_checkpoint: torch.utils.checkpoint recomputes each block during backward:
        #    AOTAutograd partitions that recompute into its own graph segment, and under
        #    reduce-overhead each segment gets captured as its own CUDA graph against the same
        #    static pool — so the recompute's outputs overwrite a tensor the original forward's
        #    backward still needs ("accessing tensor output of CUDAGraphs that has been overwritten
        #    by a subsequent run"). Same failure mode as (1), different trigger.
        #
        # Drop to plain inductor in either case: still compiled, just without the CUDA-graph
        # capture that a varying shape or a recompute makes unsound.
        compile_mode = None
        reasons = []
        if stochastic_depth:
            reasons.append("stochastic loop depth")
        if cfg.model.grad_checkpoint:
            reasons.append("grad_checkpoint")
        print(
            f"[radiance] {' and '.join(reasons)} on, so compiling without CUDA graphs "
            "(mode=None instead of 'reduce-overhead')."
        )

    model = torch.compile(raw_model, mode=compile_mode) if cfg.train.compile else raw_model

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

    optimizer = build_optimizer(model, cfg, device)
    scheduler = build_lr_scheduler(optimizer, cfg)
    print(
        f"[radiance] optimizer={cfg.train.optimizer} "
        + (
            f"(muon_lr={cfg.train.muon_lr} over {sum(1 for g in optimizer.param_groups if g.get('algorithm') == 'muon' for _ in g['params'])} "
            f"tensors, adamw lr={cfg.train.lr} over the rest)"
            if cfg.train.optimizer == "muon"
            else f"(lr={cfg.train.lr})"
        )
    )
    if cfg.model.hyper_conn_streams > 1:
        # Worth its own line: this LR is the difference between hyper-connections helping and
        # actively hurting, and it is easy to leave at `lr` by accident. See TrainConfig.
        print(
            f"[radiance] hyper-connections: {cfg.model.hyper_conn_streams} residual streams "
            f"(dynamic={cfg.model.hyper_conn_dynamic}, lr={cfg.train.hyper_conn_lr_resolved})"
        )

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
        # Checkpoints predating optimizer_type were all AdamW.
        saved_optimizer = ckpt.get("optimizer_type", "adamw")
        if saved_optimizer == cfg.train.optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            # Warn and continue rather than raise: the weights and LR schedule are still perfectly
            # resumable, and refusing to load a checkpoint because the optimizer was switched would
            # be a worse failure than restarting momentum. But say so loudly — the first steps after
            # this will show the same loss bump a from-scratch optimizer always causes.
            print(
                f"[radiance] WARNING: checkpoint was saved with optimizer={saved_optimizer!r} but "
                f"this run uses optimizer={cfg.train.optimizer!r}. Model weights and LR schedule "
                "are restored; optimizer state is NOT (their state layouts differ), so momentum "
                "restarts from zero and you should expect a transient loss spike."
            )
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
                accum_z_loss = torch.zeros((), device=device)
                accum_mtp_loss = torch.zeros((), device=device)

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
                            out = model(chunk)
                            ponder_cost, mean_loop_depth = out.ponder_cost, out.mean_loop_depth
                            moe_aux_loss = out.moe_aux_loss
                            lm_loss, z_loss = compute_loss(out.logits, chunk)
                            mtp_loss = compute_mtp_loss(raw_model, out.mtp_hidden, chunk)
                            chunk_loss = (
                                lm_loss
                                + cfg.model.ponder_weight * ponder_cost
                                + cfg.model.moe_aux_loss_weight * moe_aux_loss
                                + cfg.model.z_loss_weight * z_loss
                                + cfg.model.mtp_weight * mtp_loss
                            )

                        scaler.scale(chunk_loss * chunk_weight).backward()

                        if will_log:
                            accum_loss += chunk_loss.detach() * chunk_weight
                            accum_lm_loss += lm_loss.detach() * chunk_weight
                            accum_ponder_cost += ponder_cost.detach() * chunk_weight
                            accum_mean_loop_depth += mean_loop_depth.detach() * chunk_weight
                            accum_moe_aux_loss += moe_aux_loss.detach() * chunk_weight
                            accum_z_loss += z_loss.detach() * chunk_weight
                            accum_mtp_loss += mtp_loss.detach() * chunk_weight

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                # Loss-free MoE load balancing (cfg.model.moe_balance). Deliberately here rather
                # than inside forward: it mutates a buffer by an explicit rule with no gradient, so
                # keeping it outside the compiled region avoids a graph break on every micro-batch.
                # No-op unless MoE and bias balancing are both on.
                raw_model.update_expert_bias()
                step += 1
                step_done = True

                if will_log:
                    # Also to stdout: W&B was previously the only place a loss ever appeared, so a
                    # run with wandb.mode=disabled (sweeps, CI, quick A/Bs) produced no visible
                    # signal at all. Cheap — accum_* are already being .item()'d for the log below.
                    print(
                        f"[radiance] step {step:>6}/{cfg.train.max_steps} "
                        f"loss {accum_loss.item():.4f} lm {accum_lm_loss.item():.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                    wandb.log(
                        {
                            "train/loss": accum_loss.item(),
                            "train/lm_loss": accum_lm_loss.item(),
                            "train/ponder_cost": accum_ponder_cost.item(),
                            "train/mean_loop_depth": accum_mean_loop_depth.item(),
                            "train/moe_aux_loss": accum_moe_aux_loss.item(),
                            "train/z_loss": accum_z_loss.item(),
                            "train/mtp_loss": accum_mtp_loss.item(),
                            "train/expert_bias_spread": raw_model.expert_bias_spread(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/micro_chunk_size": micro_chunk_size,
                        },
                        step=step,
                    )

                if val_loader is not None and step % cfg.train.eval_every == 0:
                    val_loss = evaluate(
                        model, val_loader, device, device_type, dtype, cfg.train.eval_max_batches
                    )
                    print(f"[radiance] step {step:>6} val/loss {val_loss:.4f}", flush=True)
                    wandb.log({"val/loss": val_loss}, step=step)

                if step % cfg.train.save_every == 0:
                    # Keep only the latest checkpoint — remove previous ones before saving.
                    for old_ckpt in output_dir.glob("step_*.pt"):
                        old_ckpt.unlink()
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
                    previous_optimizer = optimizer
                    optimizer = migrate_optimizer_to_cpu_offload(optimizer, model, cfg, device)
                    if optimizer is not previous_optimizer:
                        # A *new* optimizer object (the plain-AdamW path swaps in CPUOffloadAdamW).
                        # Reconstructed, not resumed, so the LR trajectory continues exactly as if
                        # training had never paused - LambdaLR only needs the lambda(s) it was built
                        # with and the step to resume from. MuonWithAuxAdam instead flips its adamw
                        # groups to offloaded in place and returns itself, so its scheduler — which
                        # already points at those same param_groups — needs no rebuild.
                        scheduler = LambdaLR(optimizer, scheduler.lr_lambdas[0], last_epoch=step - 1)
                    del previous_optimizer
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
