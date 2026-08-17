from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

import wandb
from radiance import nvfp4
from radiance.config import (
    Config,
    load_config,
    resolve_device,
    resolve_dtype,
)
from radiance.data import build_dataloaders, build_tokenizer
from radiance.dpo_data import build_dpo_dataloaders
from radiance.sft_data import build_sft_dataloaders
from radiance.model import DenseTransformer, padded_vocab_size
from radiance.optim import build_optimizer, migrate_optimizer_to_cpu_offload

from .batching import (
    chunk_reduction_units,
    dpo_reference_reserve_bytes,
    estimate_batch_size,
    split_micro_batch,
)
from .checkpointing import find_resume_checkpoint, load_pretrained_weights, save_checkpoint
from .evaluation import evaluate
from .losses import (
    build_dpo_loss_fn,
    build_loss_fn,
    chunk_loss_and_metrics,
    note_dpo_z_loss_omitted,
    resolve_dpo_doc_mask,
    validate_post_training_config,
)
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


def resolve_compile_mode(raw_model: DenseTransformer, cfg: Config, device_type: str) -> str | None:
    """Which torch.compile mode this run gets: "reduce-overhead" (CUDA graphs) or None.

    mode="reduce-overhead" captures each graph as a CUDA graph sharing one memory pool, which
    assumes a *static* execution graph. Three independent things give that up:

    1. Stochastic loop depth (cfg.model.loop_count_min): replaying a different loop count
       overwrites the previous graph's gradient tensors ("accessing gradient tensor output of
       CUDAGraphs that has been overwritten by a subsequent run").
    2. grad_checkpoint: torch.utils.checkpoint recomputes each block during backward, AOTAutograd
       partitions that recompute into its own graph segment, and under reduce-overhead each
       segment is captured against the same static pool — so the recompute's outputs overwrite a
       tensor the original forward's backward still needs. Same failure mode as (1), different
       trigger.
    3. doc_attention_mask: the flex_attention BlockMask is rebuilt every step out of that batch's
       document boundaries (see DenseTransformer._doc_masks), so its tensors land at new addresses
       every step. CUDA graph trees treats them as *static* inputs — they reach the graph as
       lifted constants across the eager _doc_masks graph break — so a changed data pointer forces
       a re-record, and every re-record instantiates another cudaGraphExec whose memory the
       caching allocator never sees.

    (3) is the one that cost a run, and it is the one to be careful about, because unlike (1) and
    (2) it never raises: it just leaks and crawls. Measured on a 21-executed-block config
    (d_model 256, n_layers 6, loop_count 4) at batch 16 x grad_accum 2: **282 ms/step and +8 MB of
    process VRAM per step** against **87 ms/step and dead flat** at mode=None. torch's own
    torch.cuda.memory_reserved() stays flat through all of it — the growth is outside the caching
    allocator, which is why an earlier investigation that watched reserved memory concluded the
    model was clean and went looking in the DataLoader. Left on, it exhausts a 32 GB card in ~1600
    steps and quietly triples step time before it gets there.

    Dropping to plain inductor is nearly free: measured where CUDA graphs actually work (same
    config, doc masking off) it is 84.2 ms/step captured vs 82.7 ms/step not.
    """
    if not cfg.train.compile:
        return None

    stochastic_depth = cfg.model.loop_count_min is not None
    if stochastic_depth:
        # Stochastic loop depth means the Python `for` over the loop body runs a different number
        # of times per step, so dynamo traces one graph per distinct count. The default cache limit
        # (8) would be exceeded by a wide range and silently fall back to eager for the rest of the
        # run. Raise it to cover the range with headroom for the separate eval/generate traces.
        span = (cfg.model.loop_count_max or cfg.model.loop_count) - cfg.model.loop_count_min + 1
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 4 * span + 8)

    reasons = []
    if stochastic_depth:
        reasons.append("stochastic loop depth")
    if cfg.model.grad_checkpoint:
        reasons.append("grad_checkpoint")
    if raw_model.builds_doc_block_masks(device_type):
        reasons.append("doc_attention_mask")
    if not reasons:
        return "reduce-overhead"

    print(
        f"[radiance] {' and '.join(reasons)} on, so compiling without CUDA graphs "
        "(mode=None instead of 'reduce-overhead')."
    )
    return None


# Every scalar the step loop accumulates across a step's chunks and logs. One flat list rather
# than a set of named locals so the accumulation below is one loop over whatever the active mode's
# chunk actually reported, and the log block one comprehension: a mode that doesn't produce a term
# (DPO has no z_loss/mtp_loss; pretrain/SFT have no dpo_*) simply leaves it at the 0.0 it was
# initialised to, which is exactly what those series logged before.
_ACCUM_METRICS = (
    "loss",
    "lm_loss",
    "ponder_cost",
    "mean_loop_depth",
    "moe_aux_loss",
    "z_loss",
    "mtp_loss",
    "dpo_margin_accuracy",
    "dpo_reward_accuracy",
    "dpo_margin",
)


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
    validate_post_training_config(cfg)
    # Both before the model is built: validation so an unsupported combination fails before a
    # model is allocated, and the doc-mask resolution because DenseTransformer reads cfg.model by
    # reference and resolve_compile_mode's CUDA-graph decision asks the *built* model whether it
    # will build BlockMasks.
    resolve_dpo_doc_mask(cfg)
    note_dpo_z_loss_omitted(cfg)

    # eos_id is what recovers packed document boundaries for doc_attention_mask (see
    # model.document_ids) — data.py joins documents with exactly this token.
    raw_model = DenseTransformer(cfg.model, vocab_size=vocab_size, eos_id=tokenizer.eos_token_id).to(device)
    if cfg.train.native_bf16:
        # Parameters only, not buffers: RoPE's cos/sin cache and MoE's expert_bias carry no
        # optimizer state, so casting them buys no memory and would cost precision that nothing
        # downstream upcasts back (unlike RMSNorm's gain, which forward() already computes in
        # fp32 regardless of the gain's own storage dtype). Autograd gives each bf16 parameter a
        # bf16 .grad automatically, which is what lets optim.py's state allocators (already
        # dtype-matched to the parameter, see _new_state_like) fall out for free.
        for p in raw_model.parameters():
            p.data = p.data.to(torch.bfloat16)
        print("[radiance] train.native_bf16: parameters, gradients and optimizer moments stored in bf16")

    # Computed here (rather than only where it's consumed, near the bottom) so init_from below can
    # check "is this run resuming?" before deciding whether to apply it — resuming an interrupted
    # run of *this* config always takes priority over re-seeding from a different checkpoint.
    resume_path = find_resume_checkpoint(cfg)
    if resume_path is None and cfg.train.init_from:
        load_pretrained_weights(raw_model, cfg.train.init_from, cfg, vocab_size, device)
        print(f"[radiance] initialized model weights from {cfg.train.init_from} (fresh optimizer/scheduler/step)")

    if cfg.train.auto_batch_size:
        if device_type == "cuda":
            if cfg.train.target_effective_batch_size is None:
                cfg.train.target_effective_batch_size = cfg.train.effective_batch_size
            cfg.train.batch_size, cfg.train.grad_accum_steps = estimate_batch_size(
                raw_model, cfg, device, device_type, reserve_bytes=dpo_reference_reserve_bytes(cfg)
            )
        else:
            print(
                f"[radiance] auto_batch_size requires CUDA (device_type={device_type!r}); "
                "using configured batch_size/grad_accum_steps."
            )

    if cfg.model.fp4_linear:
        # Printed once, next to the other resolved-setting lines, because the failure mode of FP4
        # on an unsupported card is that everything falls back to bf16 and trains correctly at bf16
        # speed and quality while the config claims FP4 — a silently uninterpretable measurement.
        n_fp4 = sum(1 for m in raw_model.modules() if isinstance(m, nvfp4.FP4Linear))
        print(
            f"[radiance] nvfp4: {n_fp4} linears quantized "
            f"(grad_gemms={cfg.model.fp4_grad_gemms}, stochastic_rounding="
            f"{cfg.model.fp4_stochastic_rounding}, hadamard={cfg.model.fp4_hadamard}), "
            f"supported={nvfp4.nvfp4_supported(device)}"
        )

    compile_mode = resolve_compile_mode(raw_model, cfg, device_type)
    model = torch.compile(raw_model, mode=compile_mode) if cfg.train.compile else raw_model
    loss_fn = build_dpo_loss_fn(cfg) if cfg.dpo.enabled else build_loss_fn(cfg)

    # batch_size must be finalized (auto_batch_size, if any, already ran) before the DataLoader is built.
    if cfg.dpo.enabled:
        build_loader_fn = build_dpo_dataloaders
    elif cfg.sft.enabled:
        build_loader_fn = build_sft_dataloaders
    else:
        build_loader_fn = build_dataloaders
    train_loader, val_loader = build_loader_fn(cfg, tokenizer)

    if cfg.train.tokens_per_param is not None:
        # Same per-row unit `estimate_batch_size` uses above: respects any active sft.seq_len /
        # dpo.seq_len override and counts DPO's chosen+rejected concatenation as the two packed
        # rows it actually forwards.
        tokens_per_step = cfg.train.effective_batch_size * cfg.resolved_row_tokens
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
    # Prime the FP4 weight caches before the first forward — after any resume_from/init_from load,
    # so they describe the weights training actually starts from. Every later refresh happens after
    # optimizer.step(). No-op unless model.fp4_linear.
    nvfp4.refresh_fp4_weights(raw_model)
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
                accums = {name: torch.zeros((), device=device) for name in _ACCUM_METRICS}

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

                    # Which columns a batch carries and how a chunk's loss is built are the only
                    # things that differ between pretrain/SFT and DPO; everything the accumulation
                    # contract actually consists of — the chunk_weight normalization, the autocast
                    # block, the scaled backward, the weighted accumulation, and the enclosing
                    # try/OOM handling — lives here once, so a change to any of it cannot apply to
                    # one mode and silently miss the other.
                    # chunk_weight reconstructs exactly the loss one un-split
                    # micro_loss / grad_accum_steps backward would have produced, however many
                    # (possibly uneven) chunks the micro-batch got split into — see
                    # chunk_reduction_units for why the weight is each chunk's share of the
                    # reduction's denominator rather than its share of the rows. Summed per
                    # micro-batch rather than against cfg.train.batch_size so the weights are a
                    # true partition of 1/grad_accum_steps whatever the batch actually holds.
                    units = chunk_reduction_units(batch, cfg.dpo.enabled, cfg.sft.enabled, micro_chunk_size)
                    total_units = max(sum(units), 1)  # clamp mirrors _nll_and_logz's
                    for chunk, chunk_units in zip(
                        split_micro_batch(batch, cfg.dpo.enabled, cfg.sft.enabled, device, micro_chunk_size), units
                    ):
                        chunk_weight = chunk_units / total_units / grad_accum_steps
                        with torch.autocast(
                            device_type=device_type, dtype=dtype, enabled=dtype != torch.float32
                        ):
                            chunk_loss, metrics = chunk_loss_and_metrics(
                                model, raw_model, cfg, loss_fn, chunk
                            )

                        scaler.scale(chunk_loss * chunk_weight).backward()

                        if will_log:
                            for name, value in metrics.items():
                                accums[name] += value.detach() * chunk_weight

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
                # Re-quantize every FP4Linear's weight cache against the weights that just moved,
                # and advance the stochastic-rounding seed. Here for the same reason as the line
                # above — it writes buffers with no gradient, so keeping it out of the compiled
                # region avoids a graph break, and a validity check inside forward() would be a
                # Python branch dynamo guards on and recompiles for every step.
                #
                # Once per *optimizer* step is also the correct semantics rather than merely the
                # cheap one: every micro-batch of an accumulated step must be differentiated
                # against the same weights. Omitting this call fails silently — the forward would
                # keep using step-0 weights while the fp32 masters trained on, so the loss still
                # falls and then plateaus. No-op unless model.fp4_linear.
                nvfp4.refresh_fp4_weights(raw_model)
                step += 1
                step_done = True

                if will_log:
                    # Also to stdout: W&B was previously the only place a loss ever appeared, so a
                    # run with wandb.mode=disabled (sweeps, CI, quick A/Bs) produced no visible
                    # signal at all. Cheap — accum_* are already being .item()'d for the log below.
                    # reserved VRAM on the line too: a run whose memory climbs step over step is
                    # the signature of a torch.compile re-record (see compile_mode below), and it
                    # is otherwise invisible until the run dies of an OOM hundreds of steps later.
                    mem = (
                        f" mem {torch.cuda.memory_reserved(device) / 1e9:.2f}GB"
                        if device_type == "cuda"
                        else ""
                    )
                    print(
                        f"[radiance] step {step:>6}/{cfg.train.max_steps} "
                        f"loss {accums['loss'].item():.4f} lm {accums['lm_loss'].item():.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.3e}{mem}",
                        flush=True,
                    )
                    wandb.log(
                        {
                            **{f"train/{name}": v.item() for name, v in accums.items()},
                            "train/expert_bias_spread": raw_model.expert_bias_spread(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/micro_chunk_size": micro_chunk_size,
                        },
                        step=step,
                    )

                if val_loader is not None and step % cfg.train.eval_every == 0:
                    val_loss = evaluate(
                        model,
                        val_loader,
                        device,
                        device_type,
                        dtype,
                        cfg.train.eval_max_batches,
                        loss_fn,
                        sft=cfg.sft.enabled,
                        dpo=cfg.dpo.enabled,
                        dpo_beta=cfg.dpo.beta if cfg.dpo.enabled else None,
                        micro_chunk_size=micro_chunk_size,
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
