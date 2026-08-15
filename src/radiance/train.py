from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import wandb
from radiance import nvfp4
from radiance.config import (
    Config,
    ModelConfig,
    doc_mask_is_inert_for_dpo,
    load_config,
    resolve_device,
    resolve_dtype,
)
from radiance.data import (
    build_dataloaders,
    build_dpo_dataloaders,
    build_sft_dataloaders,
    build_tokenizer,
    dpo_cache_exists,
)
from radiance.model import DenseTransformer, checkpoint_param_bytes, padded_vocab_size, sequence_logprob_sum
from radiance.optim import build_optimizer, migrate_optimizer_to_cpu_offload, muon_orthogonalize_reserve_bytes


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


def dpo_reference_reserve_bytes(cfg: Config) -> int:
    """VRAM estimate_batch_size should hold back for a DPO reference-checkpoint load that hasn't
    happened yet.

    On a DPO cache miss, build_dpo_dataloaders loads a second full transformer (the frozen
    reference checkpoint) alongside the already-resident policy model during data prep, which runs
    *after* estimate_batch_size has already committed to a batch_size sized off currently-free
    VRAM — see docs/post-training.md's "one resident model" note, which covers the training loop
    but not this precompute pass. Returns 0 when there's nothing to reserve for: DPO disabled, or
    the packed dataset (reference log-probs included) is already cached, so no second model will
    load. No 3x grad/optimizer multiplier is needed here, unlike the policy model's reservation
    below — the reference model runs once under torch.no_grad() and is never optimized.
    """
    if not (cfg.dpo.enabled and cfg.dpo.reference_checkpoint):
        return 0
    if dpo_cache_exists(cfg):
        return 0
    try:
        return checkpoint_param_bytes(cfg.dpo.reference_checkpoint)
    except (FileNotFoundError, KeyError):
        return 0


def estimate_batch_size(
    raw_model: DenseTransformer, cfg: Config, device: str, device_type: str, reserve_bytes: int = 0
) -> tuple[int, int]:
    """Conservative analytical batch_size/grad_accum_steps for cfg.train.auto_batch_size, derived
    from free VRAM and model size rather than an expensive live probe. CUDA-only — callers must
    check device_type == "cuda" before calling this.

    reserve_bytes holds back VRAM for something not yet resident that will load before training
    starts (currently: a DPO reference checkpoint on a cache miss — see
    dpo_reference_reserve_bytes) that this call's free-VRAM snapshot can't see yet."""
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
    # The FP4 weight caches (~1.125 bytes per covered parameter — 0.5 packed plus 0.0625 of scales,
    # in each of two orientations) are deliberately *not* added here, unlike the three buffers
    # above. FP4Linear.__init__ allocates them eagerly and train() has already run .to(device) by
    # the time this is called, so mem_get_info's free_bytes already excludes them; adding them
    # again would subtract the same memory twice and hand the run a needlessly small micro-batch.
    # raw_model.fp4_cache_bytes() reports the figure for anyone sizing a run by hand.
    #
    # Muon's per-shape orthogonalize() call (optim.py's _step_muon) allocates transient
    # Newton-Schulz buffers that scale with matrix *count* at a shape (capped at _MUON_MAX_STACK),
    # not with num_params — a fine-grained MoE model's narrow expert projections can spike here far
    # out of proportion to what not_yet_allocated_bytes above already reserves. See
    # muon_orthogonalize_reserve_bytes's docstring and docs/optim.md.
    ns_reserve_bytes = muon_orthogonalize_reserve_bytes(raw_model, cfg)
    usable_bytes = (
        max(0.0, free_bytes - not_yet_allocated_bytes - ns_reserve_bytes - reserve_bytes)
        * cfg.train.vram_safety_margin
    )

    activation_dtype_bytes = 4 if cfg.train.dtype == "fp32" else 2
    bytes_per_token = raw_model.activation_bytes_per_token(activation_dtype_bytes)
    max_tokens = usable_bytes / bytes_per_token
    # A DPO "row" (pair) forwards chosen AND rejected concatenated in one call — 2x the token width
    # of one SFT/pretrain row at the same batch_size — so the token budget above buys half as many
    # pairs as it would plain sequences. batch_size here counts pairs, not sequences, for DPO.
    # `resolved_row_tokens` respects any active sft.seq_len/dpo.seq_len override instead of assuming
    # the pretraining block width.
    batch_size = max(1, int(max_tokens // cfg.resolved_row_tokens))
    grad_accum_steps = max(1, math.ceil(cfg.train.target_effective_batch_size / batch_size))

    unit = "pairs" if cfg.dpo.enabled else "sequences"
    reserve_note = f", reserve_bytes={reserve_bytes / 1e9:.2f} GB (DPO reference checkpoint)" if reserve_bytes else ""
    ns_reserve_note = f", muon_ns_reserve={ns_reserve_bytes / 1e9:.2f} GB" if ns_reserve_bytes else ""
    print(
        f"[radiance] auto_batch_size: {free_bytes / 1e9:.2f} GB free{reserve_note}{ns_reserve_note}, {num_params:,} params, "
        f"vram_safety_margin={cfg.train.vram_safety_margin} -> batch_size={batch_size} {unit}, "
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


def _loop_conditioning_signature(cfg: ModelConfig) -> tuple[str, int]:
    """The (mode, variant count) that actually allocates per-iteration parameters.

    Raw ``loop_iter_conditioning`` is inert when ``loop_multiplier == 1`` (a single iteration has
    nothing to condition on), so all three modes collapse to the same parameter allocation there.
    """
    if cfg.loop_iter_conditioning == "none" or cfg.loop_multiplier == 1:
        return ("none", 1)
    return (cfg.loop_iter_conditioning, cfg.loop_multiplier)


def load_pretrained_weights(raw_model: DenseTransformer, path: str, cfg: Config, vocab_size: int, device: str) -> None:
    """Load model weights only from a checkpoint (train.init_from) — no optimizer/scheduler/step.

    The counterpart to find_resume_checkpoint's "continue this exact run" behavior: this seeds a
    *new* run (e.g. SFT) from a previously trained model's weights, so the caller builds a fresh
    optimizer/scheduler from its own cfg.train afterward rather than restoring saved ones.

    Checks the checkpoint's saved model shape against this run's cfg.model before touching
    load_state_dict, since a mismatch there would otherwise surface as an opaque tensor-shape
    RuntimeError deep inside torch rather than a clear message naming the field that disagrees.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    source_cfg = ckpt["config"].model
    source_vocab = ckpt["model"]["token_emb.weight"].shape[0]
    mismatches = []
    if source_vocab != vocab_size:
        mismatches.append(f"vocab_size: checkpoint={source_vocab}, this run={vocab_size}")
    for field_name in (
        "d_model",
        "n_layers",
        "head_dim",
        "use_diff_attn",
        "hyper_conn_streams",
    ):
        source_val, this_val = getattr(source_cfg, field_name), getattr(cfg.model, field_name)
        if source_val != this_val:
            mismatches.append(f"model.{field_name}: checkpoint={source_val!r}, this run={this_val!r}")
    # Compare the resolved GQA count rather than the raw field: n_kv_heads=None means n_heads, so a
    # checkpoint written with the default (None) and a run that pins the resolved value must
    # still match, and only the resolved value determines the qkv projection's shape.
    source_gqa = source_cfg.n_kv_heads_resolved
    this_gqa = cfg.model.n_kv_heads_resolved
    if source_gqa != this_gqa:
        mismatches.append(
            f"model.n_kv_heads: checkpoint={source_gqa}, this run={this_gqa} (resolved; None -> n_heads)"
        )
    # Compare the *resolved* loop-conditioning allocation rather than the raw field: at
    # loop_multiplier == 1 the modes are all inert (a single iteration has nothing to condition
    # on), so a raw comparison would reject a checkpoint whose mode happened to differ.
    source_loop = _loop_conditioning_signature(source_cfg)
    this_loop = _loop_conditioning_signature(cfg.model)
    # RMSNorm._broadcast_legacy_gain lets a single 1-D gain seed every variant, so a
    # pre-conditioning checkpoint is a valid source for the norm_gains variant bank. It cannot
    # seed the lora branch (whose keys the source never had) or shrink an existing variant bank
    # back to 1-D.
    if source_loop != this_loop and not (source_loop == ("none", 1) and this_loop[0] == "norm_gains"):
        mismatches.append(
            "model.loop_iter_conditioning: "
            f"checkpoint={source_loop[0]!r} (variants={source_loop[1]}), "
            f"this run={this_loop[0]!r} (variants={this_loop[1]}) "
            "(resolved; inert when loop_multiplier == 1)"
        )
    if (
        source_loop[0] == this_loop[0] == "lora"
        and source_cfg.loop_lora_rank != cfg.model.loop_lora_rank
    ):
        mismatches.append(
            f"model.loop_lora_rank: checkpoint={source_cfg.loop_lora_rank}, "
            f"this run={cfg.model.loop_lora_rank}"
        )
    if mismatches:
        raise ValueError(
            f"train.init_from={path!r} has an incompatible model shape:\n  "
            + "\n  ".join(mismatches)
            + "\ninit_from loads weights into an already-constructed model, so shapes must match exactly."
        )
    raw_model.load_state_dict(ckpt["model"])


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
        # Same single-pass formulation as compute_loss, for the same reason: F.cross_entropy is on
        # autocast's fp32 list, so it upcast this head's whole (batch, seq, vocab_size) tensor
        # before reducing it. Each head materialises one of those, so the saving is per head.
        head_loss, _ = _nll_and_logz(logits.view(-1, logits.size(-1)), labels.view(-1))
        total = head_loss if total is None else total + head_loss
    return total / len(mtp_hidden)


def _nll_and_logz(flat_logits: torch.Tensor, flat_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean NLL over non-ignored positions, and the mean of logsumexp(logits)^2, from ONE pass.

    cross_entropy is exactly `logsumexp(x) - x[label]`, and the z-loss regulariser squares that
    same `logsumexp(x)`. Computing them separately — F.cross_entropy plus a torch.logsumexp — walks
    the largest tensor in the model twice in the forward and twice again in the backward. Deriving
    both from a single `z` halves that, and lets the whole thing stay in the compute dtype instead
    of tripping autocast's fp32 policy on log_softmax.
    """
    keep = flat_labels != -100
    # gather can't take -100, and those rows are masked out of both means below anyway.
    safe_labels = flat_labels.masked_fill(~keep, 0)
    z = torch.logsumexp(flat_logits, dim=-1).float()
    target_logit = flat_logits.gather(1, safe_labels.unsqueeze(1)).squeeze(1).float()

    keep_f = keep.to(torch.float32)
    # clamp: an all-ignored batch is degenerate (it needs seq_len < 2), and 0 beats the nan
    # F.cross_entropy returns there, which would poison the accumulated gradient.
    n_kept = keep_f.sum().clamp(min=1)
    # Identical to F.cross_entropy(ignore_index=-100) with the default 'mean' reduction, which also
    # divides by the kept count rather than the total.
    nll = ((z - target_logit) * keep_f).sum() / n_kept
    return nll, (z.square() * keep_f).sum() / n_kept


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

    **Both losses are derived from one logsumexp rather than two passes over the logits**, which is
    what makes this the cheap version. cross_entropy is exactly `logsumexp(x) - x[label]`, and
    z_loss squares that same `logsumexp(x)` — so calling F.cross_entropy *and* torch.logsumexp
    reduced the largest tensor in the model twice over, once inside cross_entropy's fused
    log_softmax and once again for z. Computing `z` once and gathering the target logit off it
    collapses that to a single reduction, forward and backward.

    Wrap this in torch.compile (see `build_loss_fn`) before judging it: inductor fuses the whole
    thing into one pass and keeps the reduction in fp32 without ever materialising an fp32 copy of
    the (batch, seq, vocab_size) logits. Measured at batch 32 x seq 512 x vocab 50304, fwd+bwd:
    22.7 ms and +8.2 GB peak for the two-pass version, 17.0 ms eager here, **5.5 ms and +5.0 GB
    compiled** — and the compiled result is *closer* to an fp32 reference than the original
    (|dlm| 2.5e-5 vs 3.4e-3), because F.cross_entropy under autocast returned a bf16-rounded loss.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    # Reduced over the vocab first, then masked — rather than flat_logits[mask], which would
    # materialise a near-full copy of the largest activation in the model just to drop one row per
    # sequence. logsumexp is internally max-subtracted so it's stable in the autocast dtype.
    return _nll_and_logz(logits.view(-1, logits.size(-1)), labels.view(-1))


def compute_sft_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SFT analogue of compute_loss: the same causal shift, plus loss_mask shifted the same way
    and folded into the ignore positions, so only supervised (assistant-turn) tokens are scored.

    loss_mask is 1 at positions data.py's SFT pipeline marked as assistant-turn tokens (or the
    trailing EOS), 0 at prompt/user-turn tokens. Position i predicts input_ids[i+1], so it should
    be scored iff *that target* is supervised — i.e. iff loss_mask[i+1] == 1 — which is exactly
    what shifting loss_mask the same way labels are shifted gives.

    With an all-ones loss_mask this is bit-identical to compute_loss on the same inputs: it's a
    strict generalization, not a parallel reimplementation, and _nll_and_logz needs no change at
    all — it already treats -100 generically anywhere in the flat label tensor.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    labels = labels.masked_fill(shifted_mask == 0, -100)
    return _nll_and_logz(logits.view(-1, logits.size(-1)), labels.view(-1))


def compute_dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The standard DPO objective (Rafailov et al. 2023): -E[logsigmoid(beta * (policy_logratio -
    ref_logratio))], mean over the batch. Each *_logp is a per-row (batch,) sequence-summed
    log-probability (see model.sequence_logprob_sum), not a mean, and the reference values are read
    from data.py's precomputed cache rather than a second live model.

    Also returns three no_grad diagnostics, logged but not optimized:
    - margin: the mean implicit reward margin (how much more the policy separates chosen from
      rejected than the reference did, in beta-scaled log-odds units).
    - margin_accuracy: fraction of the batch where the policy ranks chosen above rejected by *more
      than the reference did* (logits > 0) — this is what the loss's sigmoid argument's sign
      measures, so it can sit near 50% early in training even when the policy already ranks chosen
      above rejected in absolute terms, since it's relative to the (untrained) reference.
    - reward_accuracy: the conventional reference-independent pairwise accuracy
      (policy_chosen_logp > policy_rejected_logp) — the TRL-style number most DPO writeups mean by
      "accuracy".
    """
    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        margin_accuracy = (logits > 0).float().mean()
        reward_accuracy = (policy_chosen_logp > policy_rejected_logp).float().mean()
        margin = (
            beta * (policy_chosen_logp - ref_chosen_logp) - beta * (policy_rejected_logp - ref_rejected_logp)
        ).mean()
    return loss, margin, margin_accuracy, reward_accuracy


def compute_dpo_loss_from_logits(
    chosen_logits: torch.Tensor,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_logits: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuses sequence_logprob_sum (x2) + compute_dpo_loss into one function so build_dpo_loss_fn
    can torch.compile it as a single unit — the same reasoning build_loss_fn already applies to
    compute_loss/compute_sft_loss."""
    policy_chosen_logp = sequence_logprob_sum(chosen_logits, chosen_ids, chosen_mask)
    policy_rejected_logp = sequence_logprob_sum(rejected_logits, rejected_ids, rejected_mask)
    return compute_dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta)


def build_loss_fn(cfg: Config):
    """compute_loss (or, under cfg.sft.enabled, compute_sft_loss), compiled when the run is compiled.

    Separate from the model's own torch.compile because the loss lives outside DenseTransformer:
    it consumes the (batch, seq, vocab_size) logits the model returns, and that tensor is large
    enough that whether its reductions get fused is worth ~20% of step time on a small-d_model,
    large-vocab config. Tied to cfg.train.compile so the CPU sanity-check path stays eager.

    Not used for cfg.dpo.enabled runs — see build_dpo_loss_fn, which wraps a different-shaped
    function (compute_dpo_loss_from_logits) since DPO's loss needs both chosen and rejected logits
    plus the cached reference log-probs, not just one logits tensor and its own input_ids.
    """
    fn = compute_sft_loss if cfg.sft.enabled else compute_loss
    return torch.compile(fn) if cfg.train.compile else fn


def build_dpo_loss_fn(cfg: Config):
    """compute_dpo_loss_from_logits, compiled when the run is compiled. DPO analogue of
    build_loss_fn — kept separate rather than folded into it because the two functions take
    different-shaped arguments (one logits tensor + input_ids/loss_mask vs. two logits tensors +
    two input_ids/loss_mask pairs + two cached reference log-probs)."""
    return torch.compile(compute_dpo_loss_from_logits) if cfg.train.compile else compute_dpo_loss_from_logits


@torch.no_grad()
def evaluate(
    model: DenseTransformer,
    val_loader,
    device: str,
    device_type: str,
    dtype: torch.dtype,
    max_batches: int | None = None,
    loss_fn=compute_loss,
    sft: bool = False,
    dpo: bool = False,
    dpo_beta: float | None = None,
    micro_chunk_size: int | None = None,
) -> float:
    """Mean loss over the validation loader, capped at max_batches batches.

    The cap matters because this runs every eval_every steps: uncapped, a large validation split
    (or a streaming one, which has no length at all) makes each eval cost a meaningful fraction of
    the run. A fixed batch count also keeps val/loss comparable across configs whose val split
    sizes differ. max_batches=None keeps the original full-pass behavior.

    sft=True pulls the batch's "loss_mask" and calls loss_fn with it (compute_sft_loss's
    signature). dpo=True pulls all 6 DPO batch columns, forwards chosen+rejected concatenated
    within each chunk, and calls loss_fn with compute_dpo_loss_from_logits's signature
    (dpo_beta required). Column selection and chunk splitting are the step loop's own
    split_micro_batch/chunk_reduction_units, shared here rather than re-derived, so a fix to either
    cannot land in the step loop and silently miss eval.
    Exactly one of sft/dpo should be set, matching whichever loss_fn build_loss_fn(cfg)/
    build_dpo_loss_fn(cfg) actually built.

    micro_chunk_size is train()'s (possibly OOM-shrunk) per-forward chunk, honored here so a
    backoffed run's eval forward carries exactly the memory the step loop already validated —
    most sharply for DPO, whose concatenated chosen+rejected forward is 2x the trained width.
    A batch is split the way the step loop splits a micro-batch, and each chunk's loss is
    re-weighted by its contribution to the whole-batch reduction (row count for DPO's row-mean;
    kept-position count for compute_loss, which divides by it) so multi-chunk eval reproduces the
    un-chunked value. micro_chunk_size=None (or >= the batch's row count) is a single chunk, i.e.
    the historical single forward bit-for-bit — a run that never backoffed (chunk == batch_size)
    evals exactly as before.

    The forward + loss assembly below stays local rather than also calling chunk_loss_and_metrics:
    that function needs raw_model (for compute_mtp_loss) and unconditionally folds
    ponder_cost/moe_aux_loss/z_loss/mtp_loss into its total, which val/loss deliberately excludes
    so it stays a pure LM number (see docs/train.md) — sharing it here would either drag those
    terms into eval or need a second, eval-only mode threaded through it for no benefit.
    """
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        row_count = batch["chosen_input_ids"].size(0) if dpo else batch["input_ids"].size(0)
        chunk_n = micro_chunk_size or row_count
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32):
            units = chunk_reduction_units(batch, dpo, sft, chunk_n)
            chunks = split_micro_batch(batch, dpo, sft, device, chunk_n)
            terms: list[tuple[float, int]] = []
            for chunk, chunk_units in zip(chunks, units):
                if dpo:
                    c_ids, r_ids = chunk["chosen_input_ids"], chunk["rejected_input_ids"]
                    b = c_ids.size(0)
                    out = model(torch.cat([c_ids, r_ids], dim=0))
                    chosen_logits, rejected_logits = out.logits.split(b, dim=0)
                    chunk_loss, _, _, _ = loss_fn(
                        chosen_logits, c_ids, chunk["chosen_loss_mask"],
                        rejected_logits, r_ids, chunk["rejected_loss_mask"],
                        chunk["ref_chosen_logprob"], chunk["ref_rejected_logprob"], dpo_beta,
                    )  # margin/accuracy discarded: val/loss stays the plain DPO objective
                elif sft:
                    out = model(chunk["input_ids"])
                    loss_val, _ = loss_fn(out.logits, chunk["input_ids"], chunk["loss_mask"])  # z_loss discarded
                    chunk_loss = loss_val
                else:
                    out = model(chunk["input_ids"])
                    loss_val, _ = loss_fn(out.logits, chunk["input_ids"])  # z_loss discarded: val/loss stays pure LM
                    chunk_loss = loss_val
                # .float() before .item(): a bf16/fp16 upcast is exact, and the multi-chunk combine
                # below must not accumulate in the autocast dtype.
                terms.append((chunk_loss.float().item(), chunk_units))
            loss = (
                terms[0][0]
                if len(terms) == 1  # the historical single forward, bit-for-bit
                # clamp mirrors _nll_and_logz's: an all-ignored batch is degenerate and 0 beats nan
                # (a no-op for DPO, whose units are always >= 1 row per chunk)
                else sum(l * u for l, u in terms) / max(sum(u for _, u in terms), 1)
            )
        total += loss
        count += 1
    model.train()
    return total / count if count else float("nan")


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

_DPO_BATCH_COLUMNS = (
    "chosen_input_ids",
    "chosen_loss_mask",
    "rejected_input_ids",
    "rejected_loss_mask",
    "ref_chosen_logprob",
    "ref_rejected_logprob",
)


def split_micro_batch(batch: dict, dpo: bool, sft: bool, device: str, micro_chunk_size: int) -> list[dict]:
    """One fetched micro-batch -> the list of per-forward chunks the step loop processes.

    The only thing that differs between modes is *which columns* a batch carries — 6 parallel
    tensors for DPO, `input_ids` (+ `loss_mask` for SFT) otherwise. All of them are chunked the
    same way, along dim 0 in lockstep, so every chunk holds exactly its own rows of every column.
    micro_chunk_size >= the batch's row count yields a single chunk, i.e. the un-split forward.

    Takes plain dpo/sft booleans rather than cfg so evaluate() can share this with the step loop —
    both callers already have exactly these two booleans in hand and nothing else off cfg.
    """
    if dpo:
        columns = _DPO_BATCH_COLUMNS
    elif sft:
        columns = ("input_ids", "loss_mask")
    else:
        columns = ("input_ids",)
    chunked = {c: batch[c].to(device).split(micro_chunk_size, dim=0) for c in columns}
    return [dict(zip(columns, rows)) for rows in zip(*chunked.values())]


def chunk_reduction_units(batch: dict, dpo: bool, sft: bool, micro_chunk_size: int) -> list[int]:
    """Each chunk's share of its micro-batch's loss reduction, in that reduction's own units.

    A chunk's loss is a *mean*, so recombining chunks into the value one un-split forward would
    have produced means weighting each by how much of the denominator it contributed — and the
    denominator differs by mode. `compute_loss`/`compute_sft_loss` reduce over kept (non-ignored)
    positions via `_nll_and_logz`, so the unit is scored tokens: `seq_len - 1` per row for
    pretrain (every position but the last has a target), and `loss_mask[:, 1:].sum()` for SFT,
    which is exactly the shifted mask `compute_sft_loss` folds into its labels. `compute_dpo_loss`
    reduces over pair rows, so there the unit is rows.

    Rows are a stand-in for tokens only when every row contributes the same number of them, which
    is true for pretrain and DPO and **false for SFT**, where per-row supervised lengths vary with
    the data. Weighting SFT by rows would make a split micro-batch's gradient a row-weighted
    average of per-chunk token-means instead of the micro-batch's token-mean — a real (if small)
    difference in what the step optimises, reachable whenever OOM backoff splits a micro-batch.
    `evaluate()` shares this same function to weight its own chunks.

    Computed from the CPU batch, before `split_micro_batch` moves it: reading a `.sum()` back off
    an accelerator would force a device sync per chunk, every step, in the hot loop.

    Takes plain dpo/sft booleans rather than cfg, for the same reason as split_micro_batch above.
    """
    if dpo:
        per_row = torch.ones(batch["chosen_input_ids"].size(0))
    elif sft:
        per_row = batch["loss_mask"][:, 1:].sum(dim=1)
    else:
        input_ids = batch["input_ids"]
        per_row = torch.full((input_ids.size(0),), input_ids.size(1) - 1)
    return [int(rows.sum()) for rows in per_row.split(micro_chunk_size)]


def chunk_loss_and_metrics(
    model, raw_model: DenseTransformer, cfg: Config, loss_fn, chunk: dict
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward one chunk and build its scalar loss plus the terms to log, for either mode.

    Called from inside the step loop's autocast block (and so must not be given a chunk from a
    different device or dtype context). Returns the differentiable total the caller weights and
    backwards, and the individual `_ACCUM_METRICS` terms it produced — the caller weights and
    accumulates those identically regardless of which branch built them, which is the whole point
    of returning them by name instead of as a fixed tuple.

    DPO's branch genuinely differs in shape: it forwards chosen+rejected concatenated in one call
    and splits the logits back, against a 9-argument loss. z_loss/mtp_loss are deliberately absent
    from its total — mtp_heads > 1 is already rejected by validate_post_training_config, and
    z_loss would need its own reduction pass over the concatenated logits that isn't needed for
    DPO correctness. ponder_cost/moe_aux_loss compose in for free, being zero scalars when their
    feature is off.
    """
    if cfg.dpo.enabled:
        c_ids, r_ids = chunk["chosen_input_ids"], chunk["rejected_input_ids"]
        b = c_ids.size(0)
        out = model(torch.cat([c_ids, r_ids], dim=0))
        chosen_logits, rejected_logits = out.logits.split(b, dim=0)
        lm_loss, margin, margin_accuracy, reward_accuracy = loss_fn(
            chosen_logits, c_ids, chunk["chosen_loss_mask"],
            rejected_logits, r_ids, chunk["rejected_loss_mask"],
            chunk["ref_chosen_logprob"], chunk["ref_rejected_logprob"], cfg.dpo.beta,
        )
        chunk_loss = (
            lm_loss
            + cfg.model.ponder_weight * out.ponder_cost
            + cfg.model.moe_aux_loss_weight * out.moe_aux_loss
        )
        extra = {
            "dpo_margin_accuracy": margin_accuracy,
            "dpo_reward_accuracy": reward_accuracy,
            "dpo_margin": margin,
        }
    else:
        input_ids = chunk["input_ids"]
        out = model(input_ids)
        lm_loss, z_loss = (
            loss_fn(out.logits, input_ids, chunk["loss_mask"])
            if cfg.sft.enabled
            else loss_fn(out.logits, input_ids)
        )
        mtp_loss = compute_mtp_loss(raw_model, out.mtp_hidden, input_ids)
        # Summed in this order deliberately: it is the order the two branches were written in
        # before they shared this function, so the refactor is bit-identical rather than merely
        # equivalent-to-tolerance.
        chunk_loss = (
            lm_loss
            + cfg.model.ponder_weight * out.ponder_cost
            + cfg.model.moe_aux_loss_weight * out.moe_aux_loss
            + cfg.model.z_loss_weight * z_loss
            + cfg.model.mtp_weight * mtp_loss
        )
        extra = {"z_loss": z_loss, "mtp_loss": mtp_loss}

    metrics = {
        "loss": chunk_loss,
        "lm_loss": lm_loss,
        "ponder_cost": out.ponder_cost,
        "mean_loop_depth": out.mean_loop_depth,
        "moe_aux_loss": out.moe_aux_loss,
        **extra,
    }
    return chunk_loss, metrics


def validate_post_training_config(cfg: Config) -> None:
    """Raise clear errors for post-training mode combinations train() can't handle, rather than
    letting them fail confusingly deep inside the data pipeline or the accumulation loop."""
    if cfg.sft.enabled and cfg.dpo.enabled:
        raise ValueError("sft.enabled and dpo.enabled are mutually exclusive — pick one post-training mode.")
    if (cfg.sft.enabled or cfg.dpo.enabled) and cfg.model.mtp_heads > 1:
        # compute_mtp_loss would need the same loss_mask treatment compute_sft_loss/compute_dpo_loss
        # got — mechanically similar (shift-by-depth+1 and fold the mask into -100) but not yet
        # built for either. Raise rather than silently score prompt/rejected tokens through the
        # auxiliary heads.
        raise ValueError("sft.enabled/dpo.enabled do not support model.mtp_heads > 1 yet — set mtp_heads: 1.")
    if cfg.dpo.enabled and not cfg.dpo.reference_checkpoint:
        raise ValueError("dpo.enabled requires dpo.reference_checkpoint to be set.")


def resolve_dpo_doc_mask(cfg: Config) -> None:
    """Turn model.doc_attention_mask off for a DPO run whose packing makes it a no-op.

    See config.doc_mask_is_inert_for_dpo for why it is a no-op (and for the loop_attn_windows
    exception that keeps it on). Worth doing for two reasons, the second larger than the first:
    the BlockMask is otherwise rebuilt on every training step and across the entire
    reference-logprob precompute pass for nothing, and doc_attention_mask is one of the three
    things that force resolve_compile_mode down to mode=None — so a DPO run that skips it gets
    CUDA graphs back.

    Mutates cfg rather than branching at each use, the same idiom auto_batch_size and
    tokens_per_param already use here, which also means the value saved into the checkpoint is
    the one the run actually trained with.
    """
    if not (cfg.dpo.enabled and doc_mask_is_inert_for_dpo(cfg.model)):
        return
    cfg.model.doc_attention_mask = False
    print(
        "[radiance] dpo.enabled: turning model.doc_attention_mask off — DPO packs one pair side "
        "per row, so the document mask cannot change any scored logit, and building it every "
        "step is pure overhead (see config.doc_mask_is_inert_for_dpo)."
    )


def note_dpo_z_loss_omitted(cfg: Config) -> None:
    """Surface that model.z_loss_weight has no effect under DPO, since it defaults nonzero.

    z_loss/mtp_loss are deliberately left out of the DPO chunk loss (see docs/post-training.md) —
    z_loss would need its own reduction pass over the concatenated chosen+rejected logits that
    DPO's correctness doesn't need, and mtp_heads > 1 is already rejected outright by
    validate_post_training_config. Unlike that rejection, z_loss_weight's default is nonzero
    (1e-4), so raising here would break every existing default DPO config for a term that was
    never meant to apply to DPO — a startup note keeps the omission visible without doing that.
    """
    if cfg.dpo.enabled and cfg.model.z_loss_weight != 0:
        print(
            f"[radiance] dpo.enabled: model.z_loss_weight={cfg.model.z_loss_weight} has no effect — "
            "z_loss is deliberately omitted from the DPO loss (see docs/post-training.md) and its "
            "metric logs a flat 0. Set model.z_loss_weight: 0 to silence this note."
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
