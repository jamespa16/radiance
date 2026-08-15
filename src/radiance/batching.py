from __future__ import annotations

import math

import torch

from radiance.config import Config
from radiance.dpo_data import dpo_cache_exists
from radiance.model import DenseTransformer, checkpoint_param_bytes
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
    usable_bytes = max(0.0, free_bytes - not_yet_allocated_bytes - reserve_bytes) * cfg.train.vram_safety_margin

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
    print(
        f"[radiance] auto_batch_size: {free_bytes / 1e9:.2f} GB free{reserve_note}, {num_params:,} params, "
        f"vram_safety_margin={cfg.train.vram_safety_margin} -> batch_size={batch_size} {unit}, "
        f"grad_accum_steps={grad_accum_steps} (effective_batch_size={batch_size * grad_accum_steps}, "
        f"target={cfg.train.target_effective_batch_size})"
    )
    return batch_size, grad_accum_steps


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
