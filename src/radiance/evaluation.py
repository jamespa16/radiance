from __future__ import annotations

import torch

from radiance.model import DenseTransformer

from .batching import chunk_reduction_units, split_micro_batch
from .losses import compute_loss
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
