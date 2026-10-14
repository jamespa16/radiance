from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from radiance.model import DenseTransformer

from .batching import chunk_reduction_units, split_micro_batch
from .losses import compute_loss, forward_dpo_pair


@dataclass
class EvalResult:
    loss: float
    # Standard error of `loss`, from the spread of per-batch losses: the sampling noise of this
    # validation set, i.e. how far apart two val/loss readings must be before the gap means
    # anything. None with fewer than two batches.
    loss_stderr: float | None = None
    # Bits per UTF-8 byte of the scored targets: nats summed over every scored token, divided by
    # ln 2 and by those tokens' byte length. Unlike val/loss it does not change unit when the
    # tokenizer does, so it is what compares runs across tokenizers. None unless `token_bytes`
    # was given (pretraining only).
    bits_per_byte: float | None = None
    # DPO only: the row-weighted means of compute_dpo_loss's two accuracy diagnostics over the
    # scored val pairs — reward_accuracy (policy_chosen_logp > policy_rejected_logp) and
    # margin_accuracy (the policy separates the pair more than the reference did). val/loss alone
    # cannot say whether the policy *ranks* held-out pairs correctly; these can. None otherwise.
    dpo_reward_accuracy: float | None = None
    dpo_margin_accuracy: float | None = None


def evaluate(*args, **kwargs) -> float:
    """Mean validation loss; see evaluate_detailed, whose .loss this returns."""
    return evaluate_detailed(*args, **kwargs).loss


@torch.no_grad()
def evaluate_detailed(
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
    token_bytes: torch.Tensor | None = None,
) -> EvalResult:
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

    The loss assembly below stays local rather than also calling chunk_loss_and_metrics: that
    function needs raw_model (for compute_mtp_loss) and unconditionally folds
    ponder_cost/moe_aux_loss/z_loss/mtp_loss into its total, which val/loss deliberately excludes
    so it stays a pure LM number (see docs/train.md) — sharing it here would either drag those
    terms into eval or need a second, eval-only mode threaded through it for no benefit. The DPO
    forward itself (concatenate chosen+rejected, split the logits back) is identical either way,
    so that much is shared via losses.forward_dpo_pair.

    token_bytes (pretraining only; see data.token_byte_lengths) is each vocabulary id's UTF-8
    byte length, and turns on bits_per_byte. The byte count is taken over exactly the targets
    compute_loss scores — every token but each row's first — so the numerator and denominator
    cover the same predictions.
    """
    model.eval()
    total, count = 0.0, 0
    batch_losses: list[float] = []
    nats, n_bytes = 0.0, 0
    # DPO accuracies are per-row means, so they combine across chunks and batches by row count.
    # Accumulated on the device, and read back once after the loop rather than synced per chunk.
    reward_correct, margin_correct, n_pairs = 0.0, 0.0, 0
    if token_bytes is not None and (sft or dpo):
        raise ValueError("evaluate: token_bytes (bits/byte) is only defined for pretraining")
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
                    _, chosen_logits, rejected_logits = forward_dpo_pair(model, chunk)
                    chunk_loss, _, margin_acc, reward_acc = loss_fn(
                        chosen_logits, c_ids, chunk["chosen_loss_mask"],
                        rejected_logits, r_ids, chunk["rejected_loss_mask"],
                        chunk["ref_chosen_logprob"], chunk["ref_rejected_logprob"], dpo_beta,
                    )  # margin discarded; the accuracies are reported alongside val/loss, not in it
                    reward_correct = reward_correct + reward_acc.detach().float() * chunk_units
                    margin_correct = margin_correct + margin_acc.detach().float() * chunk_units
                    n_pairs += chunk_units
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
        batch_losses.append(loss)
        if token_bytes is not None:
            # loss is a mean over this batch's scored tokens; turn it back into a sum.
            nats += loss * sum(units)
            n_bytes += int(token_bytes[batch["input_ids"][:, 1:]].sum())
    model.train()
    if not count:
        return EvalResult(float("nan"))
    mean = total / count
    stderr = None
    if count > 1:
        var = sum((l - mean) ** 2 for l in batch_losses) / (count - 1)
        stderr = math.sqrt(var / count)
    bpb = nats / (math.log(2) * n_bytes) if token_bytes is not None and n_bytes else None
    if n_pairs:
        return EvalResult(
            mean, stderr, bpb, float(reward_correct) / n_pairs, float(margin_correct) / n_pairs
        )
    return EvalResult(mean, stderr, bpb)
