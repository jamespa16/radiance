from __future__ import annotations

import torch


def target_logit_and_logz(logits: torch.Tensor, safe_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(target_logit, z)` for cross-entropy/log-prob math, from one pass over `logits`.

    `cross_entropy` is exactly `logsumexp(x) - x[label]`, so both losses._nll_and_logz's
    mean-reduced NLL/z-loss and sequence_logprob_sum's per-row summed log-probability are built
    from the same two quantities; only the reduction differs. Computing them separately —
    F.cross_entropy plus a torch.logsumexp — walks the largest tensor in the model (batch, seq,
    vocab_size) twice in the forward and twice again in the backward.

    `safe_labels` is the caller's responsibility (not raw -100-bearing labels) because gather
    can't take -100 directly; the caller fills ignored positions with any in-range index and masks
    the corresponding output positions itself afterward.
    """
    z = torch.logsumexp(logits, dim=-1).float()
    target_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).float()
    return target_logit, z


def sequence_logprob_sum(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Per-row (batch,) sum of log p(token_{t+1} | tokens_{<=t}) at loss_mask==1 positions.

    DPO's per-sequence analogue of losses._nll_and_logz, which reduces to a single batch-wide mean
    over all flattened kept positions — the right reduction for a plain LM loss, but DPO needs one
    scalar log-probability per (prompt, completion) sequence to combine into its pairwise loss, so
    the batch dimension has to survive the reduction. Same shift-and-mask construction
    losses.compute_sft_loss uses, same target_logit_and_logz single-pass trick _nll_and_logz uses.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    keep = shifted_mask.bool()
    safe_labels = labels.masked_fill(~keep, 0)
    target_logit, z = target_logit_and_logz(logits, safe_labels)
    return ((target_logit - z) * keep.float()).sum(dim=1)
