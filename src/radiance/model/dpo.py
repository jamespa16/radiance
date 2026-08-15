from __future__ import annotations

import torch


def sequence_logprob_sum(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Per-row (batch,) sum of log p(token_{t+1} | tokens_{<=t}) at loss_mask==1 positions.

    DPO's per-sequence analogue of losses._nll_and_logz, which reduces to a single batch-wide mean
    over all flattened kept positions — the right reduction for a plain LM loss, but DPO needs one
    scalar log-probability per (prompt, completion) sequence to combine into its pairwise loss, so
    the batch dimension has to survive the reduction. Same shift-and-mask construction
    losses.compute_sft_loss uses, same single-logsumexp-pass trick _nll_and_logz uses to avoid
    walking the (batch, seq, vocab_size) logits twice.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    keep = shifted_mask.bool()
    safe_labels = labels.masked_fill(~keep, 0)
    z = torch.logsumexp(logits, dim=-1).float()
    target_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).float()
    return ((target_logit - z) * keep.float()).sum(dim=1)
