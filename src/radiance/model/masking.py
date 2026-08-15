from __future__ import annotations

import torch
def document_ids(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Which packed document each token belongs to, as an exclusive cumulative count of EOS.

    data.py's packing concatenates tokenized documents joined by exactly one eos_token_id and
    nothing else emits EOS, so document membership is fully recoverable here — no extra column in
    the packed dataset, and therefore no re-tokenizing and no cache invalidation for every dataset
    already on disk (_cache_path keys on the packed format).

    Exclusive, so a document's terminating EOS is counted as part of the document it ends rather
    than the one that follows it.
    """
    is_eos = (input_ids == eos_id).long()
    return is_eos.cumsum(dim=1) - is_eos


def build_block_mask(
    doc_ids: torch.Tensor, seq_len: int, offset: int = 0, window: int | None = None
):
    """A flex_attention BlockMask for causal + same-document (+ optionally windowed) attention.

    Built once per forward pass, outside any compiled region, and reused across every block and
    every loop iteration — exactly how the RoPE cos/sin tables are already shared. Returns a
    BlockMask that skips whole tiles where no query can attend to any key, so the document
    structure costs sparsity rather than an O(seq_len^2) materialised mask.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        causal = (q_idx + offset) >= kv_idx
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        if window is not None:
            return causal & same_doc & ((q_idx + offset) - kv_idx < window)
        return causal & same_doc

    return create_block_mask(
        mask_mod, B=doc_ids.size(0), H=None, Q_LEN=seq_len, KV_LEN=doc_ids.size(1),
        device=doc_ids.device,
    )


def sequence_logprob_sum(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Per-row (batch,) sum of log p(token_{t+1} | tokens_{<=t}) at loss_mask==1 positions.

    DPO's per-sequence analogue of train._nll_and_logz, which reduces to a single batch-wide mean
    over all flattened kept positions — the right reduction for a plain LM loss, but DPO needs one
    scalar log-probability per (prompt, completion) sequence to combine into its pairwise loss, so
    the batch dimension has to survive the reduction. Same shift-and-mask construction
    train.compute_sft_loss uses, same single-logsumexp-pass trick _nll_and_logz uses to avoid
    walking the (batch, seq, vocab_size) logits twice.

    Lives here rather than train.py because data.py's DPO reference-logprob precompute needs this
    exact primitive too, and data.py must not import train.py (train.py already imports data.py —
    that would cycle). Both data.py and train.py already import model.py, so this adds no cycle.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    keep = shifted_mask.bool()
    safe_labels = labels.masked_fill(~keep, 0)
    z = torch.logsumexp(logits, dim=-1).float()
    target_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).float()
    return ((target_logit - z) * keep.float()).sum(dim=1)


def _sparse_attn_mask(
    token_idx: torch.Tensor, seq_len: int, doc_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """Boolean (batch, 1, capacity, seq_len) mask for attention from gathered queries.

    The gathered queries sit at scattered true positions, so causality is no longer the triangle
    `is_causal=True` assumes — it is `true_position(query) >= key_position`, which has to be
    materialised. That costs batch*capacity*seq_len bools, small next to what skipping the other
    positions' whole blocks saves. Document masking, when active, is folded into the same mask
    rather than going through flex_attention, whose BlockMask assumes dense queries.
    """
    keys = torch.arange(seq_len, device=token_idx.device)
    mask = token_idx.unsqueeze(-1) >= keys  # (batch, capacity, seq_len)
    if doc_ids is not None:
        query_docs = doc_ids.gather(1, token_idx)
        mask = mask & (query_docs.unsqueeze(-1) == doc_ids.unsqueeze(1))
    return mask.unsqueeze(1)
