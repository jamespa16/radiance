from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config import ModelConfig

from .ffn import FeedForward
from .norms import RMSNorm
class ACTRouter(nn.Module):
    """Per-token halting-probability head for ACT (Graves 2016) adaptive looping.

    Normalization (RMSNorm) precedes the projection because this reads the pre-norm residual
    stream, whose norm grows with iteration count — without it the halting unit's calibration
    would drift across loop iterations.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, 1)
        # Per-iteration halting bias (cfg.loop_iter_conditioning). Without it the halting unit has
        # to infer "how deep am I" from the hidden state alone — and the RMSNorm above deliberately
        # strips the norm growth that would be its only cue. Zero-initialised, so halting behaviour
        # at init is exactly the pre-conditioning model's (proj.bias still carries Graves' -1.0).
        self.iter_bias = None
        if cfg.loop_iter_conditioning != "none" and cfg.max_loops > 1:
            self.iter_bias = nn.Parameter(torch.zeros(cfg.max_loops))

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        logits = self.proj(self.norm(x)).squeeze(-1)  # (batch, seq)
        if self.iter_bias is not None:
            logits = logits + self.iter_bias[min(variant, self.iter_bias.size(0) - 1)]
        return torch.sigmoid(logits)


def _ffn_capacity(cfg: ModelConfig, batch: int, seq_len: int) -> int:
    n_tokens = batch * seq_len
    return min(n_tokens, max(1, round(cfg.act_ffn_capacity_ratio * n_tokens)))


def _sparse_ffn_delta(
    ffn: FeedForward, h: torch.Tensor, still_running: torch.Tensor, capacity: int, variant: int = 0
) -> torch.Tensor:
    """h: (batch, seq_len, d_model) pre-FFN (post-ln2) input. still_running: (batch, seq_len) bool.
    Returns a same-shaped delta with FFN output scattered to at most `capacity` selected running
    positions and zero elsewhere (both underflow padding and overflow-dropped positions).
    """
    batch, seq_len, d_model = h.shape
    n_tokens = batch * seq_len
    flat_h = h.reshape(n_tokens, d_model)
    flat_running = still_running.reshape(n_tokens)

    # Running positions score in [1, 2); non-running in [0, 1) — running always outranks
    # non-running; ties among an overflowing running set broken by a fresh random draw each call.
    #
    # Only in training mode: a random draw is the right unbiased choice while learning (no position
    # is systematically starved of FFN across steps), but it also makes the forward pass
    # irreproducible, which at eval/generation time means val/loss and even greedy decoding move
    # run to run for identical weights and inputs. In eval the tiebreak falls back to sequence
    # order, which is deterministic. Train/eval divergence here is the same kind dropout already
    # has, and this path is documented as not bit-exact against the dense computation anyway.
    if ffn.training:
        tiebreak = torch.rand(n_tokens, device=h.device, dtype=torch.float32)
    else:
        # Descending over position, and strictly inside (0, 1) so a non-running token can never tie
        # with a running one (which scores 1 + tiebreak).
        tiebreak = torch.arange(n_tokens, 0, -1, device=h.device, dtype=torch.float32) / (n_tokens + 1)
    priority = flat_running.float() + tiebreak
    _, token_idx = torch.topk(priority, k=capacity)  # static k, compile-friendly
    valid = flat_running.index_select(0, token_idx)  # (capacity,) bool

    gathered = flat_h.index_select(0, token_idx)  # (capacity, d_model)
    ffn_out = ffn(gathered, variant) * valid.unsqueeze(-1).to(h.dtype)  # zero out padding slots

    delta_flat = flat_h.new_zeros(n_tokens, d_model).index_copy(0, token_idx, ffn_out)
    return delta_flat.view(batch, seq_len, d_model)


def _act_select(still_running: torch.Tensor, capacity: int, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick which `capacity` positions of each sequence an interior ACT iteration will compute.

    Returns (token_idx, valid), both (batch, capacity). token_idx holds true sequence positions,
    sorted ascending; valid marks which of them are actually still running (the rest are capacity
    padding, present only to keep the shape static and torch.compile-friendly).

    Selection is per *sequence*, not over the flattened batch as _sparse_ffn_delta does, because
    attention needs each row's queries to belong to that row. Still-running positions score in
    [1, 2) and halted ones in [0, 1), so running always outranks halted; ties among an overflowing
    running set are broken randomly while training (no position is systematically starved) and by
    sequence order otherwise, so eval and generation stay reproducible — the same train/eval split
    _sparse_ffn_delta already makes, and for the same reason.
    """
    batch, seq_len = still_running.shape
    if training:
        tiebreak = torch.rand(batch, seq_len, device=still_running.device, dtype=torch.float32)
    else:
        # Descending over position and strictly inside (0, 1), so a halted position can never tie
        # with a running one.
        order = torch.arange(seq_len, 0, -1, device=still_running.device, dtype=torch.float32)
        tiebreak = (order / (seq_len + 1)).expand(batch, seq_len)
    priority = still_running.float() + tiebreak
    token_idx = priority.topk(capacity, dim=1).indices
    # Sorting is not required for correctness (topk yields distinct indices either way) but keeps
    # the gathered rows in sequence order, which makes the causal mask below block-structured.
    token_idx, _ = token_idx.sort(dim=1)
    return token_idx, still_running.gather(1, token_idx)


def _block_mean(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool x's sequence dim (2) into blocks of block_size, ceil-dividing so a trailing
    partial block is averaged over only its real positions rather than being diluted by zero
    padding. x: (batch, heads, seq_len, head_dim). Returns (batch, heads, n_blocks, head_dim)
    where n_blocks = ceil(seq_len / block_size).
    """
    batch, heads, seq_len, head_dim = x.shape
    n_blocks = -(-seq_len // block_size)  # ceil
    pad = n_blocks * block_size - seq_len
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    summed = x.view(batch, heads, n_blocks, block_size, head_dim).sum(dim=3)
    counts = x.new_full((n_blocks,), block_size)
    if pad:
        counts[-1] = block_size - pad
    return summed / counts.view(1, 1, n_blocks, 1)
