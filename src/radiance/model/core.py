from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
class ModelOutput(NamedTuple):
    """DenseTransformer.forward's return value.

    A NamedTuple rather than a bare tuple so fields can be added without breaking every call site
    on the way past. `ponder_cost`/`mean_loop_depth`/`moe_aux_loss` are zero scalar tensors when
    the corresponding feature (use_router / use_moe) is off, so callers have one contract
    regardless of mode.
    """

    logits: torch.Tensor
    ponder_cost: torch.Tensor
    mean_loop_depth: torch.Tensor
    moe_aux_loss: torch.Tensor
    # Multi-token-prediction head outputs, as *hidden states* (batch, seq, d_model) rather than
    # logits: projecting them is the caller's job, one at a time, so eval and generation — which
    # never use them — pay nothing, and training holds one (batch, seq, vocab_size) tensor at a
    # time rather than mtp_heads of them. None whenever mtp_heads == 1 or we're not training.
    mtp_hidden: tuple[torch.Tensor, ...] | None = None


@dataclass
class LoopContext:
    """Per-forward, per-iteration state threaded down to the blocks.

    Six separate features (iteration conditioning, MoE iteration bias, doc masking, per-iteration
    attention windows, ACT capacity dispatch, loop K/V sharing) all need to tell a block something
    about *which pass it is on* or *what mask applies*. Bundling that into one object keeps
    TransformerBlock/CausalSelfAttention's signatures stable as those features land, instead of
    growing a new keyword argument per feature.

    Deliberately holds only non-tensor / no-grad state. Tensors that participate in autograd (the
    input-injection `anchor`, the value-residual `v_first`) stay *explicit positional arguments*
    to the checkpointed functions: torch.utils.checkpoint only tracks tensors it receives
    positionally, so a grad-requiring tensor hidden inside this dataclass would be treated as
    closure state and silently lose its recompute path under cfg.model.grad_checkpoint.
    """

    iteration: int = 0  # 0 for blocks[0]; 1..n for successive loop-body passes
    kv_cache: "KVCache | None" = None
    capacity: int | None = None  # ACT fixed-capacity dispatch; None = dense
    block_mask: Any = None  # flex_attention BlockMask (doc masking / attention windows)
    record_kv: bool = False  # have blocks hand back their post-RoPE (k, v) so an ACT iteration can
    # seed the retained K/V store the sparse iterations read from. Off by default so the common path
    # returns None there and nothing extra is kept alive for backward.

    @property
    def variant(self) -> int:
        """Index into per-iteration parameter banks (RMSNorm gains, router biases).

        Zero-based, so the loop body's first pass (iteration 1) selects variant 0 and blocks[0]
        (iteration 0) also selects 0 — blocks[0] only ever has one variant, so the clamp is what
        matters, not the collision.
        """
        return max(0, self.iteration - 1)
