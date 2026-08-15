from __future__ import annotations

import torch
import torch.nn as nn

from radiance.config import ModelConfig

from .attention import CausalSelfAttention
from .core import LoopContext
from .ffn import FeedForward, MoEFeedForward
from .hyper_connections import HyperConnection
from .norms import RMSNorm
class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        use_moe_ffn: bool = False,
        is_first: bool = False,
        n_variants: int = 1,
        hyper: bool = False,
        block_index: int = 0,
    ):
        super().__init__()
        # n_variants > 1 only for the loop body (blocks[1:]), which is the part that repeats;
        # blocks[0] runs exactly once per forward so it has nothing to be conditioned on.
        self.ln1 = RMSNorm(cfg.d_model, n_variants=n_variants)
        self.attn = CausalSelfAttention(cfg, is_first=is_first, n_variants=n_variants, block_index=block_index)
        self.ln2 = RMSNorm(cfg.d_model, n_variants=n_variants)
        self.ffn = MoEFeedForward(cfg) if use_moe_ffn else FeedForward(cfg, n_variants=n_variants)

        # cfg.hyper_conn_streams > 1: this block's two residual writes each become a hyper-connection
        # read/write pair over n parallel streams. Off by default and for any block outside the
        # trunk — MTPHead builds a TransformerBlock of its own and feeds it a plain (batch, seq,
        # d_model) tensor from outside the recursion, so it must keep the single-stream path.
        self.hyper_attn = self.hyper_ffn = None
        if hyper:
            # Two sublayers per block, and one stream of advance per loop iteration so the read
            # pattern rotates rather than repeating (see HyperConnection.__init__ for why the
            # loop body's true sublayer count is the wrong stride). blocks[0] runs exactly once,
            # so it has no iterations to advance over.
            stride = 0 if block_index == 0 else 1
            self.hyper_attn = HyperConnection(cfg, 2 * block_index, stride, n_variants)
            self.hyper_ffn = HyperConnection(cfg, 2 * block_index + 1, stride, n_variants)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: LoopContext,
        v_first: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Returns (hidden state, this block's own attention values, optional recorded (k, v)) —
        see CausalSelfAttention.forward for why the latter two come back out.

        `x` is (batch, seq, d_model) normally and (batch, seq, n_streams, d_model) when this block
        carries hyper-connections; the sublayers themselves always see the plain shape, which is
        why nothing below the block needs to know about streams.
        """
        variant = ctx.variant
        if self.hyper_attn is None:
            attn_out, v_own, recorded = self.attn(self.ln1(x, variant), cos, sin, ctx, v_first)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x, variant), variant)
            return x, v_own, recorded

        h, coeffs = self.hyper_attn.read(x, variant)
        attn_out, v_own, recorded = self.attn(self.ln1(h, variant), cos, sin, ctx, v_first)
        x = self.hyper_attn.write(x, attn_out, coeffs)
        h, coeffs = self.hyper_ffn.read(x, variant)
        x = self.hyper_ffn.write(x, self.ffn(self.ln2(h, variant), variant), coeffs)
        return x, v_own, recorded
