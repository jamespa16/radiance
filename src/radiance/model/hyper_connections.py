from __future__ import annotations

import torch
import torch.nn as nn

from radiance.config import ModelConfig
class HyperConnection(nn.Module):
    """One sublayer's hyper-connection unit (Zhu et al., ICLR 2025, arXiv:2409.19606).

    Replaces `x = x + sublayer(norm(x))` on a single residual stream with, over n streams
    H (n, d) per token:

        h_in = alpha_m^T H              # width read: which stream(s) this sublayer sees
        out  = sublayer(norm(h_in))     # unchanged
        H'   = alpha_r^T H + beta^T out # depth-connect the streams, then write the output back

    So a sublayer learns *which* stream to read, how the streams mix with each other, and how its
    output is distributed across them, instead of every sublayer reading and writing one shared
    accumulator.

    Why this architecture in particular wants it: blocks[1:] is a weight-shared loop body, so a
    6-layer model at loop_count 6 performs 62 residual writes into that single accumulator, which
    is exactly the depth regime the paper's gradient-vanishing/representation-collapse argument
    covers (and the same regime _scale_residual_init exists for). With per-iteration variants (see
    cfg.loop_iter_conditioning) each loop pass additionally learns its *own* routing, so a stream
    a given iteration never writes into carries information across the whole recursion untouched.

    **Exactly the plain residual network at initialisation.** alpha_m starts one-hot, alpha_r the
    identity, beta all-ones, and the streams start as n copies of the same vector — so every stream
    receives the same `out` and they stay equal at every depth, with the one-hot read just picking
    one of n identical copies. Both matmuls are bit-exact at those values (multiplying by exact 1.0
    and summing exact zeros), so the whole model differs from the single-stream one only by the
    final reduction in DenseTransformer._reduce_streams — which averages, making the equivalence
    bit-identical at power-of-two n and ~1 ulp otherwise.

    The `k mod n` stagger in the one-hot init is load-bearing, and the obvious alternative is a
    trap: initialise alpha_m uniformly at 1/n instead and every alpha_m/beta/alpha_r entry receives
    an *identical* gradient forever, so the n streams stay one stream for the life of the run and
    the whole feature is a no-op that looks like it is training. Staggering which stream each
    sublayer reads is what makes dL/dH differ across streams and lets them diverge.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        layer_index: int,
        variant_stride: int = 0,
        n_variants: int = 1,
        eps: float = 1e-6,
    ):
        """layer_index is this sublayer's position in the execution order (two per block, so
        adjacent sublayers read adjacent streams as the paper prescribes); variant_stride is how
        far that position advances per loop iteration, so variant v initialises as sublayer
        `layer_index + v * variant_stride`.

        The stride is 1 rather than the loop body's true unrolled sublayer count, and that is a
        deliberate correction rather than an approximation. The true count is `2 * (n_layers - 1)`,
        which is always even — so at n=2 it is congruent to 0 mod n for *every* model, and at n=4
        for every odd n_layers. In those cases each iteration reads the identical stream, the
        per-iteration routing collapses to a single pattern, and the failure is invisible: the
        model trains fine and merely gives up the loop-identity benefit that motivates having
        per-iteration variants at all. Advancing by one stream per iteration cannot alias for any
        n.
        """
        super().__init__()
        n = cfg.hyper_conn_streams
        self.n_streams = n
        self.n_variants = n_variants
        self.dynamic = cfg.hyper_conn_dynamic
        self.eps = eps

        # Static connection coefficients, all named without "dyn" so optim.py routes them to the
        # no-decay AdamW group (the paper decays the dynamic weights but not these).
        self.beta = nn.Parameter(torch.ones(n_variants, n))
        self.alpha_r = nn.Parameter(torch.eye(n).expand(n_variants, n, n).clone())
        alpha_m = torch.zeros(n_variants, n)
        for v in range(n_variants):
            alpha_m[v, (layer_index + v * variant_stride) % n] = 1.0
        self.alpha_m = nn.Parameter(alpha_m)

        if self.dynamic:
            # Zero-initialised, exactly like IterLoRA's B and the input-injection projection:
            # tanh(0) is exactly 0, so a dynamic unit is bit-identical to a static one at init.
            # Bare Parameters rather than nn.Linear deliberately — an nn.Linear here would be
            # visited by DenseTransformer._init_weights and would then have to be re-zeroed in
            # _init_inert_gates, adding a third entry to an ordering constraint that has already
            # caused two bugs in this file.
            #
            # The three projections (alpha_m, beta, alpha_r — columns [0], [1], [2:]) share one
            # tensor so the forward is a single matmul over the n-wide hidden state instead of
            # three. Measured 3.9x faster on that step at d_model 256: these are memory-bound at
            # this width, so what matters is reading the normalised hidden once rather than the
            # negligible FLOP count.
            self.dyn_proj = nn.Parameter(torch.zeros(n_variants, cfg.d_model, 2 + n))
            # One scale for beta and one shared by both alpha terms, as the paper has it.
            self.dyn_scale_beta = nn.Parameter(torch.ones(n_variants))
            self.dyn_scale_alpha = nn.Parameter(torch.ones(n_variants))

    def read(self, hidden: torch.Tensor, variant: int = 0):
        """(batch, seq, n_streams, d_model) -> ((batch, seq, d_model), write-side coefficients).

        The write's coefficients are computed here, from the hidden state *entering* the sublayer
        as the paper specifies, and handed back rather than stashed on self — keeping this module
        stateless is what lets a whole block run under torch.utils.checkpoint unchanged.
        """
        # Clamped, not wrapped: running more iterations at inference than were trained (see
        # radiance-generate --loops) should reuse the deepest learned routing rather than silently
        # cycling back to the shallow ones. Mirrors RMSNorm's per-variant gain selection.
        v = min(variant, self.n_variants - 1)
        beta, alpha_m, alpha_r = self.beta[v], self.alpha_m[v], self.alpha_r[v]

        if self.dynamic:
            # The rsqrt reduction runs in fp32 for stability, but `hidden` itself is never upcast:
            # it is the n-times-wider tensor here, and materialising an fp32 copy of it cost more
            # than the normalisation did.
            scale = torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            normed = hidden * scale.to(hidden.dtype)
            dyn = torch.tanh(normed @ self.dyn_proj[v])  # (batch, seq, n_streams, 2 + n_streams)
            dyn_m, dyn_beta, dyn_r = dyn.split([1, 1, self.n_streams], dim=-1)
            alpha_m = alpha_m + self.dyn_scale_alpha[v] * dyn_m.squeeze(-1)
            alpha_r = alpha_r + self.dyn_scale_alpha[v] * dyn_r
            beta = beta + self.dyn_scale_beta[v] * dyn_beta.squeeze(-1)

        # A matmul, not an einsum contracting dim -2: alpha_m is (n,) when static and (batch, seq,
        # n) when dynamic, and unsqueezing a row dimension onto it makes one expression cover both
        # while contracting against `hidden`'s stream dim with no permute of the wide tensor.
        # Measured 3.6x faster than the equivalent einsum.
        return (alpha_m.unsqueeze(-2) @ hidden).squeeze(-2), (beta, alpha_r)

    def write(self, hidden: torch.Tensor, out: torch.Tensor, coeffs) -> torch.Tensor:
        """(batch, seq, n_streams, d_model) + this sublayer's (batch, seq, d_model) output ->
        the updated stream tensor."""
        beta, alpha_r = coeffs
        # alpha_r is (n, n) when static and per-token (batch, seq, n, n) when dynamic, and beta
        # broadcasts over d_model either way, so one expression covers both. Moving the stream dim
        # last makes this a plain matmul rather than an einsum contracting dim -2; the movedim
        # pair are stride permutes on a tensor the matmul has to read anyway, and it measured
        # faster than either the einsum or a broadcast matmul against alpha_r.mT.
        mixed = (hidden.movedim(-2, -1) @ alpha_r).movedim(-1, -2)
        return mixed + out.unsqueeze(-2) * beta.unsqueeze(-1)
