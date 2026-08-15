from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config import ModelConfig

from .core import LoopContext
from .norms import RMSNorm, _make_iter_lora
# flex_attention's compiled kernel rejects head dimensions below this.
_FLEX_MIN_HEAD_DIM = 16

_FLEX_ATTENTION = None


def _flex_attention():
    """flex_attention, compiled or not depending on who's calling.

    Two cases, and getting this wrong breaks one of them:

    * Called from inside an already-compiled region (cfg.train.compile, the normal path): hand back
      the *raw* function and let the enclosing graph lower it. Passing a pre-compiled callable
      instead nests one torch.compile inside another, which fails at lowering with
      "convert FlexibleLayout to FixedLayout first".
    * Called eagerly (cfg.train.compile off, tests, CPU checks): compile it here. Eager
      flex_attention falls back to an unfused path that materialises the whole
      (seq_len, seq_len) score matrix — precisely the cost the BlockMask exists to avoid.

    The compiled variant is built once on first eager use and cached, so a run that never takes
    this path never pays for it.
    """
    from torch.nn.attention.flex_attention import flex_attention

    if torch.compiler.is_compiling():
        return flex_attention

    global _FLEX_ATTENTION
    if _FLEX_ATTENTION is None:
        _FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _FLEX_ATTENTION


@torch._dynamo.disable
def _diff_flex_attention(
    q1: torch.Tensor,
    k1: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    block_mask: Any,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differential attention's two flex_attention calls, forced out of the enclosing compiled
    graph — this is a correctness fix, not a defensive measure taken on spec.

    Verified directly: two flex_attention calls sharing one BlockMask, whose Q/K both trace back to
    a common upstream torch.split (exactly the qkv_proj chunking CausalSelfAttention.forward does
    for q1/k1/q2/k2), silently diverge from eager once torch.compile traces both into one inductor
    graph — the second call's output was off by up to ~0.85 absolute against an identical eager
    forward, with no error or warning. Isolated with a flex_attention-only repro independent of this
    file (a shared 5-way split feeding two flex_attention calls against one BlockMask); .contiguous()
    and .clone() on the inputs did not fix it and in some variants made *both* calls wrong instead of
    just the second, and batching the two calls into one wider flex_attention call over a doubled
    head dimension was wrong too — so this isn't a fixable data-dependency bug on our side, it's a
    scheduling assumption flex_attention's inductor lowering makes that this call pattern violates.
    torch._dynamo.disable forces a graph break here, so each call instead goes through
    _flex_attention()'s eager branch (a separately compiled, cached flex_attention) rather than
    being lowered as part of the model's single big graph — confirmed bit-identical to eager this
    way, backward included. Plain (non-differential) attention is unaffected: it only ever issues
    one flex_attention call per forward, which was already proven correct under compile.

    Re-verify this is still needed after any PyTorch upgrade (a torch.compile.disable that turns out
    unnecessary just costs a graph break, but silently trusting the compiled path here again without
    re-checking would reintroduce exactly the wrong-output bug this exists to avoid).
    """
    a1 = _flex_attention()(q1, k1, v, block_mask=block_mask, enable_gqa=enable_gqa)
    a2 = _flex_attention()(q2, k2, v, block_mask=block_mask, enable_gqa=enable_gqa)
    return a1, a2


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, n_heads, seq, head_dim); cos/sin: (seq, head_dim), broadcast over batch/heads."""
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin tables up to max_seq_len at construction time (same role the old
    learned pos_emb table played). Rotation depends only on absolute sequence position, never on
    which block or loop iteration is running, so this is built once on DenseTransformer and its
    (cos, sin) output is reused unchanged across every block and every loop iteration within a
    forward call.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim / 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        # Buffers, not parameters (not learned); persistent=False since they're deterministically
        # regenerated from head_dim/max_seq_len/theta and shouldn't bloat checkpoints.
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[offset : offset + seq_len], self.sin_cached[offset : offset + seq_len]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, is_first: bool = False, n_variants: int = 1, block_index: int = 0):
        """is_first marks blocks[0], which is always called with v_first=None (it is the block that
        *produces* v_first). Its value-residual mix would therefore never be exercised, so the
        parameter isn't created at all — otherwise it would sit in the optimizer forever collecting
        no gradient.

        block_index is this block's structural position (0 for blocks[0], 1.. for the loop body's
        weight-shared position) — only consumed by cfg.use_diff_attn, for its depth-dependent
        lambda_init. The shared loop body uses one fixed value for its position, not a per-iteration
        one (contrast loop_iter_conditioning, which does vary by iteration)."""
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.head_dim % 2 == 0, "model.head_dim must be even for RoPE's pairwise rotation"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads_resolved
        self.head_dim = cfg.d_model // cfg.n_heads
        kv_dim = self.n_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(cfg.d_model, cfg.d_model + 2 * kv_dim)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

        # Differential Attention (Ye et al. 2024, see cfg.use_diff_attn): reinterprets qkv_proj's
        # existing Q/K chunks as two half-head_dim-wide pairs (Q1/K1, Q2/K2) instead of one
        # full-width pair — two half-width heads sum back to the same total width, so qkv_proj's
        # shape above is completely unaffected by this branch. Only four small per-layer vectors
        # and a per-head output norm are new parameters.
        self.use_diff_attn = cfg.use_diff_attn
        if self.use_diff_attn:
            assert cfg.head_dim % 4 == 0, (
                "model.use_diff_attn splits each head's Q/K into two head_dim//2-wide branches, and "
                "RoPE's pairwise rotation then needs that half-width to itself be even — "
                "model.head_dim must be a multiple of 4. (DenseTransformer also validates this with "
                "a friendlier message; this assert is a defensive backstop for direct construction.)"
            )
            diff_head_dim = self.head_dim // 2
            self.diff_lambda_q1 = nn.Parameter(torch.randn(diff_head_dim) * 0.1)
            self.diff_lambda_k1 = nn.Parameter(torch.randn(diff_head_dim) * 0.1)
            self.diff_lambda_q2 = nn.Parameter(torch.randn(diff_head_dim) * 0.1)
            self.diff_lambda_k2 = nn.Parameter(torch.randn(diff_head_dim) * 0.1)
            # lambda_init(l) = 0.8 - 0.6*exp(-0.3*(l-1)), l the 1-indexed layer position (paper);
            # block_index is already 0-indexed so it *is* (l - 1). A fixed Python float, not a
            # buffer/parameter — it never changes after construction, so torch.compile specializes
            # on it like any other constant instead of treating it as a dynamic input.
            self.diff_lambda_init = 0.8 - 0.6 * math.exp(-0.3 * block_index)
            # Paper: rescale the normalized, differenced output by (1 - lambda_init) to roughly
            # match plain softmax attention's output variance (a difference of two positively
            # correlated distributions has lower variance than either alone).
            self.diff_out_scale = 1.0 - self.diff_lambda_init
            # Per-head normalization before concatenation. The paper uses a non-parametric
            # GroupNorm; this reuses the file's existing learnable RMSNorm instead, consistent with
            # how qk_norm already reuses it elsewhere — flagged in CLAUDE.md as worth sanity-checking
            # empirically against the paper's version rather than assumed equivalent.
            self.diff_norm = RMSNorm(self.head_dim)

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            # Differential attention's Q1/K1/Q2/K2 are head_dim//2 wide; qk_norm applies the same
            # (shared) norm module to both branches rather than doubling to four modules.
            qk_norm_dim = (self.head_dim // 2) if self.use_diff_attn else self.head_dim
            self.q_norm = RMSNorm(qk_norm_dim)
            self.k_norm = RMSNorm(qk_norm_dim)

        # Learned mix between this block's values and blocks[0]'s (see cfg.value_residual). A bare
        # scalar, so it lands in the optimizer's no-decay group alongside the norm gains — decaying
        # it would drag the model toward pure v_first, which is the opposite of a useful prior.
        self.value_residual = cfg.value_residual and not is_first
        if self.value_residual:
            self.value_lambda = nn.Parameter(torch.ones(1))

        # Per-head output gate (see cfg.attn_out_gate). Zero-initialised in
        # DenseTransformer._init_inert_gates, *after* _init_weights has run over every Linear.
        self.attn_out_gate = cfg.attn_out_gate
        if self.attn_out_gate:
            self.out_gate = nn.Linear(cfg.d_model, self.n_heads)

        self.qkv_lora = _make_iter_lora(cfg, cfg.d_model, cfg.d_model + 2 * kv_dim, n_variants)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: LoopContext,
        v_first: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Returns (attention output, this block's own pre-mix values, optional post-RoPE (k, v)).

        The second element is what blocks[0] hands to every later block as `v_first` for value
        residual; blocks[1:] return theirs too but nobody reads it, which keeps one signature for
        every block rather than special-casing the first.

        The third is None unless ctx.record_kv, in which case it seeds the retained K/V store that
        ACT's sparse iterations attend against (see _run_loop_body_sparse).
        """
        batch, seq_len, d_model = x.shape
        kv_cache = ctx.kv_cache

        qkv = self.qkv_proj(x)
        if self.qkv_lora is not None:
            qkv = qkv + self.qkv_lora(x, ctx.variant)
        kv_dim = self.n_kv_heads * self.head_dim

        if self.use_diff_attn:
            d = self.head_dim // 2
            q_half, k_half = d_model // 2, kv_dim // 2
            q1, q2, k1, k2, v = qkv.split([q_half, q_half, k_half, k_half, kv_dim], dim=-1)
            q1 = q1.view(batch, seq_len, self.n_heads, d).transpose(1, 2)
            q2 = q2.view(batch, seq_len, self.n_heads, d).transpose(1, 2)
            k1 = k1.view(batch, seq_len, self.n_kv_heads, d).transpose(1, 2)
            k2 = k2.view(batch, seq_len, self.n_kv_heads, d).transpose(1, 2)
            v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q, k, v = qkv.split([d_model, kv_dim, kv_dim], dim=-1)
            q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        v_own = v
        if self.value_residual and v_first is not None:
            # Mixed *before* the kv_cache write below, so cached values are already mixed and
            # generation needs no extra cache slot: within a decode step blocks[0] runs first and
            # produces this token's v_first, and past tokens' mixed values are already stored. The
            # single shared V (differential attention doesn't split V) makes this identical either way.
            lam = self.value_lambda
            v = lam * v + (1.0 - lam) * v_first

        if self.qk_norm:
            # Must precede RoPE: RMSNorm's learned per-channel weight is not rotation-invariant,
            # so normalizing after rotation would scale the rotated pairs asymmetrically.
            if self.use_diff_attn:
                q1, k1 = self.q_norm(q1), self.k_norm(k1)
                q2, k2 = self.q_norm(q2), self.k_norm(k2)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)

        if self.use_diff_attn:
            # cos/sin are already head_dim//2-wide here — DenseTransformer builds self.rope at
            # head_dim // 2 whenever use_diff_attn is set (V is never rotated, and this is the only
            # width diff-attention's Q/K ever take), so no separate table is needed.
            q1 = apply_rope(q1, cos[None, None, :, :], sin[None, None, :, :])
            k1 = apply_rope(k1, cos[None, None, :, :], sin[None, None, :, :])
            q2 = apply_rope(q2, cos[None, None, :, :], sin[None, None, :, :])
            k2 = apply_rope(k2, cos[None, None, :, :], sin[None, None, :, :])
        else:
            q = apply_rope(q, cos[None, None, :, :], sin[None, None, :, :])
            k = apply_rope(k, cos[None, None, :, :], sin[None, None, :, :])

        if kv_cache is None:
            is_causal = True
        else:
            # A non-empty cache means this call is decoding exactly one new token against
            # already-committed positions, which needs no mask (every cached key is causally
            # valid for it); an empty cache means this is the initial prefill, which needs the
            # usual triangular mask.
            is_causal = kv_cache.seq_len == 0
            if self.use_diff_attn:
                k1, k2, v = kv_cache.write3(k1, k2, v)
            else:
                k, v = kv_cache.write(k, v)

        enable_gqa = self.n_kv_heads != self.n_heads
        if self.use_diff_attn:
            if ctx.block_mask is not None:
                a1, a2 = _diff_flex_attention(q1, k1, q2, k2, v, ctx.block_mask, enable_gqa)
            else:
                dropout_p = self.dropout if self.training else 0.0
                a1 = F.scaled_dot_product_attention(
                    q1, k1, v, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=enable_gqa
                )
                a2 = F.scaled_dot_product_attention(
                    q2, k2, v, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=enable_gqa
                )
            # lambda reparameterized as exp(dot) - exp(dot) + lambda_init rather than a bare
            # parameter: at init the two exp(dot) terms nearly cancel (small random lambda_q/k),
            # so lambda starts near lambda_init in expectation while the four vectors above still
            # get a real (nonzero) gradient — the same "moves off zero from step 1" cold start
            # loop_input_injection's W_inj relies on, not a dead-parameter risk.
            lam = (
                torch.exp((self.diff_lambda_q1 * self.diff_lambda_k1).sum())
                - torch.exp((self.diff_lambda_q2 * self.diff_lambda_k2).sum())
                + self.diff_lambda_init
            )
            attn_out = self.diff_norm(a1 - lam * a2) * self.diff_out_scale
        elif ctx.block_mask is not None:
            # No dropout_p here: flex_attention has no attention-weight dropout. See
            # DenseTransformer._doc_masks, which warns once when a config combines
            # doc_attention_mask with a nonzero dropout. The FFN's residual dropout is unaffected.
            attn_out = _flex_attention()(q, k, v, block_mask=ctx.block_mask, enable_gqa=enable_gqa)
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
        attn_out = attn_out.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
        if self.attn_out_gate:
            # 2 * sigmoid, not sigmoid: with out_gate zero-initialised this is exactly 1.0, so the
            # gated model is bit-identical to the ungated one at init. A plain sigmoid would start
            # every head at 0.5 and halve the attention output.
            gate = 2.0 * torch.sigmoid(self.out_gate(x))  # (batch, seq_len, n_heads)
            attn_out = attn_out * gate.unsqueeze(-1)
        attn_out = attn_out.contiguous().view(batch, seq_len, d_model)
        # k/v here are the full-length post-RoPE tensors (after any cache concat), which is exactly
        # the shape the sparse path's retained store needs. Never populated for diff attention —
        # ctx.record_kv only ever fires from ACT's sparse path, which use_diff_attn is validated
        # against combining with (see DenseTransformer.__init__).
        recorded = None if self.use_diff_attn else ((k, v) if ctx.record_kv else None)
        return self.out_proj(attn_out), v_own, recorded

    def forward_sparse(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        retained_kv: tuple[torch.Tensor, torch.Tensor],
        token_idx: torch.Tensor,
        attn_mask: torch.Tensor,
        v_first: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attention for a gathered subset of positions (ACT's cfg.act_capacity_ratio path).

        h            (batch, capacity, d_model) — pre-normed hidden states of the selected positions
        cos/sin      (batch, capacity, head_dim) — RoPE tables gathered at those true positions
        retained_kv  full-length (batch, n_kv_heads, seq_len, head_dim) K/V from earlier iterations
        token_idx    (batch, capacity) true positions of the gathered rows
        attn_mask    (batch, 1, capacity, seq_len) from _sparse_attn_mask

        Only the selected positions' Q/K/V are computed; the fresh K/V are scattered into the
        retained store so the unselected positions keep serving whatever they last produced, and
        the gathered queries then attend against the full-length result. Returns the attention
        output for the selected rows plus the updated store.

        This is the approximation: an unselected position's retained K/V is only exactly right for
        the first block of the loop body. See cfg.act_capacity_ratio.
        """
        batch, capacity, d_model = h.shape

        qkv = self.qkv_proj(h)
        if self.qkv_lora is not None:
            qkv = qkv + self.qkv_lora(h, 0)
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([d_model, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, capacity, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, capacity, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, capacity, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.value_residual and v_first is not None:
            lam = self.value_lambda
            v = lam * v + (1.0 - lam) * v_first

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Per-row RoPE: each gathered query carries its own absolute position, so the tables are
        # indexed rather than sliced. (batch, capacity, head_dim) -> broadcast over heads.
        q = apply_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = apply_rope(k, cos.unsqueeze(1), sin.unsqueeze(1))

        k_full, v_full = retained_kv
        scatter_idx = token_idx[:, None, :, None].expand(batch, self.n_kv_heads, capacity, self.head_dim)
        # Out-of-place: the store belongs to the caller's autograd graph across iterations, and an
        # in-place write here would make backward see a mutated tensor.
        k_full = k_full.scatter(2, scatter_idx, k)
        v_full = v_full.scatter(2, scatter_idx, v)

        attn_out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=(self.n_kv_heads != self.n_heads),
        )
        attn_out = attn_out.transpose(1, 2)
        if self.attn_out_gate:
            gate = 2.0 * torch.sigmoid(self.out_gate(h))
            attn_out = attn_out * gate.unsqueeze(-1)
        attn_out = attn_out.contiguous().view(batch, capacity, d_model)
        return self.out_proj(attn_out), (k_full, v_full)


class KVCache:
    """Per-forward-call-order cache of attention K/V for incremental (one-token-at-a-time)
    generation.

    Slot count matches the number of CausalSelfAttention calls a single DenseTransformer.forward
    call makes: 1 for blocks[0] plus one per (block, loop-iteration) pair in the shared-weight
    loop body (blocks[1:] run loop_count or, in ACT mode, max_loops times — always the full
    iteration count, since attention is unconditionally dense every iteration regardless of
    per-token halting). Because blocks[1:] are weight-shared but fed an evolving hidden state,
    the same block produces different K/V on each iteration, so each (block, iteration) pair
    needs its own slot — one slot per layer is not enough once loop_count/max_loops > 1.

    Slot assignment is implicit call order, not an explicit index: begin_step() resets the
    cursor at the top of a forward call, and each write() claims the next slot. This only works
    because block execution order is identical on every call (never data-dependent).

    write3() is differential attention's variant, caching K1/K2/a single shared V instead of one
    K/V pair — still exactly one call (and one slot) per CausalSelfAttention.forward invocation,
    since V is never split between the two attention branches, so slot *count* is unaffected.
    """

    def __init__(self, num_slots: int):
        self._k: list[torch.Tensor | None] = [None] * num_slots
        self._k2: list[torch.Tensor | None] = [None] * num_slots
        self._v: list[torch.Tensor | None] = [None] * num_slots
        self._cursor = 0
        self.seq_len = 0

    def begin_step(self) -> None:
        self._cursor = 0

    def write(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self._cursor
        self._cursor += 1
        if self._k[slot] is None:
            self._k[slot], self._v[slot] = k, v
        else:
            self._k[slot] = torch.cat([self._k[slot], k], dim=2)
            self._v[slot] = torch.cat([self._v[slot], v], dim=2)
        return self._k[slot], self._v[slot]

    def write3(
        self, k1: torch.Tensor, k2: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slot = self._cursor
        self._cursor += 1
        if self._k[slot] is None:
            self._k[slot], self._k2[slot], self._v[slot] = k1, k2, v
        else:
            self._k[slot] = torch.cat([self._k[slot], k1], dim=2)
            self._k2[slot] = torch.cat([self._k2[slot], k2], dim=2)
            self._v[slot] = torch.cat([self._v[slot], v], dim=2)
        return self._k[slot], self._k2[slot], self._v[slot]
