from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from radiance.config import ModelConfig


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


def padded_vocab_size(vocab_size: int, multiple: int) -> int:
    """Round a tokenizer's vocab up to a multiple of `multiple` for the token_emb/lm_head matmuls.

    A vocab that isn't a multiple of 64/128 leaves the largest matmul in the model (the lm_head,
    d_model x vocab_size) on a ragged tile boundary, so the GPU runs a slow remainder tile. The
    extra rows are pure padding: no tokenizer id ever maps to them, so they never appear as a
    target, and the model simply learns to give them low probability (their embedding rows still
    receive gradient through the softmax denominator). This is the standard nanoGPT trick.

    `multiple <= 1` disables padding and returns vocab_size unchanged.
    """
    if multiple <= 1:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


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


class RMSNorm(nn.Module):
    """RMSNorm, optionally with one set of gains per loop iteration.

    `n_variants > 1` gives the norm a `(n_variants, d_model)` gain matrix indexed by which loop
    iteration is executing (cfg.loop_iter_conditioning). This is the cheapest way to tell a
    weight-shared loop body how deep into the recursion it is — the same trick adaLN uses to
    condition a shared block on a timestep — at ~d_model parameters per iteration.

    At `n_variants == 1` the gain keeps its original `(d_model,)` shape exactly, so a model without
    iteration conditioning is bit-identical to one from before this existed and its checkpoints
    load unchanged.
    """

    def __init__(self, d_model: int, eps: float = 1e-6, n_variants: int = 1):
        super().__init__()
        self.n_variants = n_variants
        shape = (d_model,) if n_variants == 1 else (n_variants, d_model)
        self.weight = nn.Parameter(torch.ones(*shape))
        self.eps = eps
        if n_variants > 1:
            self._register_load_state_dict_pre_hook(self._broadcast_legacy_gain, with_module=False)

    def _broadcast_legacy_gain(self, state_dict, prefix, *args) -> None:
        """Load a pre-conditioning checkpoint by copying its single gain into every variant, so
        enabling loop_iter_conditioning on an existing run starts from exactly that run's weights.
        Mirrors MoEFeedForward._upgrade_legacy_expert_keys."""
        weight = state_dict.get(f"{prefix}weight")
        if weight is not None and weight.dim() == 1:
            state_dict[f"{prefix}weight"] = weight.unsqueeze(0).expand(self.n_variants, -1).contiguous()

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        weight = self.weight
        if self.n_variants > 1:
            # Clamped, not wrapped: running more iterations at inference than were trained (see
            # radiance-generate --loops) should reuse the deepest learned gains rather than
            # silently cycling back to the shallow ones.
            weight = weight[min(variant, self.n_variants - 1)]
        return (x * weight.float()).to(dtype)


class IterLoRA(nn.Module):
    """One low-rank adapter per loop iteration, added to a shared base projection.

    The stronger arm of cfg.loop_iter_conditioning: where per-iteration norm gains can only rescale
    channels, this gives each iteration a genuine rank-r update to the projection itself, at
    n_variants * r * (in + out) parameters instead of a full weight matrix per iteration.

    B is zero-initialised (the standard LoRA convention), so the adapter contributes exactly zero
    at init and the model starts identical to the un-adapted one.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, n_variants: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_variants, in_features, rank))
        self.B = nn.Parameter(torch.zeros(n_variants, rank, out_features))
        nn.init.normal_(self.A, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        v = min(variant, self.A.size(0) - 1)
        return (x @ self.A[v]) @ self.B[v]


def _make_iter_lora(cfg: ModelConfig, in_features: int, out_features: int, n_variants: int):
    """IterLoRA when cfg selects the "lora" arm and there is more than one iteration, else None."""
    if cfg.loop_iter_conditioning != "lora" or n_variants <= 1:
        return None
    return IterLoRA(in_features, out_features, cfg.loop_lora_rank, n_variants)


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


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, is_first: bool = False, n_variants: int = 1):
        """is_first marks blocks[0], which is always called with v_first=None (it is the block that
        *produces* v_first). Its value-residual mix would therefore never be exercised, so the
        parameter isn't created at all — otherwise it would sit in the optimizer forever collecting
        no gradient."""
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

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

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
        q, k, v = qkv.split([d_model, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        v_own = v
        if self.value_residual and v_first is not None:
            # Mixed *before* the kv_cache write below, so cached values are already mixed and
            # generation needs no extra cache slot: within a decode step blocks[0] runs first and
            # produces this token's v_first, and past tokens' mixed values are already stored.
            lam = self.value_lambda
            v = lam * v + (1.0 - lam) * v_first

        if self.qk_norm:
            # Must precede RoPE: RMSNorm's learned per-channel weight is not rotation-invariant,
            # so normalizing after rotation would scale the rotated pairs asymmetrically.
            q = self.q_norm(q)
            k = self.k_norm(k)

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
            k, v = kv_cache.write(k, v)

        if ctx.block_mask is not None:
            # No dropout_p here: flex_attention has no attention-weight dropout. See
            # DenseTransformer._doc_masks, which warns once when a config combines
            # doc_attention_mask with a nonzero dropout. The FFN's residual dropout is unaffected.
            attn_out = _flex_attention()(
                q, k, v, block_mask=ctx.block_mask, enable_gqa=(self.n_kv_heads != self.n_heads)
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=(self.n_kv_heads != self.n_heads),
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
        # the shape the sparse path's retained store needs.
        recorded = (k, v) if ctx.record_kv else None
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


class FeedForward(nn.Module):
    """SwiGLU-gated MLP with configurable depth: `ffn_depth` hidden layers of width `ffn_dim`.
    The first hidden layer is gated (SiLU(gate_proj(x)) * up_proj(x)); any additional depth
    (`ffn_depth > 1`) stacks plain Linear + SiLU layers at `ffn_dim` width on top, preserving the
    "extra hidden layers deepen the MLP, independent of block count" meaning of `ffn_depth`.
    """

    def __init__(self, cfg: ModelConfig, n_variants: int = 1, ffn_dim: int | None = None):
        super().__init__()
        depth = max(1, cfg.ffn_depth)
        ffn_dim = cfg.ffn_dim if ffn_dim is None else ffn_dim
        self.gate_proj = nn.Linear(cfg.d_model, ffn_dim)
        self.up_proj = nn.Linear(cfg.d_model, ffn_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ffn_dim, ffn_dim) for _ in range(depth - 1)])
        self.down_proj = nn.Linear(ffn_dim, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.down_lora = _make_iter_lora(cfg, ffn_dim, cfg.d_model, n_variants)

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        """`variant` (which loop iteration is running) is accepted and ignored, so this and
        MoEFeedForward share one call signature and TransformerBlock/_sparse_ffn_delta never have
        to ask which kind of FFN they hold. The (*, d_model) -> (*, d_model) shape contract that
        lets the two be interchangeable is unchanged."""
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        for layer in self.hidden_layers:
            h = F.silu(layer(h))
        out = self.down_proj(h)
        if self.down_lora is not None:
            out = out + self.down_lora(h, variant)
        return self.dropout(out)


class MoERouter(nn.Module):
    """Per-token expert-routing head for MoEFeedForward. Mirrors ACTRouter's RMSNorm -> Linear
    shape, but projects to n_experts logits (softmaxed over experts) instead of a single halting
    probability, and is instantiated once per MoE FFN layer (not a model-level singleton like
    DenseTransformer.router, which is ACT's halting router — different concept, different name).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.n_experts)
        # Per-iteration routing bias (cfg.loop_iter_conditioning). Because blocks[1:] is
        # weight-shared but fed an evolving hidden state, a token's expert choice already drifts
        # across iterations for free; this makes that drift something the model can *steer* — a
        # given iteration can learn a standing preference for particular experts. Zero-initialised,
        # so routing is unchanged at init.
        self.iter_bias = None
        if cfg.loop_iter_conditioning != "none" and cfg.loop_multiplier > 1:
            self.iter_bias = nn.Parameter(torch.zeros(cfg.loop_multiplier, cfg.n_experts))

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        logits = self.proj(self.norm(x))
        if self.iter_bias is not None:
            logits = logits + self.iter_bias[min(variant, self.iter_bias.size(0) - 1)]
        # fp32 for a stable softmax under bf16/fp16 autocast, mirroring cross_entropy's own fp32
        # upcast (see activation_bytes_per_token's docstring).
        return F.softmax(logits.float(), dim=-1)  # (n_tokens, n_experts)


# MoE capacity is quantised to this many tokens so torch.compile doesn't retrace on every batch.
_CAPACITY_QUANTUM = 128


def _moe_eval_capacity(assigned: torch.Tensor, n_tokens: int) -> int:
    """True per-expert load, quantised, for cfg.moe_eval_full_capacity. See _moe_capacity."""
    # One sync to read the real load. Acceptable outside training, where the alternative is a
    # silently input-dependent result.
    #
    # Rounded up to a multiple of _CAPACITY_QUANTUM rather than used exactly: capacity is a
    # tensor *shape*, so a value that moves with the data would make torch.compile retrace the
    # MoE dispatch on nearly every eval batch. Quantising leaves a handful of distinct shapes
    # while keeping the no-drop guarantee (rounding only ever makes capacity larger).
    peak = max(1, int(assigned.sum(dim=0).max().item()))
    quantised = -(-peak // _CAPACITY_QUANTUM) * _CAPACITY_QUANTUM
    return min(n_tokens, quantised)


def _moe_capacity(cfg: ModelConfig, n_tokens: int, assigned: torch.Tensor | None = None) -> int:
    """Per-expert token capacity.

    Normally the Switch-Transformer formula, which bounds the gather's size so training throughput
    and memory don't depend on how unbalanced routing happens to be that step.

    When `assigned` is given (eval, via cfg.moe_eval_full_capacity) capacity is instead the true
    maximum per-expert load, so nothing is dropped. Dropping is a *training* throughput tradeoff;
    at inference it only discards computation. Worse, because the formula scales with n_tokens, it
    made a token's output depend on how many other tokens shared its forward pass — the same prompt
    scored differently in a batch than alone, and incremental decoding drifted from a full forward.
    """
    if assigned is not None:
        return _moe_eval_capacity(assigned, n_tokens)
    # Capped at n_tokens: a per-expert capacity above the token count is never meaningful (there
    # aren't enough tokens to fill it) and would make torch.topk's k exceed the dimension size.
    return max(1, min(n_tokens, round(cfg.moe_capacity_factor * n_tokens * cfg.moe_top_k / cfg.n_experts)))


class BatchedExperts(nn.Module):
    """`n_experts` parallel FeedForwards whose weights live in single stacked 3-D tensors, so one
    `baddbmm` per layer replaces a Python loop of `n_experts` separate small matmuls.

    Mathematically identical to `nn.ModuleList([FeedForward(cfg) for _ in range(n_experts)])` run
    one expert at a time; the difference is purely how the work is issued to the GPU. The loop
    version's cost is dominated by launch overhead and low occupancy on narrow per-expert matmuls,
    which is why it measured ~4x a single dense FFN while doing only ~2.5x its FLOPs.

    Weights are stored (in_features, out_features) — the transpose of nn.Linear's (out, in) — so
    the forward is a plain `x @ W` with no transpose per call. See MoEFeedForward's state-dict hook
    for the conversion from the old per-expert Linear layout.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # moe_expert_dim, not ffn_dim: fine-grained MoE uses many narrow experts (see
        # cfg.moe_expert_ffn_mult), keeping active parameters per token constant while multiplying
        # the number of expert combinations available.
        n_experts, d_model, ffn_dim = cfg.n_experts, cfg.d_model, cfg.moe_expert_dim
        depth = max(1, cfg.ffn_depth)
        self.mup_hidden_std = 0.02 / cfg.mup_width_mult**0.5
        self.gate_w = nn.Parameter(torch.empty(n_experts, d_model, ffn_dim))
        self.gate_b = nn.Parameter(torch.zeros(n_experts, 1, ffn_dim))
        self.up_w = nn.Parameter(torch.empty(n_experts, d_model, ffn_dim))
        self.up_b = nn.Parameter(torch.zeros(n_experts, 1, ffn_dim))
        self.hidden_w = nn.ParameterList(
            [nn.Parameter(torch.empty(n_experts, ffn_dim, ffn_dim)) for _ in range(depth - 1)]
        )
        self.hidden_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_experts, 1, ffn_dim)) for _ in range(depth - 1)]
        )
        self.down_w = nn.Parameter(torch.empty(n_experts, ffn_dim, d_model))
        self.down_b = nn.Parameter(torch.zeros(n_experts, 1, d_model))
        self.dropout = nn.Dropout(cfg.dropout)
        # DenseTransformer._init_weights only reaches nn.Linear/nn.Embedding submodules, and these
        # are raw Parameters, so match its std here explicitly — including muP's width scaling.
        for w in (self.gate_w, self.up_w, self.down_w, *self.hidden_w):
            nn.init.normal_(w, mean=0.0, std=self.mup_hidden_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_experts, capacity, d_model) -> same shape."""
        h = F.silu(torch.baddbmm(self.gate_b, x, self.gate_w)) * torch.baddbmm(self.up_b, x, self.up_w)
        for w, b in zip(self.hidden_w, self.hidden_b):
            h = F.silu(torch.baddbmm(b, h, w))
        return self.dropout(torch.baddbmm(self.down_b, h, self.down_w))


class MoEFeedForward(nn.Module):
    """Top-k routed MoE FFN, drop-in replacement for FeedForward: (*, d_model) -> (*, d_model),
    shape-agnostic over leading dims so it works both from TransformerBlock.forward's (batch,
    seq_len, d_model) path and from ACT's _sparse_ffn_delta gather path's flat (capacity, d_model)
    path with no special-casing in either.

    Router weights are Mixtral-style: full softmax over all experts, top-k selected, renormalized
    to sum to 1 over just the selected k. Dispatch mirrors _sparse_ffn_delta's gather/compute/
    scatter idiom, generalized to n_experts writers with per-expert fixed capacity (_moe_capacity)
    and drop-on-overflow (same "assigned + random tiebreak" topk priority trick as
    _sparse_ffn_delta, applied per-expert).

    Uses index_add, not _sparse_ffn_delta's index_copy, to accumulate per-expert writes into a
    shared output buffer: with top_k >= 1, a token can receive nonzero contributions from more than
    one expert (top_k=2 by default), and even with top_k=1, capacity padding for one expert can
    land on a token index another expert legitimately wrote to. index_copy would let a later
    expert's zero padding silently clobber an earlier expert's real output at that index;
    index_add sums correctly because unassigned/padding slots are explicitly zeroed by `valid`
    before being added.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.moe_top_k > cfg.n_experts:
            raise ValueError(f"model.moe_top_k ({cfg.moe_top_k}) must be <= model.n_experts ({cfg.n_experts})")
        self.cfg = cfg
        self.router = MoERouter(cfg)
        self.experts = BatchedExperts(cfg)
        # Always-on expert (DeepSeekMoE): its output is added to every token unconditionally, so
        # the routed experts don't each have to re-learn the computation every token needs.
        self.shared_expert = None
        if cfg.moe_n_shared > 0:
            self.shared_expert = FeedForward(cfg, ffn_dim=cfg.moe_n_shared * cfg.moe_shared_dim)

        # Loss-free load balancing (cfg.moe_balance): a non-learned per-expert bias added to the
        # routing logits *for top-k selection only*. Buffer, not parameter — it's updated by an
        # explicit rule after each optimizer step (DenseTransformer.update_expert_bias), never by
        # gradient, which is the entire point: it balances load without adding a term that competes
        # with the LM objective the way the aux loss does.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts), persistent=True)
        self.register_buffer("expert_load", torch.zeros(cfg.n_experts), persistent=False)

        self.aux_loss_accum: torch.Tensor | None = None
        self._aux_loss_calls = 0
        self._register_load_state_dict_pre_hook(self._upgrade_legacy_expert_keys, with_module=False)

    def _upgrade_legacy_expert_keys(self, state_dict, prefix, *args) -> None:
        """Convert checkpoints saved when experts were an nn.ModuleList of FeedForwards.

        Old layout: `{prefix}experts.{e}.{gate,up,down}_proj.{weight,bias}` with nn.Linear's
        (out_features, in_features) weights. New layout stacks them into one (n_experts, in, out)
        tensor per projection, so each old weight is transposed and the experts concatenated.
        """
        legacy = [k for k in state_dict if k.startswith(f"{prefix}experts.") and k.split(".")[-2].endswith("_proj")]
        if not legacy:
            return
        n_experts = self.cfg.n_experts
        for old_name, new_w, new_b in (
            ("gate_proj", "gate_w", "gate_b"),
            ("up_proj", "up_w", "up_b"),
            ("down_proj", "down_w", "down_b"),
        ):
            weights = [state_dict.pop(f"{prefix}experts.{e}.{old_name}.weight") for e in range(n_experts)]
            biases = [state_dict.pop(f"{prefix}experts.{e}.{old_name}.bias") for e in range(n_experts)]
            state_dict[f"{prefix}experts.{new_w}"] = torch.stack([w.t() for w in weights])
            state_dict[f"{prefix}experts.{new_b}"] = torch.stack([b.unsqueeze(0) for b in biases])
        for i in range(max(1, self.cfg.ffn_depth) - 1):
            weights = [state_dict.pop(f"{prefix}experts.{e}.hidden_layers.{i}.weight") for e in range(n_experts)]
            biases = [state_dict.pop(f"{prefix}experts.{e}.hidden_layers.{i}.bias") for e in range(n_experts)]
            state_dict[f"{prefix}experts.hidden_w.{i}"] = torch.stack([w.t() for w in weights])
            state_dict[f"{prefix}experts.hidden_b.{i}"] = torch.stack([b.unsqueeze(0) for b in biases])

    def per_expert_parameter_count(self) -> int:
        """Parameters belonging to a single *routed* expert — the stacked tensors divided by
        n_experts. Used by num_active_parameters() to discount the experts a given token never
        activates. Deliberately excludes the shared expert, which every token activates and so
        counts in full."""
        return sum(p.numel() for p in self.experts.parameters()) // self.cfg.n_experts

    def reset_aux_loss(self) -> None:
        self.aux_loss_accum = None
        self._aux_loss_calls = 0

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        orig_shape = x.shape
        d_model = orig_shape[-1]
        flat_x = x.reshape(-1, d_model)
        n_tokens = flat_x.shape[0]

        probs = self.router(flat_x, variant)  # (n_tokens, n_experts) fp32

        # Selection uses probs + expert_bias; the gating *weights* come from the unbiased probs.
        # That separation is what makes the bias loss-free: it can steer which experts a token goes
        # to without distorting how much each contributes, so it never fights the LM objective.
        selection_scores = probs + self.expert_bias if self._bias_balancing else probs
        topk_idx = selection_scores.topk(self.cfg.moe_top_k, dim=-1).indices
        topk_probs = probs.gather(1, topk_idx)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize over selected k
        weight = probs.new_zeros(n_tokens, self.cfg.n_experts).scatter_(1, topk_idx, topk_probs)

        assigned = weight > 0  # (n_tokens, n_experts)
        if self._bias_balancing and self.training:
            # Accumulated here, consumed by update_expert_bias() after the optimizer step — outside
            # the compiled graph, so no graph break in the forward.
            self.expert_load += assigned.sum(dim=0).detach().float()
        f_i = assigned.float().mean(dim=0).detach()  # fraction of tokens routed to expert i (non-diff)
        p_i = probs.mean(dim=0)  # mean full-softmax prob mass on expert i (differentiable)
        aux_loss = self.cfg.n_experts * (f_i * p_i).sum()
        self.aux_loss_accum = aux_loss if self.aux_loss_accum is None else self.aux_loss_accum + aux_loss
        self._aux_loss_calls += 1

        # Outside training, size capacity to the real load so nothing is dropped — see _moe_capacity.
        full_capacity = self.cfg.moe_eval_full_capacity and not self.training
        capacity = _moe_capacity(self.cfg, n_tokens, assigned if full_capacity else None)

        # Per-expert token selection, computed for all experts at once. Priority within the
        # assigned set is the router's own weight for this expert, not a random draw: when an
        # expert is over capacity, the tokens it drops should be the ones it was least confident
        # about. Router weights are in (0, 1], so `assigned + weight` keeps every assigned token
        # strictly above every unassigned one (which scores < 1), preserving the "assigned always
        # outranks capacity padding" property.
        #
        # This also makes routing deterministic. The previous random tiebreak ran in eval and
        # generation too, so val/loss and even greedy decoding varied run to run for the same
        # weights and inputs — which defeats comparing two configs' eval numbers.
        priority = assigned.float() + weight.detach().float()  # (n_tokens, n_experts)
        token_idx = priority.topk(capacity, dim=0).indices.t().contiguous()  # (n_experts, capacity)
        valid = assigned.t().gather(1, token_idx)  # (n_experts, capacity) bool
        gathered = flat_x[token_idx]  # (n_experts, capacity, d_model)

        expert_out = self.experts(gathered) * valid.unsqueeze(-1).to(x.dtype)
        w = (weight.t().gather(1, token_idx) * valid.to(weight.dtype)).to(x.dtype)

        # index_add, not index_copy: a token can receive nonzero contributions from more than one
        # expert (top_k >= 2 by default), and even at top_k=1 one expert's zero capacity-padding can
        # land on an index another expert legitimately wrote. index_copy would let the padding
        # clobber the real output; index_add sums correctly because `valid` zeroed the padding.
        delta = flat_x.new_zeros(n_tokens, d_model).index_add(
            0, token_idx.reshape(-1), (w.unsqueeze(-1) * expert_out).reshape(-1, d_model)
        )
        if self.shared_expert is not None:
            # Unconditional, ungated, no capacity limit: every token gets this.
            delta = delta + self.shared_expert(flat_x)
        return delta.view(orig_shape)

    @property
    def _bias_balancing(self) -> bool:
        return self.cfg.moe_balance in ("bias", "both")

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """Nudge expert_bias toward whichever experts are under-loaded, then reset the counter.

        Called once per optimizer step from train(), never inside forward — it mutates a buffer
        with no gradient, so keeping it out of the graph avoids a torch.compile break.

        The update is sign-based (DeepSeek-V3): a fixed step toward balance regardless of how far
        off an expert is. That makes it insensitive to the scale of the load imbalance and keeps
        the bias from oscillating when one batch happens to be unrepresentative.
        """
        if not self._bias_balancing or self.expert_load.sum() == 0:
            return
        mean_load = self.expert_load.mean()
        self.expert_bias -= self.cfg.moe_bias_update_rate * torch.sign(self.expert_load - mean_load)
        self.expert_load.zero_()


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
        self.attn = CausalSelfAttention(cfg, is_first=is_first, n_variants=n_variants)
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
    """

    def __init__(self, num_slots: int):
        self._k: list[torch.Tensor | None] = [None] * num_slots
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

def _maybe_checkpoint(fn, *args, enabled: bool):
    """Run fn(*args) under gradient checkpointing when `enabled`, else call it directly.

    use_reentrant=False is the non-deprecated implementation and the only one that composes with
    the rest of this model: it supports inputs that don't require grad (the cos/sin tables) and
    doesn't require the output to require grad, both of which the reentrant version chokes on.
    """
    if not enabled:
        return fn(*args)
    return checkpoint(fn, *args, use_reentrant=False)


def _run_loop_body(
    blocks: nn.ModuleList,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ctx: LoopContext,
    v_first: torch.Tensor | None = None,
    still_running: torch.Tensor | None = None,
    grad_checkpoint: bool = False,
    record_kv: list | None = None,
) -> torch.Tensor:
    """Runs `blocks` once. Attention is always fully dense. When still_running is given (and
    ctx.capacity is set), each block's FFN is dispatched through the fixed-capacity sparse path
    (_sparse_ffn_delta) instead of densely.

    v_first is blocks[0]'s attention values, for value residual. It is passed *positionally* into
    the checkpointed call rather than carried on ctx because it requires grad — see LoopContext.

    grad_checkpoint recomputes each block's activations during backward instead of storing them —
    see DenseTransformer.forward for why this matters disproportionately here.
    """
    for index, block in enumerate(blocks):
        if still_running is None:
            x, _, recorded = _maybe_checkpoint(block, x, cos, sin, ctx, v_first, enabled=grad_checkpoint)
            if record_kv is not None:
                record_kv[index] = recorded
        else:

            def run_sparse(x, cos, sin, v_first, still_running, block=block):
                variant = ctx.variant
                if block.hyper_attn is None:
                    attn_out, _, _ = block.attn(block.ln1(x, variant), cos, sin, ctx, v_first)
                    x = x + attn_out
                    return x + _sparse_ffn_delta(
                        block.ffn, block.ln2(x, variant), still_running, ctx.capacity, variant
                    )
                # Same structure with the residual writes replaced by hyper-connection read/write
                # pairs — see TransformerBlock.forward. _sparse_ffn_delta returns a delta over the
                # (batch, seq, d_model) sublayer output, which is exactly what write() expects.
                h, coeffs = block.hyper_attn.read(x, variant)
                attn_out, _, _ = block.attn(block.ln1(h, variant), cos, sin, ctx, v_first)
                x = block.hyper_attn.write(x, attn_out, coeffs)
                h, coeffs = block.hyper_ffn.read(x, variant)
                delta = _sparse_ffn_delta(
                    block.ffn, block.ln2(h, variant), still_running, ctx.capacity, variant
                )
                return block.hyper_ffn.write(x, delta, coeffs)

            x = _maybe_checkpoint(
                run_sparse, x, cos, sin, v_first, still_running, enabled=grad_checkpoint
            )
    return x


class MTPHead(nn.Module):
    """One auxiliary multi-token-prediction head (DeepSeek-V3's formulation).

    Predicts a token further ahead than the trunk does, by fusing the previous head's hidden state
    at position t with the embedding of the token the previous head was predicting, then running a
    single transformer block over the result. The unembedding is *shared* with the trunk's lm_head,
    which is what keeps a head's cost to one block rather than another d_model x vocab_size matrix.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_hidden = RMSNorm(cfg.d_model)
        self.norm_embedding = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.block = TransformerBlock(cfg, is_first=True)

    def forward(
        self,
        hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.proj(
            torch.cat([self.norm_hidden(hidden), self.norm_embedding(token_embeddings)], dim=-1)
        )
        # Fresh LoopContext: the head is outside the recursion, and passing the trunk's kv_cache
        # would let it claim cache slots that belong to the trunk's blocks.
        out, _, _ = self.block(fused, cos, sin, LoopContext())
        return out


def _shift_left(x: torch.Tensor, positions: int) -> torch.Tensor:
    """Shift a (batch, seq, ...) tensor left along seq, zero-padding the tail.

    Head j reads the embedding of the token j positions ahead; the padded tail positions have no
    real future token, and their loss contribution is masked out by compute_loss's ignore_index.
    """
    if positions == 0:
        return x
    pad = torch.zeros_like(x[:, :positions])
    return torch.cat([x[:, positions:], pad], dim=1)


def _run_loop_body_sparse(
    blocks: nn.ModuleList,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ctx: LoopContext,
    v_first: torch.Tensor | None,
    token_idx: torch.Tensor,
    valid: torch.Tensor,
    kv_store: list,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """One ACT loop iteration that computes only `capacity` positions per sequence.

    Gathers the selected positions once, runs the *entire* loop-body stack on that narrow tensor,
    then scatters the result back. Unselected positions keep their previous hidden state and serve
    their retained keys/values to everyone else — which is where the saving comes from: the attention
    projections, the attention itself, the output projection and the FFN all run at `capacity`
    instead of `seq_len`.

    kv_store is updated per block as the stack runs, so the next iteration attends against the
    freshest values each position produced.
    """
    batch = x.shape[0]
    capacity = token_idx.size(1)
    # Trailing dims are (d_model,) normally and (n_streams, d_model) under hyper-connections; the
    # gather indexes positions either way, so it just grows a dim.
    trailing = x.shape[2:]
    gather_idx = token_idx.reshape(batch, capacity, *(1,) * len(trailing)).expand(
        batch, capacity, *trailing
    )
    original = x.gather(1, gather_idx)
    h = original

    # RoPE tables and v_first are full-length; index them down to the selected positions once,
    # outside the block loop, since every block reuses them.
    cos_g, sin_g = cos[token_idx], sin[token_idx]
    v_first_g = None
    if v_first is not None:
        n_kv_heads, head_dim = v_first.size(1), v_first.size(3)
        v_first_g = v_first.gather(
            2, token_idx[:, None, :, None].expand(batch, n_kv_heads, capacity, head_dim)
        )

    variant = ctx.variant
    for index, block in enumerate(blocks):
        # Hyper-connection read/write pairs in place of the two residual writes, exactly as in
        # TransformerBlock.forward — the gathered tensor is narrower along seq but has the same
        # trailing shape, so the connection units need no special-casing for this path.
        hc_attn, hc_ffn = block.hyper_attn, block.hyper_ffn
        attn_in, coeffs = (h, None) if hc_attn is None else hc_attn.read(h, variant)
        attn_out, kv_store[index] = block.attn.forward_sparse(
            block.ln1(attn_in, variant), cos_g, sin_g, kv_store[index], token_idx, attn_mask, v_first_g
        )
        h = h + attn_out if hc_attn is None else hc_attn.write(h, attn_out, coeffs)
        ffn_in, coeffs = (h, None) if hc_ffn is None else hc_ffn.read(h, variant)
        ffn_out = block.ffn(block.ln2(ffn_in, variant), variant)
        h = h + ffn_out if hc_ffn is None else hc_ffn.write(h, ffn_out, coeffs)

    # Capacity padding rows (selected only to keep the shape static) must not overwrite anything,
    # so they are restored to what they came in as before the scatter. token_idx is distinct per
    # row, so the scatter has no colliding writes.
    h = torch.where(valid.reshape(batch, capacity, *(1,) * len(trailing)), h, original)
    return x.scatter(1, gather_idx, h)


class DenseTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, eos_id: int | None = None):
        """eos_id is what document_ids() reads the packed document boundaries off of. It defaults
        to None (masking silently off) so a checkpoint saved before doc masking existed still
        rebuilds — generate.py doesn't pass one, since a single prompt is one document anyway."""
        super().__init__()
        self.cfg = cfg
        self.eos_id = eos_id
        self._warned_dropout_with_flex = False
        self._warned_head_dim = False
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.dropout = nn.Dropout(cfg.dropout)

        # blocks[0] always stays dense (it runs once per forward, not part of the recursive loop
        # body); blocks[1:] use MoEFeedForward when use_moe is set, except every moe_dense_every-th
        # position (1-indexed within blocks[1:]), which stays dense too.
        # Only the loop body gets per-iteration parameters: blocks[0] runs exactly once per forward,
        # so there is no iteration for it to be conditioned on.
        n_variants = cfg.loop_multiplier if cfg.loop_iter_conditioning != "none" else 1
        # Hyper-connections (cfg.hyper_conn_streams) reuse n_variants as-is rather than adding a
        # second conditioning knob: at n_variants > 1 each loop iteration gets its own routing over
        # the streams, which is the same thing per-iteration norm gains do for scale.
        self.hyper_streams = cfg.hyper_conn_streams
        hyper = self.hyper_streams > 1
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, use_moe_ffn=False, is_first=True, hyper=hyper, block_index=0)]
        )
        for i in range(cfg.n_layers - 1):
            is_dense_override = cfg.use_moe and cfg.moe_dense_every and (i + 1) % cfg.moe_dense_every == 0
            self.blocks.append(
                TransformerBlock(
                    cfg,
                    use_moe_ffn=cfg.use_moe and not is_dense_override,
                    n_variants=n_variants,
                    hyper=hyper,
                    block_index=i + 1,
                )
            )

        # Re-injects the token embedding at the start of each loop iteration after the first
        # (cfg.loop_input_injection). Zero-initialised in _init_inert_gates. Only built when the
        # loop actually runs more than once: at loop_multiplier == 1 there is no second iteration
        # to inject into, so the projection would sit in the optimizer collecting no gradient.
        self.input_injection = None
        if cfg.loop_input_injection and cfg.loop_multiplier > 1:
            self.input_injection = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Plain Python list, not a second nn.ModuleList — these modules are already registered via
        # self.blocks; wrapping them again risks confusing state_dict/double-registration.
        self._moe_layers = [m for m in self.modules() if isinstance(m, MoEFeedForward)]

        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        if cfg.act_capacity_ratio < 1.0 and cfg.act_ffn_capacity_ratio < 1.0:
            raise ValueError(
                "Set only one of model.act_capacity_ratio (whole-block sparsity) and "
                "model.act_ffn_capacity_ratio (the older FFN-only version it supersedes)."
            )
        if cfg.act_capacity_ratio < 1.0 and cfg.grad_checkpoint:
            # The sparse path threads a retained K/V store across loop iterations, updating it as
            # each block runs. Under checkpointing the block would be re-executed during backward
            # and write into that store a second time, silently corrupting it. Refuse rather than
            # produce wrong gradients — and note that sparsity is itself a large activation-memory
            # reduction, so the two are largely redundant anyway.
            raise ValueError(
                "model.act_capacity_ratio < 1.0 is incompatible with model.grad_checkpoint: the "
                "sparse ACT path carries a K/V store across iterations, which recomputation during "
                "backward would write into twice. Sparsity already cuts activation memory; disable "
                "grad_checkpoint."
            )
        if cfg.act_capacity_ratio < 1.0 and not cfg.use_router:
            raise ValueError(
                "model.act_capacity_ratio only applies to router mode (model.use_router: true) — "
                "it selects which *still-running* positions to compute, and without ACT halting "
                "every position is always running."
            )

        if cfg.loop_bptt_window is not None and self.input_injection is None:
            # Truncated BPTT runs the early loop iterations under no_grad, which severs the graph
            # *upstream* of the recurrence too: blocks[0]'s only route to the loss is through those
            # iterations, so it would receive no gradient and stay frozen at its initial weights for
            # the entire run — silently, since the loss still falls (every other block trains).
            # Input injection restores the path by feeding blocks[0]'s output into the in-window
            # iterations directly. Refuse the combination rather than train a crippled model.
            raise ValueError(
                "model.loop_bptt_window requires model.loop_input_injection (and a loop that runs "
                "more than once): without the injected anchor, truncating backprop leaves blocks[0] "
                "disconnected from the loss and it never trains."
            )

        # Auxiliary multi-token-prediction heads (cfg.mtp_heads). mtp_heads=1 means ordinary
        # next-token prediction and builds none.
        self.mtp_heads = nn.ModuleList(MTPHead(cfg) for _ in range(max(0, cfg.mtp_heads - 1)))

        self.router = None
        if cfg.use_router:
            assert cfg.max_loops >= 1
            self.router = ACTRouter(cfg)

        # muP output multiplier: logits are scaled by 1/width_mult so their scale is invariant to
        # d_model. Exactly 1.0 (and skipped entirely in forward) unless cfg.mup_base_d_model is set.
        self.mup_output_mult = 1.0 / cfg.mup_width_mult

        self.apply(self._init_weights)
        # token_emb and lm_head are the *same* tensor (weight tying), and self.apply visits
        # token_emb (an nn.Embedding) before lm_head (an nn.Linear) — so under muP the Linear
        # branch of _init_weights would overwrite the embedding's std with the hidden-weight one.
        # Re-apply the embedding's own std last. No-op when width_mult == 1, where they agree.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        self._init_inert_gates()
        self._scale_residual_init()
        if self.router is not None:
            # Bias the halting unit against halting immediately (Graves ACT): sigmoid(-1) ≈ 0.27,
            # encouraging some early pondering rather than collapsing to a single pass at init.
            nn.init.constant_(self.router.proj.bias, -1.0)

    def _init_weights(self, module: nn.Module) -> None:
        # muP: hidden weights initialise at std / sqrt(width_mult) so that activation scale is
        # invariant to d_model, while the embedding's std stays fixed (its fan-in is 1 — a row
        # lookup — so it doesn't widen). width_mult is exactly 1.0 unless cfg.mup_base_d_model is
        # set, which is what keeps this a no-op for every config that hasn't opted into muP.
        hidden_std = 0.02 / self.cfg.mup_width_mult**0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_inert_gates(self) -> None:
        """Zero every projection whose feature is designed to be a no-op at initialisation.

        Must run *after* self.apply(self._init_weights), which sets every nn.Linear to
        normal(0, 0.02) and would otherwise overwrite these — the same ordering constraint
        _scale_residual_init has.

        Currently just the attention output gate: with a zeroed weight and bias the gate evaluates
        to 2 * sigmoid(0) = 1.0 for every head and every token, so a gated model's forward pass is
        bit-identical to an ungated one until training moves these weights off zero. That property
        is what makes cfg.attn_out_gate safe to default on.
        """
        for name, module in self.named_modules():
            if name.endswith("out_gate") and isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        # Input injection starts at exactly zero, so h + W_inj @ anchor == h at any loop count.
        if self.input_injection is not None:
            nn.init.zeros_(self.input_injection.weight)

    def _scale_residual_init(self) -> None:
        """Scale every projection that *writes into* the residual stream by 1/sqrt(n_residual_writes).

        Each block adds two terms to the residual stream (attn.out_proj and ffn.down_proj). With
        all of them initialized at the same std, the residual's variance grows linearly in the
        number of writes, so the deeper the stack the larger the activations entering ln_f — the
        standard GPT-2 fix is to shrink these particular projections at init so the *summed*
        residual keeps roughly unit scale.

        The count that matters here is the number of writes actually executed per forward, not
        n_layers: blocks[1:] is a weight-shared loop body re-run loop_count (or, in router mode,
        max_loops) times, so a 6-layer model at loop_count=4 performs the residual writes of a
        21-block stack and needs to be scaled as such. This is exactly the regime where the
        unscaled init hurts most, since looping multiplies depth without adding parameters.
        """
        n_blocks_executed = 1 + self.cfg.loop_multiplier * (self.cfg.n_layers - 1)
        scale = (2 * n_blocks_executed) ** -0.5
        # "down_w" catches BatchedExperts' stacked down-projection, the MoE counterpart of a dense
        # block's ffn.down_proj.
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight", "experts.down_w")):
                param.data.mul_(scale)

    def new_kv_cache(self, loop_count: int | None = None) -> KVCache:
        """Builds an empty KVCache sized for this model/config: one slot for blocks[0] plus one
        per (block, loop-iteration) pair the loop body (blocks[1:]) executes each forward call.

        `loop_count` sizes the cache for a specific iteration count instead of the config's
        maximum — needed by radiance-generate --loops, which can run *more* iterations at inference
        than training used.
        """
        multiplier = loop_count if loop_count is not None else self.cfg.loop_multiplier
        return KVCache(1 + multiplier * (self.cfg.n_layers - 1))

    def resolved_loop_count(self, override: int | None = None) -> int:
        """How many times to run the loop body on this forward pass.

        Under stochastic depth (cfg.loop_count_min/max) a training step samples uniformly from the
        configured range; eval and generation always take the top of the range, so val/loss and
        sampled text stay deterministic and reflect the model's full compute budget. When the range
        is unset both ends collapse to loop_count and this is a constant, which is what makes
        stochastic depth inert by default.
        """
        if override is not None:
            return override
        if self.cfg.use_router:
            return self.cfg.max_loops
        low = self.cfg.loop_count_min or self.cfg.loop_count
        high = self.cfg.loop_count_max or self.cfg.loop_count
        if self.training and high > low:
            return int(torch.randint(low, high + 1, (1,)).item())
        return high

    def _loop_grad_context(self, iteration: int, total: int):
        """no_grad for loop iterations outside the truncated-BPTT window, else a passthrough.

        Iterations before the last `loop_bptt_window` contribute activations but no gradient, so
        their memory is freed as the forward proceeds instead of being retained until backward.
        """
        window = self.cfg.loop_bptt_window
        if window is None or not torch.is_grad_enabled() or iteration > total - window:
            return contextlib.nullcontext()
        return torch.no_grad()

    # Runs eagerly even when the model is compiled. create_block_mask builds a BlockMask out of
    # data-dependent index tensors; traced into the enclosing graph those become intermediates with
    # an unresolved layout and inductor fails to lower them ("convert FlexibleLayout to FixedLayout
    # first"). Disabling dynamo here graph-breaks once at the top of forward — before any block runs
    # — and hands the compiled region a concrete BlockMask, which is what flex_attention wants.
    @torch._dynamo.disable
    def _doc_masks(self, input_ids: torch.Tensor, offset: int, kv_cache) -> dict[int, Any] | None:
        """One BlockMask per distinct attention window, keyed by window size, or None when document
        masking doesn't apply to this forward pass.

        Returns a dict rather than a single mask because cfg.loop_attn_windows can give each loop
        iteration a different receptive field; forward() picks per iteration. With no windows
        configured there is exactly one entry.
        """
        if not self.cfg.doc_attention_mask or self.eos_id is None:
            return None
        # Generation is a single document, so the mask would be all-ones — and flex_attention
        # against an incrementally growing cache needs different plumbing for no benefit.
        if kv_cache is not None:
            return None
        # flex_attention is impractically slow outside CUDA; CLAUDE.md's CPU sanity-check workflow
        # has to keep working, so fall back to plain causal SDPA there.
        if input_ids.device.type != "cuda":
            return None
        # flex_attention's compiled kernel requires head_dim >= 16. Real configs use 64, but the
        # tiny models used for sanity checks can go below it, and the failure mode is an
        # InductorError raised from deep inside the compiler rather than anything actionable.
        if self.cfg.head_dim < _FLEX_MIN_HEAD_DIM:
            if not self._warned_head_dim:
                print(
                    f"[radiance] note: model.doc_attention_mask is off for this run — "
                    f"flex_attention requires head_dim >= {_FLEX_MIN_HEAD_DIM}, but "
                    f"model.head_dim={self.cfg.head_dim}. Falling back to plain causal attention."
                )
                self._warned_head_dim = True
            return None

        if self.cfg.dropout > 0 and self.training and not self._warned_dropout_with_flex:
            print(
                f"[radiance] note: model.doc_attention_mask is on, so attention-weight dropout "
                f"(model.dropout={self.cfg.dropout}) is not applied — flex_attention has no "
                f"dropout_p. The FFN's residual dropout still is. Set model.dropout: 0.0 to make "
                f"this explicit."
            )
            self._warned_dropout_with_flex = True

        doc_ids = document_ids(input_ids, self.eos_id)
        seq_len = input_ids.size(1)
        windows = set(self.cfg.loop_attn_windows or []) or {None}
        return {w: build_block_mask(doc_ids, seq_len, offset, w) for w in windows}

    def _mask_for_iteration(self, masks: dict[int, Any] | None, iteration: int):
        """Select the BlockMask governing a given loop iteration (cfg.loop_attn_windows)."""
        if masks is None:
            return None
        windows = self.cfg.loop_attn_windows
        if not windows:
            return next(iter(masks.values()))
        # Clamped at the end of the list, so running more iterations than windows were configured
        # for keeps the last (widest, by convention) window rather than cycling back to a local one.
        return masks[windows[min(max(0, iteration - 1), len(windows) - 1)]]

    def _run_mtp_heads(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache,
    ) -> tuple[torch.Tensor, ...] | None:
        """Chain the auxiliary heads off the trunk's final hidden state.

        Training-only: at eval and during generation the heads contribute nothing to val/loss or to
        sampling, so skipping them keeps both paths exactly as fast as a model without MTP.
        """
        if not self.mtp_heads or not self.training or not torch.is_grad_enabled() or kv_cache is not None:
            return None

        embeddings = self.token_emb(input_ids)
        outputs, h = [], hidden
        for depth, head in enumerate(self.mtp_heads, start=1):
            # Head `depth` predicts token t+1+depth, so it is fed the embedding of token t+depth.
            h = head(h, _shift_left(embeddings, depth), cos, sin)
            outputs.append(h)
        return tuple(outputs)

    def _project_logits(self, x: torch.Tensor) -> torch.Tensor:
        """lm_head plus muP's output multiplier.

        The multiply is skipped rather than performed against 1.0 so a non-muP model's forward is
        bit-identical to one without this method — `logits * 1.0` is still a real kernel that
        rounds, and it would show up as a difference in the inert-default tests.
        """
        logits = self.lm_head(x)
        if self.mup_output_mult != 1.0:
            logits = logits * self.mup_output_mult
        return logits

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, seq, n_streams, d_model), n identical copies.

        No-op at hyper_conn_streams == 1, so a default model never grows the extra dimension and
        every downstream shape stays exactly what it was.
        """
        if self.hyper_streams == 1:
            return x
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.hyper_streams, x.size(-1))

    def _reduce_streams(self, hidden: torch.Tensor) -> torch.Tensor:
        """The inverse: combine the streams back into one (batch, seq, d_model) hidden state.

        A **mean**, where the paper sums. The difference is not cosmetic, and the reason is worth
        keeping: at init every stream holds the same vector, so a sum hands ln_f exactly n times
        the single-stream model's hidden state (verified bit-exact — the streams really are
        identical and the reduction really is n*z to the last ulp). RMSNorm is scale-invariant in
        exact arithmetic, so that ought to vanish — but `+ self.eps` sits inside the rsqrt and is
        *not* scale-invariant, so a sum silently changes ln_f's effective normalisation as a
        function of n. Measured at d_model 32 that was a 1e-3 logit shift, which would have shown
        up in an n=2 vs n=4 A/B as an architecture effect when it was really an epsilon artifact.
        Averaging keeps the reduced hidden at single-stream scale for every n, which makes the
        model exactly the plain residual network at init and is also why this repo does not need
        the paper's compensating "scale output-module init std by sqrt(n)" adjustment.
        """
        if self.hyper_streams == 1:
            return hidden
        return hidden.mean(dim=-2)

    def _reset_moe_aux_loss(self) -> None:
        for moe_ffn in self._moe_layers:
            moe_ffn.reset_aux_loss()

    def update_expert_bias(self) -> None:
        """Apply the loss-free load-balancing update to every MoE layer. Called by train() after
        optimizer.step(), so the buffer mutation stays outside the compiled graph."""
        for moe_ffn in self._moe_layers:
            moe_ffn.update_expert_bias()

    def expert_bias_spread(self) -> float:
        """Std of the balancing biases, averaged over MoE layers — a diagnostic for how hard the
        balancer is working. It should settle: a value that keeps growing means routing genuinely
        wants to collapse and the bias is the only thing preventing it."""
        if not self._moe_layers:
            return 0.0
        return float(
            sum(m.expert_bias.std().item() for m in self._moe_layers) / len(self._moe_layers)
        )

    def _collect_moe_aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        # "bias" balances load without a gradient term, so the aux loss is simply not applied —
        # returning zero here rather than weighting it by zero keeps it out of the graph entirely.
        if not self._moe_layers or self.cfg.moe_balance == "bias":
            return x.new_zeros(())
        # Averaged per-layer over its call count (once for loop_count/non-router mode, up to
        # max_loops times for ACT mode) so the aux loss magnitude doesn't scale with loop depth,
        # consistent with ponder_cost/mean_loop_depth also being mean-reduced, not summed.
        return sum(
            moe_ffn.aux_loss_accum / moe_ffn._aux_loss_calls for moe_ffn in self._moe_layers
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        loop_count: int | None = None,
    ) -> ModelOutput:
        """Returns a ModelOutput (logits, ponder_cost, mean_loop_depth, moe_aux_loss). The latter
        three are zero scalar tensors when the corresponding feature (use_router / use_moe) is off,
        so callers have one contract regardless of mode.

        kv_cache is optional and defaults to None (the training/full-sequence path, unchanged).
        When given, input_ids is the *new* chunk only (the whole prompt on the first call, one
        token per call thereafter) and attention is computed against cached past K/V plus this
        chunk; see KVCache and CausalSelfAttention.forward."""
        batch, seq_len = input_ids.shape
        offset = kv_cache.seq_len if kv_cache is not None else 0
        assert offset + seq_len <= self.cfg.max_seq_len, "sequence length exceeds max_seq_len"

        x = self.token_emb(input_ids)
        x = self.dropout(x)
        # No-op unless cfg.hyper_conn_streams > 1, in which case everything from here to ln_f
        # carries a (batch, seq, n_streams, d_model) hidden state instead of (batch, seq, d_model).
        x = self._expand_streams(x)
        cos, sin = self.rope(seq_len, offset)

        if kv_cache is not None:
            kv_cache.begin_step()

        self._reset_moe_aux_loss()

        # Gradient checkpointing is a backward-pass optimization, so it's pointless (and, with a
        # kv_cache, actively wrong — recomputation would re-append to the cache) whenever there's no
        # graph to build. Gate on both, so eval/generation always take the plain path.
        grad_checkpoint = self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled() and kv_cache is None

        # Built once and reused by every block and every loop iteration, like cos/sin.
        doc_masks = self._doc_masks(input_ids, offset, kv_cache)
        # Raw document ids, separately from the flex BlockMask: ACT's sparse path builds its own
        # (batch, 1, capacity, seq_len) mask for scattered queries, which a BlockMask can't express.
        doc_ids = None
        if self.cfg.doc_attention_mask and self.eos_id is not None and kv_cache is None:
            doc_ids = document_ids(input_ids, self.eos_id)

        # first block runs once; remaining n_layers - 1 blocks form the loop body. Its values become
        # v_first for every later block on every later iteration (cfg.value_residual).
        ctx = LoopContext(
            iteration=0, kv_cache=kv_cache, block_mask=self._mask_for_iteration(doc_masks, 1),
        )
        x, v_first, _ = _maybe_checkpoint(self.blocks[0], x, cos, sin, ctx, None, enabled=grad_checkpoint)
        if not self.cfg.value_residual:
            v_first = None

        # The anchor re-injected at every later loop iteration (cfg.loop_input_injection) is
        # blocks[0]'s *output*, not the raw embedding. Two reasons, one of them load-bearing:
        # it is the contextualised representation rather than a bare token lookup, and — critically
        # — it keeps blocks[0] connected to the loss under truncated BPTT. With loop_bptt_window
        # set, the early iterations run under no_grad, which severs the only other path from
        # blocks[0] to the loss; injecting its output into the in-window iterations restores it.
        # See the loop_bptt_window/loop_input_injection validation in __init__.
        # Reduced back to (batch, seq, d_model) under hyper-connections, so input_injection stays a
        # plain d_model -> d_model projection; its output is written into *every* stream, mirroring
        # a hyper-connection's own all-ones output distribution. W_inj is zero-initialised, so this
        # is inert at init either way.
        anchor = self._reduce_streams(x)

        if not self.cfg.use_router:
            # remaining n_layers - 1 blocks are looped, sharing weights across iterations
            n_loops = self.resolved_loop_count(loop_count)
            for n in range(1, n_loops + 1):
                if n > 1 and self.input_injection is not None:
                    # Re-anchor the recursion. From the second iteration on only: the first is
                    # already reading blocks[0]'s output, which *is* the anchor.
                    x = x + self._expand_streams(self.input_injection(anchor))
                with self._loop_grad_context(n, n_loops):
                    x = _run_loop_body(
                        self.blocks[1:],
                        x,
                        cos,
                        sin,
                        LoopContext(
                            iteration=n,
                            kv_cache=kv_cache,
                            block_mask=self._mask_for_iteration(doc_masks, n),
                        ),
                        v_first,
                        grad_checkpoint=grad_checkpoint,
                    )
            x = self.ln_f(self._reduce_streams(x))
            logits = self._project_logits(x)
            zero = x.new_zeros(())
            moe_aux_loss = self._collect_moe_aux_loss(x)
            mtp_hidden = self._run_mtp_heads(x, input_ids, cos, sin, kv_cache)
            if kv_cache is not None:
                kv_cache.seq_len += seq_len
            return ModelOutput(logits, zero, zero, moe_aux_loss, mtp_hidden)

        result = self._forward_act(
            x, cos, sin, v_first, anchor, kv_cache, grad_checkpoint, loop_count, doc_masks,
            input_ids, doc_ids,
        )
        if kv_cache is not None:
            kv_cache.seq_len += seq_len
        return result

    def _forward_act(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        v_first: torch.Tensor | None = None,
        anchor: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        grad_checkpoint: bool = False,
        loop_count: int | None = None,
        doc_masks: dict | None = None,
        input_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Adaptive Computation Time (Graves 2016) over the loop body (blocks[1:]): each token
        position gets its own halting probability per iteration, halts once its cumulative
        probability crosses 1 - halt_epsilon (or max_loops is reached), and the output is a
        probability-weighted sum of per-iteration hidden states rather than just the final one.

        Once a position halts, its state is frozen and carried forward unchanged on later
        iterations: still-running positions' causal attention keeps reading a stable key/value for
        it, and its own (recomputed but discarded) update never contributes to the output again.

        Attention is always fully dense (over every position, every iteration) regardless of
        halting. The FFN sublayer, however, is only run for every position on the first and last
        iterations; interior iterations dispatch FFN through a fixed-capacity gather/scatter
        (_sparse_ffn_delta) that processes at most `_ffn_capacity(cfg, ...)` still-running
        positions and skips FFN for the rest that iteration (see cfg.act_ffn_capacity_ratio) —
        this is opt-in (default ratio 1.0 keeps the original fully-dense behavior) and is not a
        bit-exact speedup of the dense computation: a halted position's block-to-block-evolving
        intermediate value (read as attention K/V by later internal blocks within one iteration)
        can no longer match the fully-dense computation once FFN is skipped for it anywhere in the
        stack. First/last iterations stay dense because the first has no halted positions yet to
        skip, and the last force-halts every remaining position (often at high weight), so an
        overflow drop there risks discarding a high-weight contribution.
        """
        # x is (batch, seq, d_model), or (batch, seq, n_streams, d_model) under hyper-connections —
        # so index the leading two dims rather than unpacking, and broadcast the per-position
        # halting tensors against however many trailing dims the hidden state has.
        batch, seq_len = x.shape[0], x.shape[1]
        # (batch, seq, 1) normally, (batch, seq, 1, 1) with streams.
        halt_shape = (batch, seq_len) + (1,) * (x.dim() - 2)

        cum_prob = x.new_zeros(batch, seq_len)
        n_updates = x.new_zeros(batch, seq_len)
        remainder_sum = x.new_zeros(batch, seq_len)
        still_running = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)
        accum_output = torch.zeros_like(x)
        frozen_x = x

        sparse_enabled = self.cfg.act_ffn_capacity_ratio < 1.0
        capacity = _ffn_capacity(self.cfg, batch, seq_len) if sparse_enabled else None
        max_loops = loop_count if loop_count is not None else self.cfg.max_loops

        # Whole-block sparsity (cfg.act_capacity_ratio) is a *training-time* compute optimisation,
        # in the same sense grad_checkpoint is a training-time memory one: it is off under eval,
        # no_grad and any kv_cache.
        #
        # Restricting it to training is what keeps eval and generation consistent with each other.
        # If it ran during a full forward but not during cached decoding — which cannot use it,
        # since decoding feeds one token at a time and there is nothing to select among — then the
        # same prompt would be scored one way by evaluate() and generated another way by
        # radiance-generate. Confining it to training also keeps val/loss comparable across capacity
        # ratios, since eval always measures the full dense model.
        block_sparse = (
            self.cfg.act_capacity_ratio < 1.0
            and kv_cache is None
            and self.training
            and torch.is_grad_enabled()
        )
        block_capacity = max(1, min(seq_len, round(self.cfg.act_capacity_ratio * seq_len)))
        # Seeded by the first (dense) iteration, then updated in place by each sparse one.
        kv_store: list = [None] * len(self.blocks[1:])
        sparse_mask_doc_ids = doc_ids if self.cfg.doc_attention_mask else None

        for n in range(1, max_loops + 1):
            is_first_or_last = n == 1 or n == max_loops
            if n > 1 and self.input_injection is not None:
                # Only for still-running positions: a halted position's state is frozen, and later
                # iterations must keep reading exactly the key/value it halted with. Injecting into
                # it would break that invariant and silently change what earlier-halted tokens
                # contribute to everyone else's attention.
                injected = frozen_x + self._expand_streams(self.input_injection(anchor))
                frozen_x = torch.where(still_running.reshape(halt_shape), injected, frozen_x)
            ctx = LoopContext(
                iteration=n,
                kv_cache=kv_cache,
                capacity=capacity,
                block_mask=self._mask_for_iteration(doc_masks, n),
                # The first iteration seeds the retained K/V store the sparse ones read from.
                record_kv=block_sparse and n == 1,
            )
            if block_sparse and not is_first_or_last:
                # Interior iteration: compute only the highest-priority still-running positions.
                token_idx, valid = _act_select(still_running, block_capacity, self.training)
                attn_mask = _sparse_attn_mask(token_idx, seq_len, sparse_mask_doc_ids)
                new_x = _run_loop_body_sparse(
                    self.blocks[1:], frozen_x, cos, sin, ctx, v_first,
                    token_idx, valid, kv_store, attn_mask,
                )
            elif not sparse_enabled or is_first_or_last:
                new_x = _run_loop_body(
                    self.blocks[1:], frozen_x, cos, sin, ctx, v_first,
                    grad_checkpoint=grad_checkpoint,
                    record_kv=kv_store if ctx.record_kv else None,
                )
            else:
                new_x = _run_loop_body(
                    self.blocks[1:], frozen_x, cos, sin, ctx, v_first, still_running, grad_checkpoint
                )
            # The router halts *positions*, so it reads the reduced hidden state rather than one
            # arbitrary stream — same tensor ln_f eventually sees.
            p_n = self.router(self._reduce_streams(new_x), ctx.variant)

            is_last_step = n == max_loops
            would_exceed = (cum_prob + p_n) >= (1.0 - self.cfg.halt_epsilon)
            halts_now = still_running & (would_exceed | is_last_step)

            remainder = 1.0 - cum_prob
            weight = torch.where(halts_now, remainder, p_n)
            weight = torch.where(still_running, weight, torch.zeros_like(weight))
            accum_output = accum_output + weight.reshape(halt_shape) * new_x

            n_updates = n_updates + still_running.float()
            remainder_sum = torch.where(halts_now, remainder, remainder_sum)
            cum_prob = torch.where(still_running & ~halts_now, cum_prob + p_n, cum_prob)

            frozen_x = torch.where(still_running.reshape(halt_shape), new_x, frozen_x)
            still_running = still_running & ~halts_now

        x = self.ln_f(self._reduce_streams(accum_output))
        logits = self._project_logits(x)
        ponder_cost = (n_updates + remainder_sum).mean()
        mean_loop_depth = n_updates.mean()
        moe_aux_loss = self._collect_moe_aux_loss(x)
        mtp_hidden = self._run_mtp_heads(x, input_ids, cos, sin, kv_cache)
        return ModelOutput(logits, ponder_cost, mean_loop_depth, moe_aux_loss, mtp_hidden)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_active_parameters(self) -> int:
        """Like num_parameters(), but counts only moe_top_k experts' worth of FFN parameters per
        MoE layer instead of all n_experts — the parameter count actually multiplied against any
        given token's activation, which is the right basis for Chinchilla-style tokens_per_param
        sizing (see train.py). Identical to num_parameters() when no MoE layers exist."""
        total = self.num_parameters()
        for moe_ffn in self._moe_layers:
            per_expert_params = moe_ffn.per_expert_parameter_count()
            total -= per_expert_params * (self.cfg.n_experts - self.cfg.moe_top_k)
        # MTP heads are a training-time auxiliary objective and are never run at inference, so they
        # don't multiply against tokens at serving time. Counting them would inflate the
        # tokens_per_param-derived max_steps — the same reasoning that discounts inactive experts.
        total -= sum(p.numel() for p in self.mtp_heads.parameters())
        return total

    def activation_bytes_per_token(self, activation_dtype_bytes: int) -> int:
        """Conservative (deliberately over-, not under-, estimated) activation memory per token,
        for sizing a training batch to available VRAM (see train.py's estimate_batch_size).

        Without gradient checkpointing (cfg.model.grad_checkpoint, handled separately below), every
        one of blocks[1:]'s loop passes retains its own activations for backward — loop_count
        (fixed mode) or max_loops (router mode, which always runs the dense compute for every
        iteration regardless of per-token halting) full passes over blocks[1:], plus one unlooped
        pass over blocks[0].
        Per-block cost is approximated as attention's fused-QKV/out_proj/pre-norm activations plus
        the two rotated q/k tensors RoPE retains for backward on top of those (~10 * d_model),
        plus the SwiGLU FFN's hidden-layer activations: the gated first layer retains both its
        gate_proj and up_proj outputs (2 * ffn_dim), each additional ffn_depth layer beyond the
        first retains just its own output (1 * ffn_dim each), for ~(ffn_depth + 1) * ffn_dim total
        — this ignores SDPA's memory-efficient backward (no O(seq_len^2) term) and doesn't itemize
        every temporary buffer (dropout masks, norm stats), so it already overestimates before the
        caller's own safety margin is applied. The lm_head logits (batch, seq, vocab_size) are
        counted separately since they can dominate for a large vocab relative to a small d_model,
        and always at fp32 width regardless of activation_dtype_bytes: PyTorch's autocast policy
        upcasts log_softmax (used internally by compute_loss's F.cross_entropy) to fp32 even under
        bf16/fp16 autocast, so this term doesn't shrink with a lower compute dtype the way the rest
        of the activations do.

        When a block's FFN is MoE (see MoEFeedForward), the dense (depth + 1) * ffn_dim term above
        is replaced by (depth + 1) * ffn_dim * moe_capacity_factor * moe_top_k + n_experts: total
        FFN "slots" processed across all experts scales with capacity_factor * top_k (only the
        capacity-limited, actually-dispatched tokens retain activations per expert), not n_experts
        — using n_experts would repeat the double-counting mistake num_active_parameters() avoids
        for parameter count. The + n_experts term is the router's own logits, retained for backward
        through the softmax.

        Under sparse ACT (cfg.act_capacity_ratio < 1.0) the loop body's activations scale with the
        *effective* iteration count — two dense iterations plus the interior ones at the capacity
        ratio — rather than the full loop_multiplier, plus the retained per-block K/V store the
        sparse path carries across iterations. Not modelled: the (batch, 1, capacity, seq_len)
        boolean attention mask, whose per-token cost is ratio * seq_len bytes and so isn't
        expressible in this per-token API. It is second-order against the block terms and lives
        only for the duration of one attention call, and the caller's vram_safety_margin covers it.
        """
        cfg = self.cfg
        depth = max(1, cfg.ffn_depth)
        # Hyper-connections widen the residual stream itself to n copies, so every hidden state
        # carried between sublayers costs n times what it did. Each block holds two of them (one
        # per hyper-connection write); the per-block attention/FFN terms below are computed on the
        # reduced (d_model-wide) sublayer input and so are unaffected.
        streams = self.hyper_streams
        stream_units = 2 * streams * cfg.d_model if streams > 1 else 0

        def ffn_units(block: TransformerBlock) -> float:
            if isinstance(block.ffn, MoEFeedForward):
                # moe_expert_dim, not ffn_dim: fine-grained experts are narrower, and the whole
                # point of that trade is that total dispatched width stays constant.
                routed = (depth + 1) * cfg.moe_expert_dim * cfg.moe_capacity_factor * cfg.moe_top_k
                # The shared expert has no capacity limit — every token flows through it.
                shared = (depth + 1) * cfg.moe_n_shared * cfg.moe_shared_dim if cfg.moe_n_shared else 0
                return routed + shared + cfg.n_experts
            return (depth + 1) * cfg.ffn_dim

        block0_units = 10 * cfg.d_model + stream_units + ffn_units(self.blocks[0])
        loop_body_units = sum(10 * cfg.d_model + stream_units + ffn_units(b) for b in self.blocks[1:])
        loop_multiplier = cfg.loop_multiplier
        if cfg.act_capacity_ratio < 1.0:
            # Sparse ACT: the first and last iterations run dense, the interior ones over only
            # act_capacity_ratio of each sequence — so the loop body's activations scale with the
            # *effective* iteration count, not the full one. Add the retained per-block K/V store
            # (2 * kv_dim per token per block), which the dense path doesn't carry.
            interior = max(0, loop_multiplier - 2)
            effective_loops = min(loop_multiplier, 2) + interior * cfg.act_capacity_ratio
            kv_store_units = 2 * cfg.n_kv_heads_resolved * cfg.head_dim * (cfg.n_layers - 1)
            total_block_units = block0_units + effective_loops * loop_body_units + kv_store_units
        else:
            total_block_units = block0_units + loop_multiplier * loop_body_units

        if cfg.grad_checkpoint:
            # Under per-block checkpointing only each block's *input* survives the forward pass;
            # everything else is recomputed one block at a time during backward. So the resident
            # cost is one d_model-wide tensor per block boundary, plus the transient peak of
            # recomputing whichever single block is largest.
            n_blocks_executed = 1 + loop_multiplier * (cfg.n_layers - 1)
            largest_block_units = max(
                [block0_units] + [10 * cfg.d_model + stream_units + ffn_units(b) for b in self.blocks[1:]]
            )
            # The tensor surviving at each block boundary is the residual stream, so it is
            # streams * d_model wide rather than d_model.
            total_block_units = n_blocks_executed * streams * cfg.d_model + largest_block_units
        # token embedding only (RoPE's cos/sin have no batch dim), widened to the stream count.
        embedding_units = streams * cfg.d_model
        block_bytes = activation_dtype_bytes * (total_block_units + embedding_units)
        # fp32, x3: logits + their gradient buffer + log_softmax's internal fp32 upcast working
        # buffer (empirically confirmed via a real OOM sized almost exactly to a 2x estimate during
        # GPU verification — cross_entropy's fp32 upcast needs more headroom than just logits+grad).
        logits_bytes = 4 * 3 * self.token_emb.num_embeddings
        return block_bytes + logits_bytes
