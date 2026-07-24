from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config import ModelConfig


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
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: "KVCache | None" = None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([d_model, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

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

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=(self.n_kv_heads != self.n_heads),
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """SwiGLU-gated MLP with configurable depth: `ffn_depth` hidden layers of width `ffn_dim`.
    The first hidden layer is gated (SiLU(gate_proj(x)) * up_proj(x)); any additional depth
    (`ffn_depth > 1`) stacks plain Linear + SiLU layers at `ffn_dim` width on top, preserving the
    "extra hidden layers deepen the MLP, independent of block count" meaning of `ffn_depth`.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        depth = max(1, cfg.ffn_depth)
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(cfg.ffn_dim, cfg.ffn_dim) for _ in range(depth - 1)])
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        for layer in self.hidden_layers:
            h = F.silu(layer(h))
        return self.dropout(self.down_proj(h))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 for a stable softmax under bf16/fp16 autocast, mirroring cross_entropy's own fp32
        # upcast (see activation_bytes_per_token's docstring).
        return F.softmax(self.proj(self.norm(x)).float(), dim=-1)  # (n_tokens, n_experts)


def _moe_capacity(cfg: ModelConfig, n_tokens: int) -> int:
    # Capped at n_tokens: a per-expert capacity above the token count is never meaningful (there
    # aren't enough tokens to fill it) and would make torch.topk's k exceed the dimension size.
    return max(1, min(n_tokens, round(cfg.moe_capacity_factor * n_tokens * cfg.moe_top_k / cfg.n_experts)))


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
        self.experts = nn.ModuleList([FeedForward(cfg) for _ in range(cfg.n_experts)])
        self.aux_loss_accum: torch.Tensor | None = None
        self._aux_loss_calls = 0

    def reset_aux_loss(self) -> None:
        self.aux_loss_accum = None
        self._aux_loss_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        d_model = orig_shape[-1]
        flat_x = x.reshape(-1, d_model)
        n_tokens = flat_x.shape[0]

        probs = self.router(flat_x)  # (n_tokens, n_experts) fp32
        topk_probs, topk_idx = probs.topk(self.cfg.moe_top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize over selected k
        weight = probs.new_zeros(n_tokens, self.cfg.n_experts).scatter_(1, topk_idx, topk_probs)

        assigned = weight > 0  # (n_tokens, n_experts)
        f_i = assigned.float().mean(dim=0).detach()  # fraction of tokens routed to expert i (non-diff)
        p_i = probs.mean(dim=0)  # mean full-softmax prob mass on expert i (differentiable)
        aux_loss = self.cfg.n_experts * (f_i * p_i).sum()
        self.aux_loss_accum = aux_loss if self.aux_loss_accum is None else self.aux_loss_accum + aux_loss
        self._aux_loss_calls += 1

        capacity = _moe_capacity(self.cfg, n_tokens)
        delta = flat_x.new_zeros(n_tokens, d_model)
        for e, expert in enumerate(self.experts):
            expert_weight = weight[:, e]
            assigned_e = expert_weight > 0
            # Priority within the assigned set is the router's own weight for this expert, not a
            # random draw: when an expert is over capacity, the tokens it drops should be the ones
            # it was least confident about. Router weights are in (0, 1], so `assigned + weight`
            # keeps every assigned token strictly above every unassigned one (which score < 1),
            # preserving the "assigned always outranks padding" property.
            #
            # This also makes routing deterministic. The previous random tiebreak ran in eval and
            # generation too, so val/loss and even greedy decoding varied run to run for the same
            # weights and inputs — which defeats comparing two configs' eval numbers.
            priority = assigned_e.float() + expert_weight.detach().float()
            token_idx = torch.topk(priority, k=capacity).indices
            valid = assigned_e.index_select(0, token_idx)

            gathered = flat_x.index_select(0, token_idx)
            expert_out = expert(gathered) * valid.unsqueeze(-1).to(x.dtype)
            w = (expert_weight.index_select(0, token_idx) * valid.to(expert_weight.dtype)).to(x.dtype)
            delta = delta.index_add(0, token_idx, w.unsqueeze(-1) * expert_out)

        return delta.view(orig_shape)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, use_moe_ffn: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = MoEFeedForward(cfg) if use_moe_ffn else FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: "KVCache | None" = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(self.norm(x))).squeeze(-1)  # (batch, seq)


def _ffn_capacity(cfg: ModelConfig, batch: int, seq_len: int) -> int:
    n_tokens = batch * seq_len
    return min(n_tokens, max(1, round(cfg.act_ffn_capacity_ratio * n_tokens)))


def _sparse_ffn_delta(
    ffn: FeedForward, h: torch.Tensor, still_running: torch.Tensor, capacity: int
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
    ffn_out = ffn(gathered) * valid.unsqueeze(-1).to(h.dtype)  # zero out padding slots

    delta_flat = flat_h.new_zeros(n_tokens, d_model).index_copy(0, token_idx, ffn_out)
    return delta_flat.view(batch, seq_len, d_model)


def _run_loop_body(
    blocks: nn.ModuleList,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    still_running: torch.Tensor | None = None,
    capacity: int | None = None,
    kv_cache: "KVCache | None" = None,
) -> torch.Tensor:
    """Runs `blocks` once. Attention is always fully dense. When still_running/capacity are given,
    each block's FFN is dispatched through the fixed-capacity sparse path (_sparse_ffn_delta)
    instead of densely.
    """
    for block in blocks:
        if still_running is None:
            x = block(x, cos, sin, kv_cache)
        else:
            x = x + block.attn(block.ln1(x), cos, sin, kv_cache)
            x = x + _sparse_ffn_delta(block.ffn, block.ln2(x), still_running, capacity)
    return x


class DenseTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.dropout = nn.Dropout(cfg.dropout)

        # blocks[0] always stays dense (it runs once per forward, not part of the recursive loop
        # body); blocks[1:] use MoEFeedForward when use_moe is set, except every moe_dense_every-th
        # position (1-indexed within blocks[1:]), which stays dense too.
        self.blocks = nn.ModuleList([TransformerBlock(cfg, use_moe_ffn=False)])
        for i in range(cfg.n_layers - 1):
            is_dense_override = cfg.use_moe and cfg.moe_dense_every and (i + 1) % cfg.moe_dense_every == 0
            self.blocks.append(TransformerBlock(cfg, use_moe_ffn=cfg.use_moe and not is_dense_override))

        # Plain Python list, not a second nn.ModuleList — these modules are already registered via
        # self.blocks; wrapping them again risks confusing state_dict/double-registration.
        self._moe_layers = [m for m in self.modules() if isinstance(m, MoEFeedForward)]

        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.router = None
        if cfg.use_router:
            assert cfg.max_loops >= 1
            self.router = ACTRouter(cfg)

        self.apply(self._init_weights)
        self._scale_residual_init()
        if self.router is not None:
            # Bias the halting unit against halting immediately (Graves ACT): sigmoid(-1) ≈ 0.27,
            # encouraging some early pondering rather than collapsing to a single pass at init.
            nn.init.constant_(self.router.proj.bias, -1.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        loop_multiplier = self.cfg.max_loops if self.cfg.use_router else self.cfg.loop_count
        n_blocks_executed = 1 + loop_multiplier * (self.cfg.n_layers - 1)
        scale = (2 * n_blocks_executed) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                param.data.mul_(scale)

    def new_kv_cache(self) -> KVCache:
        """Builds an empty KVCache sized for this model/config: one slot for blocks[0] plus one
        per (block, loop-iteration) pair the loop body (blocks[1:]) executes each forward call."""
        loop_multiplier = self.cfg.max_loops if self.cfg.use_router else self.cfg.loop_count
        num_slots = 1 + loop_multiplier * (self.cfg.n_layers - 1)
        return KVCache(num_slots)

    def _reset_moe_aux_loss(self) -> None:
        for moe_ffn in self._moe_layers:
            moe_ffn.reset_aux_loss()

    def _collect_moe_aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        if not self._moe_layers:
            return x.new_zeros(())
        # Averaged per-layer over its call count (once for loop_count/non-router mode, up to
        # max_loops times for ACT mode) so the aux loss magnitude doesn't scale with loop depth,
        # consistent with ponder_cost/mean_loop_depth also being mean-reduced, not summed.
        return sum(
            moe_ffn.aux_loss_accum / moe_ffn._aux_loss_calls for moe_ffn in self._moe_layers
        )

    def forward(
        self, input_ids: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, ponder_cost, mean_loop_depth, moe_aux_loss). The latter three are zero
        scalar tensors when the corresponding feature (use_router / use_moe) is off, so callers
        have one contract regardless of mode.

        kv_cache is optional and defaults to None (the training/full-sequence path, unchanged).
        When given, input_ids is the *new* chunk only (the whole prompt on the first call, one
        token per call thereafter) and attention is computed against cached past K/V plus this
        chunk; see KVCache and CausalSelfAttention.forward."""
        batch, seq_len = input_ids.shape
        offset = kv_cache.seq_len if kv_cache is not None else 0
        assert offset + seq_len <= self.cfg.max_seq_len, "sequence length exceeds max_seq_len"

        x = self.token_emb(input_ids)
        x = self.dropout(x)
        cos, sin = self.rope(seq_len, offset)

        if kv_cache is not None:
            kv_cache.begin_step()

        self._reset_moe_aux_loss()

        # first block runs once; remaining n_layers - 1 blocks form the loop body
        x = self.blocks[0](x, cos, sin, kv_cache)

        if not self.cfg.use_router:
            # remaining n_layers - 1 blocks are looped loop_count times, sharing weights across iterations
            for _ in range(self.cfg.loop_count):
                for block in self.blocks[1:]:
                    x = block(x, cos, sin, kv_cache)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            zero = x.new_zeros(())
            moe_aux_loss = self._collect_moe_aux_loss(x)
            if kv_cache is not None:
                kv_cache.seq_len += seq_len
            return logits, zero, zero, moe_aux_loss

        result = self._forward_act(x, cos, sin, kv_cache)
        if kv_cache is not None:
            kv_cache.seq_len += seq_len
        return result

    def _forward_act(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        batch, seq_len, d_model = x.shape

        cum_prob = x.new_zeros(batch, seq_len)
        n_updates = x.new_zeros(batch, seq_len)
        remainder_sum = x.new_zeros(batch, seq_len)
        still_running = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)
        accum_output = torch.zeros_like(x)
        frozen_x = x

        sparse_enabled = self.cfg.act_ffn_capacity_ratio < 1.0
        capacity = _ffn_capacity(self.cfg, batch, seq_len) if sparse_enabled else None

        for n in range(1, self.cfg.max_loops + 1):
            is_first_or_last = n == 1 or n == self.cfg.max_loops
            if not sparse_enabled or is_first_or_last:
                new_x = _run_loop_body(self.blocks[1:], frozen_x, cos, sin, kv_cache=kv_cache)
            else:
                new_x = _run_loop_body(
                    self.blocks[1:], frozen_x, cos, sin, still_running, capacity, kv_cache
                )
            p_n = self.router(new_x)

            is_last_step = n == self.cfg.max_loops
            would_exceed = (cum_prob + p_n) >= (1.0 - self.cfg.halt_epsilon)
            halts_now = still_running & (would_exceed | is_last_step)

            remainder = 1.0 - cum_prob
            weight = torch.where(halts_now, remainder, p_n)
            weight = torch.where(still_running, weight, torch.zeros_like(weight))
            accum_output = accum_output + weight.unsqueeze(-1) * new_x

            n_updates = n_updates + still_running.float()
            remainder_sum = torch.where(halts_now, remainder, remainder_sum)
            cum_prob = torch.where(still_running & ~halts_now, cum_prob + p_n, cum_prob)

            frozen_x = torch.where(still_running.unsqueeze(-1), new_x, frozen_x)
            still_running = still_running & ~halts_now

        x = self.ln_f(accum_output)
        logits = self.lm_head(x)
        ponder_cost = (n_updates + remainder_sum).mean()
        mean_loop_depth = n_updates.mean()
        moe_aux_loss = self._collect_moe_aux_loss(x)
        return logits, ponder_cost, mean_loop_depth, moe_aux_loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_active_parameters(self) -> int:
        """Like num_parameters(), but counts only moe_top_k experts' worth of FFN parameters per
        MoE layer instead of all n_experts — the parameter count actually multiplied against any
        given token's activation, which is the right basis for Chinchilla-style tokens_per_param
        sizing (see train.py). Identical to num_parameters() when no MoE layers exist."""
        total = self.num_parameters()
        for moe_ffn in self._moe_layers:
            per_expert_params = sum(p.numel() for p in moe_ffn.experts[0].parameters())
            total -= per_expert_params * (self.cfg.n_experts - self.cfg.moe_top_k)
        return total

    def activation_bytes_per_token(self, activation_dtype_bytes: int) -> int:
        """Conservative (deliberately over-, not under-, estimated) activation memory per token,
        for sizing a training batch to available VRAM (see train.py's estimate_batch_size).

        No gradient checkpointing exists anywhere in this model, so every one of blocks[1:]'s loop
        passes retains its own activations for backward — loop_count (fixed mode) or max_loops
        (router mode, which always runs the dense compute for every iteration regardless of
        per-token halting) full passes over blocks[1:], plus one unlooped pass over blocks[0].
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
        """
        cfg = self.cfg
        depth = max(1, cfg.ffn_depth)

        def ffn_units(block: TransformerBlock) -> float:
            if isinstance(block.ffn, MoEFeedForward):
                return (depth + 1) * cfg.ffn_dim * cfg.moe_capacity_factor * cfg.moe_top_k + cfg.n_experts
            return (depth + 1) * cfg.ffn_dim

        block0_units = 10 * cfg.d_model + ffn_units(self.blocks[0])
        loop_body_units = sum(10 * cfg.d_model + ffn_units(b) for b in self.blocks[1:])
        loop_multiplier = cfg.max_loops if cfg.use_router else cfg.loop_count
        total_block_units = block0_units + loop_multiplier * loop_body_units
        embedding_units = cfg.d_model  # token embedding only (RoPE's cos/sin have no batch dim)
        block_bytes = activation_dtype_bytes * (total_block_units + embedding_units)
        # fp32, x3: logits + their gradient buffer + log_softmax's internal fp32 upcast working
        # buffer (empirically confirmed via a real OOM sized almost exactly to a 2x estimate during
        # GPU verification — cross_entropy's fp32 upcast needs more headroom than just logits+grad).
        logits_bytes = 4 * 3 * self.token_emb.num_embeddings
        return block_bytes + logits_bytes
