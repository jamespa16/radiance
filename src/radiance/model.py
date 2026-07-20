from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config import ModelConfig


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

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


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
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[None, None, :, :], sin[None, None, :, :])
        k = apply_rope(k, cos[None, None, :, :], sin[None, None, :, :])

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
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


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


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
    priority = flat_running.float() + torch.rand(n_tokens, device=h.device, dtype=torch.float32)
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
) -> torch.Tensor:
    """Runs `blocks` once. Attention is always fully dense. When still_running/capacity are given,
    each block's FFN is dispatched through the fixed-capacity sparse path (_sparse_ffn_delta)
    instead of densely.
    """
    for block in blocks:
        if still_running is None:
            x = block(x, cos, sin)
        else:
            x = x + block.attn(block.ln1(x), cos, sin)
            x = x + _sparse_ffn_delta(block.ffn, block.ln2(x), still_running, capacity)
    return x


class DenseTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.router = None
        if cfg.use_router:
            assert cfg.max_loops >= 1
            self.router = ACTRouter(cfg)

        self.apply(self._init_weights)
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

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, ponder_cost, mean_loop_depth). The latter two are zero scalar tensors
        when cfg.use_router is False, so callers have one contract regardless of mode."""
        batch, seq_len = input_ids.shape
        assert seq_len <= self.cfg.max_seq_len, "sequence length exceeds max_seq_len"

        x = self.token_emb(input_ids)
        x = self.dropout(x)
        cos, sin = self.rope(seq_len)

        # first block runs once; remaining n_layers - 1 blocks form the loop body
        x = self.blocks[0](x, cos, sin)

        if not self.cfg.use_router:
            # remaining n_layers - 1 blocks are looped loop_count times, sharing weights across iterations
            for _ in range(self.cfg.loop_count):
                for block in self.blocks[1:]:
                    x = block(x, cos, sin)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            zero = x.new_zeros(())
            return logits, zero, zero

        return self._forward_act(x, cos, sin)

    def _forward_act(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                new_x = _run_loop_body(self.blocks[1:], frozen_x, cos, sin)
            else:
                new_x = _run_loop_body(self.blocks[1:], frozen_x, cos, sin, still_running, capacity)
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
        return logits, ponder_cost, mean_loop_depth

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

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
        """
        cfg = self.cfg
        depth = max(1, cfg.ffn_depth)
        block_units = 10 * cfg.d_model + (depth + 1) * cfg.ffn_dim
        loop_multiplier = cfg.max_loops if cfg.use_router else cfg.loop_count
        total_block_units = block_units * (1 + loop_multiplier * (cfg.n_layers - 1))
        embedding_units = cfg.d_model  # token embedding only (RoPE's cos/sin have no batch dim)
        block_bytes = activation_dtype_bytes * (total_block_units + embedding_units)
        # fp32, x3: logits + their gradient buffer + log_softmax's internal fp32 upcast working
        # buffer (empirically confirmed via a real OOM sized almost exactly to a 2x estimate during
        # GPU verification — cross_entropy's fp32 upcast needs more headroom than just logits+grad).
        logits_bytes = 4 * 3 * self.token_emb.num_embeddings
        return block_bytes + logits_bytes
