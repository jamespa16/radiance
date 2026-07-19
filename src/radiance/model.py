from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """MLP with configurable depth: `ffn_depth` hidden layers of width `ffn_dim`."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        depth = max(1, cfg.ffn_depth)

        layers: list[nn.Module] = [nn.Linear(cfg.d_model, cfg.ffn_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(cfg.ffn_dim, cfg.ffn_dim), nn.GELU()]
        layers += [nn.Linear(cfg.ffn_dim, cfg.d_model)]

        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ACTRouter(nn.Module):
    """Per-token halting-probability head for ACT (Graves 2016) adaptive looping.

    LayerNorm precedes the projection because this reads the pre-LN residual stream, whose norm
    grows with iteration count — without it the halting unit's calibration would drift across loop
    iterations.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(self.norm(x))).squeeze(-1)  # (batch, seq)


class DenseTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.ln_f = nn.LayerNorm(cfg.d_model)
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

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # first block runs once; remaining n_layers - 1 blocks form the loop body
        x = self.blocks[0](x)

        if not self.cfg.use_router:
            # remaining n_layers - 1 blocks are looped loop_count times, sharing weights across iterations
            for _ in range(self.cfg.loop_count):
                for block in self.blocks[1:]:
                    x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            zero = x.new_zeros(())
            return logits, zero, zero

        return self._forward_act(x)

    def _forward_act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adaptive Computation Time (Graves 2016) over the loop body (blocks[1:]): each token
        position gets its own halting probability per iteration, halts once its cumulative
        probability crosses 1 - halt_epsilon (or max_loops is reached), and the output is a
        probability-weighted sum of per-iteration hidden states rather than just the final one.

        Once a position halts, its state is frozen and carried forward unchanged on later
        iterations: still-running positions' causal attention keeps reading a stable key/value for
        it, and its own (recomputed but discarded) update never contributes to the output again.
        Compute is not actually skipped for halted positions (dense batched ops, no gather/scatter)
        — the adaptivity is in the loss signal and output composition, not wall-clock cost.
        """
        batch, seq_len, d_model = x.shape

        cum_prob = x.new_zeros(batch, seq_len)
        n_updates = x.new_zeros(batch, seq_len)
        remainder_sum = x.new_zeros(batch, seq_len)
        still_running = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)
        accum_output = torch.zeros_like(x)
        frozen_x = x

        for n in range(1, self.cfg.max_loops + 1):
            new_x = frozen_x
            for block in self.blocks[1:]:
                new_x = block(new_x)
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
        Per-block cost is approximated as attention's fused-QKV/out_proj/pre-norm activations
        (~8 * d_model) plus the FFN's hidden-layer activations (~ffn_depth * ffn_dim) — this
        ignores SDPA's memory-efficient backward (no O(seq_len^2) term) and doesn't itemize every
        temporary buffer (dropout masks, LayerNorm stats), so it already overestimates before the
        caller's own safety margin is applied. The lm_head logits (batch, seq, vocab_size) are
        counted separately since they can dominate for a large vocab relative to a small d_model,
        and always at fp32 width regardless of activation_dtype_bytes: PyTorch's autocast policy
        upcasts log_softmax (used internally by compute_loss's F.cross_entropy) to fp32 even under
        bf16/fp16 autocast, so this term doesn't shrink with a lower compute dtype the way the rest
        of the activations do.
        """
        cfg = self.cfg
        block_units = 8 * cfg.d_model + cfg.ffn_depth * cfg.ffn_dim
        loop_multiplier = cfg.max_loops if cfg.use_router else cfg.loop_count
        total_block_units = block_units * (1 + loop_multiplier * (cfg.n_layers - 1))
        embedding_units = cfg.d_model
        block_bytes = activation_dtype_bytes * (total_block_units + embedding_units)
        # fp32, x3: logits + their gradient buffer + log_softmax's internal fp32 upcast working
        # buffer (empirically confirmed via a real OOM sized almost exactly to a 2x estimate during
        # GPU verification — cross_entropy's fp32 upcast needs more headroom than just logits+grad).
        logits_bytes = 4 * 3 * self.token_emb.num_embeddings
        return block_bytes + logits_bytes
