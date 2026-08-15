from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from radiance import nvfp4
from radiance.config import ModelConfig

from .act import ACTRouter, _act_select, _ffn_capacity, _sparse_ffn_delta
from .attention import KVCache, RotaryEmbedding, _FLEX_MIN_HEAD_DIM
from .block import TransformerBlock
from .core import LoopContext, ModelOutput
from .ffn import MoEFeedForward
from .masking import _sparse_attn_mask, build_block_mask, document_ids
from .mtp import MTPHead, _shift_left
from .norms import RMSNorm
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

        # Checked here, before any block is constructed, so this raises the friendlier message
        # below instead of CausalSelfAttention.__init__'s bare assert (same condition, kept there
        # too as a defensive backstop for anyone constructing CausalSelfAttention directly).
        if cfg.use_diff_attn and cfg.head_dim % 4 != 0:
            raise ValueError(
                "model.use_diff_attn splits each head's Q/K into two head_dim//2-wide branches, and "
                f"RoPE's pairwise rotation then needs that half-width to itself be even — "
                f"model.head_dim ({cfg.head_dim}) must be a multiple of 4."
            )
        if cfg.use_diff_attn and cfg.act_capacity_ratio < 1.0:
            raise ValueError(
                "model.use_diff_attn is incompatible with model.act_capacity_ratio < 1.0: ACT's "
                "sparse path (CausalSelfAttention.forward_sparse) has no differential-attention "
                "variant yet. Set act_capacity_ratio: 1.0 or use_diff_attn: false."
            )
        if cfg.fp4_linear:
            # Checked here, before any block is constructed, for the same reason use_diff_attn's
            # checks were moved to the top: the failure would otherwise surface from inside
            # nvfp4.quantize with a much worse message.
            #
            # 128, not 64. Each of these widths serves as the *row* dimension of some GEMM —
            # dgrad makes K a row dim and wgrad makes both N and K row dims — and the
            # SWIZZLE_32_4_4 scale layout pads rows to 128.
            widths = {
                "d_model": cfg.d_model,
                "the qkv projection width (d_model + 2 * n_kv_heads * head_dim)": cfg.d_model
                + 2 * cfg.n_kv_heads_resolved * cfg.head_dim,
                "ffn_dim (d_model * ffn_mult)": cfg.ffn_dim,
            }
            bad = {name: value for name, value in widths.items() if value % 128 != 0}
            if bad:
                detail = "; ".join(f"{name} = {value}" for name, value in bad.items())
                raise ValueError(
                    f"model.fp4_linear needs every quantized width to be a multiple of 128, but {detail}. "
                    "NVFP4's block scales are stored in 128-row tiles. Adjust d_model / head_dim / "
                    "ffn_mult, or unset fp4_linear."
                )
            if cfg.use_moe:
                raise ValueError(
                    "model.fp4_linear is incompatible with model.use_moe. BatchedExperts applies its "
                    "weights with torch.baddbmm over an (n_experts, in, out) stack and _scaled_mm_v2 "
                    "is 2-D only; a per-expert Python loop would reintroduce exactly the launch "
                    "overhead the batched dispatch exists to remove (24.0 -> 3.16 ms at 32 experts). "
                    "With MoE on the experts *are* the FFN, so quantizing only attention would "
                    "produce a throughput number that looks like a failure of FP4 when FP4 never "
                    "touched the expensive part. Refusing beats a misleading measurement."
                )
            nvfp4.require_nvfp4("model.fp4_linear")

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        # Differential attention's Q/K live at head_dim // 2 (V is never rotated, and nothing else
        # in the model consumes self.rope's output), so build the table at that width directly
        # rather than building a full-width one and slicing it — RoPE's frequency spacing depends
        # on head_dim itself, so a slice of the full-width table is a *different*, wrong rotation,
        # not a valid subset of one built for the halved width.
        rope_head_dim = cfg.head_dim // 2 if cfg.use_diff_attn else cfg.head_dim
        self.rope = RotaryEmbedding(rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
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

        # Before _init_weights, so the swapped-in modules are the ones it initialises. FP4Linear
        # subclasses nn.Linear and the swap transplants the existing Parameter objects, so both
        # orderings would in fact work — but keeping it here means there is exactly one place that
        # decides which linears are quantized, the way _init_inert_gates holds the whole zero-init
        # policy. This is the ordering-constrained block; do not scatter it.
        self._convert_linears_to_fp4()

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

    def _convert_linears_to_fp4(self) -> None:
        """Swap the eligible `nn.Linear`s for `FP4Linear`, in place.

        **The rule is that FP4 covers exactly the tensors Muon owns**, and that is deliberate reuse
        rather than coincidence: `optim._MUON_EXCLUDED_SUBSTRINGS` already encodes this repo's
        judgement about which tensors are hidden linear maps and which are small tensors whose
        exact scale is load-bearing, with the reasoning written out. A second, differently-drawn
        taxonomy would be one more thing to keep in sync.

        So the routers, the attention output gate and the tied embedding stay bf16. `out_gate` is
        the sharpest case: it is zero-initialised precisely so that `2 * sigmoid(0) == 1.0`
        exactly, and quantizing it would put 4-bit noise on an exact identity.
        """
        cfg = self.cfg
        if not cfg.fp4_linear:
            return

        recipe = nvfp4.FP4Recipe(
            grad_gemms=cfg.fp4_grad_gemms,
            stochastic_rounding=cfg.fp4_stochastic_rounding,
            hadamard=cfg.fp4_hadamard,
            save_activations=cfg.fp4_save_activations,
        )
        keep_first = cfg.fp4_keep_bf16_first
        # Structural blocks, not loop iterations: blocks[1:] is weight-shared, so a block kept in
        # bf16 is bf16 on every iteration. Precision cannot vary per-iteration when weights are.
        keep_last = max(0, cfg.fp4_keep_bf16_blocks)
        bf16_blocks = set(range(cfg.n_layers - keep_last, cfg.n_layers))
        if keep_first:
            bf16_blocks.add(0)

        for index, block in enumerate(self.blocks):
            if index in bf16_blocks:
                continue
            self._convert_subtree_to_fp4(block, recipe)
        if cfg.mtp_heads > 1:
            for head in self.mtp_heads:
                self._convert_subtree_to_fp4(head, recipe)
        if cfg.fp4_lm_head and self._fp4_shape_ok(self.lm_head):
            # Shape-guarded like every other conversion. The head's out_features is the *padded
            # vocab*, which `vocab_pad_multiple: 128` (the default) already makes a multiple of 128
            # — but `vocab_pad_multiple: 1`, or a small test vocab, does not, and converting anyway
            # allocates a zero-length scale buffer and fails at the first forward rather than here.
            self.lm_head = self._as_fp4(self.lm_head, recipe)

    @staticmethod
    def _fp4_shape_ok(linear: nn.Linear) -> bool:
        """The kernels' shape contract, as it applies to a weight: rows (out_features) must fill a
        128-row scale tile and the contraction dim (in_features) must be a multiple of 64."""
        return linear.out_features % 128 == 0 and linear.in_features % 64 == 0

    @staticmethod
    def _as_fp4(linear: nn.Linear, recipe: nvfp4.FP4Recipe) -> nvfp4.FP4Linear:
        """Rebuild `linear` as an `FP4Linear`, transplanting the Parameter objects themselves so
        weight tying and any pre-existing initialisation survive the swap."""
        replacement = nvfp4.FP4Linear(
            linear.in_features, linear.out_features, bias=linear.bias is not None, recipe=recipe
        )
        replacement.weight = linear.weight
        if linear.bias is not None:
            replacement.bias = linear.bias
        return replacement

    def _convert_subtree_to_fp4(self, root: nn.Module, recipe: nvfp4.FP4Recipe) -> None:
        for parent_name, parent in root.named_modules():
            for child_name, child in list(parent.named_children()):
                if type(child) is not nn.Linear:
                    continue
                qualified = f"{parent_name}.{child_name}" if parent_name else child_name
                # See the docstring: same classification optim.py uses, for the same reasons.
                if any(token in qualified for token in ("router", "out_gate", "proj_gate")):
                    continue
                if not self._fp4_shape_ok(child):
                    continue
                setattr(parent, child_name, self._as_fp4(child, recipe))

    def fp4_cache_bytes(self) -> int:
        """Bytes held by the FP4 weight caches once `refresh_fp4_weights` has filled them.

        A fixed, per-parameter cost rather than a per-token one, so `estimate_batch_size` subtracts
        it alongside the gradient and optimizer buffers rather than folding it into
        `activation_bytes_per_token`. Roughly 1.125 bytes per covered parameter — 0.5 for the packed
        nibbles plus 0.0625 for the block scales, in each of the two orientations the three GEMMs
        need. Zero unless `fp4_linear` is set.
        """
        total = 0
        for module in self.modules():
            if isinstance(module, nvfp4.FP4Linear):
                total += sum(
                    buf.numel() * buf.element_size()
                    for name, buf in module.named_buffers()
                    if name.startswith("_w_")
                )
        return total

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
    def builds_doc_block_masks(self, device_type: str) -> bool:
        """Whether a full (non-cached) forward on `device_type` builds flex_attention BlockMasks.

        Split out of _doc_masks so train.py can answer the question *before* the first forward:
        a BlockMask rebuilt from each batch's document boundaries is incompatible with
        torch.compile's CUDA-graph mode, and that decision has to be made at compile time. Not a
        cheap property — it emits the head_dim fallback note — so call it once, not per step.
        """
        if not self.cfg.doc_attention_mask or self.eos_id is None:
            return False
        # flex_attention is impractically slow outside CUDA; CLAUDE.md's CPU sanity-check workflow
        # has to keep working, so fall back to plain causal SDPA there.
        if device_type != "cuda":
            return False
        # flex_attention's compiled kernel requires head_dim >= 16. Real configs use 64, but the
        # tiny models used for sanity checks can go below it, and the failure mode is an
        # InductorError raised from deep inside the compiler rather than anything actionable.
        # Differential attention calls flex_attention at head_dim // 2 (its Q1/K1/Q2/K2 width), not
        # the full head_dim, so the threshold has to apply to whichever width actually reaches it.
        effective_head_dim = self.cfg.head_dim // 2 if self.cfg.use_diff_attn else self.cfg.head_dim
        if effective_head_dim < _FLEX_MIN_HEAD_DIM:
            if not self._warned_head_dim:
                print(
                    f"[radiance] note: model.doc_attention_mask is off for this run — "
                    f"flex_attention requires head_dim >= {_FLEX_MIN_HEAD_DIM}, but the effective "
                    f"attention head_dim is {effective_head_dim} "
                    f"({'model.head_dim // 2 under use_diff_attn' if self.cfg.use_diff_attn else 'model.head_dim'}"
                    f"={self.cfg.head_dim}). Falling back to plain causal attention."
                )
                self._warned_head_dim = True
            return False
        return True

    @torch._dynamo.disable
    def _doc_masks(self, input_ids: torch.Tensor, offset: int, kv_cache) -> dict[int, Any] | None:
        """One BlockMask per distinct attention window, keyed by window size, or None when document
        masking doesn't apply to this forward pass.

        Returns a dict rather than a single mask because cfg.loop_attn_windows can give each loop
        iteration a different receptive field; forward() picks per iteration. With no windows
        configured there is exactly one entry.
        """
        # Generation is a single document, so the mask would be all-ones — and flex_attention
        # against an incrementally growing cache needs different plumbing for no benefit.
        if kv_cache is not None:
            return None
        if not self.builds_doc_block_masks(input_ids.device.type):
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
        counted separately since they can dominate for a large vocab relative to a small d_model —
        at this config's 50304-token vocab they are the single largest term by a wide margin. They
        scale with activation_dtype_bytes like everything else, plus one fp32-width reduction
        buffer; see the comment at the logits_bytes line for why that stopped being a flat fp32.

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
        # Differential attention (cfg.use_diff_attn) computes a second softmax attention branch —
        # its own rotated q2/k2, its own attention-output tensor, plus diff_norm's activations —
        # on top of everything plain attention already retains. Measured directly (peak allocated
        # memory, fwd+bwd, bf16, diff on vs off, delta divided out by n_layers/batch/seq/dtype-width
        # so it's comparable to the 10 * d_model figure below): the extra cost is ~7 * d_model per
        # token per block, consistently across d_model in {512, 1024} (6.99-7.00x) and stable across
        # batch/seq_len shapes — not guessed from the two-branch structure a priori.
        attn_unit_cost = 17 * cfg.d_model if cfg.use_diff_attn else 10 * cfg.d_model

        def ffn_units(block: TransformerBlock) -> float:
            if isinstance(block.ffn, MoEFeedForward):
                # moe_expert_dim, not ffn_dim: fine-grained experts are narrower, and the whole
                # point of that trade is that total dispatched width stays constant.
                routed = (depth + 1) * cfg.moe_expert_dim * cfg.moe_capacity_factor * cfg.moe_top_k
                # The shared expert has no capacity limit — every token flows through it.
                shared = (depth + 1) * cfg.moe_n_shared * cfg.moe_shared_dim if cfg.moe_n_shared else 0
                return routed + shared + cfg.n_experts
            return (depth + 1) * cfg.ffn_dim

        block0_units = attn_unit_cost + stream_units + ffn_units(self.blocks[0])
        loop_body_units = sum(attn_unit_cost + stream_units + ffn_units(b) for b in self.blocks[1:])
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
                [block0_units] + [attn_unit_cost + stream_units + ffn_units(b) for b in self.blocks[1:]]
            )
            # The tensor surviving at each block boundary is the residual stream, so it is
            # streams * d_model wide rather than d_model.
            total_block_units = n_blocks_executed * streams * cfg.d_model + largest_block_units
        # token embedding only (RoPE's cos/sin have no batch dim), widened to the stream count.
        embedding_units = streams * cfg.d_model
        block_bytes = activation_dtype_bytes * (total_block_units + embedding_units)
        # Logits + their gradient buffer, both at compute width, plus one fp32-width working buffer
        # for the loss's reduction. This used to be a flat `4 * 3` because F.cross_entropy sits on
        # autocast's fp32 list and so upcast the whole (batch, seq, vocab_size) tensor before
        # reducing it — the term didn't shrink with a lower compute dtype the way the rest of the
        # activations do. train.compute_loss no longer goes through cross_entropy (it derives both
        # the LM loss and z_loss from one logsumexp, and inductor keeps that reduction in fp32
        # without materialising an fp32 copy), so under bf16 this is 8 bytes per vocab element
        # rather than 12. Measured against the real peak on configs/tinystories.yaml: the whole
        # estimate lands 1.12x above actual, still on the intended over-estimating side, where the
        # fp32 assumption had it at 1.58x — headroom that auto_batch_size was spending on a
        # smaller micro-batch than the model actually needs.
        logits_bytes = (2 * activation_dtype_bytes + 4) * self.token_emb.num_embeddings
        return block_bytes + logits_bytes
