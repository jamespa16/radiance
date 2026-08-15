"""The NVFP4 linear layer: `FP4Linear`, its out-of-autograd quantization, and the per-step
weight-cache refresh.

Part of the `radiance.nvfp4` package - the package `__init__` re-exports everything here.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import (
    BLOCK,
    NVFP4Tensor,
    _HAS_TRITON,
    _supported_cached,
    global_scale_of,
    hadamard_matrix,
    mm,
    quantize,
    quantize_supported_shape,
)


# --------------------------------------------------------------------------------------------
# The linear layer
# --------------------------------------------------------------------------------------------


class FP4Recipe(NamedTuple):
    """Which parts of the NVFP4 training recipe are live. Mirrors the `model.fp4_*` config fields."""

    grad_gemms: bool = True
    stochastic_rounding: bool = True
    hadamard: bool = True
    save_activations: bool = True


class _FP4LinearFn(torch.autograd.Function):
    """`y = x Wᵀ + b` with all three GEMMs in NVFP4.

    **Quantization happens inside `forward`/`backward`, where autograd cannot see it, and that is
    the entire point of this class rather than a stylistic choice.** Quantizing in an
    autograd-visible way produces a gradient that is nonzero, roughly the right magnitude, and
    wrong: the encode is a step function with zero derivative everywhere, so the only surviving
    path back to `x` is the `amax` inside the global scale. Nothing raises, and the loss still
    falls. `tests/test_nvfp4.py::test_gradient_is_not_the_autograd_trap` builds that broken version
    as a negative control.

    Being outside autograd also means straight-through estimation needs no `x - x.detach()` trick
    (contrast `_counterfactual_probe_signal` in `model/ffn.py`): the gradient computed against the
    quantized operands *is* returned as the gradient of the full-precision input, which is exactly
    the identity Jacobian STE asks for.

    The weight arrives pre-quantized in both orientations, from a cache `refresh_fp4_weights`
    fills once per optimizer step. Re-quantizing it here instead measured **0.37x** — three times
    slower than bf16 — and was the single largest cost in the naive implementation.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        w_fwd_packed,
        w_fwd_scale,
        w_fwd_gs,
        w_dgrad_packed,
        w_dgrad_scale,
        w_dgrad_gs,
        hadamard,
        seed,
        recipe,
    ):
        gx = global_scale_of(x)
        xq = quantize(x, -1, global_scale=gx)
        y = mm(xq, NVFP4Tensor(w_fwd_packed, w_fwd_scale, w_fwd_gs), out_dtype=x.dtype)
        if bias is not None:
            y = y + bias.to(y.dtype)

        # Save the *quantized* activation rather than the bf16 one when the backward is going to
        # quantize it anyway. 1.125 bytes per element (0.5 packed + 0.0625 of scales) against bf16's
        # 2, and the saved tensor is the largest per-layer term in a transformer's activation
        # memory — `down_proj`'s input alone is `ffn_dim` wide.
        #
        # **This is bit-identical, not an approximation.** The backward previously computed exactly
        # `quantize(x, 0, global_scale=gx, hadamard=rot)` from this same `x`; all that changes is
        # when. It is therefore safe to default on, unlike anything that trades precision.
        #
        # It is skipped when the backward will not use it: with `grad_gemms` off, wgrad is
        # `grad_y.t() @ x` in bf16 and needs the real `x` back.
        rot = hadamard if recipe.hadamard and hadamard.numel() else None
        save_quantized = recipe.save_activations and recipe.grad_gemms and quantize_supported_shape(*x.shape)
        if save_quantized:
            xqt = quantize(x, 0, global_scale=gx, hadamard=rot)
            saved = (xqt.packed, xqt.scale, weight, w_dgrad_packed, w_dgrad_scale, w_dgrad_gs, hadamard, seed, gx)
        else:
            saved = (x, x, weight, w_dgrad_packed, w_dgrad_scale, w_dgrad_gs, hadamard, seed, gx)

        ctx.save_for_backward(*saved)
        ctx.recipe = recipe
        ctx.has_bias = bias is not None
        ctx.save_quantized = save_quantized
        # Kept off the saved tensors so `x` itself can be freed: the backward needs only these.
        ctx.x_dtype = x.dtype
        ctx.x_shape = x.shape
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_or_packed, x_or_scale, weight, wdp, wds, wdg, hadamard, seed, gx = ctx.saved_tensors
        recipe: FP4Recipe = ctx.recipe
        grad_y = grad_y.contiguous()
        x_dtype = ctx.x_dtype

        # `grad_y`'s N dimension must satisfy the kernels' shape contract too, and it need not —
        # a linear whose output width is not a multiple of 128 has a perfectly fine forward and an
        # unquantizable backward. Fall back rather than raise: the forward already ran in FP4.
        fp4_backward = recipe.grad_gemms and quantize_supported_shape(*grad_y.shape)

        if fp4_backward:
            rot = hadamard if recipe.hadamard and hadamard.numel() else None
            gg = global_scale_of(grad_y)
            # dgrad contracts over N: grad_y blocks along its last dim, W along its first (cached).
            gq = quantize(grad_y, -1, global_scale=gg, stochastic=recipe.stochastic_rounding, seed=seed)
            grad_x = mm(gq, NVFP4Tensor(wdp, wds, wdg), out_dtype=x_dtype)
            # wgrad contracts over M (tokens): both operands block along their first dim, and both
            # take the *same* rotation — `(RᵀdY)ᵀ(RᵀX) = dYᵀ(R Rᵀ)X = dYᵀX` only holds if they match.
            gqt = quantize(
                grad_y, 0, global_scale=gg, hadamard=rot,
                stochastic=recipe.stochastic_rounding, seed=seed,
            )
            if ctx.save_quantized:
                xqt = NVFP4Tensor(x_or_packed, x_or_scale, gx)
            else:
                xqt = quantize(x_or_packed, 0, global_scale=gx, hadamard=rot)
            grad_w = mm(gqt, xqt, out_dtype=torch.float32)
        else:
            grad_x = grad_y @ weight.to(grad_y.dtype)
            grad_w = grad_y.t() @ x_or_packed

        grad_b = grad_y.sum(0) if ctx.has_bias else None
        # grad_w back to the parameter's own dtype: master weights are fp32 everywhere in this
        # repo, and silently handing autograd a bf16 gradient for an fp32 parameter would halve
        # the precision of every update.
        return (
            grad_x.to(x_dtype),
            grad_w.to(weight.dtype),
            None if grad_b is None else grad_b.to(weight.dtype),
            None, None, None, None, None, None, None, None, None,
        )


class FP4Linear(nn.Linear):
    """`nn.Linear` whose three GEMMs run in NVFP4, with a per-optimizer-step weight cache.

    **Subclassing `nn.Linear` rather than replacing it is what keeps the blast radius small.**
    Every mechanism in this repo that classifies parameters does so by module type or by parameter
    name, and all of them keep working unchanged: `DenseTransformer._init_weights` dispatches on
    `isinstance(module, nn.Linear)`; `_scale_residual_init` matches `name.endswith("out_proj.weight")`;
    `_init_inert_gates` matches `out_gate`; `optim.build_muon_param_groups` routes a 2-D `.weight`
    to Muon. The `state_dict` is byte-identical to a bf16 model's, so an FP4 run resumes from a
    bf16 checkpoint and vice versa, and `generate.py` needs no change at all.

    The cache buffers are `persistent=False` (derived state, not weights) and are written in place
    with `copy_`, never reassigned — a tensor whose data pointer moves every step is what turned a
    compiled graph into a re-record treadmill in `resolve_compile_mode`'s CUDA-graph leak.
    """

    def __init__(self, in_features, out_features, bias=True, recipe: FP4Recipe | None = None):
        super().__init__(in_features, out_features, bias=bias)
        self.recipe = recipe or FP4Recipe()
        self._cache_valid = False
        n, k = out_features, in_features
        self.register_buffer("_w_fwd_packed", torch.empty(n, k // 2, dtype=torch.uint8), persistent=False)
        self.register_buffer("_w_fwd_scale", torch.empty((n // 128) * (k // 64) * 512, dtype=torch.uint8), persistent=False)
        self.register_buffer("_w_fwd_gs", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("_w_dgrad_packed", torch.empty(k, n // 2, dtype=torch.uint8), persistent=False)
        self.register_buffer("_w_dgrad_scale", torch.empty(((k + 127) // 128) * (n // 64) * 512, dtype=torch.uint8), persistent=False)
        self.register_buffer("_w_dgrad_gs", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("_hadamard", hadamard_matrix(BLOCK), persistent=False)
        self.register_buffer("_seed", torch.zeros((), dtype=torch.int64), persistent=False)
        # A stray load_state_dict must not leave the cache describing the previous weights.
        self.register_load_state_dict_post_hook(lambda module, *_: module.invalidate_fp4_cache())

    def invalidate_fp4_cache(self) -> None:
        self._cache_valid = False

    @torch.no_grad()
    def refresh_fp4_cache(self) -> None:
        """Re-quantize the weight into both orientations, in place.

        Called once per optimizer step from `train.py`, not from `forward`: a validity check inside
        `forward` would be a Python branch that dynamo guards on, forcing a recompile every step.
        """
        w = self.weight.detach()
        gs = global_scale_of(w)
        fwd = quantize(w, -1, global_scale=gs)
        dgrad = quantize(w, 0, global_scale=gs)
        self._w_fwd_packed.copy_(fwd.packed.view(torch.uint8))
        self._w_fwd_scale.copy_(fwd.scale.view(torch.uint8))
        self._w_fwd_gs.copy_(fwd.global_scale)
        self._w_dgrad_packed.copy_(dgrad.packed.view(torch.uint8))
        self._w_dgrad_scale.copy_(dgrad.scale.view(torch.uint8))
        self._w_dgrad_gs.copy_(dgrad.global_scale)
        self._cache_valid = True

    def _eligible(self, x: torch.Tensor) -> bool:
        # `nvfp4_supported` is memoised because this runs on every linear's forward — a device
        # capability query per projection per block per loop iteration is not free, and the answer
        # cannot change within a process.
        return (
            x.is_cuda
            and _HAS_TRITON
            and _supported_cached(x.device)
            and self._cache_valid
            and quantize_supported_shape(x.numel() // self.in_features, self.in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._eligible(x):
            _warn_fallback_once(self, x)
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        flat = x.reshape(-1, self.in_features).contiguous()
        y = _FP4LinearFn.apply(
            flat, self.weight, self.bias,
            self._w_fwd_packed.view(torch.float4_e2m1fn_x2), self._w_fwd_scale.view(torch.float8_e4m3fn), self._w_fwd_gs,
            self._w_dgrad_packed.view(torch.float4_e2m1fn_x2), self._w_dgrad_scale.view(torch.float8_e4m3fn), self._w_dgrad_gs,
            self._hadamard, self._seed, self.recipe,
        )
        return y.view(*x.shape[:-1], self.out_features)


_warned_fp4_fallback = False


def _warn_fallback_once(module: FP4Linear, x: torch.Tensor) -> None:
    """A silent fallback is the worst outcome here: the run trains correctly, at bf16 speed and
    bf16 quality, while the config claims FP4 — and the measurement that follows is uninterpretable.

    Most likely cause in practice is a token count that is not a multiple of 128. Under
    `act_capacity_ratio < 1.0` the gathered capacity frequently is not.
    """
    global _warned_fp4_fallback
    if _warned_fp4_fallback:
        return
    _warned_fp4_fallback = True
    tokens = x.numel() // module.in_features
    print(
        f"[radiance] nvfp4: falling back to bf16 for at least one linear "
        f"(tokens={tokens}, in_features={module.in_features}, cuda={x.is_cuda}, "
        f"cache_valid={module._cache_valid}). FP4 needs tokens % 128 == 0 and in_features % 64 == 0."
    )


def refresh_fp4_weights(model: nn.Module) -> None:
    """Re-quantize every `FP4Linear`'s weight cache. Call once per optimizer step.

    **Forgetting this call fails silently and expensively**: every forward keeps using the step-0
    quantized weights while the fp32 masters go on training, so the loss still falls (the gradients
    are real) and then plateaus. Same family as `loop_bptt_window` freezing `blocks[0]`.

    Safe under `grad_checkpoint`: the cache is read-only for the whole step, so the backward's
    recompute reads exactly what the forward did. That is the difference from
    `act_capacity_ratio`, which refuses `grad_checkpoint` because its retained K/V store is
    *written* during the block.
    """
    for module in model.modules():
        if isinstance(module, FP4Linear):
            module.refresh_fp4_cache()
            module._seed.add_(1)
