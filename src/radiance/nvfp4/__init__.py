"""NVFP4 (4-bit) GEMM primitives.

NVFP4 is Blackwell's 4-bit float format: `e2m1` elements (1 sign, 2 exponent, 1 mantissa bit,
representable magnitudes `{0, .5, 1, 1.5, 2, 3, 4, 6}`) packed two per byte, with a per-16-element
block scale in `e4m3` and one per-tensor `fp32` global scale on top. The block scale is what makes
4 bits usable at all: 16 consecutive values along the contraction dimension share an exponent, so
the format only has to represent each block's *shape*, not its magnitude.

This module is a numerics kernel library. It knows nothing about transformers; `model/` imports
it and nothing else does. It is kept out of `model/` for the reason `optim.py` was split out —
`model/` is already long and none of this is architecture.

Three facts about the hardware and about PyTorch drove the design, all measured on an RTX 5090
(sm_120, torch 2.13) rather than assumed:

- **NVFP4 is the only 4-bit format available here.** `_ScalingType.BlockWise1x32` (MXFP4) raises
  "MXFP4 scaling only supported in CUDA for B200/B300"; `BlockWise1x16` (NVFP4) works. Both scale
  tensors must additionally be in `SWIZZLE_32_4_4` layout — `NO_SWIZZLE` raises.
- **The GEMM is 5.2x bf16** at `4096x1280x3840` (1321 TFLOPS vs 226), but **quantization, not the
  GEMM, is the bottleneck**: a naive implementation measured *three times slower* than bf16
  end-to-end. Everything here is shaped by that. See `quantize`.
- **Inductor cannot lower `_scaled_mm_v2` with swizzled scales** ("does not yet support non-trivial
  swizzles"), so the GEMM is wrapped in a `torch.library.custom_op`. That keeps it *in* the dynamo
  graph as an opaque extern call, where `@torch._dynamo.disable` would break the graph and give
  back most of the win (0.493 ms vs 0.392 ms on an 8-linear stack).

The scale/global-scale split is worth stating once because it is easy to get subtly wrong. For a
block with `amax = max|x|` over its 16 elements:

    global_scale = max|x| over the whole tensor / (6 * 448)      # fp32, one per tensor
    block_scale  = e4m3(block amax / 6 / global_scale)           # the tensor the GEMM consumes
    q            = x / (block_scale * global_scale)              # in [-6, 6], encoded as e2m1

`_scaled_mm_v2` applies `block_scale` only, so the caller multiplies the output by
`global_scale_a * global_scale_b` afterwards. Passing the global scales as a second `TensorWise`
recipe entry alongside the block scales was tried and produced a wrong result.
"""

from .linear import FP4Linear, FP4Recipe, refresh_fp4_weights
from .quantize import (
    BLOCK,
    E4M3_MAX,
    FP4_MAX,
    FP4_VALUES,
    TINY,
    NVFP4Tensor,
    _block_scale,
    blocked_offset,
    dequantize,
    encode_e2m1,
    global_scale_of,
    hadamard_matrix,
    mm,
    nvfp4_mm,
    nvfp4_supported,
    pack_nibbles,
    quantize,
    quantize_reference,
    quantize_supported_shape,
    ref_quantize_colblock,
    ref_quantize_rowblock,
    ref_to_blocked,
    require_nvfp4,
    scale_denominator,
    unpack_nibbles,
)

__all__ = [
    "BLOCK",
    "E4M3_MAX",
    "FP4_MAX",
    "FP4_VALUES",
    "TINY",
    "NVFP4Tensor",
    "FP4Linear",
    "FP4Recipe",
    "blocked_offset",
    "dequantize",
    "encode_e2m1",
    "global_scale_of",
    "hadamard_matrix",
    "mm",
    "nvfp4_mm",
    "nvfp4_supported",
    "pack_nibbles",
    "quantize",
    "quantize_reference",
    "quantize_supported_shape",
    "ref_quantize_colblock",
    "ref_quantize_rowblock",
    "ref_to_blocked",
    "refresh_fp4_weights",
    "require_nvfp4",
    "scale_denominator",
    "unpack_nibbles",
]
