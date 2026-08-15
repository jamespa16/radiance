"""NVFP4 quantization: the format reference implementation and the Triton kernels.

Part of the `radiance.nvfp4` package - the package `__init__` re-exports everything here, so
`nvfp4.<name>` access from `model.py` and the tests is unchanged. The `FP4Linear` layer that
consumes these primitives lives in `linear.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# e2m1's representable magnitudes, indexed by the 3-bit magnitude code. The sign occupies bit 3.
FP4_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16

# Tile widths, picked by sweep on an RTX 5090 rather than by symmetry. The rowblock kernel wants
# BK=64 so its four scale columns are exactly one SWIZZLE_32_4_4 column tile; the colblock kernel,
# which transposes in registers, is 1.6x faster at BK=32 (841 GB/s vs 523) because the narrower
# tile keeps the transpose in fewer registers. Both want 8 warps.
_ROW_BK = 64
_COL_BK = 32

# Below this a block is treated as exactly zero. Blocks really do go to zero here — DenseTransformer's
# `input_injection` is zero-initialised, and gradients are exactly zero inside `loop_bptt_window`'s
# no-grad region — and an unguarded `x / amax` makes them NaN on step 0 rather than at some later
# point where it would be easier to notice.
TINY = 1e-12


def _scaling_types():
    """Imported lazily: these are private torch symbols that don't exist before ~2.9, and importing
    this module must not break a bf16 run on an older torch (pyproject floors at torch>=2.5)."""
    from torch._C import _ScalingType, _SwizzleType

    return _ScalingType, _SwizzleType


def nvfp4_supported(device: torch.device | str | None = None) -> bool:
    """True when this build and this GPU can actually run an NVFP4 GEMM.

    Deliberately a capability probe rather than a version assertion: everything here falls back to
    bf16 `F.linear` when it returns False, so CPU tests and non-Blackwell cards keep working. The
    corollary is that a config asking for FP4 on the wrong card trains correctly at bf16 speed and
    quality while claiming to be FP4 — `train.py` prints the resolved capability once at startup so
    that cannot pass unnoticed.
    """
    if not hasattr(torch, "_scaled_mm_v2") or not hasattr(torch, "float4_e2m1fn_x2"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        _ScalingType, _ = _scaling_types()
        if not hasattr(_ScalingType, "BlockWise1x16"):
            return False
    except ImportError:
        return False
    index = None
    if device is not None:
        device = torch.device(device)
        if device.type != "cuda":
            return False
        index = device.index
    # Blackwell (sm_100 datacenter, sm_120 GeForce) is where the FP4 tensor cores live.
    return torch.cuda.get_device_capability(index)[0] >= 10


def require_nvfp4(reason: str) -> None:
    """Raise with something actionable. Called from `DenseTransformer.__init__` when a config asks
    for FP4 on a build that has no way to provide it."""
    if nvfp4_supported():
        return
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    raise RuntimeError(
        f"{reason} needs NVFP4 support: torch>=2.9 with torch._scaled_mm_v2 and a Blackwell GPU "
        f"(compute capability >= 10.0). Got torch {torch.__version__}, device capability {cap}. "
        "Unset model.fp4_linear (or train.dtype: nvfp4) to train in bf16 instead."
    )


# --------------------------------------------------------------------------------------------
# Reference implementation
#
# Pure torch, no Triton. This is what the Triton kernels are tested *against*, bit-exactly on the
# nibble codes — so it is the definition of the format for this codebase, not a fallback path.
# --------------------------------------------------------------------------------------------


def encode_e2m1(a: torch.Tensor, *, sign: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    """Magnitudes in [0, 6] -> 4-bit nibble codes (sign in bit 3).

    The rounding ladder is round-to-nearest-**even**, matching Blackwell's `cvt.rn.satfinite.e2m1x2`
    so a future PTX fast path is a drop-in. The alternating `>` / `>=` is what implements it: at a
    tie the comparison that admits equality is the one whose upper neighbour has an even code.

        code  0    1    2    3    4    5    6    7
        value 0   0.5  1.0  1.5  2.0  3.0  4.0  6.0
        tie      .25  .75  1.25 1.75 2.5  3.5  5.0
        RNE       v    ^    v    ^    v    ^    v      (v = toward the even code below)

    `u`, when given, is a uniform [0, 1) draw per element and selects **stochastic rounding**
    instead: round up with probability equal to the position between the two bracketing levels.
    This is applied to gradient tensors only. Round-to-nearest is a biased estimator and the bias
    accumulates coherently over a run; SR trades that bias for variance, which averages out across
    the tokens in a batch. On weights and activations there is no accumulation to protect, so SR
    would add variance for nothing.
    """
    if u is None:
        code = (
            (a > 0.25).to(torch.uint8)
            + (a >= 0.75).to(torch.uint8)
            + (a > 1.25).to(torch.uint8)
            + (a >= 1.75).to(torch.uint8)
            + (a > 2.5).to(torch.uint8)
            + (a >= 3.5).to(torch.uint8)
            + (a > 5.0).to(torch.uint8)
        )
    else:
        # Index of the level at or below `a`, then round up with probability (a - lo) / (hi - lo).
        lo = (
            (a >= 0.5).to(torch.uint8)
            + (a >= 1.0).to(torch.uint8)
            + (a >= 1.5).to(torch.uint8)
            + (a >= 2.0).to(torch.uint8)
            + (a >= 3.0).to(torch.uint8)
            + (a >= 4.0).to(torch.uint8)
            + (a >= 6.0).to(torch.uint8)
        )
        levels = torch.tensor(FP4_VALUES + (FP4_MAX,), device=a.device, dtype=torch.float32)
        lo_v = levels[lo.long()]
        hi_v = levels[(lo + 1).long()]
        frac = torch.where(hi_v > lo_v, (a - lo_v) / (hi_v - lo_v).clamp(min=TINY), torch.zeros_like(a))
        code = lo + (u < frac).to(torch.uint8)
        code = code.clamp(max=7)
    return code | (sign.to(torch.uint8) << 3)


def pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    """(..., K) uint8 codes -> (..., K // 2) `float4_e2m1fn_x2`.

    Element `i` occupies bits `4 * (i % 2)` of byte `i // 2` — **low nibble first**. This is not a
    free choice; it is what the hardware reads. `tests/test_nvfp4.py` pins it by decoding the packed
    bytes back through `FP4_VALUES`.
    """
    return (code[..., 1::2] << 4 | code[..., 0::2]).view(torch.float4_e2m1fn_x2)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_nibbles`, for tests and `dequantize`."""
    raw = packed.view(torch.uint8)
    low = raw & 0x0F
    high = raw >> 4
    return torch.stack([low, high], dim=-1).reshape(*raw.shape[:-1], raw.shape[-1] * 2)


def _block_scale(amax: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """amax per block -> (e4m3 scale to hand the GEMM, fp32 round-tripped scale to divide by).

    **The round-trip is load-bearing.** The GEMM multiplies by the *rounded* e4m3 scale, so dividing
    by the unrounded one leaves a systematic per-block bias, and — worse — whenever e4m3 rounds the
    scale down, `|q|` exceeds 6 and clips. Divide by exactly what the GEMM will multiply by.

    **The rounding is toward +inf, not to nearest**, and that is deliberate rather than incidental.
    Two reasons, one numerical and one about reproducibility:

    - Rounding the scale *down* is exactly the case that makes the block's largest element exceed 6
      and clip. Rounding up cannot, so `|q| <= 6` holds by construction and the clamp downstream is
      a guard rather than a live path.
    - It removes a tie, and the tie was a real portability problem. `torch`'s fp32 -> e4m3 cast
      rounds halves to even; Triton's rounds them toward zero. That disagreement showed up as one
      differing scale byte in 327680 — rare enough to survive a casual test and frequent enough to
      hit ~3% of tensors. Bumping up whenever the cast landed below the true value gives the same
      answer whichever way the underlying cast broke the tie, so the reference and the kernel agree
      bit-for-bit without either depending on the other's rounding mode.
    """
    scale = (amax / scale_denominator(global_scale)).clamp(min=TINY, max=E4M3_MAX)
    bits = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    # 0x7E is the largest finite e4m3 magnitude (448); 0x7F is NaN, so never step onto it.
    bump = ((bits.view(torch.float8_e4m3fn).float() < scale) & (bits < 0x7E)).to(torch.uint8)
    scale_e4m3 = (bits + bump).view(torch.float8_e4m3fn)
    # A zero block would divide by zero; force its divisor to 1 and let the codes come out zero.
    divisor = torch.where(amax > 0, scale_e4m3.float(), torch.ones_like(amax))
    return scale_e4m3, divisor


def scale_denominator(global_scale: torch.Tensor) -> torch.Tensor:
    """`6 * global_scale`, precomputed on the host so the block scale is **one** fp32 division.

    This exists to make the reference and the Triton kernel agree bit-for-bit, and the reason is a
    trap worth knowing about: `tl.math.div_rn(amax, 6.0)` promotes the Python literal to fp64, so
    Triton computed `amax / 6 / gs` with a single final rounding where torch rounds twice. That
    disagreed on **32% of elements** by 1 ULP — invisible in the dequantized values, but 1 ULP is
    all it takes to flip a rounding decision at a block boundary, which is how it surfaced.

    Dividing once by a precomputed fp32 tensor removes both the literal and the extra rounding, and
    leaves nothing for the two implementations to disagree about.
    """
    return FP4_MAX * global_scale


def global_scale_of(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor fp32 global scale: `amax / (6 * 448)`, so every block scale fits in e4m3.

    Deliberately a plain torch reduction rather than part of the quantize kernel. Under
    `torch.compile` inductor fuses it into the epilogue of whatever produced `x`, which is most of
    why the compiled quantizer is 5x the eager one. Keeping it separate also leaves delayed scaling
    (reuse the previous step's amax, Transformer-Engine style) available as a one-line change.
    """
    return (x.abs().amax().float() / (FP4_MAX * E4M3_MAX)).clamp(min=TINY)


def ref_quantize_rowblock(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, K) -> packed (M, K // 2), unswizzled e4m3 scale (M, K // BLOCK).

    Blocks run along the **last** dim. This is the form an operand takes when its last dim is the
    GEMM's contraction dim: `X` and `W` in the forward, `dY` in dgrad.
    """
    m, k = x.shape
    xb = x.reshape(m, k // BLOCK, BLOCK).float()
    scale_e4m3, divisor = _block_scale(xb.abs().amax(-1, keepdim=True), global_scale)
    q = (xb / (divisor * global_scale)).clamp(-FP4_MAX, FP4_MAX)
    code = encode_e2m1(q.abs(), sign=q < 0, u=None if u is None else u.reshape(m, k // BLOCK, BLOCK))
    return pack_nibbles(code.reshape(m, k)), scale_e4m3.squeeze(-1)


def ref_quantize_colblock(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    hadamard: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, K) -> packed (K, M // 2), unswizzled e4m3 scale (K, M // BLOCK).

    Blocks run along the **first** dim, and the result is transposed: rows become `K`. This is the
    form an operand takes when its *leading* dim is the contraction dim — `W` in dgrad, and both
    `dY` and `X` in wgrad.

    It never materialises `x.t()`. A separate transpose would be a full read and write at the input
    dtype, several times the traffic of the 4-bit write it feeds.

    `hadamard` is an optional `(BLOCK, BLOCK)` orthogonal rotation applied along the contraction
    axis before the amax. The point is outlier suppression: a single large value in a block forces
    the whole block's scale up and quantizes its 15 neighbours to near-zero. Rotating mixes each
    outlier across the block. It is free here because **the rotation group and the scale block are
    the same 16 elements** — the transform happens on exactly the tile the reduction is about to
    consume. Both wgrad operands must receive the *same* rotation, since correctness rests on
    `(RᵀdY)ᵀ(RᵀX) = dYᵀ(R Rᵀ)X = dYᵀX`.
    """
    m, k = x.shape
    xb = x.reshape(m // BLOCK, BLOCK, k).float()
    if hadamard is not None:
        xb = torch.einsum("ij,bjk->bik", hadamard.t().float(), xb)
    scale_e4m3, divisor = _block_scale(xb.abs().amax(1, keepdim=True), global_scale)
    q = (xb / (divisor * global_scale)).clamp(-FP4_MAX, FP4_MAX)
    code = encode_e2m1(q.abs(), sign=q < 0, u=None if u is None else u.reshape(m // BLOCK, BLOCK, k))
    # (m // BLOCK, BLOCK, k) -> (k, m), then pack pairs along the new last dim (the old m).
    code = code.reshape(m, k).t().contiguous()
    return pack_nibbles(code), scale_e4m3.squeeze(1).t().contiguous()


def ref_to_blocked(scale: torch.Tensor) -> torch.Tensor:
    """(rows, cols) e4m3 block scales -> the flat `SWIZZLE_32_4_4` layout cuBLAS requires.

    Pads to a whole number of 128x4 tiles and rearranges each tile so the hardware's scale fetch is
    contiguous. See https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout.

    Padding is not optional: the hardware reads the full 128-row tile whether or not it was written,
    so an M that is not a multiple of 128 must still produce a padded tile. `M = 100` and `M = 64`
    both fail against an unpadded layout.
    """
    rows, cols = scale.shape
    n_row_tiles = (rows + 127) // 128
    n_col_tiles = (cols + 3) // 4
    padded_rows, padded_cols = n_row_tiles * 128, n_col_tiles * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = scale.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols] = scale
        scale = padded
    tiles = scale.view(n_row_tiles, 128, n_col_tiles, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


def blocked_offset(r: torch.Tensor, c: torch.Tensor, n_col_tiles: int) -> torch.Tensor:
    """Closed form of `ref_to_blocked`'s permutation: byte offset of scale element (r, c).

    The Triton kernels store each scale byte straight to its final address using this, rather than
    running a separate `to_blocked` pass — which would cost a full read/write of the scale tensor
    plus four extra kernel launches per linear per micro-batch.

    `tests/test_nvfp4.py::test_blocked_offset_matches_reference` asserts this agrees with
    `ref_to_blocked` element for element. Derived rather than obvious, so it is pinned, not trusted.
    """
    tile = (r // 128) * n_col_tiles + (c // 4)
    within = (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    return tile * 512 + within


def dequantize(
    packed: torch.Tensor,
    scale_e4m3: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Packed nibbles + unswizzled block scales -> fp32. Tests and diagnostics only."""
    code = unpack_nibbles(packed)
    values = torch.tensor(FP4_VALUES, device=packed.device, dtype=torch.float32)
    magnitude = values[(code & 0x07).long()]
    signed = torch.where(code >= 8, -magnitude, magnitude)
    rows, cols = signed.shape
    blocks = signed.reshape(rows, cols // BLOCK, BLOCK)
    return (blocks * (scale_e4m3.float() * global_scale).unsqueeze(-1)).reshape(rows, cols)


# --------------------------------------------------------------------------------------------
# The packed operand and the GEMM
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NVFP4Tensor:
    """One GEMM operand in NVFP4.

    The canonical layout is **rows = the operand's output index, columns = the contraction index**,
    with scale blocks running along columns. `_scaled_mm_v2` takes the contraction dim explicitly,
    so adopting that convention for every operand of every GEMM collapses the whole problem to two
    quantizers — one blocking along a tensor's last dim, one along its first:

        fwd   Y  = X Wᵀ    X rows M contract K (row)   W rows N contract K (row)
        dgrad dX = dY W    dY rows M contract N (row)  W rows K contract N (col)
        wgrad dW = dYᵀ X   dY rows N contract M (col)  X rows K contract M (col)

    Read the weight column: `W` is quantized twice and both forms are cacheable per optimizer step.
    `X` and `dY` are each quantized both ways per micro-batch — four activation-side quantizations
    per linear, which is the cost that has to be driven down to bandwidth for any of this to pay.
    """

    packed: torch.Tensor  # float4_e2m1fn_x2, (rows, contraction // 2)
    scale: torch.Tensor  # float8_e4m3fn, SWIZZLE_32_4_4, flat
    global_scale: torch.Tensor  # fp32, 0-d

    @property
    def rows(self) -> int:
        return self.packed.shape[0]

    @property
    def contraction(self) -> int:
        return self.packed.shape[1] * 2


@torch.library.custom_op("radiance::nvfp4_mm", mutates_args=())
def nvfp4_mm(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """`a @ b.T` in NVFP4, without the global scales. (M, K/2) x (N, K/2) -> (M, N) bf16.

    A `custom_op` rather than a bare call because inductor refuses to lower `_scaled_mm_v2` with
    swizzled scales. As a custom op it stays in the dynamo graph as an opaque extern call and
    inductor still fuses the elementwise work *around* it; `@torch._dynamo.disable` would instead
    break the graph, which measured 0.493 ms against this path's 0.392 ms on an 8-linear stack.
    """
    _ScalingType, _SwizzleType = _scaling_types()
    block = [_ScalingType.BlockWise1x16.value]
    swizzle = [_SwizzleType.SWIZZLE_32_4_4.value]
    return torch._scaled_mm_v2(
        a_packed,
        b_packed.t(),
        scale_a=[a_scale.view(torch.float8_e4m3fn)],
        recipe_a=block,
        swizzle_a=swizzle,
        scale_b=[b_scale.view(torch.float8_e4m3fn)],
        recipe_b=block,
        swizzle_b=swizzle,
        bias=None,
        out_dtype=torch.bfloat16,
    )


@nvfp4_mm.register_fake
def _nvfp4_mm_fake(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    return a_packed.new_empty((a_packed.shape[0], b_packed.shape[0]), dtype=torch.bfloat16)


def mm(a: NVFP4Tensor, b: NVFP4Tensor, *, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """`a @ b.T` with the global scales folded back in.

    The fold happens here, after the GEMM, rather than as a second `TensorWise` entry in the recipe
    — that was tried and gave a wrong result. It is also why this path must not run under fp16:
    the pre-fold output is ~`1/(ga*gb)` times the true product, comfortably inside bf16's exponent
    range and outside fp16's.
    """
    out = nvfp4_mm(a.packed, a.scale, b.packed, b.scale)
    return (out * (a.global_scale * b.global_scale)).to(out_dtype)


def quantize_reference(
    x: torch.Tensor,
    axis: int,
    *,
    global_scale: torch.Tensor | None = None,
    hadamard: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
) -> NVFP4Tensor:
    """Reference `quantize`: `axis=-1` blocks along the last dim, `axis=0` along the first.

    `quantize` below is the Triton implementation of exactly this; the two are asserted
    bit-identical on the nibble codes, so this stays the definition of the format.
    """
    if global_scale is None:
        global_scale = global_scale_of(x)
    if axis in (-1, x.dim() - 1):
        packed, scale = ref_quantize_rowblock(x, global_scale, u=u)
    elif axis == 0:
        packed, scale = ref_quantize_colblock(x, global_scale, hadamard=hadamard, u=u)
    else:
        raise ValueError(f"nvfp4.quantize supports axis 0 or -1, got {axis}")
    return NVFP4Tensor(packed=packed, scale=ref_to_blocked(scale), global_scale=global_scale)


# --------------------------------------------------------------------------------------------
# Triton kernels
#
# The reference above runs at roughly 350 GB/s compiled and 100 GB/s eager, against ~1.6 TB/s
# achievable on this card. That gap is the whole reason FP4 was measured *three times slower* than
# bf16 end-to-end: the GEMM is 5.2x faster, but a linear needs four activation-side quantizations
# per micro-batch and at 350 GB/s they cost more than the GEMM they feed.
#
# Tile: BM=128, BK=64. Chosen so one program produces exactly one 512-byte SWIZZLE_32_4_4 scale
# tile — 128 rows is one row tile, and BK/16 = 4 scale columns is one column tile. No cross-program
# straddling, no atomics, no second pass to swizzle.
# --------------------------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with the CUDA wheels
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _e2m1_level(code):
        """Magnitude of an e2m1 code, as arithmetic rather than a lookup — Triton has no gather over
        a constant table cheaper than this."""
        return tl.where(code <= 4, code.to(tl.float32) * 0.5, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0)))

    @triton.jit
    def _encode(q, u, STOCHASTIC: tl.constexpr):
        """Signed magnitudes in [-6, 6] -> 4-bit nibbles. Mirrors `encode_e2m1` exactly, including
        the alternating >/>= that makes the round-to-nearest ladder round-to-nearest-*even*."""
        a = tl.abs(q)
        sign = (q < 0).to(tl.uint8) << 3
        if STOCHASTIC:
            lo = (
                (a >= 0.5).to(tl.uint8) + (a >= 1.0).to(tl.uint8) + (a >= 1.5).to(tl.uint8)
                + (a >= 2.0).to(tl.uint8) + (a >= 3.0).to(tl.uint8) + (a >= 4.0).to(tl.uint8)
                + (a >= 6.0).to(tl.uint8)
            )
            lo_v = _e2m1_level(lo)
            hi_v = _e2m1_level(tl.minimum(lo + 1, 7))
            span = tl.maximum(hi_v - lo_v, 1e-12)
            frac = tl.where(hi_v > lo_v, (a - lo_v) / span, 0.0)
            code = tl.minimum(lo + (u < frac).to(tl.uint8), 7)
        else:
            code = (
                (a > 0.25).to(tl.uint8) + (a >= 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8)
                + (a >= 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a >= 3.5).to(tl.uint8)
                + (a > 5.0).to(tl.uint8)
            )
        return code | sign

    @triton.jit
    def _scale_and_divisor(amax, gs, denom):
        """Block amax -> (e4m3 scale for the GEMM, fp32 divisor for the quantizer).

        The round-trip through e4m3 is what keeps the two consistent; see `_block_scale`. The
        `amax > 0` guard is what stops an all-zero block becoming NaN.

        `div_rn` throughout, not `/`. Triton's default fp32 divide is ~1 ULP off IEEE, and both of
        these feed a rounding decision at a tie boundary — e4m3's here, e2m1's downstream. With
        bf16 inputs exact ties are common rather than rare (only 8 mantissa bits, and the divisor
        is derived from one of the inputs), so a 1-ULP divide disagrees with the reference on
        0.1-0.5% of elements. Loud enough to catch, quiet enough to have shipped."""
        # One fp32 division by a host-computed denominator; see `scale_denominator` for why the
        # obvious `amax / 6.0 / gs` disagrees with torch on 32% of elements.
        scale = tl.minimum(tl.maximum(tl.math.div_rn(amax, denom), 1e-12), 448.0)
        # Round the scale toward +inf, matching `_block_scale`. Whichever way this cast breaks a
        # tie (Triton's goes toward zero, torch's to even), bumping up when it landed below the
        # true value lands both on the same e4m3 byte. See `_block_scale` for why that matters.
        bits = tl.cast(scale, tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True)
        bump = ((bits.to(tl.float8e4nv, bitcast=True).to(tl.float32) < scale) & (bits < 0x7E)).to(tl.uint8)
        s_e4m3 = (bits + bump).to(tl.float8e4nv, bitcast=True)
        divisor = tl.where(amax > 0, s_e4m3.to(tl.float32), 1.0).to(tl.float32) * gs
        return s_e4m3, divisor

    @triton.jit
    def _pack_and_store(code, packed_ptr, row, col_half, row_stride, n_half, BM: tl.constexpr, BH: tl.constexpr):
        """(BM, 2*BH) nibbles -> (BM, BH) bytes, low nibble first, stored contiguously."""
        pair = tl.reshape(code, (BM, BH, 2))
        lo, hi = tl.split(pair)
        byte = lo | (hi << 4)
        tl.store(
            packed_ptr + row[:, None] * row_stride + col_half[None, :],
            byte,
            mask=col_half[None, :] < n_half,
        )

    @triton.jit
    def _store_scale_tile(s_e4m3, scale_ptr, tile_index, BM: tl.constexpr, NB: tl.constexpr):
        """One (128, 4) scale tile -> its 512 contiguous bytes in SWIZZLE_32_4_4 order.

        `blocked_offset`'s closed form, specialised to a single tile: the tile term drops out into
        `tile_index * 512` and only the within-tile permutation is computed here."""
        r = tl.arange(0, BM)
        c = tl.arange(0, NB)
        within = (r[:, None] % 32) * 16 + ((r[:, None] % 128) // 32) * 4 + c[None, :]
        tl.store(scale_ptr + tile_index * 512 + within, s_e4m3.to(tl.uint8, bitcast=True))

    @triton.jit
    def _quant_rowblock_kernel(
        x_ptr, packed_ptr, scale_ptr, gs_ptr, denom_ptr,
        M, K, stride_xm, stride_xk,
        n_col_tiles, seed_ptr,
        STOCHASTIC: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    ):
        """Blocks along the last dim. One program = one (128, 64) input tile = one scale tile."""
        pid_m, pid_k = tl.program_id(0), tl.program_id(1)
        NB: tl.constexpr = BK // 16
        BH: tl.constexpr = BK // 2

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_k = pid_k * BK + tl.arange(0, BK)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        gs = tl.load(gs_ptr).to(tl.float32)
        denom = tl.load(denom_ptr).to(tl.float32)

        xb = tl.reshape(x, (BM, NB, 16))
        s_e4m3, divisor = _scale_and_divisor(tl.max(tl.abs(xb), axis=2), gs, denom)
        q = tl.minimum(tl.maximum(tl.math.div_rn(xb, divisor[:, :, None]), -6.0), 6.0)

        u = tl.zeros((BM, NB, 16), dtype=tl.float32)
        if STOCHASTIC:
            lin = offs_m[:, None, None] * K + (pid_k * BK + tl.arange(0, NB)[None, :, None] * 16 + tl.arange(0, 16)[None, None, :])
            u = tl.rand(tl.load(seed_ptr), lin)
        code = tl.reshape(_encode(q, u, STOCHASTIC), (BM, BK))

        _pack_and_store(code, packed_ptr, offs_m, pid_k * BH + tl.arange(0, BH), K // 2, K // 2, BM, BH)
        _store_scale_tile(s_e4m3, scale_ptr, pid_m * n_col_tiles + pid_k, BM, NB)

    @triton.jit
    def _quant_colblock_kernel(
        x_ptr, packed_ptr, scale_ptr, gs_ptr, denom_ptr, had_ptr,
        M, K, stride_xm, stride_xk,
        n_col_tiles, seed_ptr,
        HADAMARD: tl.constexpr, STOCHASTIC: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    ):
        """Blocks along the *first* dim, output transposed: (M, K) -> packed (K, M // 2).

        Reads `x` in its natural layout — the transpose happens in registers. Materialising `x.t()`
        first would cost a full read and write at the input dtype, several times the traffic of the
        4-bit write this feeds.

        The Hadamard rotation folds in for free because **the rotation group and the scale block are
        the same 16 elements**: the rotation is applied to exactly the tile the amax is about to
        reduce. `R.T @ g` is computed as one (16, 16) x (16, BM // 16 * BK) `tl.dot`.
        """
        pid_m, pid_k = tl.program_id(0), tl.program_id(1)
        NG: tl.constexpr = BM // 16  # scale blocks (and rotation groups) along M
        BH: tl.constexpr = BM // 2

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_k = pid_k * BK + tl.arange(0, BK)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        gs = tl.load(gs_ptr).to(tl.float32)
        denom = tl.load(denom_ptr).to(tl.float32)

        # (BM, BK) -> (NG, 16, BK): 16 consecutive rows form one block along the contraction dim.
        xb = tl.reshape(x, (NG, 16, BK))
        if HADAMARD:
            # Rotate along the 16-axis: move it first so the batch collapses into one tl.dot.
            g = tl.reshape(tl.trans(xb, 1, 0, 2), (16, NG * BK))
            # R **transposed**: the rotation applied is Rᵀ, and both wgrad operands must get the
            # same one for `(RᵀdY)ᵀ(RᵀX) = dYᵀX` to hold. Loading R instead of Rᵀ here is a
            # wrong-but-finite gradient with no error raised.
            h = tl.load(had_ptr + tl.arange(0, 16)[None, :] * 16 + tl.arange(0, 16)[:, None])
            # ieee, not the default tf32: this feeds an amax and then a tie-sensitive rounding
            # decision, and tf32's 10-bit mantissa would disagree with the reference constantly.
            xb = tl.trans(tl.reshape(tl.dot(h, g, input_precision="ieee"), (16, NG, BK)), 1, 0, 2)

        s_e4m3, divisor = _scale_and_divisor(tl.max(tl.abs(xb), axis=1), gs, denom)
        q = tl.minimum(tl.maximum(tl.math.div_rn(xb, divisor[:, None, :]), -6.0), 6.0)

        u = tl.zeros((NG, 16, BK), dtype=tl.float32)
        if STOCHASTIC:
            lin = (pid_m * BM + tl.arange(0, NG)[:, None, None] * 16 + tl.arange(0, 16)[None, :, None]) * K + offs_k[None, None, :]
            u = tl.rand(tl.load(seed_ptr), lin)
        # (NG, 16, BK) -> (BK, BM): rows become K, columns become the contraction dim M.
        code = tl.trans(tl.reshape(_encode(q, u, STOCHASTIC), (BM, BK)), 1, 0)

        _pack_and_store(code, packed_ptr, offs_k, pid_m * BH + tl.arange(0, BH), M // 2, M // 2, BK, BH)
        # Scale rows are K here, so the tile grid is transposed relative to the rowblock kernel.
        _store_scale_tile_col(s_e4m3, scale_ptr, pid_k, pid_m, K, n_col_tiles, BK, NG)

    @triton.jit
    def _store_scale_tile_col(s_e4m3, scale_ptr, pid_k, pid_m, K, n_col_tiles, BK: tl.constexpr, NG: tl.constexpr):
        """Colblock scales come out (NG, BK) with rows = M-blocks; the output wants (BK, NG) with
        rows = K. Transpose in registers, then scatter with the general `blocked_offset` form.

        Unlike the rowblock case this program does *not* own exactly one tile — `BK = 64` is half a
        128-row tile and `NG = 8` spans two 4-column tiles — so the tile term stays inside the
        offset computation rather than factoring out.
        """
        s = tl.trans(s_e4m3, 1, 0)  # (BK, NG)
        r = (pid_k * BK + tl.arange(0, BK))[:, None]
        c = (pid_m * NG + tl.arange(0, NG))[None, :]
        offset = ((r // 128) * n_col_tiles + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
        tl.store(scale_ptr + offset, s.to(tl.uint8, bitcast=True), mask=r < K)


def hadamard_matrix(n: int = BLOCK, *, device=None, seed: int = 0) -> torch.Tensor:
    """Normalised `n x n` Hadamard with a fixed random sign diagonal — the `R` in `RᵀdY`, `RᵀX`.

    One matrix for the whole run, from a fixed seed, so a resumed run agrees with the original. It
    is exact in fp32, so a mismatch across resume would be a precision change rather than a
    correctness one — but the seed is fixed anyway, because it is free.
    """
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.where(torch.rand(n, 1, generator=gen) < 0.5, -1.0, 1.0).to(h.device)
    return h / n**0.5 * signs


def quantize_supported_shape(rows: int, cols: int) -> bool:
    """The kernels' shape contract: 128-row tiles and 64-wide blocks, unpadded.

    `rows % 128` is the one that bites at runtime — it is the token count, and under
    `act_capacity_ratio < 1.0` the gathered capacity frequently is not a multiple of 128. Callers
    fall back to `F.linear` rather than padding, and warn once.
    """
    return rows % 128 == 0 and cols % 64 == 0


def quantize(
    x: torch.Tensor,
    axis: int,
    *,
    global_scale: torch.Tensor | None = None,
    hadamard: torch.Tensor | None = None,
    stochastic: bool = False,
    seed: torch.Tensor | int = 0,
) -> NVFP4Tensor:
    """Quantize `x` to NVFP4. `axis=-1` blocks along the last dim, `axis=0` along the first.

    Bit-identical to `quantize_reference` on the nibble codes, which is asserted rather than assumed
    (`tests/test_nvfp4.py::test_triton_matches_reference_bit_exactly`).

    `stochastic` selects stochastic rounding, for gradient tensors only. `seed` may be a 0-d int64
    tensor so it can live in a buffer that `refresh_fp4_weights` advances once per optimizer step —
    a Python int would be a dynamo guard and force a recompile every step.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return quantize_reference(x, axis, global_scale=global_scale, hadamard=hadamard)
    if x.dim() != 2:
        raise ValueError(f"nvfp4.quantize expects a 2-D tensor, got shape {tuple(x.shape)}")
    if global_scale is None:
        global_scale = global_scale_of(x)
    seed_t = seed if isinstance(seed, torch.Tensor) else _default_seed(x.device, seed)
    denom = scale_denominator(global_scale)

    m, k = x.shape
    if axis in (-1, 1):
        if not quantize_supported_shape(m, k):
            raise ValueError(f"nvfp4.quantize rowblock needs rows % 128 == 0 and cols % 64 == 0, got {(m, k)}")
        n_row_tiles, n_col_tiles = m // 128, k // 64
        packed = torch.empty(m, k // 2, device=x.device, dtype=torch.uint8)
        # empty, not zeros: the rowblock grid covers every tile and each program writes its full
        # 512-byte tile unmasked, so there is no padding to clear. The colblock path below does
        # need zeros — its scale rows are K, which need not fill the last 128-row tile.
        scale = torch.empty(n_row_tiles * n_col_tiles * 512, device=x.device, dtype=torch.uint8)
        _quant_rowblock_kernel[(n_row_tiles, k // _ROW_BK)](
            x, packed, scale, global_scale, denom,
            m, k, x.stride(0), x.stride(1),
            n_col_tiles, seed_t,
            STOCHASTIC=stochastic, BM=128, BK=_ROW_BK, num_warps=8,
        )
    elif axis == 0:
        if not quantize_supported_shape(m, k):
            raise ValueError(f"nvfp4.quantize colblock needs rows % 128 == 0 and cols % 64 == 0, got {(m, k)}")
        n_row_tiles, n_col_tiles = (k + 127) // 128, m // 64
        packed = torch.empty(k, m // 2, device=x.device, dtype=torch.uint8)
        scale = torch.zeros(n_row_tiles * n_col_tiles * 512, device=x.device, dtype=torch.uint8)
        if hadamard is None:
            hadamard = torch.empty(0, device=x.device, dtype=torch.float32)
        _quant_colblock_kernel[(m // 128, k // _COL_BK)](
            x, packed, scale, global_scale, denom, hadamard,
            m, k, x.stride(0), x.stride(1),
            n_col_tiles, seed_t,
            HADAMARD=hadamard.numel() > 0, STOCHASTIC=stochastic, BM=128, BK=_COL_BK, num_warps=8,
        )
    else:
        raise ValueError(f"nvfp4.quantize supports axis 0 or -1, got {axis}")

    return NVFP4Tensor(
        packed=packed.view(torch.float4_e2m1fn_x2),
        scale=scale.view(torch.float8_e4m3fn),
        global_scale=global_scale,
    )


_SUPPORT_CACHE: dict[torch.device, bool] = {}


def _supported_cached(device: torch.device) -> bool:
    """Memoised `nvfp4_supported`. Hot path: called once per linear per forward."""
    if device not in _SUPPORT_CACHE:
        _SUPPORT_CACHE[device] = nvfp4_supported(device)
    return _SUPPORT_CACHE[device]


_SEED_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _default_seed(device: torch.device, value: int) -> torch.Tensor:
    """Cached 0-d seed tensor.

    The kernels read the seed through a pointer rather than as a scalar argument, so that
    `refresh_fp4_weights` can advance it in place once per optimizer step without making it a
    dynamo guard. The cache is what keeps that from costing a host-to-device copy on every call —
    materialising `torch.tensor(0, device=...)` per quantization measured ~0.02 ms, comfortably
    more than the kernel it was feeding, and it also made the whole path uncapturable by CUDA
    graphs.
    """
    key = (device, value)
    if key not in _SEED_CACHE:
        _SEED_CACHE[key] = torch.tensor(value, device=device, dtype=torch.int64)
    return _SEED_CACHE[key]
