"""NVFP4 numerics.

Read this before changing anything in `nvfp4.py`, because most of what can go wrong here goes
wrong *silently* — finite, plausibly scaled, and wrong. The tempting-but-wrong version of almost
every piece produces output that passes a shape check and an `isfinite` check:

- Quantizing inside autograd's view rather than inside an `autograd.Function` gives a gradient that
  is nonzero and roughly the right magnitude, because the encode has zero derivative but the
  `amax` in the global scale does not. The loss still falls. `test_gradient_is_not_the_autograd_trap`
  builds that broken version deliberately, as a negative control.
- Applying different Hadamard rotations to the two wgrad operands leaves the gradient wrong by a
  fixed rotation. `test_wgrad_hadamard_is_exact` therefore asserts the *mismatched* case diverges
  as well as the matched case agreeing — without the second half it would pass against `R = I`.
- Dividing by the unrounded block scale while the GEMM multiplies by the e4m3-rounded one leaves a
  per-block bias and occasional clipping. Only a bit-exact comparison against the reference sees it.

So the assertions here are mostly bit-exact (`rtol=0, atol=0`) on integer nibble codes, or paired
with a negative control. An upper bound on error, on its own, is passed by an accidental fallback
to bf16 — which is exactly the bug `_fp4_eligible` could introduce.
"""

from __future__ import annotations

import pytest
import torch

from radiance import nvfp4


requires_nvfp4 = pytest.mark.skipif(
    not nvfp4.nvfp4_supported(), reason="NVFP4 needs a Blackwell GPU and torch._scaled_mm_v2"
)


def _hadamard(n: int, device, seed: int = 0) -> torch.Tensor:
    """Normalised n x n Hadamard with a random sign diagonal. `R @ R.T == I`."""
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    gen = torch.Generator(device=device).manual_seed(seed)
    signs = torch.where(torch.rand(n, 1, device=device, generator=gen) < 0.5, -1.0, 1.0)
    return h / n**0.5 * signs


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


# --- the format itself: encoding, packing, swizzle -------------------------------------------


def test_encode_is_round_to_nearest_even():
    """The rounding ladder must be RNE, matching Blackwell's `cvt.rn.satfinite.e2m1x2` so a PTX
    fast path stays a drop-in. Every tie is checked explicitly; a plain `>` ladder gets four of
    these wrong and still looks fine on random data."""
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8)
    got = nvfp4.encode_e2m1(ties, sign=torch.zeros_like(ties, dtype=torch.bool))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_encode_matches_torch_reference_bit_exactly():
    """Against torch's own `_f32_to_floatx_unpacked`, which uses the magic-adder trick rather than
    a comparison ladder. Two independent implementations of the same rounding rule."""
    common = pytest.importorskip(
        "torch.testing._internal.common_quantized", reason="needs expecttest"
    )
    a = torch.linspace(-6.5, 6.5, 4096, dtype=torch.bfloat16).reshape(1, -1)
    expected = common._bfloat16_to_float4_e2m1fn_x2(a)
    q = a.float().clamp(-nvfp4.FP4_MAX, nvfp4.FP4_MAX)
    got = nvfp4.pack_nibbles(nvfp4.encode_e2m1(q.abs(), sign=q < 0))
    torch.testing.assert_close(got.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def test_pack_is_low_nibble_first():
    """Element i occupies bits 4*(i%2) of byte i//2. Not a free choice — it is what the hardware
    reads, and getting it backwards gives rel_err ~1.0 rather than 0.13."""
    code = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
    packed = nvfp4.pack_nibbles(code).view(torch.uint8)
    torch.testing.assert_close(packed, torch.tensor([[0x21, 0x43]], dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(nvfp4.unpack_nibbles(packed), code, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(128, 4), (256, 8), (4096, 80), (100, 16), (64, 16)])
def test_blocked_offset_matches_reference(shape):
    """`blocked_offset` is a closed form of `ref_to_blocked`'s permute/transpose chain, derived by
    hand so the Triton kernels can store each scale byte straight to its final address. Derived,
    therefore pinned. Includes two shapes that need row padding."""
    rows, cols = shape
    scale = (torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) % 200).to(
        torch.float8_e4m3fn
    )
    expected = nvfp4.ref_to_blocked(scale)

    r = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    c = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    offset = nvfp4.blocked_offset(r, c, n_col_tiles=(cols + 3) // 4)
    got = torch.zeros_like(expected)
    got[offset.reshape(-1)] = scale.reshape(-1)

    torch.testing.assert_close(got.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def test_block_scale_is_round_tripped_through_e4m3():
    """The GEMM multiplies by the e4m3-rounded scale, so the quantizer must divide by exactly that.
    Dividing by the unrounded scale leaves a per-block bias and lets |q| exceed 6 whenever e4m3
    rounds down. Checked by construction: the divisor must be exactly representable in e4m3."""
    amax = torch.rand(64, 1) * 10 + 0.1
    gs = torch.tensor(0.01)
    scale_e4m3, divisor = nvfp4._block_scale(amax, gs)
    torch.testing.assert_close(divisor, scale_e4m3.float(), rtol=0, atol=0)


def test_zero_blocks_quantize_to_zero_without_nan():
    """`amax == 0` divides by zero unless guarded. Reachable on step 0: `input_injection` is
    zero-initialised, and gradients are exactly zero inside `loop_bptt_window`'s no-grad region."""
    x = torch.randn(64, 64)
    x[:, :16] = 0.0  # one all-zero block per row, alongside live ones
    gs = nvfp4.global_scale_of(x)
    packed, scale = nvfp4.ref_quantize_rowblock(x, gs)
    out = nvfp4.dequantize(packed, scale, gs)
    assert torch.isfinite(out).all()
    assert (out[:, :16] == 0).all()
    assert out[:, 16:].abs().sum() > 0

    zeros = torch.zeros(32, 32)
    gs0 = nvfp4.global_scale_of(zeros)
    packed, scale = nvfp4.ref_quantize_rowblock(zeros, gs0)
    assert torch.isfinite(nvfp4.dequantize(packed, scale, gs0)).all()


def test_stochastic_rounding_is_unbiased_and_round_to_nearest_is_not():
    """SR trades bias for variance. The bias is what matters: round-to-nearest's error accumulates
    coherently across a run, SR's averages out. Both halves are asserted — a broken SR that just
    fell back to RN would pass the first."""
    x = torch.randn(128, 128)
    gs = nvfp4.global_scale_of(x)

    rn = nvfp4.dequantize(*nvfp4.ref_quantize_rowblock(x, gs), gs)
    acc = torch.zeros_like(x)
    draws = 200
    for i in range(draws):
        u = torch.rand(x.shape, generator=torch.Generator().manual_seed(i))
        acc += nvfp4.dequantize(*nvfp4.ref_quantize_rowblock(x, gs, u=u), gs)
    sr_mean = acc / draws

    assert (sr_mean - x).mean().abs() < (rn - x).mean().abs()
    assert _rel_err(sr_mean, x) < _rel_err(rn, x) / 2

    # RN is bit-reproducible; SR is not.
    again = nvfp4.dequantize(*nvfp4.ref_quantize_rowblock(x, gs), gs)
    torch.testing.assert_close(rn, again, rtol=0, atol=0)


def test_colblock_is_the_transpose_of_rowblock_when_unrotated():
    """The two quantizers must agree about the format; they differ only in which axis the blocks
    run along. Quantizing `x` colblock is quantizing `x.T` rowblock, without materialising `x.T`."""
    x = torch.randn(64, 32)
    gs = nvfp4.global_scale_of(x)
    col_packed, col_scale = nvfp4.ref_quantize_colblock(x, gs)
    row_packed, row_scale = nvfp4.ref_quantize_rowblock(x.t().contiguous(), gs)
    torch.testing.assert_close(
        col_packed.view(torch.uint8), row_packed.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        col_scale.view(torch.uint8), row_scale.view(torch.uint8), rtol=0, atol=0
    )


# --- the GEMM --------------------------------------------------------------------------------


@requires_nvfp4
@pytest.mark.parametrize("shape", [(256, 256, 256), (4096, 1280, 3840), (128, 64, 128)])
def test_forward_gemm_matches_fp32_reference(shape):
    """rel_err ~0.134 is the inherent cost of e2m1 with 16-wide blocks on gaussian data; 0.2 is the
    acceptance bound.

    The `not allclose to bf16` half matters as much as the bound: an upper bound alone is passed by
    an accidental fallback to a bf16 matmul, which is precisely what a bug in `_fp4_eligible` would
    produce, and it would look like a clean pass with no speedup.
    """
    m, k, n = shape
    device = "cuda"
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    reference = x.float() @ w.float().t()

    got = nvfp4.mm(nvfp4.quantize_reference(x, -1), nvfp4.quantize_reference(w, -1))

    assert got.shape == (m, n)
    assert _rel_err(got, reference) < 0.2
    assert not torch.allclose(got.float(), (x @ w.t()).float(), rtol=1e-3, atol=1e-3)


@requires_nvfp4
def test_dgrad_and_wgrad_orientations():
    """The other two GEMMs of a linear. dgrad contracts over N, wgrad over M, so the weight and both
    activations need the colblock form — this pins that the canonical operand layout actually holds
    for all three, which is the claim `NVFP4Tensor`'s docstring makes."""
    device = "cuda"
    m, k, n = 4096, 1280, 3840
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    gy = torch.randn(m, n, device=device, dtype=torch.bfloat16)

    dx = nvfp4.mm(nvfp4.quantize_reference(gy, -1), nvfp4.quantize_reference(w, 0))
    assert dx.shape == (m, k)
    assert _rel_err(dx, gy.float() @ w.float()) < 0.2

    dw = nvfp4.mm(nvfp4.quantize_reference(gy, 0), nvfp4.quantize_reference(x, 0))
    assert dw.shape == (n, k)
    assert _rel_err(dw, gy.float().t() @ x.float()) < 0.2


@requires_nvfp4
def test_wgrad_hadamard_is_exact_and_mismatched_rotations_are_not():
    """Correctness of the rotation rests on `(RᵀdY)ᵀ(RᵀX) = dYᵀ(R Rᵀ)X = dYᵀX`, which needs the
    *same* R on both operands.

    The negative control is the point of the test. Two different sign vectors give a wrong-but-finite
    gradient with no error raised, and without asserting that case diverges this test would pass
    against `R = I` — i.e. against the rotation silently not happening at all.
    """
    device = "cuda"
    m, k, n = 2048, 512, 1024
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    gy = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    reference = gy.float().t() @ x.float()

    r = _hadamard(nvfp4.BLOCK, device, seed=0)
    torch.testing.assert_close(r @ r.t(), torch.eye(nvfp4.BLOCK, device=device), rtol=1e-4, atol=1e-4)

    matched = nvfp4.mm(
        nvfp4.quantize_reference(gy, 0, hadamard=r), nvfp4.quantize_reference(x, 0, hadamard=r)
    )
    assert _rel_err(matched, reference) < 0.2

    other = _hadamard(nvfp4.BLOCK, device, seed=1)
    mismatched = nvfp4.mm(
        nvfp4.quantize_reference(gy, 0, hadamard=r), nvfp4.quantize_reference(x, 0, hadamard=other)
    )
    assert _rel_err(mismatched, reference) > 1.0


@requires_nvfp4
def test_custom_op_has_a_working_fake_kernel():
    """`register_fake` is what lets dynamo trace through the op without running it. A wrong shape
    here surfaces as a compile-time error deep inside AOTAutograd, so pin it directly."""
    device = "cuda"
    x = torch.randn(256, 128, device=device, dtype=torch.bfloat16)
    w = torch.randn(512, 128, device=device, dtype=torch.bfloat16)
    a, b = nvfp4.quantize_reference(x, -1), nvfp4.quantize_reference(w, -1)

    with torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True):
        fake = nvfp4.nvfp4_mm(a.packed, a.scale, b.packed, b.scale)
    assert fake.shape == (256, 512)
    assert fake.dtype == torch.bfloat16


# --- the Triton kernels ----------------------------------------------------------------------
#
# These are asserted **bit-exact** against the reference, on integer nibble codes and raw scale
# bytes. That bar is not gratuitous: getting to it flushed out four separate 1-ULP disagreements,
# every one of which was invisible in the dequantized values and would have shipped.
#
#   - `tl.math.div_rn(amax, 6.0)` promotes the literal to fp64, so Triton rounded once where torch
#     rounds twice — 32% of elements differed by 1 ULP. Fixed by `scale_denominator`.
#   - Triton's default fp32 divide is not correctly rounded, so `q` differed at rounding ties.
#   - Triton's fp32 -> e4m3 cast breaks ties toward zero; torch's breaks them to even. Fixed by
#     rounding the block scale toward +inf in both, which also makes `|q| <= 6` exact.
#   - `tl.dot` defaults to tf32, which the Hadamard path cannot tolerate.
#
# An "is it close enough" assertion passes all four. Only bit-exactness catches them.


@requires_nvfp4
@pytest.mark.parametrize("shape", [(1024, 512), (256, 128), (4096, 1280), (128, 64), (512, 1408)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("axis,rotate", [(-1, False), (0, False), (0, True)])
def test_triton_matches_reference_bit_exactly(shape, dtype, axis, rotate):
    m, k = shape
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    gs = nvfp4.global_scale_of(x)
    r = nvfp4.hadamard_matrix(nvfp4.BLOCK, device="cuda") if rotate else None

    got = nvfp4.quantize(x, axis, global_scale=gs, hadamard=r)
    expected = nvfp4.quantize_reference(x, axis, global_scale=gs, hadamard=r)

    torch.testing.assert_close(
        got.packed.view(torch.uint8), expected.packed.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        got.scale.view(torch.uint8), expected.scale.view(torch.uint8), rtol=0, atol=0
    )


@requires_nvfp4
@pytest.mark.parametrize("exponent", [-4, -2, 0, 2, 4])
def test_triton_matches_reference_across_magnitudes(exponent):
    """Same assertion, swept over five orders of magnitude of input scale. The e4m3 block scale
    saturates at 448 and underflows below ~2e-3, and both ends are where a rounding disagreement
    between the two implementations would hide."""
    x = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16) * (10.0**exponent)
    gs = nvfp4.global_scale_of(x)
    for axis in (-1, 0):
        got = nvfp4.quantize(x, axis, global_scale=gs)
        expected = nvfp4.quantize_reference(x, axis, global_scale=gs)
        torch.testing.assert_close(
            got.packed.view(torch.uint8), expected.packed.view(torch.uint8), rtol=0, atol=0
        )


@requires_nvfp4
def test_block_scale_rounds_up_so_nothing_clips():
    """Rounding the scale toward +inf makes `|q| <= 6` true by construction, so the downstream
    clamp is a guard and never a live path. Asserted at exactly 6.0, not merely below it — if the
    scale were rounded to nearest this would exceed 6 for some block."""
    x = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
    gs = nvfp4.global_scale_of(x)
    xb = x.reshape(1024, 512 // nvfp4.BLOCK, nvfp4.BLOCK).float()
    scale_e4m3, divisor = nvfp4._block_scale(xb.abs().amax(-1, keepdim=True), gs)
    assert (xb / (divisor * gs)).abs().max().item() <= nvfp4.FP4_MAX
    assert (scale_e4m3.float() * gs * nvfp4.FP4_MAX >= xb.abs().amax(-1, keepdim=True) - 1e-6).all()


@requires_nvfp4
def test_triton_stochastic_rounding_is_seeded_and_unbiased():
    """The kernels read the seed through a *pointer*, so `refresh_fp4_weights` can advance it in
    place without becoming a dynamo guard.

    An earlier version passed the seed tensor as a scalar argument, which handed `tl.rand` a device
    *address* as its seed. That produced perfectly plausible-looking randomness — the bias test
    below still passed — while being unseeded and unreproducible. Hence the reproducibility half.
    """
    x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    gs = nvfp4.global_scale_of(x)
    seed = torch.tensor(0, device="cuda", dtype=torch.int64)

    a = nvfp4.quantize(x, -1, global_scale=gs, stochastic=True, seed=seed)
    b = nvfp4.quantize(x, -1, global_scale=gs, stochastic=True, seed=seed)
    torch.testing.assert_close(a.packed.view(torch.uint8), b.packed.view(torch.uint8), rtol=0, atol=0)

    seed.fill_(1)
    c = nvfp4.quantize(x, -1, global_scale=gs, stochastic=True, seed=seed)
    assert not torch.equal(a.packed.view(torch.uint8), c.packed.view(torch.uint8))

    # Mutating the seed buffer in place is the mechanism `refresh_fp4_weights` uses, so the kernel
    # must see the new value without being re-traced or re-launched differently.
    seed.fill_(0)
    again = nvfp4.quantize(x, -1, global_scale=gs, stochastic=True, seed=seed)
    torch.testing.assert_close(
        a.packed.view(torch.uint8), again.packed.view(torch.uint8), rtol=0, atol=0
    )

    # And it is genuinely stochastic rounding, not noise: every code stays within one level of the
    # round-to-nearest result.
    rn = nvfp4.quantize(x, -1, global_scale=gs)
    delta = (
        nvfp4.unpack_nibbles(a.packed.view(torch.uint8)).int() & 0x07
    ) - (nvfp4.unpack_nibbles(rn.packed.view(torch.uint8)).int() & 0x07)
    assert delta.abs().max().item() <= 1
    assert delta.abs().sum().item() > 0


@requires_nvfp4
def test_triton_handles_zero_blocks():
    """Same guard as the reference, but through the kernel: a zero block must not divide by zero.
    `input_injection` is exactly zero at init, so this is a step-0 path, not a corner case."""
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    x[:, :16] = 0.0
    gs = nvfp4.global_scale_of(x)
    for axis in (-1, 0):
        got = nvfp4.quantize(x, axis, global_scale=gs)
        expected = nvfp4.quantize_reference(x, axis, global_scale=gs)
        torch.testing.assert_close(
            got.packed.view(torch.uint8), expected.packed.view(torch.uint8), rtol=0, atol=0
        )

    zeros = torch.zeros(256, 128, device="cuda", dtype=torch.bfloat16)
    gz = nvfp4.global_scale_of(zeros)
    out = nvfp4.mm(nvfp4.quantize(zeros, -1, global_scale=gz), nvfp4.quantize(zeros, -1, global_scale=gz))
    assert torch.isfinite(out).all() and out.abs().max().item() == 0.0


@requires_nvfp4
def test_ragged_shapes_are_rejected_not_silently_padded():
    """The kernels assume 128-row tiles and 64-wide blocks. `FP4Linear` checks
    `quantize_supported_shape` and falls back to `F.linear`; reaching `quantize` with a bad shape
    is a bug, so it raises rather than producing a subtly wrong layout."""
    assert nvfp4.quantize_supported_shape(4096, 1280)
    assert not nvfp4.quantize_supported_shape(100, 1280)
    assert not nvfp4.quantize_supported_shape(4096, 100)
    with pytest.raises(ValueError, match="rows % 128"):
        nvfp4.quantize(torch.randn(100, 128, device="cuda"), -1)


def test_unsupported_axis_raises():
    # axis=1 on a 2-D tensor *is* the last dim and is valid; only a genuinely out-of-range axis
    # should raise. Getting this backwards would have made the guard reject the rowblock path.
    nvfp4.quantize_reference(torch.randn(32, 32), 1)
    with pytest.raises(ValueError, match="axis 0 or -1"):
        nvfp4.quantize_reference(torch.randn(32, 32), 2)


# --- FP4Linear, the weight cache, and model integration --------------------------------------


def _fp4_linear(in_f=256, out_f=512, bias=True, **kw):
    layer = nvfp4.FP4Linear(in_f, out_f, bias=bias, **kw).cuda()
    layer.refresh_fp4_cache()
    return layer


@requires_nvfp4
def test_gradient_is_not_the_autograd_trap():
    """The headline test, and the reason `_FP4LinearFn` exists at all.

    Quantizing in an autograd-visible way does not fail loudly. The encode is a step function with
    zero derivative, so the only surviving path back to `x` runs through the differentiable `amax`
    inside the global scale — which yields a gradient that is nonzero, plausibly scaled, and
    unrelated to the true one. Training with it makes the loss fall, slowly and wrongly.

    Both arms are needed. Without the broken arm, a future refactor that reintroduced the trap
    would still pass the "close to the reference" assertion on many shapes; without the good arm,
    the test says nothing about `_FP4LinearFn`.
    """
    torch.manual_seed(0)
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    layer = _fp4_linear(256, 512, bias=False)
    grad_out = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)

    # fp32 reference gradient.
    xr = x.detach().float().requires_grad_(True)
    yr = torch.nn.functional.linear(xr, layer.weight.detach().float())
    yr.backward(grad_out.float())

    # The real path.
    xg = x.detach().clone().requires_grad_(True)
    layer(xg).backward(grad_out)

    # The trap: quantize/dequantize *inside* the graph, so autograd differentiates the quantizer.
    xb = x.detach().clone().requires_grad_(True)
    gs = nvfp4.global_scale_of(xb)  # differentiable amax — the only live path
    blocks = xb.reshape(256, 256 // nvfp4.BLOCK, nvfp4.BLOCK).float()
    scale = (blocks.abs().amax(-1, keepdim=True) / nvfp4.FP4_MAX / gs).clamp(min=nvfp4.TINY)
    q = (blocks / (scale * gs)).clamp(-nvfp4.FP4_MAX, nvfp4.FP4_MAX)
    codes = torch.bucketize(q.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda"))
    fake = torch.where(q < 0, -1.0, 1.0) * codes.float() * (scale * gs)  # step function: zero grad
    torch.nn.functional.linear(
        fake.reshape(256, 256).bfloat16(), layer.weight.to(torch.bfloat16)
    ).backward(grad_out)

    def cosine(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return (a @ b / (a.norm() * b.norm())).item()

    trap = cosine(xb.grad, xr.grad)
    real = cosine(xg.grad, xr.grad)
    assert xb.grad.abs().sum() > 0, "the trap must produce a nonzero gradient — that is what makes it a trap"
    assert trap < 0.5, f"the trap gradient should be unrelated to the truth, got cosine {trap}"
    assert real > 0.95, f"_FP4LinearFn's gradient should track the fp32 reference, got cosine {real}"


@requires_nvfp4
def test_weight_cache_is_stale_until_refreshed():
    """Catches a forgotten `refresh_fp4_weights` call in `train.py`.

    That omission is silent and expensive: the forward keeps using step-0 weights while the fp32
    masters go on training, so the loss falls (the gradients are real) and then plateaus.
    """
    layer = _fp4_linear(256, 512)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    before = layer(x).clone()

    with torch.no_grad():
        layer.weight.mul_(1.5)
    torch.testing.assert_close(layer(x), before, rtol=0, atol=0)  # stale, deliberately

    pointers = [layer._w_fwd_packed.data_ptr(), layer._w_dgrad_packed.data_ptr(), layer._w_fwd_scale.data_ptr()]
    layer.refresh_fp4_cache()
    assert not torch.equal(layer(x), before)
    # Written in place: a cache whose data pointer moves each step is what turned a compiled graph
    # into a re-record treadmill in resolve_compile_mode's CUDA-graph leak.
    assert pointers == [layer._w_fwd_packed.data_ptr(), layer._w_dgrad_packed.data_ptr(), layer._w_fwd_scale.data_ptr()]


@requires_nvfp4
def test_fp4_buffers_stay_out_of_the_state_dict():
    """The cache is derived state, not weights. Keeping it non-persistent is what lets an FP4 run
    resume from a bf16 checkpoint and vice versa, and lets `generate.py` load one unchanged."""
    fp4 = nvfp4.FP4Linear(256, 512)
    plain = torch.nn.Linear(256, 512)
    assert set(fp4.state_dict()) == set(plain.state_dict()) == {"weight", "bias"}
    fp4.load_state_dict(plain.state_dict(), strict=True)
    plain.load_state_dict(fp4.state_dict(), strict=True)


@requires_nvfp4
def test_loading_a_state_dict_invalidates_the_cache():
    """Otherwise a `load_state_dict` between refreshes leaves the cache describing the old weights
    — the stale-cache failure above, reached by a different route."""
    layer = _fp4_linear(256, 512)
    assert layer._cache_valid
    layer.load_state_dict(layer.state_dict())
    assert not layer._cache_valid


@requires_nvfp4
def test_falls_back_to_bf16_on_ragged_token_counts():
    """`M % 128 != 0` has no valid scale-tile layout. Falling back keeps the model correct; the
    one-time warning is what keeps the *measurement* interpretable, since a silent fallback trains
    at bf16 speed and quality while the config claims FP4."""
    layer = _fp4_linear(256, 512)
    x = torch.randn(100, 256, device="cuda", dtype=torch.bfloat16)  # 100 % 128 != 0
    expected = torch.nn.functional.linear(x, layer.weight.to(x.dtype), layer.bias.to(x.dtype))
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


@requires_nvfp4
def test_grad_gemms_off_keeps_the_backward_in_bf16():
    """`fp4_grad_gemms: false` is the control arm that separates "FP4 forward is fine, the gradient
    path is what hurts" from "FP4 hurts"."""
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    grads = {}
    for name, recipe in [("fp4", nvfp4.FP4Recipe(grad_gemms=True)), ("bf16", nvfp4.FP4Recipe(grad_gemms=False))]:
        torch.manual_seed(1)
        layer = _fp4_linear(256, 512, recipe=recipe)
        xg = x.clone().requires_grad_(True)
        layer(xg).backward(grad_out)
        grads[name] = xg.grad.clone()
    assert not torch.equal(grads["fp4"], grads["bf16"])
    cos = (grads["fp4"].float().flatten() @ grads["bf16"].float().flatten()) / (
        grads["fp4"].float().norm() * grads["bf16"].float().norm()
    )
    assert cos.item() > 0.9


# --- model + config integration ---------------------------------------------------------------


def _model_cfg(**kw):
    from radiance.config import ModelConfig

    base = dict(d_model=256, head_dim=64, n_layers=4, ffn_mult=2.0, ffn_depth=1, dropout=0.0, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def test_fp4_off_leaves_the_model_structurally_untouched():
    """The FP4 analogue of `test_inert_defaults`' job. FP4 cannot be *numerically* inert, so the
    contract is structural instead: with the flag off, no `FP4Linear` exists anywhere."""
    from radiance.model import DenseTransformer

    model = DenseTransformer(_model_cfg(), vocab_size=128)
    assert not any(isinstance(m, nvfp4.FP4Linear) for m in model.modules())
    assert model.fp4_cache_bytes() == 0


@requires_nvfp4
def test_conversion_skips_block_zero_the_last_block_and_the_gates():
    """The conversion policy, pinned. `blocks[1:]` is weight-shared so `fp4_keep_bf16_blocks`
    counts *structural* blocks; `out_gate` stays bf16 because its zero-init makes
    `2 * sigmoid(0) == 1.0` exactly and 4-bit noise on an exact identity is not a trade."""
    from radiance.model import DenseTransformer

    model = DenseTransformer(
        _model_cfg(fp4_linear=True, d_model=1280, head_dim=64,
                   fp4_keep_bf16_first=True, fp4_keep_bf16_blocks=1, fp4_lm_head=False),
        vocab_size=256,
    )
    fp4_names = [n for n, m in model.named_modules() if isinstance(m, nvfp4.FP4Linear)]
    blocks = sorted({n.split(".")[1] for n in fp4_names if n.startswith("blocks.")})
    assert blocks == ["1", "2"], f"expected blocks 1-2 quantized (0 first, 3 last kept), got {blocks}"
    assert not any("out_gate" in n or "router" in n for n in fp4_names)
    assert not any(n == "lm_head" for n in fp4_names)
    assert model.fp4_cache_bytes() > 0

    # The shipped defaults are maximum coverage: every block, plus lm_head.
    wide = DenseTransformer(_model_cfg(fp4_linear=True, d_model=1280, head_dim=64), vocab_size=256)
    wide_names = [n for n, m in wide.named_modules() if isinstance(m, nvfp4.FP4Linear)]
    assert sorted({n.split(".")[1] for n in wide_names if n.startswith("blocks.")}) == ["0", "1", "2", "3"]
    assert isinstance(wide.lm_head, nvfp4.FP4Linear)
    # Still never the gates or routers, at any coverage setting.
    assert not any("out_gate" in n or "router" in n for n in wide_names)


@requires_nvfp4
def test_fp4_lm_head_can_be_turned_off():
    from radiance.model import DenseTransformer

    off = DenseTransformer(_model_cfg(fp4_linear=True, d_model=1280, head_dim=64, fp4_lm_head=False), vocab_size=256)
    on = DenseTransformer(_model_cfg(fp4_linear=True, d_model=1280, head_dim=64, fp4_lm_head=True), vocab_size=256)
    assert not isinstance(off.lm_head, nvfp4.FP4Linear)
    assert isinstance(on.lm_head, nvfp4.FP4Linear)


@requires_nvfp4
def test_ragged_widths_raise_with_the_offending_field_named():
    from radiance.model import DenseTransformer

    with pytest.raises(ValueError, match="multiple of 128"):
        DenseTransformer(_model_cfg(fp4_linear=True, d_model=320, head_dim=64), vocab_size=256)


@requires_nvfp4
def test_fp4_with_moe_raises():
    """Refusing beats a misleading measurement: with MoE on the experts *are* the FFN, so leaving
    them bf16 would mean FP4 never touched the expensive part."""
    from radiance.model import DenseTransformer

    with pytest.raises(ValueError, match="use_moe"):
        DenseTransformer(
            _model_cfg(fp4_linear=True, d_model=1280, head_dim=64, use_moe=True, n_experts=4),
            vocab_size=256,
        )


def test_nvfp4_dtype_sugar_and_its_guards():
    from radiance.config import Config, ModelConfig, TrainConfig, _apply_dtype_sugar, resolve_dtype

    assert resolve_dtype("nvfp4") is torch.bfloat16  # an alias: autocast is still bf16
    cfg = _apply_dtype_sugar(Config(model=ModelConfig(), train=TrainConfig(dtype="nvfp4")))
    assert cfg.model.fp4_linear

    with pytest.raises(ValueError, match="fp16"):
        _apply_dtype_sugar(Config(model=ModelConfig(fp4_linear=True), train=TrainConfig(dtype="fp16")))


def test_contradictory_config_raises(tmp_path):
    """`train.dtype: nvfp4` plus an explicit `model.fp4_linear: false` should not resolve silently
    in either direction."""
    import yaml

    from radiance.config import load_config

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"train": {"dtype": "nvfp4"}, "model": {"fp4_linear": False}}))
    with pytest.raises(ValueError, match="contradict"):
        load_config(str(path))


@requires_nvfp4
def test_refresh_fp4_weights_advances_the_stochastic_rounding_seed():
    """The seed must move per optimizer step, or the same rounding pattern repeats and reintroduces
    exactly the bias stochastic rounding exists to remove."""
    from radiance.model import DenseTransformer

    model = DenseTransformer(_model_cfg(fp4_linear=True, d_model=1280, head_dim=64), vocab_size=256).cuda()
    seeds = []
    for _ in range(3):
        nvfp4.refresh_fp4_weights(model)
        seeds.append(next(m._seed.item() for m in model.modules() if isinstance(m, nvfp4.FP4Linear)))
    assert seeds == [1, 2, 3]


@requires_nvfp4
def test_saving_activations_packed_is_bit_identical():
    """`fp4_save_activations` stores the quantized activation for backward (1.125 bytes/element)
    instead of the bf16 one (2), and it is the only `fp4_*` knob that defaults on.

    That is justified by this test rather than by convention: the backward already computed exactly
    `quantize(x, 0, global_scale=gx, hadamard=rot)` from this same `x`, so moving it into the
    forward changes *when*, not *what*. `rtol=0, atol=0` — anything less would let a real numerical
    change hide behind "close enough", and a memory optimisation has no business changing results.
    """
    torch.manual_seed(0)
    x = torch.randn(512, 1280, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16)

    results = {}
    for save in (False, True):
        torch.manual_seed(1)
        layer = nvfp4.FP4Linear(
            1280, 2560,
            recipe=nvfp4.FP4Recipe(save_activations=save, stochastic_rounding=False),
        ).cuda()
        layer.refresh_fp4_cache()
        xg = x.clone().requires_grad_(True)
        layer(xg).backward(grad_out)
        results[save] = (xg.grad.clone(), layer.weight.grad.clone())

    torch.testing.assert_close(results[True][0], results[False][0], rtol=0, atol=0)
    torch.testing.assert_close(results[True][1], results[False][1], rtol=0, atol=0)


@requires_nvfp4
def test_saving_activations_packed_uses_less_memory():
    """The other half of the contract: it has to actually save memory, or it is complexity for
    nothing. Measured on the saved tensors themselves rather than on peak, so the assertion is not
    at the mercy of allocator behaviour."""
    sizes = {}
    for save in (False, True):
        torch.manual_seed(1)
        layer = nvfp4.FP4Linear(
            1280, 2560, recipe=nvfp4.FP4Recipe(save_activations=save, stochastic_rounding=False)
        ).cuda()
        layer.refresh_fp4_cache()
        x = torch.randn(4096, 1280, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        # forward() reshapes its output, so grad_fn is the view; the FP4 node is one hop back.
        node = out.grad_fn
        while not hasattr(node, "saved_tensors") or not node.saved_tensors:
            node = node.next_functions[0][0]
        # Total, not max: the fp32 weight (13.1 MB here) is larger than the activation and is saved
        # either way, so a per-tensor maximum would compare the weight against itself.
        sizes[save] = sum(t.numel() * t.element_size() for t in node.saved_tensors)

    bf16_activation_bytes = 4096 * 1280 * 2
    saved_bytes = sizes[False] - sizes[True]
    assert saved_bytes > 0.5 * bf16_activation_bytes, (
        f"packed saving should return most of the activation's {bf16_activation_bytes / 1e6:.1f} MB, "
        f"got {saved_bytes / 1e6:.1f} MB (totals {sizes})"
    )


@requires_nvfp4
def test_grad_gemms_off_keeps_the_bf16_activation():
    """With the backward in bf16, wgrad is `grad_y.t() @ x` and needs the real activation — the
    packed form cannot substitute. The skip must be automatic rather than the caller's job."""
    torch.manual_seed(1)
    layer = nvfp4.FP4Linear(
        1280, 2560, recipe=nvfp4.FP4Recipe(grad_gemms=False, save_activations=True)
    ).cuda()
    layer.refresh_fp4_cache()
    x = torch.randn(512, 1280, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    out.backward(torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16))
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@requires_nvfp4
def test_lm_head_conversion_is_shape_guarded():
    """`fp4_lm_head` defaults on, but the head's out_features is the *padded vocab* — which
    `vocab_pad_multiple: 128` makes a multiple of 128 and `vocab_pad_multiple: 1` does not.

    Regression test: converting it unguarded allocates a zero-length scale buffer and fails at the
    first forward instead of quietly staying bf16. Caught by `tests/test_compile.py` the moment the
    coverage defaults were flipped, on a model with a 64-token test vocab.
    """
    from radiance.model import DenseTransformer

    ragged = DenseTransformer(_model_cfg(fp4_linear=True, d_model=1280, head_dim=64), vocab_size=64)
    assert not isinstance(ragged.lm_head, nvfp4.FP4Linear)
    ids = torch.randint(0, 64, (2, 64), device="cuda")
    ragged = ragged.cuda()
    nvfp4.refresh_fp4_weights(ragged)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.isfinite(ragged(ids).logits).all()

    aligned = DenseTransformer(_model_cfg(fp4_linear=True, d_model=1280, head_dim=64), vocab_size=50304)
    assert isinstance(aligned.lm_head, nvfp4.FP4Linear)
