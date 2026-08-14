# `nvfp4.py` — NVFP4 4-bit GEMMs

Behind `cfg.model.fp4_linear` (opt-in). NVFP4 is Blackwell's 4-bit float: `e2m1` elements packed two per byte, a
per-16-element `e4m3` block scale, and one per-tensor fp32 global scale. `FP4Linear` runs all three of a linear's
GEMMs (forward, dgrad, wgrad) on the FP4 tensor cores while master weights, gradients and optimizer state stay fp32
and autocast stays bf16 — **only the GEMM operands are 4-bit.** `train.dtype: nvfp4` is sugar that sets `fp4_linear`
(`config._apply_dtype_sugar`).

The module is separate from `model.py` for the reason `optim.py` was split out: ~600 lines of Triton with nothing to
do with architecture. See `configs/fineweb_500m_nvfp4.yaml`, and [results.md](results.md) for the throughput A/B.

**NVFP4 is the only 4-bit format available on this card.** `_ScalingType.BlockWise1x32` (MXFP4) raises "MXFP4 scaling
only supported in CUDA for B200/B300" on sm_120; `BlockWise1x16` (NVFP4) works. Both scale tensors must additionally
be in `SWIZZLE_32_4_4` layout — `NO_SWIZZLE` raises.

## The GEMM is 5.2x bf16 and the quantization is the bottleneck

`_scaled_mm_v2` at `4096x1280x3840` measures 1321 TFLOPS against bf16's 226. But a linear needs *four*
activation-side quantizations per micro-batch, and a naive implementation measured **0.37x — three times slower than
bf16**. Two things fixed that, and both are load-bearing rather than optimisations:

- **The weight cache.** `refresh_fp4_weights(raw_model)` re-quantizes every `FP4Linear`'s weight into both
  orientations once per optimizer step, called from `train.py` right after `optimizer.step()` next to
  `update_expert_bias()` and for the same reason — it writes buffers with no gradient, so keeping it outside the
  compiled region avoids a graph break, and a validity check inside `forward` would be a Python branch dynamo guards
  on and recompiles for every step. Once per *optimizer* step is also the correct semantics, since every micro-batch
  of an accumulated step must differentiate against the same weights. **Omitting the call fails silently**: the
  forward keeps using step-0 weights while the fp32 masters train on, so the loss still falls and then plateaus — the
  same family as `loop_bptt_window` freezing `blocks[0]`.
- **`torch.library.custom_op`, not `@torch._dynamo.disable`.** Inductor refuses to lower `_scaled_mm_v2` with
  swizzled scales ("does not yet support non-trivial swizzles"), so the GEMM has to be opaque either way — but a
  custom op stays *in* the dynamo graph as an extern call and inductor still fuses the elementwise work around it,
  where a graph break does not. Measured 0.392 ms vs 0.493 ms on an 8-linear stack.

## The operand convention keeps this to two kernels

`_scaled_mm_v2` takes `contraction_dim` explicitly, so every operand of every GEMM is stored the same way: rows = its
output index, columns = the contraction index, scale blocks along columns.

| GEMM | operand A | operand B |
|---|---|---|
| fwd `Y = X Wᵀ` | `X` rows `M`, contract `K` — **rowblock** | `W` rows `N`, contract `K` — **rowblock** |
| dgrad `dX = dY W` | `dY` rows `M`, contract `N` — **rowblock** | `W` rows `K`, contract `N` — **colblock** |
| wgrad `dW = dYᵀ X` | `dY` rows `N`, contract `M` — **colblock** | `X` rows `K`, contract `M` — **colblock** |

So `W` is quantized twice (both cached) and `X`/`dY` twice each per micro-batch. The colblock kernel never
materialises a transpose — it reads the natural `(M, K)` layout with coalesced loads and writes packed bytes at
`out[k, m//2]`, since a separate `.t().contiguous()` would cost several times the traffic of the 4-bit write it
feeds.

**The Triton kernels emit the swizzled scale layout directly**, using `blocked_offset`'s closed form, rather than
running a separate `to_blocked` pass — which would add four kernel launches per linear per micro-batch. Tile widths
were picked by sweep, not symmetry: rowblock `BK=64` (its four scale columns are exactly one `SWIZZLE_32_4_4` column
tile), colblock `BK=32` (1.6x faster than 64 — 841 GB/s vs 523 — because the narrower tile keeps the in-register
transpose in fewer registers), both at 8 warps. Both forms of a 4096x1280 input cost **0.028 ms** against a
two-kernel bandwidth floor of ~0.020.

## Bit-exactness against the pure-torch reference

Getting there flushed out four separate 1-ULP disagreements, every one invisible in the dequantized values. This is
why `tests/test_nvfp4.py` compares integer nibble codes at `rtol=0, atol=0` rather than asserting closeness — an "is
it close enough" assertion passes all four:

- `tl.math.div_rn(amax, 6.0)` **promotes the Python literal to fp64**, so Triton rounded once where torch rounds
  twice; 32% of elements differed by 1 ULP. Fixed by `scale_denominator`, which precomputes `6 * global_scale` on the
  host so both sides do one fp32 division.
- Triton's default fp32 `/` is not correctly rounded, so `q` differed at rounding ties. `div_rn`.
- Triton's fp32 -> e4m3 cast breaks ties **toward zero**; torch's breaks them **to even**. Fixed by rounding the
  block scale toward +inf in both — which also makes `|q| <= 6` exact, so the downstream clamp is a guard rather than
  a live path. (Rounding to nearest and matching the tie behaviour was tried first and is *worse*: ceil is robust to
  a 1-ULP disagreement, nearest is not.)
- `tl.dot` defaults to tf32, which the Hadamard rotation cannot tolerate — `input_precision="ieee"`.

Exact ties are common rather than rare here, which is why these mattered: bf16 inputs have 8 mantissa bits and the
divisor is derived from one of them.

## Quantization sits inside `_FP4LinearFn`, where autograd cannot see it

That is mandatory rather than stylistic. Quantizing in an autograd-visible way produces a gradient that is nonzero,
plausibly scaled, and wrong: the encode is a step function with zero derivative, so the only surviving path back to
`x` is the `amax` inside the global scale. Nothing raises and the loss still falls.
`test_gradient_is_not_the_autograd_trap` builds that broken version deliberately as a negative control — the trap's
gradient measures cosine < 0.5 against an fp32 reference where the real path measures > 0.95. Being outside autograd
also means straight-through needs no `x - x.detach()` trick (contrast `_counterfactual_probe_signal`): the gradient
computed against the quantized operands *is* the gradient returned for the full-precision input, which is exactly the
identity Jacobian STE asks for.

## The recipe

Follows NVIDIA's "Pretraining LLMs with NVFP4": **stochastic rounding on gradient tensors only** (round-to-nearest's
bias accumulates coherently over a run; weights and activations have no accumulation to protect, so SR there would
add variance for nothing), and a **random Hadamard rotation on the wgrad operands** to spread outliers before the
per-16 amax picks a scale. The rotation is free because **the rotation group and the scale block are the same 16
elements** — it happens in-register on exactly the tile the reduction is about to consume. Both wgrad operands must
get the *same* rotation, since correctness rests on `(RᵀdY)ᵀ(RᵀX) = dYᵀ(R Rᵀ)X = dYᵀX`; two different sign vectors
give a wrong-but-finite gradient with no error, so `test_wgrad_hadamard_is_exact_and_mismatched_rotations_are_not`
asserts the mismatched case diverges as well as the matched case agreeing — without that half it would pass against
`R = I`.

## Integration

**`FP4Linear` subclasses `nn.Linear`**, which keeps the blast radius small: every mechanism here that classifies
parameters does so by module type or parameter name, and all keep working unchanged — `_init_weights`'
`isinstance(module, nn.Linear)`, `_scale_residual_init`'s `endswith("down_proj.weight")`, `_init_inert_gates`, and
`optim.build_muon_param_groups`. The cache buffers are `persistent=False`, so **`state_dict` is byte-identical to a
bf16 model's** — an FP4 run resumes from a bf16 checkpoint and vice versa, and `generate.py` needs no change.

**Which linears convert: exactly the tensors Muon owns.** Deliberate reuse rather than coincidence —
`optim._MUON_EXCLUDED_SUBSTRINGS` already encodes this repo's judgement about which tensors are hidden linear maps
and which are small tensors whose exact scale is load-bearing. So `qkv_proj`/`out_proj`/the FFN
projections/`MTPHead.proj` convert, while the routers, `out_gate` and the tied embedding stay bf16. `out_gate` is the
sharpest case: it is zero-initialised precisely so `2 * sigmoid(0) == 1.0` exactly, and 4-bit noise on an exact
identity is not a trade.

Three knobs narrow that coverage and **all three default to maximum coverage rather than to the cautious setting**:
`fp4_keep_bf16_first` (default `false`) would keep `blocks[0]` bf16, `fp4_keep_bf16_blocks` (default `0`) keeps the
last k **structural** blocks — `blocks[1:]` is weight-shared, so a bf16 block is bf16 on every loop iteration and
there is deliberately no per-iteration variant of the knob — and `fp4_lm_head` (default `true`) quantizes the head.
That is a throughput decision made on throughput evidence alone (~4% of the step), and it departs from two settings
that exist for real reasons: NVIDIA's recipe keeps the final blocks in high precision, and `lm_head` is weight-tied
to `token_emb`, the tensor `embed_lr`'s decomposition attributes ~93% of the largest quality win in this repo to.
**The quality side is unmeasured**, so if an FP4 run regresses on `val/loss`, turn coverage back in this order:
`fp4_lm_head: false`, then `fp4_keep_bf16_blocks: 1`, then `fp4_keep_bf16_first: true`.

## Guards and fallbacks

**Two things raise at `DenseTransformer.__init__`.** Every quantized width must be a multiple of **128** (`d_model`,
the qkv width, `ffn_dim`) — each serves as a *row* dim in some GEMM and the swizzle pads rows to 128. And
`fp4_linear` + `use_moe` is refused: `BatchedExperts` applies its weights with `baddbmm` over an `(n_experts, in,
out)` stack and `_scaled_mm_v2` is 2-D only, so a per-expert loop would reintroduce exactly the launch overhead the
batched dispatch exists to remove (24.0 -> 3.16 ms at 32 experts). With MoE on the experts *are* the FFN, so
quantizing only attention would produce a number that looks like a failure of FP4 when FP4 never touched the
expensive part — refusing beats a misleading measurement.

At runtime `FP4Linear` falls back to plain `F.linear` when the token count is not a multiple of 128, off CUDA, or on
a non-Blackwell card, with a **one-time warning**. The warning is the point: a silent fallback trains correctly at
bf16 speed and quality while the config claims FP4, which makes the resulting measurement uninterpretable. Note
`act_capacity_ratio < 1.0` produces a gathered capacity that frequently is not a multiple of 128.

**Compiled FP4 does not match eager FP4 to bf16 noise, and that is expected.** Quantization is a step function, so a
sub-ULP input difference — which inductor legitimately produces by reassociating an elementwise chain — can push a
value across a rounding boundary and flip its 4-bit code by a whole level. Stacked over several FP4 layers this
reaches ~0.75 on logits whose bf16-only compile difference is 0.04. The meaningful invariant, and what
`tests/test_compile.py` asserts, is that the compile difference stays well below the **quantization error itself**
(0.75 against 3.71); the exact checks live on the quantizer (bit-identical under compile) and on a single layer given
a fixed input (bf16 noise), which is where an inductor bug would show up cleanly. Do not "fix" the end-to-end test by
tightening its tolerance until it fails.

`fp4_cache_bytes()` reports the caches' footprint (~1.125 bytes per covered parameter — 0.5 packed plus 0.0625 of
scales, in each of two orientations). It is *not* per-token, so it deliberately does not live in
`activation_bytes_per_token` — and it is deliberately *not* added to `estimate_batch_size`'s
`not_yet_allocated_bytes` either, unlike the gradient and optimizer buffers next to it there. Those three really are
unallocated when the estimate runs; these buffers are allocated eagerly in `FP4Linear.__init__` and have already
moved to the device with the model, so `mem_get_info` has counted them and adding them again would subtract the same
~0.48 GB twice. The function exists for sizing a run by hand.

`torch._C._ScalingType`, `_SwizzleType` and `aten::_scaled_mm_v2` are **private, undocumented, torch-2.13-era
surface**. `pyproject.toml`'s `torch>=2.5` floor is deliberately unchanged — bf16 runs must keep working on older
torch — so `nvfp4_supported()` probes capability at runtime and `require_nvfp4` raises only when a config actually
asks for FP4. Re-verify after any torch upgrade, exactly as `_diff_flex_attention` already asks for.
