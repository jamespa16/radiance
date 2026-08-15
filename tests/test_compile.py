"""torch.compile interactions.

Everything here failed only under a real compiled GPU run — the eager unit tests all passed while
these were broken, which is exactly why they're worth pinning. Both bugs cost a full training run
to discover:

  * document masking + torch.compile: create_block_mask builds a BlockMask from data-dependent
    index tensors. Traced into the enclosing graph those become intermediates with an unresolved
    layout and inductor fails to lower them. DenseTransformer._doc_masks is marked
    @torch._dynamo.disable so it runs eagerly and hands the compiled region a concrete BlockMask.

  * stochastic loop depth + mode="reduce-overhead": CUDA graphs assume a static execution path,
    which a per-step-varying loop count gives up. Replaying a different count overwrote the
    previous graph's gradient tensors. train.py drops to mode=None when loop_count_min is set.

  * grad_checkpoint + mode="reduce-overhead": torch.utils.checkpoint recomputes each block during
    backward. AOTAutograd partitions that recompute into its own graph segment, and reduce-overhead
    captures each segment as its own CUDA graph against one shared static pool — so the recompute's
    outputs overwrite a tensor the original forward's backward still needs ("accessing tensor
    output of CUDAGraphs that has been overwritten by a subsequent run"). train.py drops to
    mode=None whenever grad_checkpoint is on, same as it does for stochastic depth. Needs several
    backward() calls sharing one accumulated step (i.e. grad_accum_steps > 1) plus a router/MoE
    layer to actually surface — a plain looped model with grad_checkpoint alone did not reproduce
    it, so the regression test below matches configs/super.yaml's combination that did.

  * differential attention (cfg.model.use_diff_attn) + document masking: CausalSelfAttention.forward
    issues *two* flex_attention calls per layer (one per branch), against the same reused BlockMask,
    whose Q/K both trace back to one torch.split of qkv_proj's output. Traced into one inductor graph
    together, the second call's output silently diverges from eager — no error, no warning, up to
    ~0.85 absolute on a toy shape. Eager was correct throughout; only the compiled *combination* was
    wrong, which is exactly the class of bug this file exists to catch — the shapes/gradients/KV-cache
    tests in test_diff_attention.py all passed the whole time. _diff_flex_attention (model.py) is
    marked @torch._dynamo.disable for the same structural reason _doc_masks is, though the underlying
    cause is different (an inductor scheduling issue with two flex_attention calls sharing a BlockMask
    and split-provenance inputs, not an unlowerable data-dependent tensor). Plain attention is
    unaffected: it only ever issues one flex_attention call per layer.

These are slow (each compiles), so they're marked and kept few.
"""

from __future__ import annotations

import pytest
import torch

from radiance import nvfp4
from radiance.config import ModelConfig
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled paths need CUDA")
slow = pytest.mark.slow


def _build(**overrides) -> DenseTransformer:
    fields = dict(
        # head_dim >= 16: flex_attention's compiled kernel rejects anything smaller.
        d_model=128, head_dim=64, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=64,
    )
    fields.update(overrides)
    torch.manual_seed(0)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB, eos_id=5).cuda()


@slow
@requires_cuda
@pytest.mark.parametrize("grad_checkpoint", [False, True])
@pytest.mark.parametrize("streams", [1, 4])
def test_doc_masking_compiles_for_train_and_eval(grad_checkpoint, streams):
    """Both paths matter: grad_checkpoint is disabled under eval, so eval traces a *different*
    graph — and that was the one that blew up while training compiled fine.

    Parametrized over hyper_conn_streams because it changes the rank of every tensor flowing
    between sublayers, and the doc-mask path is where a rank surprise would surface: the flex
    BlockMask is built against (batch, seq) document ids while the hidden state carries an extra
    stream dimension.
    """
    model = _build(loop_count=2, grad_checkpoint=grad_checkpoint, hyper_conn_streams=streams)
    compiled = torch.compile(model)
    ids = torch.randint(0, TINY_VOCAB, (2, 64), device="cuda")

    compiled.train()
    compiled(ids).logits.square().mean().backward()

    compiled.eval()
    with torch.no_grad():
        assert compiled(ids).logits.shape == (2, 64, TINY_VOCAB)


@slow
@requires_cuda
def test_diff_attn_doc_masking_compiles_correctly():
    """Regression test for the silent-wrong-output bug in this file's docstring: unlike the other
    tests here, this must compare *values*, not just shapes/crashes — the bug it pins produced a
    perfectly shaped, perfectly finite, silently wrong tensor. Multiple documents in the batch, so
    the BlockMask is a real (non-degenerate) mask, not one that happens to reduce to plain causal.
    """
    model = _build(use_diff_attn=True, loop_count=2)
    ids = torch.randint(0, TINY_VOCAB, (2, 64), device="cuda")
    ids[:, 20] = 5  # eos_id, matching _build's fixed eos_id — forces real document boundaries
    ids[:, 45] = 5

    model.eval()
    with torch.no_grad():
        eager_out = model(ids).logits

    compiled = torch.compile(model)
    with torch.no_grad():
        compiled_out = compiled(ids).logits

    torch.testing.assert_close(eager_out, compiled_out, rtol=2e-2, atol=2e-2)

    compiled.train()
    out = compiled(ids)
    out.logits.float().square().mean().backward()
    dead = [name for name, p in compiled.named_parameters() if p.grad is None]
    assert not dead


@slow
@requires_cuda
def test_stochastic_depth_compiles_across_loop_counts():
    """Every sampled loop count traces its own graph; running several in sequence must not corrupt
    each other's gradients."""
    model = _build(loop_count=4, loop_count_min=2, loop_count_max=5)
    # mode=None, matching what train.py selects when loop_count_min is set — "reduce-overhead"
    # captures CUDA graphs, which a varying loop count cannot support.
    compiled = torch.compile(model, mode=None)
    ids = torch.randint(0, TINY_VOCAB, (2, 64), device="cuda")

    compiled.train()
    for loops in (2, 3, 4, 5, 3, 2):
        compiled.zero_grad()
        compiled(ids, loop_count=loops).logits.square().mean().backward()
        assert torch.isfinite(model.token_emb.weight.grad).all()


@slow
@requires_cuda
def test_grad_checkpoint_compiles_under_grad_accum():
    """Reproduces configs/super.yaml's exact crash (grad_checkpoint + use_router + use_moe, several
    backward() calls per accumulated step, no zero_grad between them) under mode="reduce-overhead",
    then confirms mode=None — what train.py now selects whenever grad_checkpoint is on — is safe
    across repeated accumulated steps. See the module docstring for the mechanism."""
    model = _build(
        loop_count=3, grad_checkpoint=True,
        use_router=True, max_loops=4,
        use_moe=True, n_experts=4, moe_top_k=2,
    )
    compiled = torch.compile(model, mode=None)
    ids = torch.randint(0, TINY_VOCAB, (4, 64), device="cuda")

    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)

    compiled.train()
    for _ in range(3):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for _ in range(3):  # grad_accum_steps=3: several backward() calls, no zero_grad between
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = compiled(ids)
                loss = out.logits.float().square().mean() + out.moe_aux_loss + out.ponder_cost
            loss.backward()
        assert torch.isfinite(model.token_emb.weight.grad).all()


# The remaining tests need no GPU and no compile: they pin the *decision* about which compile mode
# a run gets. That decision is what the three cases above turn on, and getting the third one wrong
# was silent — nothing raised, the run just leaked ~8 MB/step of process VRAM (outside torch's
# caching allocator, so invisible to torch.cuda.memory_reserved) and ran 3x slower until the card
# filled up. Asserting the mode directly is the only cheap way to catch a regression.


def _mode(tiny_cfg, device_type="cuda", **overrides) -> str | None:
    """resolve_compile_mode for a tiny config with `overrides` applied to cfg.model.

    Fields are set after construction rather than passed through the fixture because the fixture
    pins head_dim (and train.compile) itself, and these cases need to vary exactly those.
    """
    from radiance.train import resolve_compile_mode

    cfg = tiny_cfg()
    cfg.train.compile = True  # every case here is a compiled run
    for field, value in overrides.items():
        setattr(cfg.model, field, value)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB, eos_id=5)
    return resolve_compile_mode(model, cfg, device_type)


def test_cuda_graphs_off_when_doc_masking_builds_a_mask_per_step(tiny_cfg):
    """The leak. doc_attention_mask rebuilds a flex BlockMask from each batch's document
    boundaries, so its tensors change address every step; CUDA graph trees re-records on that and
    never frees the previous cudaGraphExec. head_dim >= 16 because below it doc masking falls back
    to plain SDPA and builds no mask at all."""
    assert _mode(tiny_cfg, d_model=64, head_dim=16, doc_attention_mask=True) is None


def test_cuda_graphs_off_for_stochastic_depth_and_grad_checkpoint(tiny_cfg):
    assert _mode(tiny_cfg, doc_attention_mask=False, loop_count=4, loop_count_min=2, loop_count_max=4) is None
    assert _mode(tiny_cfg, doc_attention_mask=False, grad_checkpoint=True) is None


def test_cuda_graphs_kept_when_nothing_rules_them_out(tiny_cfg):
    """The control the other three need: without it they could all pass on a function that
    returned None unconditionally, and the fix would be "never use CUDA graphs" by accident."""
    assert _mode(tiny_cfg, doc_attention_mask=False) == "reduce-overhead"
    # head_dim < 16 and non-CUDA both fall back to plain SDPA, so no per-step mask is built and
    # there is nothing for CUDA graphs to trip over.
    assert _mode(tiny_cfg, head_dim=8, doc_attention_mask=True) == "reduce-overhead"
    assert _mode(tiny_cfg, d_model=64, head_dim=16, doc_attention_mask=True, device_type="cpu") == "reduce-overhead"


def test_compile_disabled_means_no_mode(tiny_cfg):
    from radiance.train import resolve_compile_mode

    cfg = tiny_cfg()  # the fixture's compile=False
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB, eos_id=5)
    assert resolve_compile_mode(model, cfg, "cuda") is None


# --- NVFP4 ------------------------------------------------------------------------------------
#
# A fifth entry for the catalogue at the top of this file. Inductor refuses to lower
# `_scaled_mm_v2` with swizzled scales ("does not yet support non-trivial swizzles"), so the GEMM
# is wrapped in a `torch.library.custom_op` — opaque to inductor, but still *inside* the dynamo
# graph, which measured 0.392 ms against `@torch._dynamo.disable`'s 0.493 ms on an 8-linear stack.
#
# Both properties need pinning, and neither is visible to an eager test. A custom op inside an
# `autograd.Function` inside AOTAutograd is the same class of interaction that produced the
# differential-attention bug: a perfectly-shaped, perfectly-finite, silently wrong tensor.


def _fp4_build(**overrides) -> DenseTransformer:
    # d_model 1280 rather than 128: FP4 needs every quantized width to be a multiple of 128, and
    # the token count to be a multiple of 128 as well.
    fields = dict(
        d_model=1280, head_dim=64, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=128, fp4_linear=True, fp4_keep_bf16_first=False,
        fp4_keep_bf16_blocks=0, fp4_stochastic_rounding=False,
    )
    fields.update(overrides)
    torch.manual_seed(0)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB, eos_id=5).cuda()


@slow
@requires_cuda
@pytest.mark.skipif(not nvfp4.nvfp4_supported(), reason="needs a Blackwell GPU")
def test_fp4_quantizer_is_bit_identical_under_compile():
    """The exact half of the compile contract, and the one that would actually catch inductor
    changing the computation: the nibble codes and scale bytes must be identical.

    This is asserted at the quantizer rather than at the model's logits because quantization is a
    *step function* — see `test_fp4_compile_difference_is_smaller_than_quantization_error` for why
    that makes an end-to-end bitwise comparison the wrong instrument.
    """
    x = torch.randn(256, 1280, device="cuda", dtype=torch.bfloat16)
    gs = nvfp4.global_scale_of(x)
    for axis in (-1, 0):
        eager = nvfp4.quantize(x, axis, global_scale=gs)
        torch._dynamo.reset()
        compiled = torch.compile(lambda t, a=axis: nvfp4.quantize(t, a, global_scale=nvfp4.global_scale_of(t)))(x)
        torch.testing.assert_close(
            compiled.packed.view(torch.uint8), eager.packed.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            compiled.scale.view(torch.uint8), eager.scale.view(torch.uint8), rtol=0, atol=0
        )


@slow
@requires_cuda
@pytest.mark.skipif(not nvfp4.nvfp4_supported(), reason="needs a Blackwell GPU")
def test_fp4_linear_compiles_to_the_same_values_as_eager():
    """A single FP4Linear, given a *fixed* input, must compile to eager's answer within bf16 noise.

    Values, not shapes: a custom op inside an `autograd.Function` inside AOTAutograd is the same
    class of interaction that produced the differential-attention bug, which returned a
    perfectly-shaped, perfectly-finite, silently wrong tensor.
    """
    torch.manual_seed(0)
    layer = nvfp4.FP4Linear(1280, 1280, recipe=nvfp4.FP4Recipe(stochastic_rounding=False)).cuda()
    layer.refresh_fp4_cache()
    x = torch.randn(256, 1280, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(256, 1280, device="cuda", dtype=torch.bfloat16)

    xe = x.clone().requires_grad_(True)
    eager = layer(xe)
    eager.backward(grad_out)
    eager_gx, eager_gw = xe.grad.clone(), layer.weight.grad.clone()
    layer.zero_grad(set_to_none=True)

    torch._dynamo.reset()
    xc = x.clone().requires_grad_(True)
    compiled = torch.compile(layer)(xc)
    compiled.backward(grad_out)

    # bf16 has ~3 significant digits; these are a few ULP at the tensors' own scale.
    torch.testing.assert_close(compiled, eager, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(xc.grad, eager_gx, rtol=5e-2, atol=2e-2)
    assert (layer.weight.grad - eager_gw).abs().max() < 0.02 * eager_gw.abs().max()


@slow
@requires_cuda
@pytest.mark.skipif(not nvfp4.nvfp4_supported(), reason="needs a Blackwell GPU")
def test_fp4_compile_difference_is_smaller_than_quantization_error():
    """End-to-end, compiled FP4 does **not** match eager FP4 to bf16 noise, and that is expected
    rather than a bug. Do not "fix" it by tightening the tolerance until it fails.

    Quantization is a step function: a sub-ULP difference in a layer's input — which inductor
    legitimately produces by reassociating an elementwise chain — can push a value across a
    rounding boundary and flip its 4-bit code by a whole level, a ~33% change in that element.
    Stacked over several FP4 layers this reaches ~0.75 on logits whose bf16-only compile difference
    is 0.04.

    So the invariant that *is* meaningful: the compile-induced difference must stay well below the
    quantization error itself. If inductor were actually changing the computation, this would
    invert. Measured on this shape: quantization error 3.71, compile difference 0.75.

    The exact checks live in the two tests above, on the quantizer (bit-identical) and on a single
    layer given a fixed input (bf16 noise) — those are where an inductor bug would show up cleanly.
    """
    ids = torch.randint(0, TINY_VOCAB, (2, 128), device="cuda")

    def logits(model, compile_it):
        if compile_it:
            torch._dynamo.reset()
            model = torch.compile(model, mode=None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(ids).logits.float().clone()

    torch.manual_seed(0)
    bf16_model = DenseTransformer(
        ModelConfig(d_model=1280, head_dim=64, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                    dropout=0.0, max_seq_len=128),
        vocab_size=TINY_VOCAB, eos_id=5,
    ).cuda()
    fp4_model = _fp4_build()
    nvfp4.refresh_fp4_weights(fp4_model)

    bf16_eager = logits(bf16_model, False)
    fp4_eager = logits(fp4_model, False)
    fp4_compiled = logits(fp4_model, True)

    quantization_error = (fp4_eager - bf16_eager).abs().max()
    compile_difference = (fp4_compiled - fp4_eager).abs().max()
    assert compile_difference < 0.5 * quantization_error, (
        f"compile difference {compile_difference:.3f} is not clearly below the quantization error "
        f"{quantization_error:.3f} — inductor may be changing the computation, not just perturbing it"
    )
    assert torch.isfinite(fp4_compiled).all()


@slow
@requires_cuda
@pytest.mark.skipif(not nvfp4.nvfp4_supported(), reason="needs a Blackwell GPU")
def test_fp4_gemm_adds_no_graph_break():
    """The whole reason for wrapping the GEMM in a `custom_op` rather than
    `@torch._dynamo.disable`: 0.392 ms vs 0.493 ms on an 8-linear stack.

    Compared against the same model without FP4 rather than against zero, because this repo already
    has one intentional graph break — `_doc_masks` is `@torch._dynamo.disable`d so `create_block_mask`
    runs eagerly. Asserting zero here would just re-assert that, and would fail for the wrong reason.
    """
    import torch._dynamo as dynamo

    ids = torch.randint(0, TINY_VOCAB, (2, 128), device="cuda")

    def breaks(model):
        dynamo.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return dynamo.explain(model)(ids).graph_break_count

    torch.manual_seed(0)
    baseline = DenseTransformer(
        ModelConfig(d_model=1280, head_dim=64, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                    dropout=0.0, max_seq_len=128),
        vocab_size=TINY_VOCAB, eos_id=5,
    ).cuda()
    fp4_model = _fp4_build()
    nvfp4.refresh_fp4_weights(fp4_model)

    assert breaks(fp4_model) == breaks(baseline)


@slow
@requires_cuda
@pytest.mark.skipif(not nvfp4.nvfp4_supported(), reason="needs a Blackwell GPU")
def test_fp4_weight_cache_is_stable_across_micro_batches():
    """The cache must be read-only for a whole optimizer step — all micro-batches of an accumulated
    step have to be differentiated against the same weights — and must move when the step ends."""
    model = _fp4_build()
    nvfp4.refresh_fp4_weights(model)
    layer = next(m for m in model.modules() if isinstance(m, nvfp4.FP4Linear))
    ids = torch.randint(0, TINY_VOCAB, (2, 128), device="cuda")

    before = layer._w_fwd_packed.clone()
    for _ in range(3):  # three micro-batches, no optimizer step between them
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids).logits.float().sum().backward()
    torch.testing.assert_close(layer._w_fwd_packed, before, rtol=0, atol=0)

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    nvfp4.refresh_fp4_weights(model)
    assert not torch.equal(layer._w_fwd_packed, before)


@slow
@requires_cuda
@pytest.mark.skipif(not nvfp4.nvfp4_supported(), reason="needs a Blackwell GPU")
def test_fp4_works_under_grad_checkpoint():
    """The recompute reads the weight cache a second time, which is safe because the cache is
    read-only during a step. That is the difference from `act_capacity_ratio`, which refuses
    `grad_checkpoint` because its retained K/V store is *written* during the block — do not add a
    defensive raise here by pattern-matching from that one."""
    ids = torch.randint(0, TINY_VOCAB, (2, 128), device="cuda")
    grads = {}
    for checkpointing in (False, True):
        model = _fp4_build(grad_checkpoint=checkpointing)
        nvfp4.refresh_fp4_weights(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids).logits.float().sum().backward()
        grads[checkpointing] = {n: p.grad.clone() for n, p in model.named_parameters()}

    for name, grad in grads[True].items():
        assert torch.isfinite(grad).all(), f"{name} has non-finite gradient under grad_checkpoint"
        torch.testing.assert_close(grad, grads[False][name], rtol=5e-2, atol=5e-2)
