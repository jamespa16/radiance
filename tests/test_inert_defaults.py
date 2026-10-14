"""Features that default *on* must be exactly inert at initialisation.

This repo's convention is that a new capability ships enabled, with its parameters defaulting to a
setting that makes it a mathematical no-op — so every pre-existing config keeps training exactly as
it did, and the feature only engages once someone configures it or training moves its weights.

That property is worth testing rather than asserting in a comment, because it is easy to break by
accident: an init helper that runs in the wrong order, a gate that can't quite reach 1.0, a scale
that's 0.999 instead of 1. Each test here pins one such construction. `assert_close` with zero
tolerance (not `allclose`) is deliberate — "inert" means bit-identical, not approximately equal.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


MODES = {
    "dense": dict(loop_count=1),
    "looped": dict(loop_count=3),
    "router": dict(use_router=True, max_loops=4),
    "router_sparse": dict(use_router=True, max_loops=4, act_ffn_capacity_ratio=0.5),
    "moe": dict(use_moe=True, n_experts=4, loop_count=2),
    "gqa": dict(n_kv_heads=2, loop_count=2),
    "grad_checkpoint": dict(loop_count=2, grad_checkpoint=True),
    "hyper": dict(loop_count=2, hyper_conn_streams=4),
}


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    cfg = ModelConfig(
        d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=32, **overrides,
    )
    torch.manual_seed(seed)
    return DenseTransformer(cfg, vocab_size=TINY_VOCAB)


def _logits(model: DenseTransformer, ids: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(1234)  # same dropout/tiebreak stream on both sides of a comparison
    return model(ids).logits


@pytest.mark.parametrize("mode", list(MODES))
def test_value_residual_is_inert_at_init(tiny_ids, mode):
    """lambda initialises to exactly 1.0, so v = 1*v + 0*v_first is the un-mixed value."""
    ids = tiny_ids(batch=2, seq=16)
    on = _build(value_residual=True, **MODES[mode]).train()
    off = _build(value_residual=False, **MODES[mode]).train()
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k in off.state_dict()})

    torch.testing.assert_close(_logits(on, ids), _logits(off, ids), rtol=0, atol=0)


@pytest.mark.parametrize("mode", list(MODES))
def test_attn_out_gate_is_inert_at_init(tiny_ids, mode):
    """2 * sigmoid(0) == 1.0 exactly, given out_gate is zero-initialised after _init_weights."""
    ids = tiny_ids(batch=2, seq=16)
    on = _build(attn_out_gate=True, **MODES[mode]).train()
    off = _build(attn_out_gate=False, **MODES[mode]).train()
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k in off.state_dict()})

    torch.testing.assert_close(_logits(on, ids), _logits(off, ids), rtol=0, atol=0)


def test_out_gate_is_exactly_one_at_init():
    """The gate value itself, not just its effect — 2*sigmoid can hit 1.0 where sigmoid cannot."""
    model = _build(attn_out_gate=True)
    x = torch.randn(2, 5, model.cfg.d_model)
    for block in model.blocks:
        gate = 2.0 * torch.sigmoid(block.attn.out_gate(x))
        torch.testing.assert_close(gate, torch.ones_like(gate), rtol=0, atol=0)


def test_value_lambda_is_exactly_one_at_init():
    # blocks[0] deliberately has none — it produces v_first rather than mixing against it.
    model = _build(value_residual=True)
    for block in model.blocks[1:]:
        assert block.attn.value_lambda.item() == 1.0


# --- the other half of the contract: inert at init, but not inert forever -------------------


def test_value_residual_changes_output_once_lambda_moves(tiny_ids):
    """An 'inert' feature that stays inert under training would be dead code."""
    ids = tiny_ids(batch=2, seq=16)
    model = _build(value_residual=True, loop_count=2).eval()
    with torch.no_grad():
        before = model(ids).logits
        for block in model.blocks[1:]:
            block.attn.value_lambda.fill_(0.5)
        after = model(ids).logits
    assert (before - after).abs().max() > 1e-5


def test_attn_out_gate_changes_output_once_weights_move(tiny_ids):
    ids = tiny_ids(batch=2, seq=16)
    model = _build(attn_out_gate=True, loop_count=2).eval()
    with torch.no_grad():
        before = model(ids).logits
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.out_gate.weight, std=0.02)
        after = model(ids).logits
    assert (before - after).abs().max() > 1e-5


_PROJECT_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _PROJECT_DEVICES)
@pytest.mark.parametrize("base", [8, 16])
def test_mup_multiplier_on_lm_head_input_is_bit_identical(device, base):
    """_project_logits applies a power-of-two muP multiplier to lm_head's input rather than to its
    output. That is a throughput change and must be nothing else: logits and both gradients
    (into the hidden state and into the tied weight) bit-identical to the logits-side multiply,
    under bf16 autocast on CUDA as well as fp32 — the autocast cast and the bf16 output rounding
    are where a non-power-of-two scale would stop commuting.

    base 8 (d_model 32 -> mult 0.25) and 16 (0.5) are both powers of two; the non-power-of-two
    case keeps the logits-side multiply and needs no test of its own."""
    cfg = ModelConfig(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1, dropout=0.0,
                      max_seq_len=32, mup_base_d_model=base)
    torch.manual_seed(0)
    model = DenseTransformer(cfg, vocab_size=TINY_VOCAB).to(device)
    assert model._mup_mult_exact_on_input
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def run(exact_on_input: bool):
        model._mup_mult_exact_on_input = exact_on_input
        model.zero_grad(set_to_none=True)
        x = torch.randn(2, 16, cfg.d_model, device=device, generator=torch.Generator(device).manual_seed(5))
        x.requires_grad_(True)
        with autocast:
            logits = model._project_logits(x)
        logits.float().square().mean().backward()
        return logits.detach(), x.grad, model.lm_head.weight.grad.clone()

    for got, want in zip(run(True), run(False)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
