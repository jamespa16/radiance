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
