"""Whole-block ACT sparsity (cfg.act_capacity_ratio).

This is the mechanism that finally makes router mode save wall-clock rather than just producing a
loss signal. It is an *approximation* — see tests/test_act_kv_invariance.py for exactly why reusing
a halted position's K/V is only exact for the first block of the loop body — so what's testable is
not equality with the dense model but a set of structural guarantees:

  * disabled (ratio 1.0) it is bit-identical to dense
  * it is training-only, so eval and generation stay dense and stay consistent with each other
  * capacity padding never corrupts a position that wasn't selected
  * the configuration errors that would silently produce wrong gradients are refused
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer
from radiance.model.act import _act_select
from radiance.model.masking import _sparse_attn_mask
from tests.conftest import TINY_VOCAB


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    fields = dict(d_model=64, head_dim=16, n_layers=4, ffn_mult=2.0, ffn_depth=1, dropout=0.0,
                  max_seq_len=64, use_router=True, max_loops=5)
    fields.update(overrides)
    torch.manual_seed(seed)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB, eos_id=5)


# --- selection and masking --------------------------------------------------------------------


def test_selection_prefers_still_running_positions():
    """Halted positions are only selected once the running ones run out, so capacity is never
    wasted on a position whose output would be discarded anyway."""
    still_running = torch.tensor([[True, False, True, False, True, False]])
    token_idx, valid = _act_select(still_running, capacity=3, training=False)

    assert sorted(token_idx[0].tolist()) == [0, 2, 4]
    assert valid.all()


def test_selection_pads_with_halted_positions_when_under_capacity():
    """Shapes stay static (compile-friendly) by padding; `valid` marks the padding so it can be
    discarded rather than written back."""
    still_running = torch.tensor([[True, False, False, False]])
    token_idx, valid = _act_select(still_running, capacity=3, training=False)

    assert token_idx.shape == (1, 3)
    assert valid.sum() == 1
    assert token_idx[0][valid[0]].tolist() == [0]


def test_selection_is_deterministic_outside_training():
    """Otherwise val/loss and greedy decoding would move run to run for identical weights."""
    still_running = torch.rand(2, 16) > 0.5
    first, _ = _act_select(still_running, capacity=6, training=False)
    second, _ = _act_select(still_running, capacity=6, training=False)
    torch.testing.assert_close(first, second)


def test_selection_is_per_sequence():
    """Attention needs each row's queries to belong to that row, so selection cannot be over the
    flattened batch the way _sparse_ffn_delta's is."""
    still_running = torch.tensor([[True, True, False, False], [False, False, True, True]])
    token_idx, _ = _act_select(still_running, capacity=2, training=False)

    assert sorted(token_idx[0].tolist()) == [0, 1]
    assert sorted(token_idx[1].tolist()) == [2, 3]


def test_sparse_mask_is_causal_on_true_positions():
    """The gathered queries sit at scattered positions, so causality is
    true_position(query) >= key_position, not the triangle is_causal assumes."""
    token_idx = torch.tensor([[1, 4]])
    mask = _sparse_attn_mask(token_idx, seq_len=6)[0, 0]

    assert mask[0].tolist() == [True, True, False, False, False, False]   # query at position 1
    assert mask[1].tolist() == [True, True, True, True, True, False]      # query at position 4


def test_sparse_mask_respects_document_boundaries():
    doc_ids = torch.tensor([[0, 0, 1, 1]])
    token_idx = torch.tensor([[3]])
    mask = _sparse_attn_mask(token_idx, seq_len=4, doc_ids=doc_ids)[0, 0]

    assert mask[0].tolist() == [False, False, True, True]


# --- end-to-end behaviour ---------------------------------------------------------------------


def test_ratio_one_is_bit_identical_to_dense(tiny_ids):
    ids = tiny_ids(batch=2, seq=32)
    dense = _build().train()
    off = _build(act_capacity_ratio=1.0).train()

    torch.manual_seed(7)
    a = dense(ids).logits
    torch.manual_seed(7)
    b = off(ids).logits
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0.75, 0.5, 0.25])
def test_sparse_is_training_only(tiny_ids, ratio):
    """Under eval the model must be exactly the dense one — that's what keeps evaluate() and
    radiance-generate consistent with each other, and val/loss comparable across ratios."""
    ids = tiny_ids(batch=2, seq=32)
    sparse = _build(act_capacity_ratio=ratio).eval()
    dense = _build().eval()

    with torch.no_grad():
        torch.testing.assert_close(sparse(ids).logits, dense(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0.75, 0.5, 0.25])
def test_sparse_trains(tiny_ids, ratio):
    """Finite output and a gradient for every parameter — the sparse path must not orphan any."""
    ids = tiny_ids(batch=2, seq=32)
    model = _build(act_capacity_ratio=ratio).train()

    out = model(ids)
    assert torch.isfinite(out.logits).all()
    (out.logits.square().mean() + out.ponder_cost).backward()

    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"no gradient for {dead}"


def test_generation_matches_a_full_forward(tiny_ids):
    """Both are dense (sparsity is training-only), so incremental decoding must still agree with a
    single forward — the invariant the KVCache's implicit slot assignment rests on."""
    ids = tiny_ids(batch=1, seq=12)
    model = _build(act_capacity_ratio=0.25).eval()

    with torch.no_grad():
        full = model(ids).logits
        cache = model.new_kv_cache()
        parts = [model(ids[:, :6], kv_cache=cache).logits]
        for t in range(6, 12):
            parts.append(model(ids[:, t : t + 1], kv_cache=cache).logits)

    torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-4, atol=1e-4)


# --- refused configurations -------------------------------------------------------------------


def test_grad_checkpoint_combination_is_refused():
    """Recomputation during backward would write into the retained K/V store a second time,
    silently corrupting it — so this raises rather than producing wrong gradients."""
    with pytest.raises(ValueError, match="grad_checkpoint"):
        _build(act_capacity_ratio=0.5, grad_checkpoint=True)


def test_both_capacity_ratios_is_refused():
    with pytest.raises(ValueError, match="act_capacity_ratio"):
        _build(act_capacity_ratio=0.5, act_ffn_capacity_ratio=0.5)


def test_requires_router_mode():
    """Without ACT halting every position is always running, so there is nothing to select."""
    with pytest.raises(ValueError, match="use_router"):
        _build(act_capacity_ratio=0.5, use_router=False)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(dict(use_moe=True, n_experts=4), id="moe"),
        pytest.param(dict(n_kv_heads=2), id="gqa"),
        pytest.param(dict(loop_iter_conditioning="lora"), id="lora"),
        pytest.param(dict(value_residual=False, attn_out_gate=False), id="no_attn_extras"),
        pytest.param(dict(hyper_conn_streams=4), id="hyper"),
    ],
)
def test_sparse_composes_with_other_features(tiny_ids, extra):
    """The gathered tensor is (batch, capacity, d_model), which satisfies the same
    (*, d_model) -> (*, d_model) contract MoEFeedForward and FeedForward share — so the sparse path
    needs no special-casing per FFN type, and GQA/LoRA/value-residual ride along unchanged."""
    model = _build(act_capacity_ratio=0.5, **extra).train()
    out = model(tiny_ids(batch=2, seq=32))

    assert torch.isfinite(out.logits).all()
    (out.logits.square().mean() + out.ponder_cost + out.moe_aux_loss).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_capacity_padding_does_not_corrupt_unselected_positions(tiny_ids):
    """When fewer positions are running than capacity, the padding rows are selected anyway to keep
    the shape static. They must be restored before the scatter, or they would overwrite live state
    with a recomputed value."""
    ids = tiny_ids(batch=2, seq=32)
    # halt_epsilon this large forces everything to halt on the first iteration, so every interior
    # iteration is pure padding and must be a complete no-op.
    model = _build(act_capacity_ratio=0.5, halt_epsilon=0.99).train()
    dense = _build(halt_epsilon=0.99).train()

    torch.manual_seed(11)
    sparse_logits = model(ids).logits
    torch.manual_seed(11)
    dense_logits = dense(ids).logits

    torch.testing.assert_close(sparse_logits, dense_logits, rtol=1e-5, atol=1e-5)


def test_activation_estimate_shrinks_with_capacity():
    """auto_batch_size sizes the batch off activation_bytes_per_token, so if that doesn't know
    about sparsity the freed memory goes unused. Predicted ratios track the measured peaks closely
    (0.81/0.71 predicted vs 0.83/0.73 measured at d_model 512, n_layers 8, max_loops 6)."""
    def estimate(ratio):
        cfg = ModelConfig(d_model=512, head_dim=64, n_layers=8, ffn_mult=4.0, ffn_depth=2,
                          dropout=0.0, max_seq_len=512, use_router=True, max_loops=6,
                          act_capacity_ratio=ratio)
        torch.manual_seed(0)
        # A realistic vocab, not TINY_VOCAB: the logits term doesn't scale with the capacity ratio,
        # so a toy vocab lets the block terms dominate and the ratios come out lower than any real
        # model's. These bands are calibrated against the measured configuration.
        return DenseTransformer(cfg, vocab_size=50304).activation_bytes_per_token(2)

    dense, half, quarter = estimate(1.0), estimate(0.5), estimate(0.25)
    assert quarter < half < dense
    assert 0.75 < half / dense < 0.9
    assert 0.65 < quarter / dense < 0.8


def test_legacy_ffn_only_ratio_still_works(tiny_ids):
    """act_ffn_capacity_ratio is superseded but must keep working for existing configs."""
    model = _build(act_ffn_capacity_ratio=0.5).train()
    out = model(tiny_ids(batch=2, seq=32))
    assert torch.isfinite(out.logits).all()
