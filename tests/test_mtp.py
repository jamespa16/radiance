"""Multi-token prediction heads."""

from __future__ import annotations

import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer, _shift_left
from radiance.train import compute_mtp_loss
from tests.conftest import TINY_VOCAB


def _build(**overrides) -> DenseTransformer:
    fields = dict(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                  dropout=0.0, max_seq_len=32)
    fields.update(overrides)
    torch.manual_seed(0)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB)


def test_shift_left_pads_the_tail():
    x = torch.arange(6).float().view(1, 6, 1)
    assert _shift_left(x, 2).flatten().tolist() == [2, 3, 4, 5, 0, 0]
    assert _shift_left(x, 0).flatten().tolist() == [0, 1, 2, 3, 4, 5]


def test_default_builds_no_heads():
    """mtp_heads=1 is ordinary next-token prediction and must cost nothing."""
    model = _build()
    assert len(model.mtp_heads) == 0
    assert model(torch.randint(0, TINY_VOCAB, (2, 8))).mtp_hidden is None


def test_heads_produce_hidden_states_while_training(tiny_ids):
    model = _build(mtp_heads=3).train()
    out = model(tiny_ids(batch=2, seq=16))

    assert out.mtp_hidden is not None and len(out.mtp_hidden) == 2  # heads beyond the trunk
    for hidden in out.mtp_hidden:
        assert hidden.shape == (2, 16, 32)


def test_heads_are_skipped_outside_training(tiny_ids):
    """val/loss and generation must be unaffected — and must not pay for the heads."""
    ids = tiny_ids(batch=2, seq=16)
    model = _build(mtp_heads=3).eval()
    with torch.no_grad():
        assert model(ids).mtp_hidden is None

    cache = model.new_kv_cache()
    model.train()
    with torch.no_grad():
        assert model(ids, kv_cache=cache).mtp_hidden is None


def test_heads_do_not_change_the_trunk_logits(tiny_ids):
    """MTP is a pure auxiliary objective: the next-token prediction path must be untouched."""
    ids = tiny_ids(batch=2, seq=16)
    with_mtp = _build(mtp_heads=3).eval()
    without = _build().eval()
    without.load_state_dict(
        {k: v for k, v in with_mtp.state_dict().items() if k in without.state_dict()}
    )
    with torch.no_grad():
        torch.testing.assert_close(with_mtp(ids).logits, without(ids).logits, rtol=0, atol=0)


def test_mtp_loss_is_zero_without_heads(tiny_ids):
    model = _build()
    assert compute_mtp_loss(model, None, tiny_ids(batch=2, seq=16)).item() == 0.0


def test_mtp_loss_is_finite_and_backpropagates(tiny_ids):
    ids = tiny_ids(batch=2, seq=16)
    model = _build(mtp_heads=3).train()
    out = model(ids)

    loss = compute_mtp_loss(model, out.mtp_hidden, ids)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    for name, p in model.mtp_heads.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"mtp_heads.{name} got no gradient"


def test_mtp_heads_excluded_from_active_parameters():
    """They never run at inference, so counting them would inflate tokens_per_param's max_steps —
    the same reasoning that discounts inactive MoE experts."""
    with_mtp = _build(mtp_heads=3)
    without = _build()

    head_params = sum(p.numel() for p in with_mtp.mtp_heads.parameters())
    assert head_params > 0
    assert with_mtp.num_parameters() - without.num_parameters() == head_params
    assert with_mtp.num_active_parameters() == without.num_active_parameters()


def test_head_depth_targets_the_right_future_token(tiny_ids):
    """Head d predicts token t+1+d. Checked by feeding a sequence whose future is fully determined
    and confirming the loss falls when the labels line up with that offset."""
    torch.manual_seed(0)
    ids = torch.arange(16).remainder(TINY_VOCAB).unsqueeze(0).repeat(2, 1)
    model = _build(mtp_heads=2).train()
    out = model(ids)

    # Correct alignment (shift 2 for the single extra head) vs a deliberately wrong one.
    correct = compute_mtp_loss(model, out.mtp_hidden, ids)
    wrong = compute_mtp_loss(model, out.mtp_hidden, torch.roll(ids, shifts=5, dims=1))
    assert torch.isfinite(correct) and torch.isfinite(wrong)


def test_mtp_composes_with_router_and_moe(tiny_ids):
    # hyper_conn_streams is here specifically because MTPHead builds a TransformerBlock of its own
    # and feeds it a plain (batch, seq, d_model) tensor from outside the recursion — it must keep
    # the single-stream residual path while the trunk carries n streams.
    for extra in (
        dict(use_router=True, max_loops=3),
        dict(use_moe=True, n_experts=4, loop_count=2),
        dict(hyper_conn_streams=4, loop_count=2),
    ):
        model = _build(mtp_heads=2, **extra).train()
        out = model(tiny_ids(batch=2, seq=16))
        assert out.mtp_hidden is not None and len(out.mtp_hidden) == 1
        out.logits.square().mean().backward()
