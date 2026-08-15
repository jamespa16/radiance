"""compute_loss's z-loss term and build_lr_scheduler's two schedule shapes."""

from __future__ import annotations

import pytest
import torch

from radiance.config import Config, TrainConfig
from radiance.train import build_lr_scheduler
from radiance.losses import compute_loss


def test_z_loss_matches_masked_reference():
    """The efficient form (reduce over vocab, then mask) must equal the naive masked-gather form.

    compute_loss deliberately avoids flat_logits[mask], which would copy nearly the whole logits
    tensor — the largest activation in the model — to drop one row per sequence. This pins that the
    optimisation didn't change the number.
    """
    torch.manual_seed(3)
    logits = torch.randn(2, 8, 50)
    input_ids = torch.randint(0, 50, (2, 8))

    _, z_loss = compute_loss(logits, input_ids)

    # Reference: the final position of each sequence has no next-token target (ignore_index).
    keep = torch.ones(2, 8, dtype=torch.bool)
    keep[:, -1] = False
    expected = (torch.logsumexp(logits.view(-1, 50), -1).square() * keep.view(-1)).sum() / keep.sum()

    torch.testing.assert_close(z_loss, expected)


def test_lm_loss_ignores_final_position():
    """lm_loss must average over exactly the (seq - 1) positions that have a target."""
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 20)
    input_ids = torch.randint(0, 20, (1, 6))

    lm_loss, _ = compute_loss(logits, input_ids)
    expected = torch.nn.functional.cross_entropy(logits[0, :-1], input_ids[0, 1:])

    torch.testing.assert_close(lm_loss, expected)


def test_single_pass_loss_matches_two_pass_reference():
    """Both losses come from one logsumexp; that must equal computing them independently.

    This is the identity the fast path rests on — cross_entropy is `logsumexp(x) - x[label]`, so
    the z-loss regulariser's own logsumexp is the same reduction and the second pass over the
    largest tensor in the model was redundant. If a future change reintroduces a separate
    F.cross_entropy call the numbers should still match, so what this really pins is that the
    hand-rolled NLL is exactly cross_entropy's mean-over-kept-positions and not, say, a mean over
    all positions including the ignored ones.
    """
    torch.manual_seed(7)
    logits = torch.randn(3, 9, 40)
    input_ids = torch.randint(0, 40, (3, 9))

    lm_loss, z_loss = compute_loss(logits, input_ids)

    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((3, 1), -100)], dim=1).view(-1)
    flat = logits.view(-1, 40)
    ref_lm = torch.nn.functional.cross_entropy(flat, labels, ignore_index=-100)
    keep = (labels != -100).to(flat.dtype)
    ref_z = (torch.logsumexp(flat, dim=-1).square() * keep).sum() / keep.sum()

    torch.testing.assert_close(lm_loss, ref_lm)
    torch.testing.assert_close(z_loss, ref_z)


def test_single_pass_loss_gradients_match_cross_entropy():
    """Equal losses aren't enough — the gradient reaching the logits has to match too."""
    torch.manual_seed(11)
    logits = torch.randn(2, 7, 30, requires_grad=True)
    input_ids = torch.randint(0, 30, (2, 7))

    compute_loss(logits, input_ids)[0].backward()
    ours = logits.grad.clone()

    ref_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((2, 1), -100)], dim=1)
    torch.nn.functional.cross_entropy(
        ref_logits.view(-1, 30), labels.view(-1), ignore_index=-100
    ).backward()

    torch.testing.assert_close(ours, ref_logits.grad)


def test_all_ignored_batch_is_zero_not_nan():
    """A batch with no targets at all must not poison the accumulated gradient with nan.

    F.cross_entropy returns nan for an all-ignored batch (0/0). Only reachable at seq_len 1, but a
    nan here propagates through the whole accumulated step, so the kept-count is clamped instead.
    """
    lm_loss, z_loss = compute_loss(torch.randn(2, 1, 12), torch.randint(0, 12, (2, 1)))

    assert torch.isfinite(lm_loss) and lm_loss.item() == 0.0
    assert torch.isfinite(z_loss)


def _lr_curve(**train_kwargs) -> list[float]:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    cfg = Config(train=TrainConfig(**train_kwargs))
    scheduler = build_lr_scheduler(optimizer, cfg)
    curve = []
    for _ in range(train_kwargs["max_steps"]):
        curve.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return curve


def test_wsd_holds_then_decays():
    curve = _lr_curve(
        max_steps=100, warmup_ratio=0.1, lr_schedule="wsd", wsd_decay_ratio=0.2, min_lr_ratio=0.1
    )
    assert curve[0] < curve[5] < curve[9]                      # warmup ramps
    assert all(lr == pytest.approx(1.0) for lr in curve[10:80])  # stable phase holds full LR
    assert curve[80] > curve[90] > curve[99]                    # decay phase
    assert curve[99] >= 0.1 - 1e-9                              # never below min_lr_ratio


def test_cosine_is_unchanged_and_is_the_default():
    """Cosine stays the default: switching it would silently reshape every existing config's LR."""
    assert TrainConfig().lr_schedule == "cosine"
    curve = _lr_curve(max_steps=100, warmup_ratio=0.1, min_lr_ratio=0.1)
    assert curve[0] < curve[9]                                  # warmup
    assert curve[10] == pytest.approx(1.0)                      # peak right after warmup
    assert all(a >= b - 1e-9 for a, b in zip(curve[10:], curve[11:]))  # monotone decay after peak
    assert curve[-1] == pytest.approx(0.1, abs=1e-3)


def test_first_step_is_not_at_zero_lr():
    """LambdaLR evaluates at step 0 to set the first optimizer.step()'s LR; a bare step/warmup ramp
    would waste that entire update at lr=0."""
    for schedule in ("cosine", "wsd"):
        curve = _lr_curve(max_steps=100, warmup_ratio=0.1, lr_schedule=schedule)
        assert curve[0] > 0.0


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="lr_schedule"):
        _lr_curve(max_steps=10, lr_schedule="linear")
