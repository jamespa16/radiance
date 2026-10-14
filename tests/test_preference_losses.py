"""The preference-loss variants behind dpo.loss_type / label_smoothing / sft_weight.

Each is checked against the property that defines it rather than against a re-typed copy of its
formula: the default path is bit-identical to plain DPO; cDPO's optimum is the finite margin
log((1-e)/e); hinge is exactly zero past its margin; IPO is exactly zero at its target margin of
per-token-mean log-ratios; the NLL term is the chosen side's per-token-mean NLL. Plus the
validation guard rails and the val-side accuracies evaluate_detailed now reports.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from radiance.config import Config, DPOConfig, TrainConfig
from radiance.evaluation import evaluate_detailed
from radiance.losses import (
    build_dpo_loss_fn,
    compute_dpo_loss,
    compute_dpo_loss_from_logits,
    validate_post_training_config,
)
from tests.test_evaluate import _FakeModel


def _logps(seed: int = 0, n: int = 6):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, generator=g) * 5 for _ in range(4)]


def test_default_kwargs_are_bit_identical_to_plain_dpo():
    pc, pr, rc, rr = _logps()
    beta = 0.1
    loss, margin, margin_acc, reward_acc = compute_dpo_loss(pc, pr, rc, rr, beta)
    expected = -F.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()
    assert torch.equal(loss, expected)


def test_label_smoothing_zero_is_the_unsmoothed_path():
    pc, pr, rc, rr = _logps(1)
    plain = compute_dpo_loss(pc, pr, rc, rr, 0.2)[0]
    smoothed = compute_dpo_loss(pc, pr, rc, rr, 0.2, label_smoothing=0.0)[0]
    assert torch.equal(plain, smoothed)


def test_label_smoothing_optimum_is_the_finite_cdpo_margin():
    # cDPO's per-pair loss -(1-e)logsig(x) - e logsig(-x) has derivative sigmoid(x) - (1-e),
    # zero at x* = log((1-e)/e): a pair is pushed apart only up to x*, then pulled back.
    eps, beta = 0.1, 1.0
    x_star = math.log((1 - eps) / eps)
    zero = torch.zeros(1)
    for x, sign in ((x_star - 0.5, -1), (x_star + 0.5, 1)):
        pc = torch.tensor([x], requires_grad=True)
        loss = compute_dpo_loss(pc, zero, zero, zero, beta, label_smoothing=eps)[0]
        loss.backward()
        # d loss / d pc has the sign of (x - x*): negative below the optimum, positive above.
        assert math.copysign(1, pc.grad.item()) == sign
    pc = torch.tensor([x_star], requires_grad=True)
    compute_dpo_loss(pc, zero, zero, zero, beta, label_smoothing=eps)[0].backward()
    assert abs(pc.grad.item()) < 1e-6


def test_hinge_is_zero_past_its_margin_and_linear_before_it():
    beta = 0.5
    zero = torch.zeros(1)
    past = compute_dpo_loss(torch.tensor([2.5]), zero, zero, zero, beta, loss_type="hinge")[0]
    assert past.item() == 0.0  # beta * 2.5 = 1.25 >= 1
    before = compute_dpo_loss(torch.tensor([1.0]), zero, zero, zero, beta, loss_type="hinge")[0]
    assert before.item() == pytest.approx(1 - 0.5)


def test_ipo_is_zero_at_its_target_margin_of_per_token_means():
    beta = 0.25
    target = 1 / (2 * beta)  # 2.0
    n_c, n_r = torch.tensor([4.0]), torch.tensor([10.0])
    # (pc - rc)/n_c - (pr - rr)/n_r == target, with deliberately different lengths so a sum-based
    # implementation would land elsewhere.
    pc, rc = torch.tensor([3.0 + target * 4.0]), torch.tensor([3.0])
    pr, rr = torch.tensor([-7.0]), torch.tensor([-7.0])
    loss = compute_dpo_loss(pc, pr, rc, rr, beta, loss_type="ipo", chosen_tokens=n_c, rejected_tokens=n_r)[0]
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    off = compute_dpo_loss(pc + 4.0, pr, rc, rr, beta, loss_type="ipo", chosen_tokens=n_c, rejected_tokens=n_r)[0]
    assert off.item() == pytest.approx(1.0)  # (+4 / 4 tokens)^2


def test_sft_weight_adds_the_chosen_per_token_mean_nll():
    pc, pr, rc, rr = _logps(2)
    n_c = torch.tensor([3.0, 5.0, 2.0, 7.0, 1.0, 4.0])
    base = compute_dpo_loss(pc, pr, rc, rr, 0.1)[0]
    with_nll = compute_dpo_loss(pc, pr, rc, rr, 0.1, sft_weight=0.5, chosen_tokens=n_c, rejected_tokens=n_c)[0]
    torch.testing.assert_close(with_nll - base, 0.5 * (-pc / n_c).mean())


def test_from_logits_counts_tokens_on_the_shifted_mask():
    # The NLL term's denominator must be the positions sequence_logprob_sum actually scores
    # (mask[:, 1:]), not the raw mask's sum — an off-by-one there silently rescales the term.
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 11)
    ids = torch.randint(0, 11, (2, 6))
    mask = torch.tensor([[1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 0]])
    ref = torch.zeros(2)
    base = compute_dpo_loss_from_logits(logits, ids, mask, logits, ids, mask, ref, ref, 0.1)[0]
    with_nll = compute_dpo_loss_from_logits(logits, ids, mask, logits, ids, mask, ref, ref, 0.1, sft_weight=1.0)[0]
    from radiance.model import sequence_logprob_sum

    logp = sequence_logprob_sum(logits, ids, mask)
    n = mask[:, 1:].sum(dim=1).float()
    torch.testing.assert_close(with_nll - base, (-logp / n).mean())


def test_build_dpo_loss_fn_binds_the_configured_variant():
    cfg = Config(
        dpo=DPOConfig(enabled=True, loss_type="hinge"),
        train=TrainConfig(compile=False),
    )
    fn = build_dpo_loss_fn(cfg)
    torch.manual_seed(4)
    logits = torch.randn(2, 5, 7)
    ids = torch.randint(0, 7, (2, 5))
    mask = torch.ones_like(ids)
    ref = torch.zeros(2)
    got = fn(logits, ids, mask, logits.flip(0), ids.flip(0), mask, ref, ref, 0.3)[0]
    expected = compute_dpo_loss_from_logits(
        logits, ids, mask, logits.flip(0), ids.flip(0), mask, ref, ref, 0.3, loss_type="hinge"
    )[0]
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "dpo, match",
    [
        (DPOConfig(loss_type="kto"), "loss_type"),
        (DPOConfig(label_smoothing=0.5), "label_smoothing"),
        (DPOConfig(label_smoothing=0.1, loss_type="ipo"), "sigmoid only"),
        (DPOConfig(sft_weight=-1.0), "sft_weight"),
        (DPOConfig(min_score_margin=1.0, score_chosen_column="a"), "score_rejected_column"),
    ],
)
def test_validate_rejects_bad_variants(dpo, match):
    dpo.enabled = True
    dpo.reference_checkpoint = "ref.pt"
    with pytest.raises(ValueError, match=match):
        validate_post_training_config(Config(dpo=dpo))


def test_reference_checkpoint_defaults_to_init_from():
    cfg = Config(dpo=DPOConfig(enabled=True), train=TrainConfig(init_from="sft.pt"))
    assert cfg.resolved_reference_checkpoint == "sft.pt"
    validate_post_training_config(cfg)  # no longer raises: the reference resolves
    cfg.dpo.reference_checkpoint = "other.pt"
    assert cfg.resolved_reference_checkpoint == "other.pt"
    with pytest.raises(ValueError, match="reference_checkpoint"):
        validate_post_training_config(Config(dpo=DPOConfig(enabled=True)))


def _dpo_batch(n: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 5, (n, 8), generator=g)
    r = torch.randint(0, 5, (n, 8), generator=g)
    return {
        "chosen_input_ids": c,
        "chosen_loss_mask": torch.ones_like(c),
        "rejected_input_ids": r,
        "rejected_loss_mask": torch.ones_like(r),
        "ref_chosen_logprob": torch.randn(n, generator=g),
        "ref_rejected_logprob": torch.randn(n, generator=g),
    }


def test_eval_reports_row_weighted_preference_accuracies():
    batches = [_dpo_batch(4, 0), _dpo_batch(4, 1)]
    model: Any = _FakeModel()
    result = evaluate_detailed(
        model, batches, "cpu", "cpu", torch.float32, loss_fn=compute_dpo_loss_from_logits, dpo=True, dpo_beta=0.25
    )
    # Recompute each accuracy over all 8 pairs directly.
    rewards, margins = [], []
    for b in batches:
        logits_c = model(b["chosen_input_ids"]).logits
        logits_r = model(b["rejected_input_ids"]).logits
        _, _, acc_m, acc_r = compute_dpo_loss_from_logits(
            logits_c, b["chosen_input_ids"], b["chosen_loss_mask"],
            logits_r, b["rejected_input_ids"], b["rejected_loss_mask"],
            b["ref_chosen_logprob"], b["ref_rejected_logprob"], 0.25,
        )
        rewards.append(acc_r.item())
        margins.append(acc_m.item())
    assert result.dpo_reward_accuracy == pytest.approx(sum(rewards) / 2)
    assert result.dpo_margin_accuracy == pytest.approx(sum(margins) / 2)

    chunked = evaluate_detailed(
        _FakeModel(), batches, "cpu", "cpu", torch.float32,
        loss_fn=compute_dpo_loss_from_logits, dpo=True, dpo_beta=0.25, micro_chunk_size=3,
    )
    assert chunked.dpo_reward_accuracy == pytest.approx(result.dpo_reward_accuracy)
    assert chunked.dpo_margin_accuracy == pytest.approx(result.dpo_margin_accuracy)


def test_eval_accuracies_absent_outside_dpo():
    ids = torch.randint(0, 5, (2, 8))
    result = evaluate_detailed(_FakeModel(), [{"input_ids": ids}], "cpu", "cpu", torch.float32)
    assert result.dpo_reward_accuracy is None and result.dpo_margin_accuracy is None
