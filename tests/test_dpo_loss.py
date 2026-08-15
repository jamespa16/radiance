"""DPO loss primitives: model.sequence_logprob_sum (the per-sequence reduction DPO needs, where
train._nll_and_logz only gives a batch-wide mean) and train.compute_dpo_loss (the pairwise DPO
objective itself), plus train.validate_post_training_config's guard rails.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from radiance.config import Config, DPOConfig, ModelConfig, SFTConfig
from radiance.model import sequence_logprob_sum
from radiance.losses import compute_dpo_loss, note_dpo_z_loss_omitted, validate_post_training_config


def test_sequence_logprob_sum_matches_hand_built_log_softmax_reference():
    torch.manual_seed(10)
    logits = torch.randn(3, 8, 20)
    input_ids = torch.randint(0, 20, (3, 8))
    loss_mask = torch.randint(0, 2, (3, 8))

    result = sequence_logprob_sum(logits, input_ids, loss_mask)

    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((3, 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((3, 1))], dim=1)
    keep = shifted_mask.bool()
    log_probs = torch.log_softmax(logits, dim=-1)
    safe_labels = labels.masked_fill(~keep, 0)
    target_logprob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    expected = (target_logprob * keep.float()).sum(dim=1)

    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


def test_sequence_logprob_sum_masked_positions_contribute_nothing():
    torch.manual_seed(11)
    logits = torch.randn(2, 8, 20)
    input_ids = torch.randint(0, 20, (2, 8))
    loss_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 2)

    result = sequence_logprob_sum(logits, input_ids, loss_mask)

    perturbed_logits = logits.clone()
    # Same reasoning as test_sft_loss.py: position i predicts target i+1, so corrupting positions
    # whose *target* is masked (shifted_mask[i] = loss_mask[i+1] == 0, i.e. positions 0..2) must
    # not move the result at all.
    perturbed_logits[:, :3, :] = torch.randn(2, 3, 20) * 100
    perturbed_result = sequence_logprob_sum(perturbed_logits, input_ids, loss_mask)

    torch.testing.assert_close(result, perturbed_result, rtol=0, atol=1e-5)


def test_sequence_logprob_sum_zero_mask_row_is_zero():
    logits = torch.randn(1, 5, 10)
    input_ids = torch.randint(0, 10, (1, 5))
    loss_mask = torch.zeros(1, 5, dtype=torch.long)

    result = sequence_logprob_sum(logits, input_ids, loss_mask)

    torch.testing.assert_close(result, torch.zeros(1))


def test_compute_dpo_loss_equals_log_2_when_policy_matches_reference():
    """policy_logp == ref_logp on both sides means the policy hasn't moved relative to the
    reference at all, so the DPO objective must equal exactly -logsigmoid(0) = log(2), and the
    gradient must push chosen up and rejected down by equal and opposite amounts (not zero — DPO
    always has *some* gradient at this point, since being tied isn't optimal)."""
    torch.manual_seed(12)
    ref_chosen = torch.randn(5)
    ref_rejected = torch.randn(5)
    policy_chosen = ref_chosen.clone().requires_grad_(True)
    policy_rejected = ref_rejected.clone().requires_grad_(True)

    loss, margin, margin_accuracy, reward_accuracy = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    torch.testing.assert_close(loss, torch.tensor(math.log(2)))
    torch.testing.assert_close(margin, torch.zeros(()))
    torch.testing.assert_close(margin_accuracy, torch.zeros(()))  # logits == 0 is not > 0
    # reward_accuracy is reference-independent, so tying policy to reference doesn't pin it to any
    # particular value — cross-check against the same comparison compute_dpo_loss makes internally.
    torch.testing.assert_close(reward_accuracy, (policy_chosen > policy_rejected).float().mean())

    loss.backward()
    torch.testing.assert_close(policy_chosen.grad, -policy_rejected.grad)
    assert torch.all(policy_chosen.grad != 0)


def test_compute_dpo_loss_matches_hand_built_logsigmoid_reference():
    torch.manual_seed(13)
    policy_chosen = torch.randn(4)
    policy_rejected = torch.randn(4)
    ref_chosen = torch.randn(4)
    ref_rejected = torch.randn(4)
    beta = 0.3

    loss, _, _, _ = compute_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)

    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    expected = -F.logsigmoid(logits).mean()

    torch.testing.assert_close(loss, expected)


def test_compute_dpo_loss_accuracy_and_margin_sanity():
    beta = 0.5
    ref_chosen = torch.zeros(1)
    ref_rejected = torch.zeros(1)

    # Policy strongly separates chosen from rejected in the preferred direction.
    _, margin, margin_accuracy, reward_accuracy = compute_dpo_loss(
        torch.tensor([5.0]), torch.tensor([-5.0]), ref_chosen, ref_rejected, beta
    )
    assert margin_accuracy.item() == 1.0
    assert reward_accuracy.item() == 1.0
    assert margin.item() > 0

    # Policy separates them the wrong way.
    _, margin2, margin_accuracy2, reward_accuracy2 = compute_dpo_loss(
        torch.tensor([-5.0]), torch.tensor([5.0]), ref_chosen, ref_rejected, beta
    )
    assert margin_accuracy2.item() == 0.0
    assert reward_accuracy2.item() == 0.0
    assert margin2.item() < 0


def _dpo_model_cfg() -> ModelConfig:
    return ModelConfig(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1, dropout=0.0, max_seq_len=32)


def test_validate_post_training_config_raises_on_both_sft_and_dpo_enabled():
    cfg = Config(model=_dpo_model_cfg(), sft=SFTConfig(enabled=True), dpo=DPOConfig(enabled=True))
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_post_training_config(cfg)


def test_validate_post_training_config_raises_on_dpo_without_reference_checkpoint():
    cfg = Config(model=_dpo_model_cfg(), dpo=DPOConfig(enabled=True, dataset="foo/bar"))
    with pytest.raises(ValueError, match="reference_checkpoint"):
        validate_post_training_config(cfg)


def test_validate_post_training_config_raises_on_dpo_with_mtp_heads():
    cfg = Config(
        model=ModelConfig(
            d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1, dropout=0.0, max_seq_len=32,
            mtp_heads=2,
        ),
        dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint="foo.pt"),
    )
    with pytest.raises(ValueError, match="mtp_heads"):
        validate_post_training_config(cfg)


def test_validate_post_training_config_ok_when_only_dpo_enabled():
    cfg = Config(
        model=_dpo_model_cfg(),
        dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint="foo.pt"),
    )
    validate_post_training_config(cfg)  # must not raise


def test_note_dpo_z_loss_omitted_prints_when_dpo_enabled_with_default_z_loss_weight(capsys):
    cfg = Config(
        model=_dpo_model_cfg(),
        dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint="foo.pt"),
    )
    note_dpo_z_loss_omitted(cfg)
    assert "z_loss_weight" in capsys.readouterr().out


def test_note_dpo_z_loss_omitted_silent_when_z_loss_weight_zeroed(capsys):
    model_cfg = _dpo_model_cfg()
    model_cfg.z_loss_weight = 0
    cfg = Config(
        model=model_cfg,
        dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint="foo.pt"),
    )
    note_dpo_z_loss_omitted(cfg)
    assert capsys.readouterr().out == ""


def test_note_dpo_z_loss_omitted_silent_when_dpo_disabled(capsys):
    note_dpo_z_loss_omitted(Config(model=_dpo_model_cfg()))
    assert capsys.readouterr().out == ""
