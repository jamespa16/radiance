"""compute_sft_loss: a strict generalization of compute_loss, not a parallel reimplementation.

_nll_and_logz already treats -100 generically anywhere in the flat label tensor (see
test_loss_and_schedule.py), so compute_sft_loss needs no changes there — only a labels tensor
that additionally carries the shifted loss_mask's zeros. These tests pin that construction.
"""

from __future__ import annotations

import torch

from radiance.train import compute_loss, compute_sft_loss


def test_all_ones_mask_matches_compute_loss():
    """With nothing masked, compute_sft_loss must be bit-identical to compute_loss — proof it's a
    strict generalization of the pretrain loss, not a divergent reimplementation."""
    torch.manual_seed(0)
    logits = torch.randn(3, 10, 40)
    input_ids = torch.randint(0, 40, (3, 10))
    loss_mask = torch.ones(3, 10, dtype=torch.long)

    lm_loss, z_loss = compute_loss(logits, input_ids)
    sft_lm_loss, sft_z_loss = compute_sft_loss(logits, input_ids, loss_mask)

    torch.testing.assert_close(lm_loss, sft_lm_loss, rtol=0, atol=0)
    torch.testing.assert_close(z_loss, sft_z_loss, rtol=0, atol=0)


def test_masked_positions_contribute_nothing_to_the_loss():
    """Perturbing logits at positions whose *target* is masked out must not move the loss at all —
    proof those positions are genuinely excluded, not merely down-weighted."""
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 20)
    input_ids = torch.randint(0, 20, (2, 8))
    # Mask out the second half of each row (prompt-like); only the tail is "assistant" (1).
    loss_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 2)

    lm_loss, z_loss = compute_sft_loss(logits, input_ids, loss_mask)

    perturbed_logits = logits.clone()
    # Position i's target is input_ids[i+1]; masking loss_mask[i+1] == 0 means position i itself
    # (predicting a masked target) must not matter. Corrupt exactly the positions whose target is
    # masked: shifted_mask[i] = loss_mask[i+1], so positions 0..2 (predicting masked targets 1..3)
    # are free to change; scramble them arbitrarily.
    perturbed_logits[:, :3, :] = torch.randn(2, 3, 20) * 100

    perturbed_lm_loss, perturbed_z_loss = compute_sft_loss(perturbed_logits, input_ids, loss_mask)

    torch.testing.assert_close(lm_loss, perturbed_lm_loss, rtol=0, atol=0)
    torch.testing.assert_close(z_loss, perturbed_z_loss, rtol=0, atol=0)


def test_unmasked_positions_match_hand_built_cross_entropy_reference():
    """The kept positions' loss must equal a hand-built F.cross_entropy reference that applies
    the same -100 labels (causal shift AND the prompt mask), independent of compute_sft_loss's
    own internals."""
    torch.manual_seed(2)
    logits = torch.randn(2, 6, 15)
    input_ids = torch.randint(0, 15, (2, 6))
    loss_mask = torch.tensor([[0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1]])

    lm_loss, _ = compute_sft_loss(logits, input_ids, loss_mask)

    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((2, 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((2, 1))], dim=1)
    labels = labels.masked_fill(shifted_mask == 0, -100)
    expected = torch.nn.functional.cross_entropy(logits.view(-1, 15), labels.view(-1), ignore_index=-100)

    torch.testing.assert_close(lm_loss, expected)


def test_gradient_flows_only_to_unmasked_positions():
    """Equal losses aren't enough — the gradient reaching the logits must be zero at every
    position whose target is masked out, and match cross_entropy's gradient elsewhere."""
    torch.manual_seed(3)
    logits = torch.randn(2, 7, 12, requires_grad=True)
    input_ids = torch.randint(0, 12, (2, 7))
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1, 1]])

    compute_sft_loss(logits, input_ids, loss_mask)[0].backward()
    ours = logits.grad.clone()

    ref_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((2, 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((2, 1))], dim=1)
    labels = labels.masked_fill(shifted_mask == 0, -100)
    torch.nn.functional.cross_entropy(
        ref_logits.view(-1, 12), labels.view(-1), ignore_index=-100
    ).backward()

    torch.testing.assert_close(ours, ref_logits.grad)

    # And directly: every masked target position's row of the gradient must be exactly zero.
    shifted_mask_bool = shifted_mask.bool()
    assert torch.all(ours[~shifted_mask_bool] == 0.0)
    assert torch.any(ours[shifted_mask_bool] != 0.0)
