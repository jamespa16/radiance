"""evaluate()'s micro_chunk_size handling: a backoffed run's eval forward must carry exactly the
per-forward chunk size the step loop already validated (notably DPO's concatenated chosen+rejected
batch), while the val/loss value must remain the un-chunked batch mean."""

from __future__ import annotations

from typing import Any

import torch

from radiance.evaluation import evaluate
from radiance.losses import compute_dpo_loss_from_logits, compute_loss, compute_sft_loss


class _FakeModel:
    """Deterministic stand-in for DenseTransformer: logits depend only on each row's own ids, so a
    chunked forward over a row produces exactly the per-row logits the un-chunked forward would."""

    def __init__(self, vocab: int = 5) -> None:
        self.vocab = vocab
        self.calls: list[int] = []

    def eval(self) -> Any:
        return self

    def train(self, mode: bool = True) -> Any:
        return self

    def __call__(self, x: torch.Tensor) -> Any:
        self.calls.append(x.size(0))
        base = torch.arange(self.vocab, dtype=torch.float32)
        logits = base[None, None, :] * ((x + 1).float() / 10.0)[:, :, None]
        return type("Output", (), {"logits": logits})()


def test_evaluate_pretrain_single_chunk_is_one_forward():
    ids = torch.randint(0, 5, (4, 8))
    model: Any = _FakeModel()

    evaluate(model, [{"input_ids": ids}], "cpu", "cpu", torch.float32, loss_fn=compute_loss)

    assert model.calls == [4]


def test_evaluate_pretrain_multi_chunk_matches_single_forward():
    ids = torch.randint(0, 5, (4, 8))
    reference_model: Any = _FakeModel()
    reference = evaluate(
        reference_model, [{"input_ids": ids}], "cpu", "cpu", torch.float32, loss_fn=compute_loss
    )
    assert reference_model.calls == [4]

    chunked_model: Any = _FakeModel()
    chunked = evaluate(
        chunked_model,
        [{"input_ids": ids}],
        "cpu",
        "cpu",
        torch.float32,
        loss_fn=compute_loss,
        micro_chunk_size=1,
    )

    assert chunked_model.calls == [1, 1, 1, 1]
    assert abs(chunked - reference) < 1e-5


def test_evaluate_pretrain_kept_position_reweighting_across_uneven_batches():
    batches = [
        {"input_ids": torch.randint(0, 5, (4, 8))},
        {"input_ids": torch.randint(0, 5, (3, 5))},
    ]
    reference_model: Any = _FakeModel()
    reference = evaluate(
        reference_model, batches, "cpu", "cpu", torch.float32, loss_fn=compute_loss
    )

    chunked_model: Any = _FakeModel()
    chunked = evaluate(
        chunked_model,
        batches,
        "cpu",
        "cpu",
        torch.float32,
        loss_fn=compute_loss,
        micro_chunk_size=2,
    )

    assert chunked_model.calls == [2, 2, 2, 1]
    assert abs(chunked - reference) < 1e-5


def test_evaluate_sft_multi_chunk_matches_single_forward():
    ids = torch.randint(0, 5, (4, 8))
    loss_mask = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 1],
        ]
    )
    batch = {"input_ids": ids, "loss_mask": loss_mask}

    reference_model: Any = _FakeModel()
    reference = evaluate(
        reference_model,
        [batch],
        "cpu",
        "cpu",
        torch.float32,
        loss_fn=compute_sft_loss,
        sft=True,
    )
    assert reference_model.calls == [4]

    chunked_model: Any = _FakeModel()
    chunked = evaluate(
        chunked_model,
        [batch],
        "cpu",
        "cpu",
        torch.float32,
        loss_fn=compute_sft_loss,
        sft=True,
        micro_chunk_size=2,
    )

    assert chunked_model.calls == [2, 2]
    assert abs(chunked - reference) < 1e-5


def test_evaluate_dpo_multi_chunk_matches_single_forward():
    chosen_ids = torch.randint(0, 5, (4, 8))
    rejected_ids = torch.randint(0, 5, (4, 8))
    batch = {
        "chosen_input_ids": chosen_ids,
        "chosen_loss_mask": torch.ones_like(chosen_ids),
        "rejected_input_ids": rejected_ids,
        "rejected_loss_mask": torch.ones_like(rejected_ids),
        "ref_chosen_logprob": torch.randn(4),
        "ref_rejected_logprob": torch.randn(4),
    }

    reference_model: Any = _FakeModel()
    reference = evaluate(
        reference_model,
        [batch],
        "cpu",
        "cpu",
        torch.float32,
        loss_fn=compute_dpo_loss_from_logits,
        dpo=True,
        dpo_beta=0.25,
    )
    # 4 chosen + 4 rejected are concatenated into one forward.
    assert reference_model.calls == [8]

    chunked_model: Any = _FakeModel()
    chunked = evaluate(
        chunked_model,
        [batch],
        "cpu",
        "cpu",
        torch.float32,
        loss_fn=compute_dpo_loss_from_logits,
        dpo=True,
        dpo_beta=0.25,
        micro_chunk_size=2,
    )

    # Each DPO chunk forwards 2 chosen + 2 rejected = 4 rows, never the un-backoffed 8.
    assert chunked_model.calls == [4, 4]
    assert abs(chunked - reference) < 1e-5
