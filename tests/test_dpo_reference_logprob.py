"""data._add_reference_logprobs: the one-time precompute pass that lets DPO training hold only the
policy model in memory. Cross-checks the precompute pipeline (load checkpoint -> forward -> cache)
against the primitive it's built from (model.sequence_logprob_sum), and pins _dpo_cache_path's
reference-checkpoint-identity keying (mtime+size, not content) that invalidates the cache when the
reference model changes.
"""

from __future__ import annotations

import os

import torch
from datasets import Dataset, DatasetDict

from radiance.config import Config, DPOConfig
from radiance.data import _add_reference_logprobs, _dpo_cache_path
from radiance.model import DenseTransformer, sequence_logprob_sum
from radiance.optim import build_optimizer
from radiance.train import build_lr_scheduler, save_checkpoint
from tests.conftest import TINY_VOCAB


class _FakeTokenizer:
    # Deliberately outside [1, TINY_VOCAB) so it never collides with the synthetic random ids
    # below — on CPU doc_attention_mask always falls back to plain SDPA anyway (see
    # model.py/CLAUDE.md), so this is belt-and-suspenders, not load-bearing for this test.
    eos_token_id = 0


def _make_packed_split(seq_len: int, n: int = 6, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    chosen_ids = torch.randint(1, TINY_VOCAB, (n, seq_len), generator=gen)
    rejected_ids = torch.randint(1, TINY_VOCAB, (n, seq_len), generator=gen)
    chosen_mask = torch.zeros(n, seq_len, dtype=torch.long)
    chosen_mask[:, seq_len // 2 :] = 1
    rejected_mask = torch.zeros(n, seq_len, dtype=torch.long)
    rejected_mask[:, seq_len // 2 :] = 1
    split = Dataset.from_dict(
        {
            "chosen_input_ids": chosen_ids.tolist(),
            "chosen_loss_mask": chosen_mask.tolist(),
            "rejected_input_ids": rejected_ids.tolist(),
            "rejected_loss_mask": rejected_mask.tolist(),
        }
    )
    # _add_reference_logprobs' real caller (_load_or_build_dpo_packed, via
    # _tokenize_and_filter_dpo) always hands it an already torch-formatted dataset — match that
    # here rather than the plain-Python-list default Dataset.from_dict produces.
    split.set_format(type="torch")
    return split, chosen_ids, chosen_mask, rejected_ids, rejected_mask


def test_add_reference_logprobs_matches_direct_sequence_logprob_sum(tmp_path, tiny_cfg):
    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    raw_model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    raw_model.eval()

    optimizer = build_optimizer(raw_model, cfg, "cpu")
    scheduler = build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ckpt_path = tmp_path / "reference.pt"
    save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, scaler, step=0, cfg=cfg)

    split, chosen_ids, chosen_mask, rejected_ids, rejected_mask = _make_packed_split(cfg.data.seq_len)
    packed = DatasetDict({"train": split})

    dpo_cfg = Config(
        model=cfg.model,
        data=cfg.data,
        train=cfg.train,
        dpo=DPOConfig(enabled=True, dataset="unused", reference_checkpoint=str(ckpt_path)),
    )

    result = _add_reference_logprobs(packed, dpo_cfg, _FakeTokenizer(), device="cpu")

    with torch.no_grad():
        expected_chosen = sequence_logprob_sum(raw_model(chosen_ids).logits, chosen_ids, chosen_mask)
        expected_rejected = sequence_logprob_sum(raw_model(rejected_ids).logits, rejected_ids, rejected_mask)

    torch.testing.assert_close(
        torch.tensor(result["train"]["ref_chosen_logprob"], dtype=torch.float32),
        expected_chosen,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        torch.tensor(result["train"]["ref_rejected_logprob"], dtype=torch.float32),
        expected_rejected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_dpo_cache_path_differs_with_reference_checkpoint_size(tmp_path):
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"x" * 100)
    ckpt_b = tmp_path / "b.pt"
    ckpt_b.write_bytes(b"x" * 200)

    cfg_a = Config(dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint=str(ckpt_a)))
    cfg_b = Config(dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint=str(ckpt_b)))

    assert _dpo_cache_path(cfg_a) != _dpo_cache_path(cfg_b)


def test_dpo_cache_path_differs_with_reference_checkpoint_mtime_only(tmp_path):
    ckpt = tmp_path / "ref.pt"
    ckpt.write_bytes(b"x" * 100)

    cfg = Config(dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint=str(ckpt)))
    path_before = _dpo_cache_path(cfg)

    st = ckpt.stat()
    os.utime(ckpt, (st.st_atime, st.st_mtime + 5))  # same size, different mtime, no real-clock reliance
    path_after = _dpo_cache_path(cfg)

    assert path_before != path_after
