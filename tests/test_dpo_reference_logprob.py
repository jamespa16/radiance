"""data._add_reference_logprobs: the one-time precompute pass that lets DPO training hold only the
policy model in memory. Cross-checks the precompute pipeline (load checkpoint -> forward -> cache)
against the primitive it's built from (model.sequence_logprob_sum), and pins _dpo_cache_path's
reference-checkpoint-identity keying (mtime+size, not content) that invalidates the cache when the
reference model changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from datasets import Dataset, DatasetDict

from radiance.config import Config, DPOConfig, DataConfig
from radiance.dpo_data import _add_reference_logprobs, _dpo_cache_path, dpo_cache_exists
from radiance.model import DenseTransformer, checkpoint_param_bytes, sequence_logprob_sum
from radiance.optim import build_optimizer
from radiance.train import build_lr_scheduler
from radiance.batching import dpo_reference_reserve_bytes
from radiance.checkpointing import save_checkpoint
from tests.conftest import TINY_VOCAB


class _FakeTokenizer:
    # Deliberately outside [1, TINY_VOCAB) so it never collides with the synthetic random ids
    # below — on CPU doc_attention_mask always falls back to plain SDPA anyway (see
    # model/attention.py/CLAUDE.md), so this is belt-and-suspenders, not load-bearing for this test.
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


def test_add_reference_logprobs_rejects_mismatched_tokenizer(tmp_path, tiny_cfg):
    # A reference checkpoint saved under a different data.tokenizer would have its log-probs
    # gathered over a mismatched vocabulary: numerically-plausible, semantically meaningless,
    # and silent. It must raise here, before the (worse) loss starts looking plausible. The
    # happy path - same tokenizer on both sides - is the first test above, which saves the
    # reference under `cfg` and builds the DPO cfg from the same `cfg.data`.
    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    raw_model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    raw_model.eval()

    optimizer = build_optimizer(raw_model, cfg, "cpu")
    scheduler = build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ckpt_path = tmp_path / "reference.pt"
    save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, scaler, step=0, cfg=cfg)

    dpo_cfg = Config(
        model=cfg.model,
        data=DataConfig(seq_len=cfg.data.seq_len, tokenizer="some/other-tokenizer"),
        train=cfg.train,
        dpo=DPOConfig(enabled=True, dataset="unused", reference_checkpoint=str(ckpt_path)),
    )
    split, *_ = _make_packed_split(cfg.data.seq_len)

    with pytest.raises(ValueError, match="mismatched vocabulary"):
        _add_reference_logprobs(DatasetDict({"train": split}), dpo_cfg, _FakeTokenizer(), device="cpu")


class _OOMSimulator:
    """Wraps the real reference model and turns CUDA OOM into something testable on CPU:
    calls with more than `oom_above` rows raise torch.cuda.OutOfMemoryError (the exception the
    precompute's backoff catches), smaller calls pass through to the real model unchanged — so
    the values computed on the retried, chunked path are the real ones the happy-path test
    cross-checks. Records every call's row count so the backoff is observable, not just
    crash-free."""

    def __init__(self, real, oom_above: int):
        self.real = real
        self.oom_above = oom_above
        self.calls: list[int] = []

    @property
    def cfg(self):
        # The precompute reads (and turns doc_attention_mask off on) the reference model's config;
        # delegating rather than copying keeps that write landing on the real model, which is the
        # one that actually runs the forward.
        return self.real.cfg

    def __call__(self, x):
        n = x.size(0)
        self.calls.append(n)
        if n > self.oom_above:
            raise torch.cuda.OutOfMemoryError(f"simulated: {n} rows do not fit")
        return self.real(x)


def _save_reference_checkpoint(tmp_path, cfg) -> Path:
    raw_model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    raw_model.eval()
    optimizer = build_optimizer(raw_model, cfg, "cpu")
    scheduler = build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ckpt_path = tmp_path / "reference.pt"
    save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, scaler, step=0, cfg=cfg)
    return ckpt_path


def test_add_reference_logprobs_backs_off_on_oom(tmp_path, tiny_cfg, monkeypatch):
    # An OOM here used to crash the run at startup: this pass runs before the training loop, so
    # the step loop's micro_chunk_size/CPU-offload backoff never sees it. The fix is the same
    # idiom locally — process the fetched batch in chunks, halve on OOM, retry. Rows are
    # independent, so the chunked path must be bit-for-bit the reference values.
    import radiance.dpo_data as dpo_data_mod

    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    ckpt_path = _save_reference_checkpoint(tmp_path, cfg)
    split, chosen_ids, chosen_mask, rejected_ids, rejected_mask = _make_packed_split(
        cfg.data.seq_len, n=8
    )
    packed = DatasetDict({"train": split})
    dpo_cfg = Config(
        model=cfg.model,
        data=cfg.data,
        train=cfg.train,
        dpo=DPOConfig(
            enabled=True, dataset="unused", reference_checkpoint=str(ckpt_path), reference_batch_size=8
        ),
    )

    real_load = dpo_data_mod.load_transformer_from_checkpoint
    simulators: list[_OOMSimulator] = []

    def fake_load(*args, **kwargs):
        ref, ref_cfg = real_load(*args, **kwargs)
        simulators.append(_OOMSimulator(ref, oom_above=8))
        return simulators[-1], ref_cfg

    monkeypatch.setattr(dpo_data_mod, "load_transformer_from_checkpoint", fake_load)

    result = _add_reference_logprobs(packed, dpo_cfg, _FakeTokenizer(), device="cpu")

    # The precompute loads exactly one model. First attempt on the fetched 8-row batch (16 rows:
    # both sides) OOMed; after the halved retry every call carries at most 8 rows (a 4-row chunk,
    # both sides) — two of them to cover the 8 rows.
    assert len(simulators) == 1
    calls = simulators[0].calls
    # The first attempt carried the whole fetched batch (16 rows: both sides) and OOMed; after
    # the halved retry the 8 rows are covered by two <=8-row calls — nothing bigger than a
    # half-batch is ever attempted again (sticky backoff).
    assert calls[0] == 16
    assert len(calls) == 3 and sum(calls[1:]) == 16 and all(call <= 8 for call in calls[1:])

    # The expected values come from the *reference checkpoint's* weights (saved before backoff),
    # so load it once more, un-wrapped, for the cross-check — the happy-path test above does the same.
    ref_model, _ = real_load(str(ckpt_path), "cpu", eos_id=0)
    ref_model.eval()
    with torch.no_grad():
        expected_chosen = sequence_logprob_sum(ref_model(chosen_ids).logits, chosen_ids, chosen_mask)
        expected_rejected = sequence_logprob_sum(
            ref_model(rejected_ids).logits, rejected_ids, rejected_mask
        )

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


def test_add_reference_logprobs_reraises_when_even_one_row_does_not_fit(tmp_path, tiny_cfg, monkeypatch):
    # The backoff has a floor of one row; if a single row pair still OOMs there is nothing left
    # to halve, so it must re-raise the OOM (with a note) rather than spin or silently drop rows.
    import radiance.dpo_data as dpo_data_mod

    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    ckpt_path = _save_reference_checkpoint(tmp_path, cfg)
    split, *_ = _make_packed_split(cfg.data.seq_len, n=4)
    packed = DatasetDict({"train": split})
    dpo_cfg = Config(
        model=cfg.model,
        data=cfg.data,
        train=cfg.train,
        dpo=DPOConfig(
            enabled=True, dataset="unused", reference_checkpoint=str(ckpt_path), reference_batch_size=4
        ),
    )

    real_load = dpo_data_mod.load_transformer_from_checkpoint

    def fake_load(*args, **kwargs):
        ref, ref_cfg = real_load(*args, **kwargs)
        return _OOMSimulator(ref, oom_above=-1), ref_cfg  # -1: every call, down to one row

    monkeypatch.setattr(dpo_data_mod, "load_transformer_from_checkpoint", fake_load)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _add_reference_logprobs(packed, dpo_cfg, _FakeTokenizer(), device="cpu")


def test_add_reference_logprobs_resets_chunk_size_per_split(tmp_path, tiny_cfg, monkeypatch):
    # A backoff forced by one split's data must not handicap the next split, which gets its own
    # fresh, unstressed DataLoader loop and may never hit the same OOM. oom_above=8 only ever
    # trips on the first (16-row, both sides) call of the *first* split processed; if chunk_size
    # stayed sticky across splits, the second split's first call would arrive already halved.
    import radiance.dpo_data as dpo_data_mod

    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    ckpt_path = _save_reference_checkpoint(tmp_path, cfg)
    train_split, *_ = _make_packed_split(cfg.data.seq_len, n=8, seed=0)
    val_split, *_ = _make_packed_split(cfg.data.seq_len, n=8, seed=1)
    packed = DatasetDict({"train": train_split, "validation": val_split})
    dpo_cfg = Config(
        model=cfg.model,
        data=cfg.data,
        train=cfg.train,
        dpo=DPOConfig(
            enabled=True, dataset="unused", reference_checkpoint=str(ckpt_path), reference_batch_size=8
        ),
    )

    real_load = dpo_data_mod.load_transformer_from_checkpoint
    simulators: list[_OOMSimulator] = []

    def fake_load(*args, **kwargs):
        ref, ref_cfg = real_load(*args, **kwargs)
        # OOMs only on the very first call ever made (the train split's first attempt); every
        # later call, including the validation split's first, must be allowed through.
        simulators.append(_OOMSimulator(ref, oom_above=8))
        return simulators[-1], ref_cfg

    monkeypatch.setattr(dpo_data_mod, "load_transformer_from_checkpoint", fake_load)

    _add_reference_logprobs(packed, dpo_cfg, _FakeTokenizer(), device="cpu")

    # Exactly one reference model is loaded and shared across both splits, so calls accumulate in
    # split order: train's 16-row call OOMs, backs off to two 8-row calls (as in the backoff test
    # above) — then validation's *first* call must be the full un-backed-off 16 rows again, not a
    # residual halved size carried over from train. Had chunk_size stayed sticky, calls[3] would
    # be 8 (from a chunk_size of 4) instead, and it would never OOM a second time.
    assert simulators[0].calls[:3] == [16, 8, 8]
    assert simulators[0].calls[3] == 16


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


def _dpo_cfg(tmp_path, ref: Path) -> Config:
    # Explicit cache_dir under tmp_path so these tests (which touch the filesystem, unlike the
    # pure path-comparison tests above) write nothing into the repo.
    return Config(
        dpo=DPOConfig(
            enabled=True,
            dataset="foo/bar",
            reference_checkpoint=str(ref),
            cache_dir=str(tmp_path / "cache"),
        )
    )


def test_dpo_cache_path_reuses_cache_when_reference_checkpoint_missing(tmp_path):
    # The checkpoint is only needed to COMPUTE the reference log-probs, which run once and land
    # in the cache; archiving the checkpoint afterwards must not break a run that only consumes
    # the cached dataset. With the file gone its identity (path+mtime+size) is no longer
    # computable, so _dpo_cache_path resolves the cache by the single ref-* entry under the base
    # digest instead.
    ref = tmp_path / "ref.pt"
    ref.write_bytes(b"x" * 100)

    cfg = _dpo_cfg(tmp_path, ref)
    path = _dpo_cache_path(cfg)
    path.mkdir(parents=True)  # a cache built while the checkpoint was present
    ref.unlink()

    assert _dpo_cache_path(cfg) == path


def test_dpo_cache_path_raises_when_reference_checkpoint_missing_and_no_cache(tmp_path):
    # File gone and no cache built: the reference log-probs can neither be loaded nor computed.
    ref = tmp_path / "ref.pt"
    with pytest.raises(ValueError, match="does not exist"):
        _dpo_cache_path(_dpo_cfg(tmp_path, ref))


def test_dpo_cache_path_raises_ambiguous_when_reference_checkpoint_missing(tmp_path):
    # Two different reference checkpoints used with the same dataset/config leave two caches
    # under the one base digest. With the checkpoint archived, which one of them was built
    # against it is not knowable, so _dpo_cache_path must raise rather than reuse a cache that
    # may hold the wrong reference model's log-probs.
    ref_a = tmp_path / "a.pt"
    ref_b = tmp_path / "b.pt"
    ref_a.write_bytes(b"x" * 100)
    ref_b.write_bytes(b"x" * 200)

    _dpo_cache_path(_dpo_cfg(tmp_path, ref_a)).mkdir(parents=True)
    _dpo_cache_path(_dpo_cfg(tmp_path, ref_b)).mkdir(parents=True)
    ref_a.unlink()

    with pytest.raises(ValueError, match="not possible to tell which"):
        _dpo_cache_path(_dpo_cfg(tmp_path, ref_a))


def test_checkpoint_param_bytes_matches_real_state_dict_bytes(tmp_path, tiny_cfg):
    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    raw_model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    ckpt_path = _save_reference_checkpoint(tmp_path, cfg)

    expected = sum(t.numel() * t.element_size() for t in raw_model.state_dict().values())
    assert checkpoint_param_bytes(str(ckpt_path)) == expected


def test_dpo_cache_exists_false_without_cache_dir():
    cfg = Config(dpo=DPOConfig(enabled=True, dataset="foo/bar", reference_checkpoint="unused.pt"))
    assert dpo_cache_exists(cfg) is False


def test_dpo_cache_exists_true_once_the_cache_path_is_built(tmp_path):
    ref = tmp_path / "ref.pt"
    ref.write_bytes(b"x" * 100)
    cfg = _dpo_cfg(tmp_path, ref)

    assert dpo_cache_exists(cfg) is False
    _dpo_cache_path(cfg).mkdir(parents=True)
    assert dpo_cache_exists(cfg) is True


def test_dpo_cache_exists_false_rather_than_raising_when_ambiguous(tmp_path):
    # dpo_cache_exists is a batch-sizing probe, not the real cache lookup — an ambiguous or
    # missing reference checkpoint should read as "no cache" here rather than propagate
    # _dpo_cache_path's raise; build_dpo_dataloaders still raises it on the real build path.
    ref_a = tmp_path / "a.pt"
    ref_b = tmp_path / "b.pt"
    ref_a.write_bytes(b"x" * 100)
    ref_b.write_bytes(b"x" * 200)
    _dpo_cache_path(_dpo_cfg(tmp_path, ref_a)).mkdir(parents=True)
    _dpo_cache_path(_dpo_cfg(tmp_path, ref_b)).mkdir(parents=True)
    ref_a.unlink()

    assert dpo_cache_exists(_dpo_cfg(tmp_path, ref_a)) is False


def test_dpo_reference_reserve_bytes_zero_when_dpo_disabled():
    assert dpo_reference_reserve_bytes(Config()) == 0


def test_dpo_reference_reserve_bytes_zero_on_cache_hit(tmp_path, tiny_cfg):
    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    ckpt_path = _save_reference_checkpoint(tmp_path, cfg)
    dpo_cfg = _dpo_cfg(tmp_path, ckpt_path)
    _dpo_cache_path(dpo_cfg).mkdir(parents=True)  # simulate an already-built cache

    assert dpo_reference_reserve_bytes(dpo_cfg) == 0


def test_dpo_reference_reserve_bytes_equals_checkpoint_param_bytes_on_cache_miss(tmp_path, tiny_cfg):
    cfg = tiny_cfg(_train=dict(optimizer="adamw"))
    ckpt_path = _save_reference_checkpoint(tmp_path, cfg)
    dpo_cfg = _dpo_cfg(tmp_path, ckpt_path)

    assert dpo_reference_reserve_bytes(dpo_cfg) == checkpoint_param_bytes(str(ckpt_path))
