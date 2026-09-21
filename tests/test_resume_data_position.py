"""Resuming a run: the data position, and checkpoint writes that survive being interrupted.

Two failures this pins down, both of which would only surface hours into a long run:

  * A resumed streaming disk-cache run replayed its data from the start. StreamingPackedDataset
    yields every cached block before the live stream, and with disk_cache_max_gb covering the run
    the cache holds everything trained so far — so a restart at hour 30 trained the remainder of
    the run on the first 30 hours' data again. The fix records the batches drawn in the checkpoint
    and skips them on resume.
  * The old checkpoint was deleted *before* the new one was written, so a save that died mid-write
    left no checkpoint at all and resume_from: auto quietly restarted from step 0.

Every document here is exactly one packed block (15 words + eos at seq_len 16) of unique words,
and word "wN" tokenizes to id N + 2, so a block's first token names it — identical across
DataLoader worker processes, which each hold their own tokenizer copy.
"""

from __future__ import annotations

import shutil

import pytest
import torch
from datasets import DatasetDict, IterableDataset

import radiance.checkpointing as ckpt_mod
import radiance.data as d
import radiance.train as train_mod
from radiance.checkpointing import find_resume_checkpoint, prune_checkpoints, save_checkpoint
from radiance.config import Config, DataConfig, ModelConfig, TrainConfig, WandbConfig
from radiance.model import DenseTransformer
from radiance.optim import build_optimizer

WORDS_PER_DOC = 15  # + eos = one seq_len-16 block per document
N_DOCS = 240


class _IndexTok:
    """'wN' -> N + 2 (0 unused, 1 = eos). Stateless, so every worker process agrees on ids."""

    eos_token_id = 1

    def __init__(self, n_words: int):
        self._len = n_words + 2

    def __len__(self):
        return self._len

    def __call__(self, texts):
        return {"input_ids": [[int(w[1:]) + 2 for w in t.split()] for t in texts]}


def _docs(n_docs: int) -> list[str]:
    return [" ".join(f"w{i * WORDS_PER_DOC + j}" for j in range(WORDS_PER_DOC)) for i in range(n_docs)]


@pytest.fixture
def fake_stream(monkeypatch):
    """A single-corpus streaming dataset with 4 data shards. Two workers need more than 2: HF
    splits an IterableDataset across DataLoader workers itself, *on top of* the raw.shard() in
    StreamingPackedDataset, so at 2 shards the second worker is left with none."""
    docs = _docs(N_DOCS)
    q = N_DOCS // 4
    shards = [docs[i * q : (i + 1) * q] for i in range(4)]

    def gen(shards):
        for shard in shards:
            for text in shard:
                yield {"text": text}

    def fake(name, streaming=False, split=None, **_kw):
        assert streaming
        ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
        return ds if split else DatasetDict({"train": ds})

    monkeypatch.setattr(d, "load_dataset", fake)


def _cfg(cache_dir, num_workers: int, shuffle_buffer_size: int = 1, **train) -> Config:
    return Config(
        data=DataConfig(
            dataset="test/stream",
            seq_len=16,
            streaming=True,
            cache_dir=str(cache_dir),
            disk_cache_max_gb=1.0,
            disk_cache_shard_size=3,  # small and odd, so skips land mid-shard as well as on a boundary
            shuffle_buffer_size=shuffle_buffer_size,  # 1 = order preserved, so a skip can be exact
            num_workers=num_workers,
        ),
        train=TrainConfig(batch_size=2, device="cpu", compile=False, auto_batch_size=False, **train),
    )


def _draw(cfg: Config, n_batches: int, resume_batches: int | None = None) -> list[int]:
    """Block ids (first token) of the first n_batches a fresh loader yields."""
    loader, _ = d.build_dataloaders(cfg, _IndexTok(N_DOCS * WORDS_PER_DOC))
    if resume_batches is not None:
        loader.dataset.set_resume_position(resume_batches)
    ids: list[int] = []
    for i, batch in enumerate(loader):
        if i == n_batches:
            break
        ids.extend(batch["input_ids"][:, 0].tolist())
    return ids


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("drop_cache", [False, True], ids=["cached", "cache-deleted"])
def test_resume_continues_the_stream_exactly(fake_stream, tmp_path, num_workers, drop_cache):
    """With an order-preserving shuffle buffer, every worker's stream picks up exactly where the
    first run left it: nothing replayed, nothing skipped. 'cache-deleted' forces the whole skip
    through the live-stream path instead of the cached-shard fast path.

    Compared per worker, not as one sequence: a fresh DataLoader always starts its round-robin at
    worker 0, so after a resume the workers' batches interleave from a different starting worker.
    Which blocks each worker yields — the property that decides what gets trained on — is exact."""
    before, after = 7, 9  # odd, so the two workers' shares differ by one batch
    workers = max(num_workers, 1)
    bs = 2
    ref = _draw(_cfg(tmp_path / "ref", num_workers), 40)
    ref_by_worker = [
        [x for i, x in enumerate(ref) if (i // bs) % workers == w] for w in range(workers)
    ]

    cache = tmp_path / "cache"
    first = _draw(_cfg(cache, num_workers), before)
    if drop_cache:
        shutil.rmtree(cache)
    resumed = _draw(_cfg(cache, num_workers), after, resume_batches=before)

    assert not set(first) & set(resumed), "resumed run replayed blocks the first run trained on"
    for stream in ref_by_worker:
        got = [x for x in first + resumed if x in set(stream)]
        assert got == stream[: len(got)], "a worker's stream skipped or reordered blocks across the resume"
    assert sum(len([x for x in first + resumed if x in set(s)]) for s in ref_by_worker) == (before + after) * bs


def test_without_a_resume_position_the_cache_replays_from_the_start(fake_stream, tmp_path):
    """The failure the fix exists for — pins that the test above is not vacuous."""
    first = _draw(_cfg(tmp_path, 0), 10)
    again = _draw(_cfg(tmp_path, 0), 10)
    assert again == first


def test_resume_overlap_is_bounded_by_the_shuffle_buffer(fake_stream, tmp_path):
    """With a real shuffle buffer the skip is applied pre-shuffle, so it is exact only up to the
    buffer: at most shuffle_buffer_size already-trained blocks can recur."""
    buffer = 8
    first = _draw(_cfg(tmp_path, 0, shuffle_buffer_size=buffer), 20)
    resumed = _draw(_cfg(tmp_path, 0, shuffle_buffer_size=buffer), 20, resume_batches=20)
    assert len(set(first) & set(resumed)) <= buffer


# --- checkpoint writes ------------------------------------------------------------------------


def _save(path, step: int, batches: int = 0) -> None:
    cfg = Config(model=ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=16), train=TrainConfig(device="cpu"))
    model = DenseTransformer(cfg.model, vocab_size=64)
    optimizer = build_optimizer(model, cfg, "cpu")
    scheduler = train_mod.build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    save_checkpoint(path, model, optimizer, scheduler, scaler, step, cfg, train_batches_drawn=batches)


def test_interrupted_save_leaves_the_previous_checkpoint_resumable(tmp_path, monkeypatch):
    _save(tmp_path / "step_2.pt", step=2)

    real_save = torch.save

    def dies_mid_write(obj, f, *a, **kw):
        f.write(b"truncated")
        raise OSError("No space left on device")

    monkeypatch.setattr(ckpt_mod.torch, "save", dies_mid_write)
    with pytest.raises(OSError):
        _save(tmp_path / "step_4.pt", step=4)
    monkeypatch.setattr(ckpt_mod.torch, "save", real_save)

    assert not (tmp_path / "step_4.pt").exists()
    cfg = Config(train=TrainConfig(output_dir=str(tmp_path), resume_from="auto"))
    resume = find_resume_checkpoint(cfg)
    assert resume == tmp_path / "step_2.pt"
    assert torch.load(resume, weights_only=False)["step"] == 2

    # The next good save, then prune, clears both the old checkpoint and the stale .tmp.
    _save(tmp_path / "step_6.pt", step=6)
    prune_checkpoints(tmp_path, keep=tmp_path / "step_6.pt")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_6.pt"]


# --- end to end through train() ---------------------------------------------------------------


class _FakeWandb:
    def __init__(self):
        self.run = None

    def init(self, **kwargs):
        self.run = object()

    def log(self, data, step=None):
        pass

    def finish(self):
        self.run = None


def test_train_records_and_restores_the_data_position(fake_stream, tmp_path, monkeypatch, capsys):
    tok = _IndexTok(N_DOCS * WORDS_PER_DOC)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg: tok)
    monkeypatch.setattr(train_mod, "wandb", _FakeWandb())

    restored: list[int] = []
    real_set = d.StreamingPackedDataset.set_resume_position

    def spy(self, batches_drawn):
        restored.append(batches_drawn)
        real_set(self, batches_drawn)

    monkeypatch.setattr(d.StreamingPackedDataset, "set_resume_position", spy)

    def run(max_steps: int) -> None:
        cfg = _cfg(
            tmp_path / "cache", 0,
            max_steps=max_steps, grad_accum_steps=2, save_every=2, eval_every=10_000, log_every=10_000,
            output_dir=str(tmp_path / "out"), resume_from="auto",
        )
        cfg.model = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=16, vocab_pad_multiple=1)
        cfg.wandb = WandbConfig(mode="disabled")
        train_mod.train(cfg)

    run(max_steps=4)
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["step_4.pt"]
    assert torch.load(out / "step_4.pt", weights_only=False)["train_batches_drawn"] == 8
    assert restored == []

    run(max_steps=6)
    assert restored == [8]
    assert "resuming data stream after 8 batches" in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == ["step_6.pt"]
    assert torch.load(out / "step_6.pt", weights_only=False)["train_batches_drawn"] == 12
