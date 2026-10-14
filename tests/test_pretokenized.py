"""radiance-pretokenize and the data.pretokenized_dir loader.

What this pins, in the order a real run depends on it:

  * The held-out set is exactly data.py's carve (the first eval_split_size documents), and none of
    those documents reach train even though the shuffled train stream passes through their file.
  * An interrupted pretokenize, re-run, writes byte-identical files to an uninterrupted one — the
    point of the tool is surviving the kind of outage that stopped V1.
  * Training resumes at the exact block: no block repeated, none skipped, across an epoch boundary.
  * train() records the position and restores it through the sampler.

Every document is one packed 16-token block (15 words + eos) of unique words, and word "wN"
tokenizes to id N + 2, so a block's first token names the document it came from.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from datasets import DatasetDict, IterableDataset

import radiance.data as d
import radiance.pretokenize as pt
import radiance.train as train_mod
from radiance.config import Config, DataConfig, ModelConfig, TrainConfig, WandbConfig

WORDS_PER_DOC = 15
SEQ_LEN = 16
N_DOCS = 400
N_FILES = 4
EVAL_DOCS = 20


class _IndexTok:
    eos_token_id = 1

    def __len__(self):
        return N_DOCS * WORDS_PER_DOC + 2

    def __call__(self, texts):
        return {"input_ids": [[int(w[1:]) + 2 for w in t.split()] for t in texts]}


def _doc_id(first_token: int) -> int:
    return (first_token - 2) // WORDS_PER_DOC


def _docs() -> list[str]:
    return [" ".join(f"w{i * WORDS_PER_DOC + j}" for j in range(WORDS_PER_DOC)) for i in range(N_DOCS)]


@pytest.fixture
def fake_corpus(monkeypatch):
    docs = _docs()
    per_file = N_DOCS // N_FILES
    files = [docs[i * per_file : (i + 1) * per_file] for i in range(N_FILES)]

    def gen(files):
        for f in files:
            for text in f:
                yield {"text": text}

    def fake(name, streaming=False, split=None, **_kw):
        assert streaming
        ds = IterableDataset.from_generator(gen, gen_kwargs={"files": files})
        return ds if split else DatasetDict({"train": ds})

    monkeypatch.setattr(pt, "load_dataset", fake)
    monkeypatch.setattr(pt, "build_tokenizer", lambda cfg: _IndexTok())


def _cfg(out, **train) -> Config:
    return Config(
        data=DataConfig(
            dataset="test/corpus", tokenizer="test-tok", seq_len=SEQ_LEN, eval_split_size=EVAL_DOCS,
            shuffle_buffer_size=16, num_workers=0, pretokenized_dir=str(out),
        ),
        train=TrainConfig(batch_size=4, device="cpu", compile=False, auto_batch_size=False, **train),
    )


def _blocks(path, dtype="uint16") -> list[int]:
    ids = np.fromfile(path, dtype=dtype)
    return [_doc_id(int(b[0])) for b in ids.reshape(-1, SEQ_LEN)]


def test_val_is_the_carve_and_never_reaches_train(fake_corpus, tmp_path):
    meta = pt.pretokenize(_cfg(tmp_path), train_tokens=10**9, num_proc=1)
    val_docs = _blocks(tmp_path / "val.bin")
    assert val_docs == list(range(EVAL_DOCS)), "val must be the first eval_split_size documents, in order"
    train_docs = _blocks(tmp_path / meta["train"]["files"][0])
    assert not set(train_docs) & set(val_docs)
    assert sorted(train_docs) == list(range(EVAL_DOCS, N_DOCS)), "every other document, exactly once"
    assert train_docs != sorted(train_docs), "train order should be shuffled, not stream order"
    assert meta["train"]["excluded_val_docs"] == EVAL_DOCS


def test_quota_stops_early(fake_corpus, tmp_path):
    meta = pt.pretokenize(_cfg(tmp_path), train_tokens=50 * SEQ_LEN, num_proc=1)
    # Stops at the first tokenize batch that reaches the quota — not before it.
    assert 50 * SEQ_LEN <= meta["train"]["tokens"] < N_DOCS * SEQ_LEN


def test_interrupted_pretokenize_resumes_to_identical_files(fake_corpus, tmp_path, monkeypatch):
    ref = pt.pretokenize(_cfg(tmp_path / "ref"), train_tokens=10**9, num_proc=1)
    ref_bytes = (tmp_path / "ref" / ref["train"]["files"][0]).read_bytes()

    monkeypatch.setattr(pt, "_TOKENIZE_BATCH", 8)
    monkeypatch.setattr(pt, "_FLUSH_EVERY_TOKENS", 5 * SEQ_LEN)
    real_append = pt._append_docs
    calls = {"n": 0}

    def dies_after_a_while(f, tokenizer, texts, dtype):
        calls["n"] += 1
        if calls["n"] == 20:
            # A torn write past the last durable position, then the outage.
            np.asarray([7, 7, 7], dtype=dtype).tofile(f)
            raise ConnectionError("network went away")
        return real_append(f, tokenizer, texts, dtype)

    out = tmp_path / "resumed"
    monkeypatch.setattr(pt, "_append_docs", dies_after_a_while)
    with pytest.raises(ConnectionError):
        pt.pretokenize(_cfg(out), train_tokens=10**9, num_proc=1)
    assert not (out / d.PRETOKENIZED_META).exists(), "an incomplete directory must not look complete"
    progress = json.loads((out / "train_000.json").read_text())
    assert 0 < progress["tokens"] < ref["train"]["tokens"]

    monkeypatch.setattr(pt, "_append_docs", real_append)
    resumed = pt.pretokenize(_cfg(out), train_tokens=10**9, num_proc=1)
    assert (out / resumed["train"]["files"][0]).read_bytes() == ref_bytes


def _interrupt(out, monkeypatch, **kwargs):
    """Start a pretokenize into `out` and kill it after a few durable flushes."""
    monkeypatch.setattr(pt, "_TOKENIZE_BATCH", 8)
    monkeypatch.setattr(pt, "_FLUSH_EVERY_TOKENS", 5 * SEQ_LEN)
    real_append = pt._append_docs
    calls = {"n": 0}

    def dies(f, tokenizer, texts, dtype):
        calls["n"] += 1
        if calls["n"] == 20:
            raise ConnectionError("network went away")
        return real_append(f, tokenizer, texts, dtype)

    monkeypatch.setattr(pt, "_append_docs", dies)
    with pytest.raises(ConnectionError):
        pt.pretokenize(_cfg(out, **kwargs), train_tokens=10**9, num_proc=1)
    monkeypatch.setattr(pt, "_append_docs", real_append)


def test_resume_keeps_the_recorded_partition_whatever_num_proc_says(fake_corpus, tmp_path, monkeypatch):
    # A worker's progress is a document count into its own shard: resumed under a different
    # worker count it would skip the wrong documents and append to the wrong file.
    ref = pt.pretokenize(_cfg(tmp_path / "ref"), train_tokens=10**9, num_proc=1)
    ref_bytes = (tmp_path / "ref" / ref["train"]["files"][0]).read_bytes()
    out = tmp_path / "resumed"
    _interrupt(out, monkeypatch)
    assert json.loads((out / "plan.json").read_text())["workers"] == 1

    resumed = pt.pretokenize(_cfg(out), train_tokens=10**9, num_proc=4)
    assert resumed["train"]["files"] == ["train_000.bin"]
    assert (out / "train_000.bin").read_bytes() == ref_bytes


def test_resume_refuses_a_plan_that_changes_the_output(fake_corpus, tmp_path, monkeypatch):
    out = tmp_path / "resumed"
    _interrupt(out, monkeypatch)
    with pytest.raises(ValueError, match="seed: 42 -> 1"):
        pt.pretokenize(_cfg(out, seed=1), train_tokens=10**9, num_proc=1)
    with pytest.raises(ValueError, match="target"):
        pt.pretokenize(_cfg(out), train_tokens=10**8, num_proc=1)


def test_resume_without_a_plan_refuses_a_different_worker_count(fake_corpus, tmp_path, monkeypatch):
    # A directory interrupted before plan.json existed: its worker count is the progress files'.
    out = tmp_path / "resumed"
    _interrupt(out, monkeypatch)
    (out / "plan.json").unlink()
    with pytest.raises(ValueError, match="--num-proc 1"):
        pt.pretokenize(_cfg(out), train_tokens=10**9, num_proc=2)


def test_sampler_resumes_at_the_exact_block_across_epochs():
    n, bs = 23, 4  # 23 blocks -> 5 full batches (20 blocks) per epoch
    reference = d.EpochPermutationSampler(n, bs, seed=3)
    stream = [i for _ in range(5) for i in reference]
    assert len(stream) == 5 * 20
    for epoch in range(3):
        # Distinct blocks within an epoch; the 3 left over differ per epoch rather than being
        # starved forever.
        assert len(set(stream[epoch * 20 : (epoch + 1) * 20])) == 20
    assert stream[:20] != stream[20:40]

    for drawn_batches in (0, 3, 5, 7, 12):
        resumed = d.EpochPermutationSampler(n, bs, seed=3)
        assert resumed.set_resume_position(drawn_batches * bs) == drawn_batches * bs
        got = [i for _ in range(2) for i in resumed]  # the rest of this epoch, then the next
        start = drawn_batches * bs
        assert got == stream[start : start + len(got)]
        assert len(got) == (20 - start % 20) + 20


def test_loader_reads_blocks_and_validates_the_tokenizer(fake_corpus, tmp_path):
    cfg = _cfg(tmp_path)
    pt.pretokenize(cfg, train_tokens=10**9, num_proc=1)
    train_loader, val_loader = d.build_dataloaders(cfg, _IndexTok())
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (4, SEQ_LEN) and batch["input_ids"].dtype == torch.int64
    # Unshuffled and complete: two passes give the same blocks, and every held-out block.
    first = [_doc_id(int(b[0])) for batch in val_loader for b in batch["input_ids"]]
    second = [_doc_id(int(b[0])) for batch in val_loader for b in batch["input_ids"]]
    assert first == second == list(range(EVAL_DOCS))

    cfg.data.tokenizer = "some-other-tokenizer"
    with pytest.raises(ValueError, match="tokenized with"):
        d.build_dataloaders(cfg, _IndexTok())


def test_train_resumes_the_pretokenized_position_exactly(fake_corpus, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    pt.pretokenize(_cfg(data_dir), train_tokens=10**9, num_proc=1)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg: _IndexTok())

    seen: list[int] = []
    real_getitem = d.PretokenizedDataset.__getitem__

    def spy(self, idx):
        item = real_getitem(self, idx)
        if len(self) > EVAL_DOCS:  # train blocks only
            seen.append(_doc_id(int(item["input_ids"][0])))
        return item

    monkeypatch.setattr(d.PretokenizedDataset, "__getitem__", spy)

    def run(max_steps: int) -> None:
        cfg = _cfg(
            data_dir, max_steps=max_steps, grad_accum_steps=2, save_every=3, eval_every=10_000,
            log_every=10_000, output_dir=str(tmp_path / "out"), resume_from="auto",
        )
        cfg.model = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=SEQ_LEN, vocab_pad_multiple=1)
        cfg.wandb = WandbConfig(mode="disabled")
        train_mod.train(cfg)

    # 380 train blocks -> 95 batches/epoch; 60 steps x 2 micro-batches = 120 batches crosses it.
    run(max_steps=30)
    first = list(seen)
    seen.clear()
    run(max_steps=60)
    assert "resuming pretokenized data after 240 blocks" in capsys.readouterr().out
    sampler = d.EpochPermutationSampler(N_DOCS - EVAL_DOCS, 4, seed=42)
    reference = [i for _ in range(2) for i in sampler]
    blocks_by_index = _blocks(data_dir / "train_000.bin")
    expected = [blocks_by_index[i] for i in reference[: 120 * 4]]
    assert first + seen == expected


def test_sampler_resume_maps_epochs_across_a_batch_size_change():
    n = 23
    old = d.EpochPermutationSampler(n, 4, seed=3)  # 20-block epochs
    stream = [list(old) for _ in range(3)]
    new = d.EpochPermutationSampler(n, 6, seed=3)  # 18-block epochs
    # 1 full epoch + 7 blocks drawn at batch_size 4 is epoch 1, offset 7 at any batch_size.
    assert new.set_resume_position(27, drawn_batch_size=4) == 18 + 7
    got = list(new)
    assert got == stream[1][7:18], "the rest of epoch 1, in its own order"
    # An offset past the new epoch's end lands at that end: the next iter() starts epoch 2.
    assert new.set_resume_position(20 + 19, drawn_batch_size=4) == 18 + 18


def test_train_resume_under_a_different_batch_size_neither_repeats_nor_skips(
    fake_corpus, tmp_path, monkeypatch, capsys
):
    # auto_batch_size may re-estimate on resume; the old batch count times the new batch_size
    # would land the seek on the wrong block.
    data_dir = tmp_path / "data"
    pt.pretokenize(_cfg(data_dir), train_tokens=10**9, num_proc=1)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg: _IndexTok())
    seen: list[int] = []
    real_getitem = d.PretokenizedDataset.__getitem__

    def spy(self, idx):
        item = real_getitem(self, idx)
        if len(self) > EVAL_DOCS:
            seen.append(_doc_id(int(item["input_ids"][0])))
        return item

    monkeypatch.setattr(d.PretokenizedDataset, "__getitem__", spy)

    def run(max_steps: int, batch_size: int) -> None:
        cfg = _cfg(
            data_dir, max_steps=max_steps, grad_accum_steps=2, save_every=3, eval_every=10_000,
            log_every=10_000, output_dir=str(tmp_path / "out"), resume_from="auto",
        )
        cfg.train.batch_size = batch_size
        cfg.model = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=SEQ_LEN, vocab_pad_multiple=1)
        cfg.wandb = WandbConfig(mode="disabled")
        train_mod.train(cfg)

    n = N_DOCS - EVAL_DOCS  # 380 blocks: 380-block epochs at batch_size 4, 378 at 6
    blocks_by_index = _blocks(data_dir / "train_000.bin")
    epoch = lambda e, length: [blocks_by_index[i] for i in torch.randperm(n, generator=torch.Generator().manual_seed(42 + e))[:length].tolist()]
    run(max_steps=30, batch_size=4)  # 240 blocks
    first = list(seen)
    seen.clear()
    run(max_steps=45, batch_size=6)  # 15 steps x 2 x 6 = 180 more: 138 to finish epoch 0, 42 of epoch 1
    assert "drawn at batch_size 4, now 6" in capsys.readouterr().out
    assert first == epoch(0, 380)[:240]
    assert seen == epoch(0, 378)[240:] + epoch(1, 378)[:42]


def test_decay_branch_anneals_a_kept_checkpoint_without_touching_the_main_run(
    fake_corpus, tmp_path, monkeypatch
):
    """The V2 workflow end to end: a WSD run keeps step 10 (keep_every), and a --decay-from branch
    off it resumes its optimizer and data position, anneals to min_lr_ratio over the requested
    steps, and saves its final step into its own directory."""
    from radiance.checkpointing import configure_decay_branch

    data_dir = tmp_path / "data"
    pt.pretokenize(_cfg(data_dir), train_tokens=10**9, num_proc=1)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg: _IndexTok())
    out = tmp_path / "out"

    def cfg_for_run() -> Config:
        cfg = _cfg(
            data_dir, max_steps=23, grad_accum_steps=2, save_every=5, keep_every=10,
            eval_every=10_000, log_every=10_000, output_dir=str(out), resume_from="auto",
            lr_schedule="wsd", warmup_ratio=0.1, min_lr_ratio=0.1,
        )
        cfg.model = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=SEQ_LEN, vocab_pad_multiple=1)
        cfg.wandb = WandbConfig(mode="disabled")
        return cfg

    train_mod.train(cfg_for_run())
    # keep_every spares 10 and 20; the final step (23, not a save_every multiple) is saved too.
    assert sorted(p.name for p in out.glob("step_*.pt")) == ["step_10.pt", "step_20.pt", "step_23.pt"]

    branch_cfg = cfg_for_run()
    configure_decay_branch(branch_cfg, str(out / "step_10.pt"), decay_steps=4)
    assert branch_cfg.train.wsd_decay_start_ratio == pytest.approx(1.0)  # stable phase
    train_mod.train(branch_cfg)

    branch_dir = out / "decay_from_10"
    assert [p.name for p in branch_dir.glob("step_*.pt")] == ["step_14.pt"]
    final = torch.load(branch_dir / "step_14.pt", weights_only=False)
    assert final["train_batches_drawn"] == 14 * 2, "branch must continue the checkpoint's data position"
    sched = final["scheduler"]
    assert sched["_last_lr"][0] / sched["base_lrs"][0] == pytest.approx(0.1)
    assert sorted(p.name for p in out.glob("step_*.pt")) == ["step_10.pt", "step_20.pt", "step_23.pt"]

    # Re-running the same command resumes the branch, not the source checkpoint.
    again = cfg_for_run()
    configure_decay_branch(again, str(out / "step_10.pt"), decay_steps=4)
    assert again.train.resume_from == "auto" and again.train.output_dir == str(branch_dir)
