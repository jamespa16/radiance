"""data.dataset_mix: training on a weighted blend of several corpora at once.

Covers the three things that are new here:
  * load_dataset_mix — parsing + validation of the mix YAML file (weights, per-entry text_column);
  * the mix itself — each corpus is tokenized+packed independently and the packed blocks are
    combined so that a corpus contributes ~its weight of the training tokens, for the
    non-streaming, streaming, and streaming+disk-cache paths alike;
  * config plumbing — data.dataset_mix takes precedence over data.dataset and a relative mix
    path resolves against the config file's directory.

The corpora are local HF datasets (monkeypatched load_dataset) with a word-level fake tokenizer,
so "alpha" and "beta" map to distinct token ids and a mix's weight shows up as the alpha/beta
token ratio of the produced batches — the exact property the feature promises.
"""

from __future__ import annotations

import textwrap

import pytest
import torch
from datasets import Dataset, DatasetDict, IterableDataset
import yaml

from radiance.config import Config, MixDatasetConfig, load_config
from radiance import data as d


class _WordTok:
    """Deterministic word-level tokenizer: one id per whitespace-separated word (0=eos reserved)."""

    eos_token_id = 1

    def __init__(self):
        self._id: dict[str, int] = {}
        self._next = 2

    def _encode(self, s: str):
        out = []
        for w in s.split():
            if w not in self._id:
                self._id[w] = self._next
                self._next += 1
            out.append(self._id[w])
        return out

    def id_for(self, word: str) -> int:
        self._encode(word)
        return self._id[word]

    def __call__(self, texts):
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        return {"input_ids": [self._encode(t) for t in texts]}


def _to_iterable(ds):
    # from_generator wants a generator *function* (re-runnable, picklable), not a generator object.
    rows = list(ds)

    def gen():
        for row in rows:
            yield row

    return IterableDataset.from_generator(gen)


@pytest.fixture
def local_store(monkeypatch):
    """Two local corpora, monkeypatched in as load_dataset. A uses column 'text', B uses 'body',
    so a mix that tokenizes the wrong column for B produces no beta tokens at all."""
    store = {
        "test/corpusA": DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["alpha"] * 512}),  # 512 docs -> 64 packed blocks
                "validation": Dataset.from_dict({"text": ["alpha"] * 64}),
            }
        ),
        "test/corpusB": DatasetDict(
            {
                "train": Dataset.from_dict({"body": ["beta"] * 512}),  # one-word docs, col 'body'
                "validation": Dataset.from_dict({"body": ["beta"] * 64}),
            }
        ),
    }

    def fake(name, streaming=False, split=None, **_kw):
        dd = store[name]
        if streaming:
            out = DatasetDict({k: _to_iterable(v) for k, v in dd.items()})
            return out[split] if split else out
        return dd[split] if split else dd

    monkeypatch.setattr(d, "load_dataset", fake)
    return store


def _write_mix(path, entries):
    path.write_text(yaml.safe_dump(entries))
    return str(path)


def _mix_cfg(tiny_cfg, tmp_path, mix_entries, **data_overrides):
    # seq_len is pinned to 16 by the tiny_cfg fixture (can't be overridden via _data); with one-word
    # docs that still packs each corpus to several blocks, plenty for a ratio check.
    mix_path = _write_mix(tmp_path / "mix.yaml", mix_entries)
    overrides = dict(dataset_mix=mix_path, cache_dir=str(tmp_path / "cache"))
    overrides.update(data_overrides)
    return tiny_cfg(_data=overrides, _train=dict(seed=42))


def _alpha_beta_ratio(ids: torch.Tensor, alpha_id: int, beta_id: int) -> float:
    a = (ids == alpha_id).sum().item()
    b = (ids == beta_id).sum().item()
    return a / (a + b) if (a + b) else float("nan")


def _collect(loader, max_rows: int | None = None) -> torch.Tensor:
    rows = []
    total = 0
    for batch in loader:
        rows.append(batch["input_ids"])
        total += batch["input_ids"].shape[0]
        if max_rows is not None and total >= max_rows:
            break
    return torch.cat(rows, dim=0) if rows else torch.empty(0, dtype=torch.long)


# --- load_dataset_mix parsing / validation --------------------------------------


def test_load_dataset_mix_parses_and_defaults_text_column(local_store, tiny_cfg, tmp_path):
    path = _write_mix(
        tmp_path / "mix.yaml",
        [
            {"dataset": "test/corpusA", "weight": 2.0},
            {"dataset": "test/corpusB", "text_column": "body"},  # weight defaults to 1.0
        ],
    )
    entries = d.load_dataset_mix(path, tiny_cfg())

    assert entries[0] == MixDatasetConfig(dataset="test/corpusA", weight=2.0, text_column="text")
    assert entries[1] == MixDatasetConfig(dataset="test/corpusB", weight=1.0, text_column="body")


@pytest.mark.parametrize(
    "content, match",
    [
        (yaml.safe_dump([]), "non-empty"),
        (yaml.safe_dump({"dataset": "x"}), "non-empty"),  # a mapping, not a list
        (yaml.safe_dump([{"weight": 1.0}]), "non-empty 'dataset'"),  # missing dataset
        (yaml.safe_dump([{"dataset": "x", "weight": 0}]), "> 0"),
        (yaml.safe_dump([{"dataset": "x", "weight": -1}]), "> 0"),
    ],
)
def test_load_dataset_mix_rejects_bad_files(local_store, tiny_cfg, tmp_path, content, match):
    path = tmp_path / "mix.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        d.load_dataset_mix(str(path), tiny_cfg())


# --- non-streaming mix -----------------------------------------------------------


def test_nonstreaming_mix_is_weighted_and_uses_each_corpus_column(local_store, tiny_cfg, tmp_path):
    tok = _WordTok()
    alpha_id, beta_id = tok.id_for("alpha"), tok.id_for("beta")
    cfg = _mix_cfg(tiny_cfg, tmp_path, [{"dataset": "test/corpusA"}, {"dataset": "test/corpusB", "text_column": "body"}])

    train_loader, val_loader = d.build_dataloaders(cfg, tok)

    assert val_loader is not None  # both corpora have a validation split
    train_ids = _collect(train_loader)  # finite epoch (non-streaming)
    assert train_ids.shape[1] == cfg.data.seq_len
    ratio = _alpha_beta_ratio(train_ids, alpha_id, beta_id)
    assert 0.4 <= ratio <= 0.6, f"expected ~50/50 alpha/beta, got {ratio:.3f}"

    val_ids = _collect(val_loader, max_rows=50)
    assert (val_ids == alpha_id).any() and (val_ids == beta_id).any()  # both corpora reached val too


def test_nonstreaming_mix_honors_weights(local_store, tiny_cfg, tmp_path):
    tok = _WordTok()
    alpha_id, beta_id = tok.id_for("alpha"), tok.id_for("beta")
    cfg = _mix_cfg(
        tiny_cfg, tmp_path,
        [{"dataset": "test/corpusA", "weight": 3.0}, {"dataset": "test/corpusB", "text_column": "body", "weight": 1.0}],
    )

    train_loader, _ = d.build_dataloaders(cfg, tok)
    ratio = _alpha_beta_ratio(_collect(train_loader), alpha_id, beta_id)
    assert 0.65 <= ratio <= 0.85, f"expected ~75/25 alpha/beta, got {ratio:.3f}"


def test_single_dataset_path_unchanged_when_no_mix(local_store, tiny_cfg, tmp_path):
    """data.dataset (no mix) still works after the per-corpus refactor: only corpusA's tokens."""
    tok = _WordTok()
    alpha_id, beta_id = tok.id_for("alpha"), tok.id_for("beta")
    cfg = tiny_cfg(_data=dict(dataset="test/corpusA", cache_dir=str(tmp_path / "cache")), _train=dict(seed=42))

    train_loader, val_loader = d.build_dataloaders(cfg, tok)
    ids = _collect(train_loader)
    assert (ids == alpha_id).any()
    assert not (ids == beta_id).any()  # B is not in this run at all
    assert val_loader is not None


# --- streaming mix ---------------------------------------------------------------


def test_streaming_mix_is_weighted(local_store, tiny_cfg, tmp_path):
    tok = _WordTok()
    alpha_id, beta_id = tok.id_for("alpha"), tok.id_for("beta")
    cfg = _mix_cfg(
        tiny_cfg, tmp_path,
        [{"dataset": "test/corpusA"}, {"dataset": "test/corpusB", "text_column": "body"}],
        streaming=True,
    )

    train_loader, _ = d.build_dataloaders(cfg, tok)
    ratio = _alpha_beta_ratio(_collect(train_loader, max_rows=200), alpha_id, beta_id)
    assert 0.4 <= ratio <= 0.6, f"expected ~50/50 alpha/beta, got {ratio:.3f}"


# --- streaming + bounded disk cache mix ------------------------------------------


def test_streaming_disk_cache_mix_is_weighted(local_store, tiny_cfg, tmp_path):
    tok = _WordTok()
    alpha_id, beta_id = tok.id_for("alpha"), tok.id_for("beta")
    cfg = _mix_cfg(
        tiny_cfg, tmp_path,
        [{"dataset": "test/corpusA"}, {"dataset": "test/corpusB", "text_column": "body"}],
        streaming=True,
        disk_cache_max_gb=0.001,  # ~1 MB: small enough to force eviction, big enough to cache
        disk_cache_shard_size=2,
    )

    train_loader, _ = d.build_dataloaders(cfg, tok)
    ratio = _alpha_beta_ratio(_collect(train_loader, max_rows=200), alpha_id, beta_id)
    assert 0.4 <= ratio <= 0.6, f"expected ~50/50 alpha/beta, got {ratio:.3f}"


# --- _WeightedInterleave unit behavior -------------------------------------------


def test_weighted_interleave_is_weighted_and_drops_empty_sources():
    a = [{"x": "a"}] * 10
    empty = []
    b = [{"x": "b"}] * 10
    # empty source in the middle: must be dropped and not stall the iterator
    ds = d._WeightedInterleave([a, empty, b], [1.0, 5.0, 1.0], seed=0)
    it = iter(ds)
    seen = [next(it) for _ in range(2000)]
    frac_a = sum(1 for x in seen if x["x"] == "a") / len(seen)
    assert 0.4 <= frac_a <= 0.6  # ~50/50 between a and b (empty dropped, renormalised)


def test_weighted_interleave_reiterates_exhausted_streaming_source():
    # A source shorter than the run must replay, not silently shrink the mix to the other source.
    a = [{"x": "a"}] * 2
    b = [{"x": "b"}] * 100
    ds = d._WeightedInterleave([a, b], [1.0, 1.0], seed=1)
    it = iter(ds)
    seen = [next(it) for _ in range(400)]
    # a is replayed; over 400 draws both must appear with roughly equal share
    frac_a = sum(1 for x in seen if x["x"] == "a") / len(seen)
    assert 0.4 <= frac_a <= 0.6


# --- config plumbing -------------------------------------------------------------


def test_load_config_resolves_relative_dataset_mix_against_config_dir(tmp_path):
    (tmp_path / "mix.yaml").write_text(yaml.safe_dump([{"dataset": "test/corpusA"}]))
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(textwrap.dedent(
        """
        run_name: mix-run
        data:
          dataset_mix: mix.yaml   # bare filename -> resolved against this file's directory
        """
    ))
    cfg = load_config(str(cfg_path))
    assert cfg.data.dataset_mix == str(tmp_path / "mix.yaml")


def test_load_config_rejects_dataset_mix_and_dataset_together(tmp_path):
    # A copy-pasted config that leaves data.dataset set alongside data.dataset_mix is a silent
    # trap (the mix wins, dataset is ignored), so load_config rejects it rather than resolving.
    (tmp_path / "mix.yaml").write_text(yaml.safe_dump([{"dataset": "test/corpusA"}]))
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(textwrap.dedent(
        """
        run_name: mix-run
        data:
          dataset_mix: mix.yaml
          dataset: test/corpusB   # leftover from a previous experiment
        """
    ))
    with pytest.raises(ValueError, match="both set"):
        load_config(str(cfg_path))


def test_dataset_mix_takes_precedence_over_dataset(local_store, tiny_cfg, tmp_path):
    tok = _WordTok()
    alpha_id, beta_id = tok.id_for("alpha"), tok.id_for("beta")
    # data.dataset points at corpusA, but the mix pulls in corpusB too.
    cfg = _mix_cfg(
        tiny_cfg, tmp_path,
        [{"dataset": "test/corpusA"}, {"dataset": "test/corpusB", "text_column": "body"}],
        dataset="test/corpusA",
    )
    train_loader, _ = d.build_dataloaders(cfg, tok)
    ids = _collect(train_loader, max_rows=200)
    assert (ids == alpha_id).any() and (ids == beta_id).any()
