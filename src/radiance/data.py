from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import yaml
from pathlib import Path

import torch
from datasets import DatasetDict, IterableDataset, interleave_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from radiance.config import Config, MixDatasetConfig
logger = logging.getLogger(__name__)


def build_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We tokenize full documents and pack/re-chunk them into seq_len blocks ourselves (see
    # _tokenize_and_pack), so no single call ever feeds a document straight into the model —
    # the tokenizer's own "longer than model_max_length" warning is a false positive here.
    tokenizer.model_max_length = int(1e30)
    return tokenizer


def _tokenize_and_pack(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config, text_column: str | None = None):
    # A mix entry may tokenize a different column than the shared data.text_column; a single
    # dataset (or an entry that omits text_column) falls back to the shared value, unchanged.
    text_column = text_column or cfg.data.text_column
    seq_len = cfg.data.seq_len

    def tokenize_fn(batch):
        return tokenizer(batch[text_column])

    map_kwargs = {} if isinstance(dataset, IterableDataset) else {"num_proc": cfg.data.num_workers or None}

    # A streaming IterableDataset's .column_names goes to None after a .map() call (streaming
    # doesn't infer schema), so remove_columns=dataset.column_names would silently become a
    # no-op post-tokenize — fall back to the columns we know the prior step actually produced.
    input_columns = dataset.column_names if dataset.column_names is not None else [text_column]
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=input_columns,
        **map_kwargs,
    )

    def group_fn(batch):
        concatenated = [tok for ids in batch["input_ids"] for tok in ids + [tokenizer.eos_token_id]]
        n_blocks = len(concatenated) // seq_len
        concatenated = concatenated[: n_blocks * seq_len]
        blocks = [concatenated[i : i + seq_len] for i in range(0, len(concatenated), seq_len)]
        return {"input_ids": blocks}

    tokenized_columns = tokenized.column_names
    if tokenized_columns is None:
        tokenized_columns = list(tokenizer(["placeholder"]).keys())
    packed = tokenized.map(
        group_fn,
        batched=True,
        remove_columns=tokenized_columns,
        **map_kwargs,
    )
    if not isinstance(packed, IterableDataset):
        packed.set_format(type="torch", columns=["input_ids"])
    return packed


def _packed_cache_path(dataset_id: str, text_column: str, cfg: Config) -> Path:
    # Keyed per corpus (not per run) so a dataset used in several mixes — or in a mix and on its
    # own — shares one tokenized+packed cache entry, and changing one mix entry never invalidates
    # the others. For the single-dataset case this is byte-identical to the pre-mix cache key, so
    # existing caches keep working.
    key = "|".join(
        [
            dataset_id,
            cfg.data.tokenizer,
            text_column,
            str(cfg.data.seq_len),
            str(cfg.data.eval_split_size),
        ]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cfg.data.cache_dir) / digest


def split_off_eval(train_split, eval_split_size: int):
    """When a dataset has no validation split, carve a deterministic held-out slice off the
    front of train instead (same order every run, so eval numbers stay comparable across runs).
    No-op (val_split=None) unless eval_split_size > 0."""
    if eval_split_size <= 0:
        return train_split, None
    val_split = train_split.take(eval_split_size)
    train_split = train_split.skip(eval_split_size)
    return train_split, val_split


_BYTES_PER_GB = 1_000_000_000


def _disk_cache_max_bytes(cfg: Config) -> int | None:
    return int(cfg.data.disk_cache_max_gb * _BYTES_PER_GB) if cfg.data.disk_cache_max_gb else None


def _load_or_build_packed(cfg: Config, tokenizer: PreTrainedTokenizerBase):
    return _load_or_build_packed_for_dataset(cfg.data.dataset, cfg.data.text_column, cfg, tokenizer)


def _require_train_split(raw, dataset_id: str) -> None:
    if "train" not in raw:
        raise ValueError(
            f"dataset {dataset_id!r} has no 'train' split; the data pipeline (and every "
            "data.dataset_mix entry) needs one"
        )


def _load_or_build_packed_for_dataset(
    dataset_id: str, text_column: str, cfg: Config, tokenizer: PreTrainedTokenizerBase
):
    """Load (or tokenize+pack and cache) one corpus's packed train/validation blocks.

    Per-corpus rather than per-run so a data.dataset_mix caches and reloads each dataset
    independently — changing one entry's weight, or adding a corpus, never re-tokenises the
    others. The disk-cache streaming path is deliberately NOT handled here: it builds torch
    StreamingPackedDataset wrappers rather than HF datasets, so it stays in build_dataloaders.
    Returns {"train": <packed>, "validation": <packed>|None}. Both return paths hand back
    `input_ids` in torch format so a corpus loaded from an existing cache is row-for-row the same
    type as one just built: `_tokenize_and_pack` applies set_format to a freshly built dataset,
    and the cache-hit branch re-applies it after load_from_disk because HF does not persist
    set_format to disk. The DataLoader's collate (torch.as_tensor) is a no-op on torch rows; a
    mix reorders/shuffles these per corpus and its interleaved result is normalized by the collate
    regardless of the per-corpus format.
    """
    if cfg.data.streaming and not cfg.data.disk_cache_max_gb:
        raw = load_dataset(dataset_id, streaming=True)
        _require_train_split(raw, dataset_id)
        train_split = raw["train"]
        val_split = raw.get("validation")
        if val_split is None:
            train_split, val_split = split_off_eval(train_split, cfg.data.eval_split_size)

        train_split = train_split.shuffle(seed=cfg.train.seed, buffer_size=cfg.data.shuffle_buffer_size)
        train_packed = _tokenize_and_pack(train_split, tokenizer, cfg, text_column)
        train_packed = train_packed.shuffle(seed=cfg.train.seed, buffer_size=cfg.data.shuffle_buffer_size)
        val_packed = _tokenize_and_pack(val_split, tokenizer, cfg, text_column) if val_split is not None else None

        return {"train": train_packed, "validation": val_packed}

    cache_path = _packed_cache_path(dataset_id, text_column, cfg)
    if cfg.data.cache_dir and cache_path.exists():
        packed = load_from_disk(str(cache_path))
        # load_from_disk drops the format _tokenize_and_pack set before save_to_disk (HF doesn't
        # persist set_format to disk); re-apply it so cached rows are torch like freshly built ones.
        packed.set_format(type="torch", columns=["input_ids"])
        return packed

    raw = load_dataset(dataset_id)
    _require_train_split(raw, dataset_id)
    train_split = raw["train"]
    val_split = raw.get("validation")
    if val_split is None:
        train_split, val_split = split_off_eval(train_split, cfg.data.eval_split_size)

    packed = DatasetDict({"train": _tokenize_and_pack(train_split, tokenizer, cfg, text_column)})
    if val_split is not None:
        packed["validation"] = _tokenize_and_pack(val_split, tokenizer, cfg, text_column)

    if cfg.data.cache_dir:
        packed.save_to_disk(str(cache_path))

    return packed


def _streaming_cache_digest(cfg: Config, dataset_id: str, text_column: str) -> str:
    # Per-corpus namespace so a mix gives each dataset its own cache dir + lock (and its own
    # bounded budget via StreamingPackedDataset._per_namespace_budget). For the single-dataset
    # case this is byte-identical to the pre-mix digest, so existing streaming caches keep working.
    key = "|".join(
        [
            dataset_id,
            cfg.data.tokenizer,
            text_column,
            str(cfg.data.seq_len),
            str(cfg.data.num_workers),
            str(cfg.data.eval_split_size),
        ]
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _read_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"n_raw_consumed": 0, "next_shard_idx": 0, "shards": []}


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    tmp_path = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(manifest))
    os.replace(tmp_path, manifest_path)


class _CacheLock:
    """Advisory PID lockfile guarding a cache namespace against two concurrently-running
    training processes — per-worker file isolation only protects workers within one run."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path

    def acquire(self) -> None:
        if self.lock_path.exists():
            try:
                pid = int(self.lock_path.read_text().strip())
            except ValueError:
                pid = None
            if pid is not None and pid != os.getpid():
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True  # process exists, just owned by someone else
                if alive:
                    raise RuntimeError(
                        f"Streaming disk cache at {self.lock_path.parent} is locked by a "
                        f"running process (pid {pid}). Two training runs can't safely share "
                        f"the same streaming cache namespace concurrently."
                    )
        self.lock_path.write_text(str(os.getpid()))

    def release(self) -> None:
        try:
            if self.lock_path.exists() and int(self.lock_path.read_text().strip()) == os.getpid():
                self.lock_path.unlink()
        except (ValueError, OSError):
            pass


class StreamingPackedDataset(torch.utils.data.IterableDataset):
    """Streams `cfg.data.dataset`, tokenizes+packs it into fixed `seq_len` blocks, and
    maintains a bounded, ring-buffer-style on-disk cache so repeated runs against the same
    dataset/config don't re-fetch/re-tokenize data already streamed before.

    Each DataLoader worker owns a private manifest + set of shard files (no cross-worker
    locking needed); a single lockfile per cache namespace guards against two concurrently
    running *processes* sharing the same namespace, which per-worker isolation doesn't cover.
    """

    def __init__(
        self,
        cfg: Config,
        tokenizer: PreTrainedTokenizerBase,
        split: str,
        num_splits_in_use: int,
        carve_eval_from_train: bool = False,
        dataset_id: str | None = None,
        text_column: str | None = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.split = split
        self.num_splits_in_use = num_splits_in_use
        self.carve_eval_from_train = carve_eval_from_train
        # A mix passes one corpus's id/column per wrapper; the single-dataset path omits both and
        # falls back to the shared config values (identical cache namespace, identical behavior).
        self.dataset_id = dataset_id or cfg.data.dataset
        self.text_column = text_column or cfg.data.text_column

        self.cache_dir = Path(cfg.data.cache_dir) / _streaming_cache_digest(cfg, self.dataset_id, self.text_column) / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = _CacheLock(self.cache_dir / ".lock")
        self._lock.acquire()

    def __del__(self):
        self._lock.release()

    def _manifest_path(self, worker_id: int, num_workers: int) -> Path:
        return self.cache_dir / f"manifest_w{worker_id}_of_{num_workers}.json"

    def _shard_path(self, worker_id: int, shard_idx: int) -> Path:
        return self.cache_dir / f"shard_w{worker_id}_{shard_idx:06d}.pt"

    def _per_namespace_budget(self, num_workers: int) -> int:
        return _disk_cache_max_bytes(self.cfg) // (num_workers * self.num_splits_in_use)

    def _flush(self, manifest: dict, manifest_path: Path, block_buffer: list, worker_id: int, budget: int) -> None:
        if not block_buffer:
            return
        shard_idx = manifest["next_shard_idx"]
        manifest["next_shard_idx"] += 1
        shard_path = self._shard_path(worker_id, shard_idx)
        torch.save(block_buffer, shard_path)
        manifest["shards"].append(
            {"file": shard_path.name, "n_blocks": len(block_buffer), "n_bytes": shard_path.stat().st_size}
        )

        total_bytes = sum(s["n_bytes"] for s in manifest["shards"])
        while total_bytes > budget and len(manifest["shards"]) > 1:
            oldest = manifest["shards"].pop(0)
            (self.cache_dir / oldest["file"]).unlink(missing_ok=True)
            total_bytes -= oldest["n_bytes"]

        _write_manifest(manifest_path, manifest)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        manifest_path = self._manifest_path(worker_id, num_workers)
        manifest = _read_manifest(manifest_path)
        budget = self._per_namespace_budget(num_workers)

        shuffle_buf: list[dict] = []
        buffer_size = self.cfg.data.shuffle_buffer_size

        def shuffled(items):
            for item in items:
                if len(shuffle_buf) < buffer_size:
                    shuffle_buf.append(item)
                    continue
                idx = random.randrange(buffer_size)
                yield shuffle_buf[idx]
                shuffle_buf[idx] = item
            random.shuffle(shuffle_buf)
            yield from shuffle_buf
            shuffle_buf.clear()

        def source():
            for shard in manifest["shards"]:
                blocks = torch.load(self.cache_dir / shard["file"])
                for block in blocks:
                    yield {"input_ids": block}

            if self.carve_eval_from_train:
                raw = load_dataset(self.dataset_id, split="train", streaming=True)
                if self.split == "validation":
                    raw = raw.take(self.cfg.data.eval_split_size)
                else:
                    raw = raw.skip(self.cfg.data.eval_split_size)
            else:
                raw = load_dataset(self.dataset_id, split=self.split, streaming=True)
            if num_workers > 1:
                raw = raw.shard(num_shards=num_workers, index=worker_id)
            raw = raw.skip(manifest["n_raw_consumed"])

            seq_len = self.cfg.data.seq_len
            eos_id = self.tokenizer.eos_token_id
            text_column = self.text_column
            token_buffer: list[int] = []
            block_buffer: list[list[int]] = []
            raw_consumed_since_flush = 0
            n_yielded = 0

            # Keep the raw-example tokenize batch small relative to disk_cache_shard_size: a
            # flush can only happen between tokenize-batch boundaries (raw_consumed_since_flush
            # is only known safe to persist once we stop actively draining block_buffer from the
            # current batch's token_buffer), so a large tokenize batch would let many more blocks
            # accumulate in memory than ever make it to disk before a short run's generator gets
            # abandoned. 8 examples still gets most of batched tokenization's throughput benefit.
            tokenize_batch_size = 8

            def drain_blocks():
                nonlocal token_buffer, block_buffer, raw_consumed_since_flush, n_yielded
                while len(token_buffer) >= seq_len:
                    block = token_buffer[:seq_len]
                    token_buffer = token_buffer[seq_len:]
                    block_buffer.append(block)
                    n_yielded += 1
                    yield {"input_ids": block}

                    if len(block_buffer) >= self.cfg.data.disk_cache_shard_size:
                        manifest["n_raw_consumed"] += raw_consumed_since_flush
                        self._flush(manifest, manifest_path, block_buffer, worker_id, budget)
                        block_buffer, raw_consumed_since_flush = [], 0

            raw_batch = []
            for example in raw:
                raw_batch.append(example)
                if len(raw_batch) < tokenize_batch_size:
                    continue
                token_lists = self.tokenizer([ex[text_column] for ex in raw_batch])["input_ids"]
                for ids in token_lists:
                    token_buffer.extend(ids)
                    token_buffer.append(eos_id)
                raw_consumed_since_flush += len(raw_batch)
                raw_batch = []

                yield from drain_blocks()

            if raw_batch:
                token_lists = self.tokenizer([ex[text_column] for ex in raw_batch])["input_ids"]
                for ids in token_lists:
                    token_buffer.extend(ids)
                    token_buffer.append(eos_id)
                raw_consumed_since_flush += len(raw_batch)

                yield from drain_blocks()

            manifest["n_raw_consumed"] += raw_consumed_since_flush
            self._flush(manifest, manifest_path, block_buffer, worker_id, budget)

            if n_yielded == 0 and manifest["shards"]:
                logger.warning(
                    "[radiance] streaming disk cache: worker %d/%d for split %r has fully "
                    "consumed its raw data partition — later epochs will only replay the "
                    "%d bytes currently cached, not fetch new data.",
                    worker_id,
                    num_workers,
                    self.split,
                    sum(s["n_bytes"] for s in manifest["shards"]),
                )

        yield from shuffled(source())


def _pretrain_collate(batch):
    input_ids = torch.stack([torch.as_tensor(ex["input_ids"]) for ex in batch])
    return {"input_ids": input_ids}


def _dataloader_kwargs(cfg: Config) -> dict:
    return dict(
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        collate_fn=_pretrain_collate,
        drop_last=True,
    )


def _normalized(weights: list[float]) -> list[float]:
    """Mix weights -> probabilities. interleave_datasets requires probabilities that sum to 1, and
    a mix subset (e.g. the validation interleave) re-normalises over just its members."""
    total = sum(weights)
    return [w / total for w in weights]


def load_dataset_mix(path: str, cfg: Config) -> list[MixDatasetConfig]:
    """Read a data.dataset_mix YAML file and return its validated, resolved entries.

    The file is a YAML list; each item is a mapping with a required `dataset` and optional
    `weight` (default 1.0) and `text_column` (default: the shared data.text_column). Weights are
    validated finite and > 0; text_column is resolved against cfg.data.text_column so the rest of
    the pipeline never sees a None.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"dataset mix file {path!r} must be a non-empty YAML list of "
            "{{dataset, [weight], [text_column]}} entries"
        )
    entries: list[MixDatasetConfig] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get("dataset"):
            raise ValueError(f"dataset mix entry {i} in {path!r} must be a mapping with a non-empty 'dataset'")
        weight = float(item.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"dataset mix entry {i} ({item['dataset']!r}) has weight {weight!r}; "
                "weights must be finite and > 0"
            )
        entries.append(
            MixDatasetConfig(
                dataset=item["dataset"],
                weight=weight,
                text_column=item.get("text_column") or cfg.data.text_column,
            )
        )
    return entries


class _WeightedInterleave(torch.utils.data.IterableDataset):
    """Weighted, effectively-infinite interleave of per-corpus packed-block streams.

    Each draw picks a source with probability proportional to its mix weight and yields the next
    block from that source; an exhausted source is re-iterated (a streaming corpus replays its
    cache / refetches, matching the single-dataset streaming path's later-epoch behavior) and a
    source that yields nothing is dropped, renormalising the rest. Per-source iterators are made
    inside __iter__ so each DataLoader worker gets its own, and each underlying source shards
    across workers itself (HF streaming / StreamingPackedDataset), so the interleave is
    worker-safe. The per-worker RNG is seeded from (seed, worker_id) so the mix is deterministic
    for a given seed.
    """

    def __init__(self, sources: list, weights: list[float], seed: int):
        self.sources = sources
        total = sum(weights)
        self.probs = [w / total for w in weights]
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id)
        iters = [iter(s) for s in self.sources]
        active = list(range(len(self.sources)))
        while active:
            probs = [self.probs[i] for i in active]
            total = sum(probs)
            pick = active[rng.choices(range(len(active)), weights=[p / total for p in probs], k=1)[0]]
            try:
                item = next(iters[pick])
            except StopIteration:
                iters[pick] = iter(self.sources[pick])
                try:
                    item = next(iters[pick])
                except StopIteration:
                    active.remove(pick)  # empty source: drop it and renormalise the rest
                    continue
            yield item


_SPLIT_SET_CACHE: dict[str, frozenset[str]] = {}


def _dataset_splits(dataset_id: str) -> frozenset[str]:
    """The split names a (possibly remote) dataset advertises, cached per dataset_id so a corpus
    pays for the remote metadata/stream setup once per process, not on every probe. Streaming
    load is used (metadata only, no download) to match how the streaming path otherwise resolves
    splits. The per-worker stream in StreamingPackedDataset.__iter__ still loads the data itself;
    this only dedupes the split-set probe (e.g. a corpus listed more than once in a mix)."""
    splits = _SPLIT_SET_CACHE.get(dataset_id)
    if splits is None:
        splits = frozenset(load_dataset(dataset_id, streaming=True).keys())
        _SPLIT_SET_CACHE[dataset_id] = splits
    return splits


def _has_validation_split(dataset_id: str) -> bool:
    return "validation" in _dataset_splits(dataset_id)


def _build_mixed_disk_cache_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase, entries, weights):
    # The bounded-cache streaming path: one StreamingPackedDataset per (corpus, split). Each sizes
    # its own cache budget from num_splits_in_use, so passing the total number of (corpus, split)
    # namespaces keeps the combined footprint within disk_cache_max_gb. We over-count (assume 2
    # splits per corpus, even those without validation) rather than under-count it: the budget is
    # a ceiling, so a smaller real footprint is safe, an underestimated one is not.
    num_splits_in_use = len(entries) * 2

    train_sources = []
    val_sources = []
    val_weights = []
    for e in entries:
        has_real_validation = _has_validation_split(e.dataset)
        carve = not has_real_validation and cfg.data.eval_split_size > 0
        has_validation = has_real_validation or carve
        train_sources.append(
            StreamingPackedDataset(
                cfg, tokenizer, split="train", num_splits_in_use=num_splits_in_use,
                carve_eval_from_train=carve, dataset_id=e.dataset, text_column=e.text_column,
            )
        )
        if has_validation:
            val_sources.append(
                StreamingPackedDataset(
                    cfg, tokenizer, split="validation", num_splits_in_use=num_splits_in_use,
                    carve_eval_from_train=carve, dataset_id=e.dataset, text_column=e.text_column,
                )
            )
            val_weights.append(e.weight)

    loader_kwargs = _dataloader_kwargs(cfg)
    train_ds = _WeightedInterleave(train_sources, weights, cfg.train.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)
    if val_sources:
        val_ds = _WeightedInterleave(val_sources, val_weights, cfg.train.seed + 1)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)
    else:
        val_loader = None
    return train_loader, val_loader


def _build_mixed_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase):
    """Build train/val loaders for a data.dataset_mix (see DataConfig.dataset_mix).

    Each corpus is tokenized+packed independently (per-corpus cache, per-entry text_column) and
    the packed blocks are then mixed:
      - non-streaming: in-memory corpora are per-corpus shuffled and combined with
        interleave_datasets(stopping_strategy='all_exhausted') — a finite epoch in which every
        corpus is covered and each contributes ~its weight of the blocks;
      - streaming (with or without disk_cache_max_gb): the live per-corpus block streams are
        combined with _WeightedInterleave — effectively infinite, each draw from a corpus chosen
        by weight (the disk-cache variant wraps per-corpus StreamingPackedDatasets).
    The mixed validation split is the same mix restricted to the corpora that have one (a corpus
    without a validation split — and no data.eval_split_size carve — simply doesn't contribute to
    val/loss; this mirrors the single-dataset behavior).
    """
    entries = load_dataset_mix(cfg.data.dataset_mix, cfg)
    weights = [e.weight for e in entries]
    loader_kwargs = _dataloader_kwargs(cfg)

    if cfg.data.streaming and cfg.data.disk_cache_max_gb:
        return _build_mixed_disk_cache_dataloaders(cfg, tokenizer, entries, weights)

    packed = [
        _load_or_build_packed_for_dataset(e.dataset, e.text_column, cfg, tokenizer) for e in entries
    ]

    if cfg.data.streaming:
        train_ds = _WeightedInterleave([p["train"] for p in packed], weights, cfg.train.seed)
        val_parts = [(i, p["validation"]) for i, p in enumerate(packed) if p["validation"] is not None]
        val_ds = (
            _WeightedInterleave([v for _, v in val_parts], [weights[i] for i, _ in val_parts], cfg.train.seed + 1)
            if val_parts else None
        )
    else:
        train_parts = []
        for i, p in enumerate(packed):
            t = p["train"]
            if len(t) == 0:
                logger.warning(
                    "[radiance] mix corpus %r packs to 0 blocks at seq_len=%d; skipped from the mix",
                    entries[i].dataset, cfg.data.seq_len,
                )
                continue
            train_parts.append((i, t))
        if not train_parts:
            raise ValueError(
                "the data.dataset_mix packs to no training blocks — check data.seq_len against "
                "the corpora's sizes (a corpus smaller than one seq_len packs to nothing)"
            )
        train_ds = interleave_datasets(
            [t.shuffle(seed=cfg.train.seed + 1000 * i) for i, t in train_parts],
            probabilities=_normalized([weights[i] for i, _ in train_parts]),
            seed=cfg.train.seed,
            stopping_strategy="all_exhausted",
        )
        # Non-streaming corpora only carry a "validation" key when one exists (see
        # _load_or_build_packed_for_dataset), so probe with .get — a corpus with no validation
        # split and no eval_split_size carve would otherwise raise KeyError here.
        val_parts = [
            (i, p["validation"]) for i, p in enumerate(packed)
            if p.get("validation") is not None and len(p.get("validation")) > 0
        ]
        if val_parts:
            val_ds = interleave_datasets(
                [v.shuffle(seed=cfg.train.seed + 2000 * i) for i, v in val_parts],
                probabilities=_normalized([weights[i] for i, _ in val_parts]),
                seed=cfg.train.seed + 1,
                stopping_strategy="all_exhausted",
            )
        else:
            val_ds = None

    # Non-streaming train_ds is a finite map-style interleave, so shuffle per epoch like the
    # single-dataset path (shuffle=not cfg.data.streaming); the streaming train_ds is an
    # IterableDataset and must stay shuffle=False.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=not cfg.data.streaming, **loader_kwargs
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)
        if val_ds is not None else None
    )
    return train_loader, val_loader


def build_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[DataLoader, DataLoader | None]:
    # A data.dataset_mix drives the pipeline instead of the single data.dataset.
    if cfg.data.dataset_mix:
        return _build_mixed_dataloaders(cfg, tokenizer)

    loader_kwargs = _dataloader_kwargs(cfg)

    if cfg.data.streaming and cfg.data.disk_cache_max_gb:
        raw = load_dataset(cfg.data.dataset, streaming=True)
        has_real_validation = "validation" in raw
        carve_eval = not has_real_validation and cfg.data.eval_split_size > 0
        has_validation = has_real_validation or carve_eval
        num_splits_in_use = 2 if has_validation else 1

        train_ds = StreamingPackedDataset(
            cfg, tokenizer, split="train", num_splits_in_use=num_splits_in_use, carve_eval_from_train=carve_eval
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)

        val_loader = None
        if has_validation:
            val_ds = StreamingPackedDataset(
                cfg,
                tokenizer,
                split="validation",
                num_splits_in_use=num_splits_in_use,
                carve_eval_from_train=carve_eval,
            )
            val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)

        return train_loader, val_loader

    packed = _load_or_build_packed(cfg, tokenizer)

    train_ds = packed["train"]
    val_ds = packed.get("validation")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=not cfg.data.streaming, **loader_kwargs
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader
