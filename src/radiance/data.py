from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path

import torch
from datasets import DatasetDict, IterableDataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from radiance.config import Config

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


def _tokenize_and_pack(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    text_column = cfg.data.text_column
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


def _cache_path(cfg: Config) -> Path:
    key = "|".join(
        [
            cfg.data.dataset,
            cfg.data.tokenizer,
            cfg.data.text_column,
            str(cfg.data.seq_len),
            str(cfg.data.eval_split_size),
        ]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cfg.data.cache_dir) / digest


def _split_off_eval(train_split, eval_split_size: int):
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
    if cfg.data.streaming and not cfg.data.disk_cache_max_gb:
        raw = load_dataset(cfg.data.dataset, streaming=True)
        train_split = raw["train"]
        val_split = raw.get("validation")
        if val_split is None:
            train_split, val_split = _split_off_eval(train_split, cfg.data.eval_split_size)

        train_split = train_split.shuffle(seed=cfg.train.seed, buffer_size=cfg.data.shuffle_buffer_size)
        train_packed = _tokenize_and_pack(train_split, tokenizer, cfg)
        train_packed = train_packed.shuffle(seed=cfg.train.seed, buffer_size=cfg.data.shuffle_buffer_size)
        val_packed = _tokenize_and_pack(val_split, tokenizer, cfg) if val_split is not None else None

        return {"train": train_packed, "validation": val_packed}

    cache_path = _cache_path(cfg)
    if cfg.data.cache_dir and cache_path.exists():
        packed = load_from_disk(str(cache_path))
        packed.set_format(type="torch", columns=["input_ids"])
        return packed

    raw = load_dataset(cfg.data.dataset)
    train_split = raw["train"]
    val_split = raw.get("validation")
    if val_split is None:
        train_split, val_split = _split_off_eval(train_split, cfg.data.eval_split_size)

    packed = DatasetDict({"train": _tokenize_and_pack(train_split, tokenizer, cfg)})
    if val_split is not None:
        packed["validation"] = _tokenize_and_pack(val_split, tokenizer, cfg)

    if cfg.data.cache_dir:
        packed.save_to_disk(str(cache_path))

    return packed


def _streaming_cache_digest(cfg: Config) -> str:
    key = "|".join(
        [
            cfg.data.dataset,
            cfg.data.tokenizer,
            cfg.data.text_column,
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
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.split = split
        self.num_splits_in_use = num_splits_in_use
        self.carve_eval_from_train = carve_eval_from_train

        self.cache_dir = Path(cfg.data.cache_dir) / _streaming_cache_digest(cfg) / split
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
                raw = load_dataset(self.cfg.data.dataset, split="train", streaming=True)
                if self.split == "validation":
                    raw = raw.take(self.cfg.data.eval_split_size)
                else:
                    raw = raw.skip(self.cfg.data.eval_split_size)
            else:
                raw = load_dataset(self.cfg.data.dataset, split=self.split, streaming=True)
            if num_workers > 1:
                raw = raw.shard(num_shards=num_workers, index=worker_id)
            raw = raw.skip(manifest["n_raw_consumed"])

            seq_len = self.cfg.data.seq_len
            eos_id = self.tokenizer.eos_token_id
            text_column = self.cfg.data.text_column
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


def build_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[DataLoader, DataLoader | None]:
    def collate(batch):
        input_ids = torch.stack([torch.as_tensor(ex["input_ids"]) for ex in batch])
        return {"input_ids": input_ids}

    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        collate_fn=collate,
        drop_last=True,
    )

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


# --- SFT (post-training) ----------------------------------------------------------------------
#
# Same overall shape as the pretrain path above (tokenize -> pack into fixed seq_len blocks ->
# cache to disk -> collate), with two differences carried alongside input_ids at every stage:
# each example is chat-formatted (see _format_sft_messages/_tokenize_sft_example) rather than
# plain text, and a per-token loss_mask travels with it so only assistant-turn tokens (and the
# trailing EOS) are scored. Packing multiple examples per block and letting doc_attention_mask
# isolate them via the same EOS-boundary detection it already uses for pretraining documents is
# deliberate — see CLAUDE.md/the SFT design notes for why no model.py change is needed for that.


def _format_sft_messages(example: dict, cfg: Config) -> list[dict]:
    """Normalize one dataset row to a [{"role": ..., "content": ...}, ...] turn list.

    Reads cfg.sft.messages_column directly when set (the standard shape for chat-formatted HF
    datasets, e.g. HuggingFaceH4/no_robots). When cfg.sft.instruction_column is set instead
    (Alpaca-style datasets with separate instruction/input/output columns, no native messages
    column), builds an equivalent 2-turn list from those columns.
    """
    if cfg.sft.instruction_column is not None:
        instruction = example[cfg.sft.instruction_column]
        extra_input = example.get(cfg.sft.input_column) if cfg.sft.input_column else None
        prompt = f"{instruction}\n{extra_input}" if extra_input else instruction
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example[cfg.sft.output_column]},
        ]
    return example[cfg.sft.messages_column]


def _tokenize_sft_example(
    messages: list[dict], tokenizer: PreTrainedTokenizerBase, cfg: Config
) -> tuple[list[int], list[int]]:
    """Tokenize one chat example to (ids, loss_mask).

    mask is 0 for user/system turns (not supervised) and 1 for assistant turns plus the trailing
    EOS (supervised, including EOS so the model learns to stop). Attention is not restricted
    here — a response still attends causally to its own prompt within the example; only the loss
    is masked, downstream in train.compute_sft_loss. Turn markers (cfg.sft.user_prefix/
    assistant_prefix) are plain text tokenized through the existing vocab, not new special tokens.
    """
    ids: list[int] = []
    mask: list[int] = []
    for turn in messages:
        is_assistant = turn["role"] == "assistant"
        prefix = cfg.sft.assistant_prefix if is_assistant else cfg.sft.user_prefix
        turn_ids = tokenizer(prefix + turn["content"])["input_ids"]
        ids.extend(turn_ids)
        mask.extend([int(is_assistant)] * len(turn_ids))
    ids.append(tokenizer.eos_token_id)
    mask.append(1)
    return ids, mask


def format_sft_prompt(user_message: str, cfg: Config) -> str:
    """The same turn-formatting _tokenize_sft_example applies, for a single open user turn —
    used by generate.py so prompting an SFT checkpoint matches how it was trained."""
    return cfg.sft.user_prefix + user_message + cfg.sft.assistant_prefix


def _tokenize_and_pack_sft(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    seq_len = cfg.sft.seq_len or cfg.data.seq_len

    def tokenize_fn(batch):
        keys = list(batch.keys())
        rows = [dict(zip(keys, values)) for values in zip(*batch.values())]
        ids_list, mask_list = [], []
        for example in rows:
            ids, mask = _tokenize_sft_example(_format_sft_messages(example, cfg), tokenizer, cfg)
            ids_list.append(ids)
            mask_list.append(mask)
        return {"input_ids": ids_list, "loss_mask": mask_list}

    map_kwargs = {"num_proc": cfg.data.num_workers or None}
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        **map_kwargs,
    )

    def group_fn(batch):
        ids_concat: list[int] = []
        mask_concat: list[int] = []
        for ids, mask in zip(batch["input_ids"], batch["loss_mask"]):
            ids_concat.extend(ids)
            mask_concat.extend(mask)
        n_blocks = len(ids_concat) // seq_len
        ids_concat = ids_concat[: n_blocks * seq_len]
        mask_concat = mask_concat[: n_blocks * seq_len]
        id_blocks = [ids_concat[i : i + seq_len] for i in range(0, len(ids_concat), seq_len)]
        mask_blocks = [mask_concat[i : i + seq_len] for i in range(0, len(mask_concat), seq_len)]
        return {"input_ids": id_blocks, "loss_mask": mask_blocks}

    packed = tokenized.map(
        group_fn,
        batched=True,
        remove_columns=tokenized.column_names,
        **map_kwargs,
    )
    packed.set_format(type="torch", columns=["input_ids", "loss_mask"])
    return packed


def _sft_cache_path(cfg: Config) -> Path:
    seq_len = cfg.sft.seq_len or cfg.data.seq_len
    key = "|".join(
        [
            cfg.sft.dataset,
            cfg.data.tokenizer,
            str(seq_len),
            cfg.sft.messages_column,
            str(cfg.sft.instruction_column),
            str(cfg.sft.input_column),
            str(cfg.sft.output_column),
            cfg.sft.user_prefix,
            cfg.sft.assistant_prefix,
            str(cfg.sft.eval_split_size),
        ]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cfg.sft.cache_dir) / digest


def _load_or_build_sft_packed(cfg: Config, tokenizer: PreTrainedTokenizerBase):
    cache_path = _sft_cache_path(cfg)
    if cfg.sft.cache_dir and cache_path.exists():
        packed = load_from_disk(str(cache_path))
        packed.set_format(type="torch", columns=["input_ids", "loss_mask"])
        return packed

    raw = load_dataset(cfg.sft.dataset)
    train_split = raw["train"]
    val_split = raw.get("validation")
    if val_split is None:
        train_split, val_split = _split_off_eval(train_split, cfg.sft.eval_split_size)

    packed = DatasetDict({"train": _tokenize_and_pack_sft(train_split, tokenizer, cfg)})
    if val_split is not None:
        packed["validation"] = _tokenize_and_pack_sft(val_split, tokenizer, cfg)

    if cfg.sft.cache_dir:
        packed.save_to_disk(str(cache_path))

    return packed


def build_sft_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[DataLoader, DataLoader | None]:
    """SFT analogue of build_dataloaders: same (train_loader, val_loader) contract, but each
    batch additionally carries "loss_mask" alongside "input_ids" (see module docstring above).

    No streaming support yet — instruction datasets are small enough to tokenize/cache up front,
    and StreamingPackedDataset's bounded-disk-cache machinery isn't worth replicating for this
    until a dataset actually needs it.
    """
    if cfg.data.streaming:
        raise ValueError(
            "sft.enabled does not support data.streaming yet — instruction datasets are small "
            "enough to tokenize/cache up front. Set data.streaming: false for an SFT run."
        )
    if not cfg.sft.dataset:
        raise ValueError("sft.enabled requires sft.dataset to be set.")

    def collate(batch):
        input_ids = torch.stack([torch.as_tensor(ex["input_ids"]) for ex in batch])
        loss_mask = torch.stack([torch.as_tensor(ex["loss_mask"]) for ex in batch])
        return {"input_ids": input_ids, "loss_mask": loss_mask}

    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        collate_fn=collate,
        drop_last=True,
    )

    packed = _load_or_build_sft_packed(cfg, tokenizer)
    train_ds = packed["train"]
    val_ds = packed.get("validation")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, **loader_kwargs)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader
