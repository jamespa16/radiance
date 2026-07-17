from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from radiance.config import Config


def build_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_and_pack(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    text_column = cfg.data.text_column
    seq_len = cfg.data.seq_len

    def tokenize_fn(batch):
        return tokenizer(batch[text_column])

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=cfg.data.num_workers or None,
    )

    def group_fn(batch):
        concatenated = [tok for ids in batch["input_ids"] for tok in ids + [tokenizer.eos_token_id]]
        n_blocks = len(concatenated) // seq_len
        concatenated = concatenated[: n_blocks * seq_len]
        blocks = [concatenated[i : i + seq_len] for i in range(0, len(concatenated), seq_len)]
        return {"input_ids": blocks}

    packed = tokenized.map(
        group_fn,
        batched=True,
        remove_columns=tokenized.column_names,
        num_proc=cfg.data.num_workers or None,
    )
    packed.set_format(type="torch", columns=["input_ids"])
    return packed


def _cache_path(cfg: Config) -> Path:
    key = "|".join(
        [cfg.data.dataset, cfg.data.tokenizer, cfg.data.text_column, str(cfg.data.seq_len)]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cfg.data.cache_dir) / digest


def _load_or_build_packed(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    cache_path = _cache_path(cfg)
    if cfg.data.cache_dir and cache_path.exists():
        packed = load_from_disk(str(cache_path))
        packed.set_format(type="torch", columns=["input_ids"])
        return packed

    raw = load_dataset(cfg.data.dataset)
    train_split = raw["train"]
    val_split = raw.get("validation")

    packed = DatasetDict({"train": _tokenize_and_pack(train_split, tokenizer, cfg)})
    if val_split is not None:
        packed["validation"] = _tokenize_and_pack(val_split, tokenizer, cfg)

    if cfg.data.cache_dir:
        packed.save_to_disk(str(cache_path))

    return packed


def build_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[DataLoader, DataLoader | None]:
    packed = _load_or_build_packed(cfg, tokenizer)

    train_ds = packed["train"]
    val_ds = packed.get("validation")

    def collate(batch):
        input_ids = torch.stack([ex["input_ids"] for ex in batch])
        return {"input_ids": input_ids}

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate,
            drop_last=True,
        )

    return train_loader, val_loader
