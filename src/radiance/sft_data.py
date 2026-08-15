from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from radiance.config import Config

from .data import _split_off_eval

# SFT (post-training): same overall shape as the pretrain path (tokenize -> pack into fixed
# seq_len blocks -> cache to disk -> collate), with two differences carried alongside input_ids at
# every stage: each example is chat-formatted (see _format_sft_messages/_tokenize_sft_example)
# rather than plain text, and a per-token loss_mask travels with it so only assistant-turn tokens
# (and the trailing EOS) are scored. Packing multiple examples per block and letting
# doc_attention_mask isolate them via the same EOS-boundary detection it already uses for
# pretraining documents is deliberate — see CLAUDE.md/the SFT design notes for why no model
# change is needed for that.

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
    messages: list[dict], tokenizer: PreTrainedTokenizerBase, user_prefix: str, assistant_prefix: str
) -> tuple[list[int], list[int]]:
    """Tokenize one chat example to (ids, loss_mask).

    mask is 0 for user/system turns (not supervised) and 1 for assistant turns plus the trailing
    EOS (supervised, including EOS so the model learns to stop). Attention is not restricted
    here — a response still attends causally to its own prompt within the example; only the loss
    is masked, downstream in losses.compute_sft_loss/compute_dpo_loss. Turn markers (user_prefix/
    assistant_prefix) are plain text tokenized through the existing vocab, not new special tokens.

    Takes the prefixes as explicit strings rather than reading cfg.sft.user_prefix/assistant_prefix
    directly, so DPO's data pipeline can reuse this unchanged with cfg.dpo's own prefixes instead of
    being hardcoded to the sft config block.
    """
    ids: list[int] = []
    mask: list[int] = []
    for turn in messages:
        is_assistant = turn["role"] == "assistant"
        prefix = assistant_prefix if is_assistant else user_prefix
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


def format_chat_prompt(user_message: str, cfg: Config) -> str:
    """Generalizes format_sft_prompt to also cover a DPO checkpoint.

    sft.enabled and dpo.enabled are mutually exclusive on any one run, so a DPO checkpoint's own
    saved cfg.sft.enabled is False even though it expects the same turn template it was
    (transitively) SFT'd with. Picks whichever post-training mode the checkpoint's own config has
    on; raises if neither, mirroring format_sft_prompt's implicit prior contract that only an
    SFT-trained checkpoint made sense to prompt this way.
    """
    if cfg.sft.enabled:
        return format_sft_prompt(user_message, cfg)
    if cfg.dpo.enabled:
        return cfg.dpo.user_prefix + user_message + cfg.dpo.assistant_prefix
    raise ValueError(
        "format_chat_prompt requires a checkpoint trained with sft.enabled: true or dpo.enabled: true"
    )


def _tokenize_and_pack_sft(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    seq_len = cfg.sft.seq_len or cfg.data.seq_len

    def tokenize_fn(batch):
        keys = list(batch.keys())
        rows = [dict(zip(keys, values)) for values in zip(*batch.values())]
        ids_list, mask_list = [], []
        for example in rows:
            ids, mask = _tokenize_sft_example(
                _format_sft_messages(example, cfg), tokenizer, cfg.sft.user_prefix, cfg.sft.assistant_prefix
            )
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
