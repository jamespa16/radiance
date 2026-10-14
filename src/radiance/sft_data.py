from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from radiance.config import Config

from .data import pin_memory_for, split_off_eval

# SFT (post-training): same overall shape as the pretrain path (tokenize -> pack into fixed
# seq_len blocks -> cache to disk -> collate), with two differences carried alongside input_ids at
# every stage: each example is chat-formatted (see _format_sft_messages/tokenize_sft_example)
# rather than plain text, and a per-token loss_mask travels with it so only assistant-turn tokens
# (and the trailing EOS) are scored. Packing multiple examples per block and letting
# doc_attention_mask isolate them via the same EOS-boundary detection it already uses for
# pretraining documents is deliberate — see CLAUDE.md/the SFT design notes for why no model
# change is needed for that.

def load_chat_dataset(dataset: str, subset: str | None = None) -> DatasetDict:
    """A post-training dataset by name: an HF hub id (with an optional config `subset`), a local
    `save_to_disk` directory (what radiance-rlaif writes), or a local .json/.jsonl file.

    Shared by the SFT and DPO pipelines. A local directory holding a single `Dataset` rather than a
    `DatasetDict` is treated as its train split. Any other directory (raw data files, a cloned hub
    repo) goes to `load_dataset` as it always did, as does a hub id — the no-subset call is the
    bare `load_dataset(dataset)` it always was.
    """
    path = Path(dataset)
    # save_to_disk writes exactly one of these at the top level: dataset_dict.json for a
    # DatasetDict, state.json for a single Dataset.
    if path.is_dir() and ((path / "dataset_dict.json").is_file() or (path / "state.json").is_file()):
        loaded = load_from_disk(str(path))
        return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})
    if path.is_file() and path.suffix in (".json", ".jsonl"):
        return load_dataset("json", data_files=str(path))
    if subset is not None:
        return load_dataset(dataset, subset)
    return load_dataset(dataset)


def select_splits(
    raw: DatasetDict,
    dataset: str,
    train_split: str,
    eval_split: str | None,
    eval_fallbacks: tuple[str, ...],
    eval_split_size: int,
    max_examples: int | None,
):
    """(train, eval-or-None) from a loaded DatasetDict, by the sft/dpo split fields.

    An explicit eval_split must exist. Otherwise the first of `eval_fallbacks` the dataset has is
    used, and failing that eval_split_size rows are carved off the train split. max_examples caps
    the *train* side, after the eval carve (so the held-out rows are the same whatever the cap) and
    after a fixed-seed shuffle (so a cap samples the split rather than its head — hub datasets are
    often sorted by source).
    """
    available = ", ".join(sorted(raw.keys()))
    if train_split not in raw:
        raise ValueError(
            f"dataset {dataset!r} has no {train_split!r} split (it has: {available}). Set train_split."
        )
    train = raw[train_split]
    if len(train) == 0:
        raise ValueError(f"dataset {dataset!r}:{train_split} is empty.")
    if eval_split is not None:
        if eval_split not in raw:
            raise ValueError(f"dataset {dataset!r} has no {eval_split!r} split (it has: {available}).")
        val = raw[eval_split]
    else:
        val = next((raw[s] for s in eval_fallbacks if s in raw and s != train_split), None)
        if val is None:
            if eval_split_size > 0 and eval_split_size >= len(train):
                # Small generated datasets (a radiance-rlaif round over few prompts) hit this; the
                # alternative is an IndexError from inside datasets' take().
                raise ValueError(
                    f"eval_split_size={eval_split_size} would carve all {len(train)} rows of "
                    f"{dataset!r}:{train_split} off as validation, leaving nothing to train on."
                )
            train, val = split_off_eval(train, eval_split_size)
    if max_examples is not None and len(train) > max_examples:
        train = train.shuffle(seed=0).select(range(max_examples))
    return train, val


# radiance-rlaif's pair directories carry this manifest; defined here rather than in rlaif.py so
# the data pipelines can recognise one without importing the generation stack.
RLAIF_MANIFEST_NAME = "rlaif_manifest.json"
# Bumped when tokenize_chat_prompt's prompt ids change, since radiance-rlaif pairs store sampled
# ids that are only valid after the exact prompt they were sampled with. 2: the assistant prefix's
# merged-away tail (gpt2's bare-space token) is left to the policy instead of ending the prompt.
RLAIF_PROMPT_IDS_VERSION = 2


def local_dataset_fingerprint(dataset: str) -> str | None:
    """An identity for a radiance-rlaif pair directory's contents (latest mtime + total size over
    its files), or None for anything else. Cache keys built from the path alone go stale the moment
    the directory is rewritten — the normal case for radiance-rlaif, whose round-N output directory
    is regenerated in place. Same path+mtime+size idea dpo_data uses for the reference checkpoint.

    Limited to rlaif output (recognised by its manifest) so every other local dataset keeps the
    path-only key it had before this existed: fingerprinting those too would orphan their caches,
    and with them DPO reference log-probs that cannot be recomputed once the reference checkpoint
    is gone. A hand-maintained local dataset rewritten in place still needs its cache cleared."""
    path = Path(dataset)
    if not (path / RLAIF_MANIFEST_NAME).is_file():
        return None
    files = [p for p in path.rglob("*") if p.is_file()]
    stats = [p.stat() for p in files]
    return f"{max((s.st_mtime for s in stats), default=0)}|{sum(s.st_size for s in stats)}"


def split_key_extras(section) -> list[str]:
    """Cache-key parts for the split/subset/max_examples fields, and a radiance-rlaif pair
    directory's fingerprint — each included only when it differs from its default (or, for the
    fingerprint, only for rlaif output), so every cache key built before these existed hashes
    exactly as it did and those caches stay valid."""
    extras = []
    if section.subset is not None:
        extras.append(f"subset={section.subset}")
    if section.train_split != "train":
        extras.append(f"train_split={section.train_split}")
    if section.eval_split is not None:
        extras.append(f"eval_split={section.eval_split}")
    if section.max_examples is not None:
        extras.append(f"max_examples={section.max_examples}")
    fingerprint = local_dataset_fingerprint(section.dataset)
    if fingerprint is not None:
        extras.append(f"local={fingerprint}")
    return extras


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


def tokenize_sft_example(
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


def tokenize_chat_prompt(
    messages: list[dict], tokenizer: PreTrainedTokenizerBase, user_prefix: str, assistant_prefix: str
) -> tuple[list[int], list[int]]:
    """A conversation plus an open assistant turn, as (ids, loss_mask): the token-level form of
    format_chat_messages, built turn by turn exactly as tokenize_sft_example builds them (and with
    its mask, so the open turn's prefix tokens are 1, as an assistant turn's are).

    The open prefix stops at the last token a reply leaves alone. SFT tokenizes prefix and reply
    together, and BPE merges the prefix's tail into the reply's first word: with gpt2,
    "\\n\\nAssistant: " alone ends in a bare-space token (220), but "\\n\\nAssistant: Hello" ends
    ":", " Hello" — so a prompt ending in 220 is one SFT never trained after. Dropping the
    merged-away tail (found by tokenizing the prefix with a probe word after it) leaves the policy
    to sample " Hello" itself, as it learned to; a reply that does start with whitespace can still
    produce the bare token.

    radiance-rlaif samples after these ids and the DPO pipeline rebuilds its pairs on them, which
    is what makes a pair the policy's own token sequence rather than a re-tokenization of its text.
    """
    ids, mask = tokenize_sft_example(messages, tokenizer, user_prefix, assistant_prefix)
    ids, mask = ids[:-1], mask[:-1]  # the conversation is not over: no EOS
    prefix = tokenizer(assistant_prefix)["input_ids"]
    probe = tokenizer(assistant_prefix + "Hello")["input_ids"]
    keep = 0
    while keep < len(prefix) and keep < len(probe) and prefix[keep] == probe[keep]:
        keep += 1
    prefix = prefix[:keep]
    return ids + prefix, mask + [1] * len(prefix)


def format_sft_prompt(user_message: str, cfg: Config) -> str:
    """The same turn-formatting tokenize_sft_example applies, for a single open user turn —
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


def format_chat_messages(messages: list[dict], cfg: Config) -> str:
    """Multi-turn analog of format_chat_prompt, for an OpenAI-style [{"role", "content"}, ...]
    conversation (radiance.serve). Concatenates (prefix + content) per turn using the same
    user_prefix/assistant_prefix convention tokenize_sft_example trains on, then appends a bare
    trailing assistant_prefix (no content) to prompt generation.

    "system" (and any role other than "assistant") gets user_prefix: tokenize_sft_example has no
    distinct system-turn handling, so a checkpoint was never trained on a separate system marker —
    folding it into user_prefix matches what the model actually saw during SFT/DPO.
    """
    if not messages:
        raise ValueError("format_chat_messages requires at least one message")
    user_prefix, assistant_prefix = _chat_prefixes(cfg, "format_chat_messages")
    parts = [
        (assistant_prefix if m["role"] == "assistant" else user_prefix) + m["content"] for m in messages
    ]
    parts.append(assistant_prefix)
    return "".join(parts)


def _chat_prefixes(cfg: Config, caller: str) -> tuple[str, str]:
    """(user_prefix, assistant_prefix) of whichever post-training mode the checkpoint's own config
    has on — see format_chat_prompt for why a DPO checkpoint reads cfg.dpo's."""
    if cfg.sft.enabled:
        return cfg.sft.user_prefix, cfg.sft.assistant_prefix
    if cfg.dpo.enabled:
        return cfg.dpo.user_prefix, cfg.dpo.assistant_prefix
    raise ValueError(f"{caller} requires a checkpoint trained with sft.enabled: true or dpo.enabled: true")


def chat_prompt_ids(messages: list[dict], tokenizer: PreTrainedTokenizerBase, cfg: Config) -> list[int]:
    """The ids to sample a chat reply after: tokenize_chat_prompt under the checkpoint's own turn
    template. What `radiance-generate --chat` and radiance-serve's /v1/chat/completions prompt
    with, so a served or sampled checkpoint sees the same prompt boundary radiance-rlaif's pairs
    and judge evaluation do — tokenizing format_chat_messages' text whole would end the prompt in
    the assistant prefix's merged-away trailing-space token (see tokenize_chat_prompt), a context
    SFT never trained after."""
    if not messages:
        raise ValueError("chat_prompt_ids requires at least one message")
    user_prefix, assistant_prefix = _chat_prefixes(cfg, "chat_prompt_ids")
    ids, _ = tokenize_chat_prompt(messages, tokenizer, user_prefix, assistant_prefix)
    return ids


def _tokenize_and_pack_sft(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    seq_len = cfg.sft.seq_len or cfg.data.seq_len

    def tokenize_fn(batch):
        keys = list(batch.keys())
        rows = [dict(zip(keys, values)) for values in zip(*batch.values())]
        ids_list, mask_list = [], []
        for example in rows:
            ids, mask = tokenize_sft_example(
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
            *split_key_extras(cfg.sft),
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

    raw = load_chat_dataset(cfg.sft.dataset, cfg.sft.subset)
    train_split, val_split = select_splits(
        raw,
        cfg.sft.dataset,
        cfg.sft.train_split,
        cfg.sft.eval_split,
        ("validation",),
        cfg.sft.eval_split_size,
        cfg.sft.max_examples,
    )

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
        pin_memory=pin_memory_for(cfg),
    )

    packed = _load_or_build_sft_packed(cfg, tokenizer)
    train_ds = packed["train"]
    val_ds = packed.get("validation")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, **loader_kwargs)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader
