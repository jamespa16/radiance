from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, doc_mask_is_inert_for_dpo, resolve_device, resolve_dtype
from radiance.model import load_transformer_from_checkpoint, sequence_logprob_sum

from .data import _split_off_eval
from .sft_data import _tokenize_sft_example

# DPO (preference post-training): different packing strategy from SFT's above, deliberately.
# SFT packs many chat examples per fixed-seq_len block, relying on
# doc_attention_mask/model.document_ids' EOS-boundary detection to isolate them from each
# other. DPO instead packs exactly one preference pair per dataset row — a (prompt+chosen)
# sequence and a (prompt+rejected) sequence, each independently padded to seq_len by appending
# repeated eos_token_id at the tail (loss_mask=0 on the padding). This is "packing of one" rather
# than SFT's "many examples per block", chosen because DPO's loss needs the two halves of a pair
# to survive DataLoader shuffling together — keeping both in the same row makes that automatic,
# where packing multiple different pairs per block would require reconstructing pairing from
# document ids after shuffling, which is fragile. It costs some padding-waste efficiency
# relative to SFT's packing, an accepted tradeoff at this dataset scale (thousands to tens of
# thousands of rows) and the same one reference DPO implementations (e.g. TRL) make.
#
# Note this padding scheme needs no doc_attention_mask support at all for correctness, unlike
# SFT's multi-example blocks where it's load-bearing: attention is strictly causal and the real
# (scored) content always precedes its own trailing padding within a row, so the real tokens can
# never attend to padding regardless of whether document-boundary masking is active. Whether the
# reference-logprob precompute pass below sets eos_id on its model is therefore a free choice,
# not a correctness requirement.

def _format_dpo_pair(example: dict, cfg: Config) -> tuple[list[dict], list[dict]]:
    """Normalize one dataset row to (chosen_messages, rejected_messages).

    Reads cfg.dpo.chosen_column/rejected_column directly as pre-built [{"role","content"}, ...]
    message lists that already include the prompt turn (the shape e.g. argilla/dpo-mix-7k uses)
    when cfg.dpo.prompt_column is unset. When it's set instead (Alpaca/orca-style datasets with a
    separate prompt column and plain-string chosen/rejected completions, e.g. Intel/orca_dpo_pairs),
    builds a shared 1-turn user prompt from prompt_column (+ optional system_column) and appends
    each side's completion as its own final assistant turn. Mirrors _format_sft_messages'
    instruction_column fallback.
    """
    if cfg.dpo.prompt_column is not None:
        prompt = example[cfg.dpo.prompt_column]
        if cfg.dpo.system_column:
            system = example.get(cfg.dpo.system_column)
            if system:
                prompt = f"{system}\n{prompt}"
        prompt_turns = [{"role": "user", "content": prompt}]
        chosen = prompt_turns + [{"role": "assistant", "content": example[cfg.dpo.chosen_column]}]
        rejected = prompt_turns + [{"role": "assistant", "content": example[cfg.dpo.rejected_column]}]
        return chosen, rejected
    return example[cfg.dpo.chosen_column], example[cfg.dpo.rejected_column]


def _tokenize_dpo_row(
    example: dict, tokenizer: PreTrainedTokenizerBase, cfg: Config, seq_len: int
) -> dict | None:
    """One dataset row -> a dict of 4 seq_len-padded lists, or None if either side's un-padded
    length exceeds seq_len (dropped, not truncated).

    Dropping is deliberate: truncating the completion would score a partial response as fully
    chosen/rejected, and truncating the prompt would silently change what the response is
    conditioned on — both corrupt the training signal in a way that's worse than losing the
    example. Reuses _tokenize_sft_example verbatim for both sides (now that it takes explicit
    prefixes rather than reading cfg.sft.* directly) — DPO's per-side (ids, loss_mask) construction
    is identical to SFT's single-completion case, just called twice against a shared prompt prefix.
    """
    chosen_msgs, rejected_msgs = _format_dpo_pair(example, cfg)
    chosen_ids, chosen_mask = _tokenize_sft_example(
        chosen_msgs, tokenizer, cfg.dpo.user_prefix, cfg.dpo.assistant_prefix
    )
    rejected_ids, rejected_mask = _tokenize_sft_example(
        rejected_msgs, tokenizer, cfg.dpo.user_prefix, cfg.dpo.assistant_prefix
    )
    if len(chosen_ids) > seq_len or len(rejected_ids) > seq_len:
        return None

    eos_id = tokenizer.eos_token_id

    def pad(ids: list[int], mask: list[int]) -> tuple[list[int], list[int]]:
        n_pad = seq_len - len(ids)
        return ids + [eos_id] * n_pad, mask + [0] * n_pad

    chosen_ids, chosen_mask = pad(chosen_ids, chosen_mask)
    rejected_ids, rejected_mask = pad(rejected_ids, rejected_mask)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_loss_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_loss_mask": rejected_mask,
    }


def _tokenize_and_filter_dpo(dataset, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    seq_len = cfg.dpo.seq_len or cfg.data.seq_len
    n_before = len(dataset)

    def batch_fn(batch):
        keys = list(batch.keys())
        rows = [dict(zip(keys, values)) for values in zip(*batch.values())]
        out = {
            "chosen_input_ids": [],
            "chosen_loss_mask": [],
            "rejected_input_ids": [],
            "rejected_loss_mask": [],
        }
        for example in rows:
            tokenized = _tokenize_dpo_row(example, tokenizer, cfg, seq_len)
            if tokenized is None:
                continue
            for k, v in tokenized.items():
                out[k].append(v)
        return out

    map_kwargs = {"num_proc": cfg.data.num_workers or None}
    tokenized = dataset.map(
        batch_fn,
        batched=True,
        remove_columns=dataset.column_names,
        **map_kwargs,
    )
    # Computed post-hoc from before/after dataset length rather than an in-map counter: a
    # closure-captured counter would only see whichever worker's shard it ran in under
    # num_proc > 1, silently undercounting.
    n_dropped = n_before - len(tokenized)
    if n_dropped:
        print(
            f"[radiance] dpo data prep: dropped {n_dropped}/{n_before} pairs exceeding seq_len={seq_len}",
            flush=True,
        )
    tokenized.set_format(type="torch", columns=list(tokenized.column_names))
    return tokenized


def _add_reference_logprobs(
    packed: DatasetDict, cfg: Config, tokenizer: PreTrainedTokenizerBase, device: str
) -> DatasetDict:
    """One-time pass: forward the frozen reference checkpoint over every packed pair and cache
    each side's summed log-probability as a new column.

    Not a datasets.map() call — a model forward pass can't run inside a .map() multiprocessing
    worker the way plain tokenization can — so this is a plain batched DataLoader loop instead.
    Training itself never loads this model: it's built, used, and freed entirely within this
    function, which is what keeps DPO training at exactly one resident model's VRAM cost.
    """
    ref_model, ref_cfg = load_transformer_from_checkpoint(
        cfg.dpo.reference_checkpoint, device, eos_id=tokenizer.eos_token_id
    )
    # The checkpoint was trained with whatever tokenizer its own saved config names. If that
    # differs from this run's data.tokenizer, its vocab ids don't correspond to this run's packed
    # ids, and every reference logprob below is computed over a mismatched vocabulary - and
    # sequence_logprob_sum will happily gather from a no-smaller ref vocab and return
    # numerically-plausible but semantically meaningless values, silently corrupting the DPO loss
    # from step 1 with nothing to compare them against. data.tokenizer is the tokenizer identity
    # the cache keys above already treat as canonical, so compare it here, where both configs are
    # in hand, rather than letting a mismatched reference surface as a plausible-looking loss.
    if ref_cfg.data.tokenizer != cfg.data.tokenizer:
        raise ValueError(
            f"dpo.reference_checkpoint {cfg.dpo.reference_checkpoint!r} was trained with "
            f"data.tokenizer={ref_cfg.data.tokenizer!r}, but this run uses "
            f"data.tokenizer={cfg.data.tokenizer!r}: a mismatched vocabulary, so the reference "
            "log-probabilities would be meaningless. Point dpo.reference_checkpoint at a "
            "checkpoint trained with the same data.tokenizer."
        )
    # Same skip train() applies to the policy model, for the same reason and against the same
    # predicate — the rows scored here are exactly the rows trained on, so a mask that cannot
    # change a scored logit there cannot change one here either. Applied to the *reference*
    # checkpoint's own model config, which is whatever that run was trained with and need not
    # match this run's.
    if doc_mask_is_inert_for_dpo(ref_model.cfg):
        ref_model.cfg.doc_attention_mask = False
    result = {}
    # This pass runs before the training loop, so its OOMs never reach the step loop's backoff
    # (micro_chunk_size / CPU-offload): the reference model is a second resident model on top of
    # the policy's, reference_batch_size is hand-picked, and an OOM here would just crash the run
    # at startup. It therefore carries that idiom locally: a fetched batch is processed in
    # chunks of `chunk_size`, and an OOM halves the size and retries the batch. Rows are
    # independent here — one row is one (prompt, completion) and its summed log-probability never
    # reads another row's logits — so chunked calls are mathematically identical to the single
    # call. `chunk_size` is sticky across batches, like micro_chunk_size: a size that doesn't
    # fit once doesn't fit later, and backoff never rebuilds anything, only changes how many
    # forward calls one fetched batch takes.
    device_type = device.split(":")[0]
    dtype = resolve_dtype(cfg.train.dtype)
    for split_name, split in packed.items():
        # Reset per split rather than carrying a backoff from one split into the next: nothing
        # about a split boundary here has the running step loop's "a size that doesn't fit once
        # doesn't fit later" invariant (see comment above) — each split starts a fresh, unstressed
        # DataLoader loop, so a shrink forced by e.g. the train split's data shouldn't handicap
        # validation/test, which may never hit the same OOM.
        chunk_size = cfg.dpo.reference_batch_size
        loader = DataLoader(split, batch_size=cfg.dpo.reference_batch_size, shuffle=False)
        chosen_logprobs: list[torch.Tensor] = []
        rejected_logprobs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                c_ids = batch["chosen_input_ids"].to(device)
                c_mask = batch["chosen_loss_mask"].to(device)
                r_ids = batch["rejected_input_ids"].to(device)
                r_mask = batch["rejected_loss_mask"].to(device)
                b = c_ids.size(0)
                while True:
                    batch_chosen: list[torch.Tensor] = []
                    batch_rejected: list[torch.Tensor] = []
                    try:
                        for start in range(0, b, chunk_size):
                            n = min(chunk_size, b - start)
                            c_chunk, c_chunk_mask = c_ids[start : start + n], c_mask[start : start + n]
                            r_chunk, r_chunk_mask = r_ids[start : start + n], r_mask[start : start + n]
                            with torch.autocast(
                                device_type=device_type, dtype=dtype, enabled=dtype != torch.float32
                            ):
                                logits = ref_model(torch.cat([c_chunk, r_chunk], dim=0)).logits
                            chosen_logits, rejected_logits = logits.split(n, dim=0)
                            batch_chosen.append(
                                sequence_logprob_sum(chosen_logits, c_chunk, c_chunk_mask).cpu()
                            )
                            batch_rejected.append(
                                sequence_logprob_sum(rejected_logits, r_chunk, r_chunk_mask).cpu()
                            )
                        break
                    except torch.cuda.OutOfMemoryError:
                        # The failed call's output tensors (still referenced by name) are the
                        # biggest live allocation; drop them before the retry's forward allocates.
                        logits = chosen_logits = rejected_logits = None
                        if chunk_size == 1:
                            torch.cuda.empty_cache()
                            print(
                                f"[radiance] CUDA OOM in DPO reference-logprob precompute even at batch "
                                f"size 1: a single row pair does not fit with the policy model resident. "
                                f"Lower dpo.reference_batch_size (currently {cfg.dpo.reference_batch_size}) "
                                "or shorten the sequence length, and relaunch."
                            )
                            raise
                        torch.cuda.empty_cache()
                        chunk_size = max(1, chunk_size // 2)
                        print(
                            f"[radiance] CUDA OOM in DPO reference-logprob precompute, backing off batch "
                            f"size to {chunk_size} and retrying."
                        )
                chosen_logprobs.append(torch.cat(batch_chosen))
                rejected_logprobs.append(torch.cat(batch_rejected))
        split = split.add_column("ref_chosen_logprob", torch.cat(chosen_logprobs).tolist())
        split = split.add_column("ref_rejected_logprob", torch.cat(rejected_logprobs).tolist())
        result[split_name] = split

    del ref_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return DatasetDict(result)


def _dpo_cache_path(cfg: Config) -> Path:
    """Where the packed DPO dataset (reference log-prob columns included) is cached on disk.

    Under `cache_dir`, keyed on every field this pipeline derives from, in two levels rather
    than one digest: a base digest over the dataset/tokenizer/columns/prefixes fields, and a
    `ref-*` subdirectory digesting the reference checkpoint's identity (path+mtime+size, not a
    content hash — hashing a multi-hundred-MB-to-multi-GB checkpoint on every run start would
    defeat the point of a fast cache-hit path, and this repo doesn't content-hash checkpoint
    files anywhere else either. Changing the reference model (even to a same-path file with new
    contents) invalidates the cache since either mtime or size will differ).

    The checkpoint is only ever needed to *compute* the reference log-probs; once that pass has
    run and the packed dataset is cached, no later step touches the file. If it has since been
    archived or moved away, its identity is no longer computable, so the cache built while it
    was present is resolved by listing `ref-*` under the base digest instead. That is
    unambiguous only when exactly one such cache exists for this base key; with zero there is
    nothing to load and no checkpoint to rebuild from, and with several it is not knowable which
    reference model's log-probs a given cache holds — reusing the wrong one corrupts the DPO
    loss silently — so both raise.
    """
    seq_len = cfg.dpo.seq_len or cfg.data.seq_len
    base_key = "|".join(
        [
            cfg.dpo.dataset,
            cfg.data.tokenizer,
            str(seq_len),
            str(cfg.dpo.prompt_column),
            str(cfg.dpo.system_column),
            cfg.dpo.chosen_column,
            cfg.dpo.rejected_column,
            cfg.dpo.user_prefix,
            cfg.dpo.assistant_prefix,
            str(cfg.dpo.eval_split_size),
        ]
    )
    base_dir = Path(cfg.dpo.cache_dir) / hashlib.sha256(base_key.encode()).hexdigest()[:16]

    ref_path = Path(cfg.dpo.reference_checkpoint)
    try:
        ref_stat = ref_path.stat()
    except FileNotFoundError:
        candidates = (
            sorted(d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("ref-"))
            if base_dir.is_dir()
            else []
        )
        if not candidates:
            raise ValueError(
                f"dpo.reference_checkpoint {cfg.dpo.reference_checkpoint!r} does not exist, and "
                f"no cached DPO dataset for this dataset/config was found under "
                f"{cfg.dpo.cache_dir!r}: the reference log-probabilities are computed from that "
                "checkpoint, so there is nothing to load and nothing to rebuild it from. Restore "
                "the checkpoint (or point dpo.reference_checkpoint at its new location)."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"dpo.reference_checkpoint {cfg.dpo.reference_checkpoint!r} does not exist, and "
                f"{len(candidates)} cached DPO datasets built against different reference "
                f"checkpoints were found under {base_dir}: it is not possible to tell which "
                "reference model the reference log-probabilities were computed against, and "
                "reusing the wrong one would silently corrupt the DPO loss. Point "
                "dpo.reference_checkpoint at the intended (still-present) checkpoint to use its "
                f"cache, or delete the ones you no longer need ({', '.join(c.name for c in candidates)})."
            )
        return candidates[0]
    ref_key = f"{cfg.dpo.reference_checkpoint}|{ref_stat.st_mtime}|{ref_stat.st_size}"
    return base_dir / ("ref-" + hashlib.sha256(ref_key.encode()).hexdigest()[:16])


def dpo_cache_exists(cfg: Config) -> bool:
    """Whether _load_or_build_dpo_packed will hit the disk cache rather than rebuild it.

    Lets a caller (train.py's batch-size estimate) know, before touching any GPU memory, whether a
    DPO cache miss is about to load a second full transformer for reference-logprob precompute.
    An ambiguous or missing reference checkpoint is reported here as "no cache" rather than raised
    — build_dpo_dataloaders still raises _dpo_cache_path's real error on the actual build path.
    """
    if not cfg.dpo.cache_dir:
        return False
    try:
        return _dpo_cache_path(cfg).exists()
    except ValueError:
        return False


def _load_or_build_dpo_packed(cfg: Config, tokenizer: PreTrainedTokenizerBase):
    cache_path = _dpo_cache_path(cfg)
    if cfg.dpo.cache_dir and cache_path.exists():
        packed = load_from_disk(str(cache_path))
        packed.set_format(type="torch")
        return packed

    raw = load_dataset(cfg.dpo.dataset)
    train_split = raw["train"]
    val_split = raw.get("validation")
    if val_split is None:
        val_split = raw.get("test")
    if val_split is None:
        train_split, val_split = _split_off_eval(train_split, cfg.dpo.eval_split_size)

    packed = DatasetDict({"train": _tokenize_and_filter_dpo(train_split, tokenizer, cfg)})
    if val_split is not None:
        packed["validation"] = _tokenize_and_filter_dpo(val_split, tokenizer, cfg)

    device = resolve_device(cfg.train.device)
    packed = _add_reference_logprobs(packed, cfg, tokenizer, device)

    if cfg.dpo.cache_dir:
        packed.save_to_disk(str(cache_path))

    # add_column (inside _add_reference_logprobs) doesn't necessarily preserve the "torch" format
    # set on the original 4 columns, and never applies to the 2 columns it just added — re-apply
    # unconditionally so this freshly-built path returns tensors exactly like the cache-hit path
    # above, rather than plain Python lists.
    packed.set_format(type="torch")

    return packed


def build_dpo_dataloaders(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[DataLoader, DataLoader | None]:
    """DPO analogue of build_sft_dataloaders: same (train_loader, val_loader) contract, but each
    batch carries chosen/rejected input_ids+loss_mask plus their cached reference log-probs (see
    module docstring above)."""
    if not cfg.dpo.dataset:
        raise ValueError("dpo.enabled requires dpo.dataset to be set.")
    if not cfg.dpo.reference_checkpoint:
        raise ValueError("dpo.enabled requires dpo.reference_checkpoint to be set.")

    def collate(batch):
        cols = [
            "chosen_input_ids",
            "chosen_loss_mask",
            "rejected_input_ids",
            "rejected_loss_mask",
            "ref_chosen_logprob",
            "ref_rejected_logprob",
        ]
        return {c: torch.stack([torch.as_tensor(ex[c]) for ex in batch]) for c in cols}

    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        collate_fn=collate,
        drop_last=True,
    )

    packed = _load_or_build_dpo_packed(cfg, tokenizer)
    train_ds = packed["train"]
    val_ds = packed.get("validation")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, **loader_kwargs)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader
