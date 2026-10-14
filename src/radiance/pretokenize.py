"""Tokenize a run's training data to local disk once, before training: `radiance-pretokenize`.

V1 streamed fineweb over the network for all 47 hours, so a network outage could stop the run and
one did. This moves every network read to before the run starts. The output is flat token files
that `data.pretokenized_dir` trains from (see data.PretokenizedDataset), which also buys two
things streaming could not give:

  * **A global shuffle.** Training draws a seeded permutation over every block, not a local
    shuffle buffer (EpochPermutationSampler).
  * **An exact resume.** The data position is one integer, so a resumed run neither repeats nor
    skips a block.

Layout of `data.pretokenized_dir`:

    val.bin              the held-out documents, in stream order
    train_NNN.bin        one file per worker: documents joined by EOS, same packing as data.py
    train_NNN.json       that worker's progress (documents consumed, tokens written)
    plan.json            the partition that progress is relative to (workers, quota, seed, ...)
    meta.json           written last; its presence is what marks the directory complete

Each worker reads a disjoint 1/P of the corpus's files (`IterableDataset.shard`) in a seeded
shuffled order (`.shuffle`, which permutes files as well as buffering documents), so the tokens
come from across the whole corpus rather than its first few files — fineweb's files are ordered
by crawl dump. The held-out set is exactly what data.py carves: the first `data.eval_split_size`
documents of the unshuffled stream, so val/loss is scored on the same documents as V1's. Those
documents are removed from training by content hash, since the shuffled stream can reach them.

Interrupting it is safe. Progress is flushed with each worker's file, so re-running the same
command truncates each file to its last recorded position and continues from there; the seeded
shuffle replays the same order, so the result is identical to an uninterrupted run. That holds only
under the partition the progress was recorded against, so the first run writes it to plan.json and
a re-run adopts its worker count (whatever --num-proc says) and refuses any other change.

    radiance-pretokenize --config configs/v2.yaml
    radiance-pretokenize --config configs/v2.yaml --train-tokens 2e8   # a small test directory
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from datasets import load_dataset

from radiance.config import Config, load_config
from radiance.data import (
    PRETOKENIZED_FORMAT,
    PRETOKENIZED_META,
    build_tokenizer,
    token_byte_lengths,
)
from radiance.size import size_config

# Documents per tokenizer call. Large enough that batched encoding amortises its overhead.
_TOKENIZE_BATCH = 256
# Progress is made durable (file fsynced, then position recorded) about this often per worker.
_FLUSH_EVERY_TOKENS = 50_000_000
# Headroom over the tokens the run will draw: each file loses its sub-block tail at read time.
_HEADROOM = 1.01


def run_tokens(cfg: Config, n_vocab: int) -> int:
    """Tokens `train()` will draw for this config, derived the same way it derives max_steps."""
    tokens_per_step = cfg.train.effective_batch_size * cfg.resolved_row_tokens
    if cfg.train.tokens_per_param is not None:
        active = size_config(cfg, n_vocab)["active"]
        steps = max(1, round(cfg.train.tokens_per_param * active / tokens_per_step))
    else:
        steps = cfg.train.max_steps
    return steps * tokens_per_step


def _doc_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _token_dtype(n_vocab: int) -> str:
    return "uint16" if n_vocab <= 2**16 else "uint32"


def _append_docs(f, tokenizer, texts: list[str], dtype: str) -> tuple[int, int]:
    """Tokenize `texts`, write them to `f` joined by EOS, return (tokens, utf-8 bytes)."""
    ids = tokenizer(texts)["input_ids"]
    eos = tokenizer.eos_token_id
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.chain(doc, (eos,)) for doc in ids),
        dtype=dtype,
        count=sum(len(doc) + 1 for doc in ids),
    )
    flat.tofile(f)
    return len(flat), sum(len(t.encode("utf-8")) for t in texts)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


_PLAN = "plan.json"


def _resolve_plan(out: Path, requested: dict) -> dict:
    """The partition this directory is built under: `requested` on a fresh directory (recorded to
    plan.json before any worker starts), the recorded plan on a re-run.

    A worker's progress is a document count into *its* shard of the stream, so it means something
    only under the worker count, quota and stream order it was recorded with. Resumed under a
    different --num-proc, worker i would stream a different file set, skip a count of documents
    consumed from another, and append to the old train_i.bin: duplicated and mixed corpus
    segments, files the new plan never finishes, and no error. So the worker count (a resource
    choice) is taken from the recorded plan, and every other field (what the output contains) must
    match.
    """
    path = out / _PLAN
    if not path.exists():
        stale = sorted(out.glob("train_*.json"))
        if stale and len(stale) != requested["workers"]:
            raise ValueError(
                f"{out} holds progress for {len(stale)} workers from a run that predates {_PLAN}, "
                f"but this run would use {requested['workers']}: their positions mean nothing under "
                f"a different partition. Re-run with --num-proc {len(stale)}, or delete {out}."
            )
        _write_json(path, requested)
        return requested
    recorded = json.loads(path.read_text())
    changed = [k for k in requested if k not in ("workers", "quota") and recorded.get(k) != requested[k]]
    if changed:
        diffs = ", ".join(f"{k}: {recorded.get(k)!r} -> {requested[k]!r}" for k in changed)
        raise ValueError(
            f"{out} was started under a different plan ({diffs}); the progress recorded there does "
            f"not apply to this one. Restore the original settings to finish it, or delete {out}."
        )
    if recorded["workers"] != requested["workers"]:
        print(
            f"[radiance] resuming {out} with its recorded {recorded['workers']} workers, not "
            f"{requested['workers']}: its progress is relative to that partition",
            flush=True,
        )
    return recorded


def _val_stream(cfg: Config):
    """The held-out documents, chosen exactly as data.py chooses them: a real validation split if
    the dataset has one, else the first eval_split_size documents of train. Returns (docs, carved)."""
    splits = load_dataset(cfg.data.dataset, streaming=True)
    if "validation" in splits:
        return splits["validation"], False
    if cfg.data.eval_split_size <= 0:
        return None, False
    return splits["train"].take(cfg.data.eval_split_size), True


def _write_val(cfg: Config, out: Path, dtype: str, tokenizer) -> tuple[dict | None, set[bytes]]:
    docs, carved = _val_stream(cfg)
    if docs is None:
        return None, set()
    column = cfg.data.text_column
    texts = [ex[column] for ex in docs]
    path = out / "val.bin"
    with open(path, "wb") as f:
        tokens, n_bytes = 0, 0
        for i in range(0, len(texts), _TOKENIZE_BATCH):
            t, b = _append_docs(f, tokenizer, texts[i : i + _TOKENIZE_BATCH], dtype)
            tokens, n_bytes = tokens + t, n_bytes + b
    # Only a carve can overlap train; a real validation split is disjoint by construction.
    exclude = {_doc_hash(t) for t in texts} if carved else set()
    return {"file": path.name, "docs": len(texts), "tokens": tokens, "bytes": n_bytes}, exclude


def _write_train_shard(
    cfg: Config, index: int, num_shards: int, quota: int, out: Path, dtype: str, exclude: set[bytes]
) -> dict:
    """One worker: stream its 1/num_shards of the corpus to train_{index}.bin until `quota` tokens.

    Top-level (not a closure) so the spawn start method can pickle it.
    """
    tokenizer = build_tokenizer(cfg)
    bin_path = out / f"train_{index:03d}.bin"
    progress_path = out / f"train_{index:03d}.json"
    progress = {"docs": 0, "tokens": 0, "bytes": 0, "excluded": 0, "done": False}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    if progress["done"]:
        return progress

    stream = (
        load_dataset(cfg.data.dataset, split="train", streaming=True)
        .shard(num_shards=num_shards, index=index)
        .shuffle(seed=cfg.train.seed + index, buffer_size=cfg.data.shuffle_buffer_size)
    )
    # Resume: drop whatever was written after the last durable position, and skip the documents
    # that produced what remains. islice, not .skip(): see docs/data.md on HF's skip under sharding.
    docs_iter = itertools.islice(stream, progress["docs"], None)
    column = cfg.data.text_column

    with open(bin_path, "ab") as f:
        f.truncate(progress["tokens"] * np.dtype(dtype).itemsize)
        f.seek(0, os.SEEK_END)
        since_flush = 0
        exhausted = False
        while progress["tokens"] < quota:
            batch = list(itertools.islice(docs_iter, _TOKENIZE_BATCH))
            if not batch:
                exhausted = True
                break
            texts = [ex[column] for ex in batch]
            kept = [t for t in texts if _doc_hash(t) not in exclude] if exclude else texts
            tokens, n_bytes = _append_docs(f, tokenizer, kept, dtype) if kept else (0, 0)
            progress["docs"] += len(batch)
            progress["excluded"] += len(texts) - len(kept)
            progress["tokens"] += tokens
            progress["bytes"] += n_bytes
            since_flush += tokens
            if since_flush >= _FLUSH_EVERY_TOKENS:
                f.flush()
                os.fsync(f.fileno())
                _write_json(progress_path, progress)
                since_flush = 0
                print(f"[radiance] worker {index}: {progress['tokens'] / 1e6:,.0f}M / {quota / 1e6:,.0f}M tokens", flush=True)
        f.flush()
        os.fsync(f.fileno())
    progress["done"] = True
    progress["exhausted"] = exhausted
    _write_json(progress_path, progress)
    return progress


def _shard_process(*args) -> None:
    """Process entry point for one worker: run it, then hard-exit.

    A worker stops at its quota partway through a live HF stream, and tearing that stream down
    is not bounded: its in-flight parquet reads fail against closing sockets and retry with
    backoff, and the process cannot exit until they give up — measured at ~40 s on a healthy
    connection, and indefinitely long on a bad one, with the worker's output already complete.
    os._exit skips that teardown. It is safe only because _write_train_shard has fsynced its
    token file and written its progress JSON before returning; nothing else is left to flush.
    """
    code = 0
    try:
        _write_train_shard(*args)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def pretokenize(cfg: Config, train_tokens: int | None = None, num_proc: int | None = None) -> dict:
    if not cfg.data.pretokenized_dir:
        raise ValueError("data.pretokenized_dir is not set in this config; it names the output directory")
    if cfg.data.dataset_mix:
        raise ValueError("radiance-pretokenize handles a single data.dataset, not a data.dataset_mix")
    out = Path(cfg.data.pretokenized_dir)
    if (out / PRETOKENIZED_META).exists():
        raise FileExistsError(f"{out} is already complete ({PRETOKENIZED_META} exists); delete it to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg)
    n_vocab = len(tokenizer)
    dtype = _token_dtype(n_vocab)
    target = train_tokens if train_tokens is not None else math.ceil(run_tokens(cfg, n_vocab) * _HEADROOM)

    print(f"[radiance] held-out set -> {out / 'val.bin'}", flush=True)
    val, exclude = _write_val(cfg, out, dtype, tokenizer)

    n_files = load_dataset(cfg.data.dataset, split="train", streaming=True).num_shards
    workers = max(1, min(num_proc or cfg.data.num_workers or 1, n_files))
    plan = _resolve_plan(
        out,
        {
            "dataset": cfg.data.dataset,
            "text_column": cfg.data.text_column,
            "tokenizer": cfg.data.tokenizer,
            "eval_split_size": cfg.data.eval_split_size,
            "seed": cfg.train.seed,
            "shuffle_buffer_size": cfg.data.shuffle_buffer_size,
            "dtype": dtype,
            "target": target,
            "workers": workers,
            "quota": math.ceil(target / workers),
        },
    )
    workers, quota = plan["workers"], plan["quota"]
    print(
        f"[radiance] {target / 1e9:.2f}B train tokens from {cfg.data.dataset} ({n_files:,} files) "
        f"with {workers} workers, {quota / 1e6:,.0f}M each -> {out}",
        flush=True,
    )
    if workers == 1:
        shards = [_write_train_shard(cfg, 0, 1, quota, out, dtype, exclude)]
    else:
        # spawn, not fork: each worker builds its own tokenizer and dataset stream, and a fast
        # tokenizer's thread pool does not survive a fork. Plain Processes rather than a pool so
        # each worker can hard-exit (see _shard_process); results come back through the progress
        # files, which are durable before any worker exits.
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_shard_process, args=(cfg, i, workers, quota, out, dtype, exclude))
            for i in range(workers)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        failed = [i for i, proc in enumerate(procs) if proc.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"pretokenize workers {failed} failed (tracebacks above). Progress is saved: "
                "re-run the same command to continue from where each worker stopped."
            )
        shards = [json.loads((out / f"train_{i:03d}.json").read_text()) for i in range(workers)]

    for i, shard in enumerate(shards):
        if shard.get("exhausted") and shard["tokens"] < quota:
            print(f"[radiance] WARNING: worker {i}'s partition ran out at {shard['tokens']:,} of {quota:,} tokens")
    files = [f"train_{i:03d}.bin" for i, shard in enumerate(shards) if shard["tokens"]]
    meta = {
        "format": PRETOKENIZED_FORMAT,
        "dataset": cfg.data.dataset,
        "text_column": cfg.data.text_column,
        "tokenizer": cfg.data.tokenizer,
        "eos_id": tokenizer.eos_token_id,
        "n_vocab": n_vocab,
        "dtype": dtype,
        "seed": cfg.train.seed,
        "eval_split_size": cfg.data.eval_split_size,
        "val": val,
        "train": {
            "files": files,
            "tokens": sum(s["tokens"] for s in shards),
            "docs": sum(s["docs"] for s in shards),
            "bytes": sum(s["bytes"] for s in shards),
            "excluded_val_docs": sum(s["excluded"] for s in shards),
        },
    }
    if val:
        _check_byte_table(tokenizer, n_vocab, out / val["file"], dtype, val)
    _write_json(out / PRETOKENIZED_META, meta)
    print(
        f"[radiance] done: {meta['train']['tokens'] / 1e9:.3f}B train tokens in {len(files)} files, "
        f"{val['tokens'] if val else 0:,} val tokens; {meta['train']['excluded_val_docs']} held-out "
        "documents removed from train",
        flush=True,
    )
    return meta


def _check_byte_table(tokenizer, n_vocab: int, val_path: Path, dtype: str, val: dict) -> None:
    """Cross-check val/bits_per_byte's denominator against the real text before any run uses it:
    the per-token byte lengths summed over the tokenized held-out set must equal its UTF-8 size."""
    lengths = token_byte_lengths(tokenizer, n_vocab)
    if lengths is None:
        print("[radiance] tokenizer is not byte-level BPE, so val/bits_per_byte will not be logged")
        return
    ids = np.fromfile(val_path, dtype=dtype).astype(np.int64)
    from_tokens = int(lengths.numpy()[ids].sum())
    val["bytes_from_tokens"] = from_tokens
    status = "exact" if from_tokens == val["bytes"] else "MISMATCH — val/bits_per_byte will be off"
    print(f"[radiance] byte table check: {from_tokens:,} bytes from tokens vs {val['bytes']:,} from text ({status})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a config's training data to local disk.")
    parser.add_argument("--config", required=True, help="config whose data section to tokenize; "
                        "data.pretokenized_dir names the output directory")
    parser.add_argument("--train-tokens", type=float, help="train tokens to write (default: what the "
                        "config's run will draw, +1%%)")
    parser.add_argument("--num-proc", type=int, help="worker processes (default: data.num_workers)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pretokenize(cfg, int(args.train_tokens) if args.train_tokens else None, args.num_proc)
    # With --num-proc 1 the worker ran in this process, and its abandoned stream would stall
    # interpreter exit exactly as _shard_process describes. meta.json is written; nothing is left.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
