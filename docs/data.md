# `data.py` — dataset loading, tokenization, packing

`build_tokenizer(cfg)` loads an `AutoTokenizer`. `build_dataloaders(cfg, tokenizer)` calls
`datasets.load_dataset(cfg.data.dataset)` (expects a HF `user/dataset` with `train`/`validation` splits), tokenizes,
then **packs**: concatenates all tokenized examples (joined by EOS) into one long stream and chunks it into
fixed-length `seq_len` blocks, discarding the remainder. This is standard causal-LM packing — sequences are *not*
padded per example, so `seq_len` and `model.max_seq_len` should generally match.

## Disk cache

The tokenized+packed result is cached under `cfg.data.cache_dir` (`.gitignore`d), keyed by a hash of
`dataset`/`tokenizer`/`text_column`/`seq_len`. Changing any of those four produces a new cache entry automatically;
`cache_dir: null`/empty disables caching.

## Validation split fallback

If the dataset has no `validation` split, `data.eval_split_size` (default 0, disabled) carves a deterministic slice
of that many examples off the *front* of `train` (same slice every run, so eval numbers stay comparable across runs);
those examples are excluded from training. No-op whenever a real `validation` split exists — it only ever acts as a
fallback.

## Streaming

`data.streaming: true` switches both splits to `datasets` streaming mode: `load_dataset(..., streaming=True)` plus a
shuffle buffer (`data.shuffle_buffer_size`, default 1000, HF's own default) applied to the raw stream and again after
packing, avoiding both the full download and the disk cache (`cache_dir` is ignored unless `disk_cache_max_gb` is
also set). `DataLoader` `shuffle` is forced off for the streaming train loader — ordering comes from the shuffle
buffer, not a sampler. With `num_workers > 0`, HF shards the stream across workers automatically but duplicates data
across workers (with a warning) if the dataset doesn't have enough underlying file shards.

`data.prefetch_factor` (default 2, applied to every `DataLoader`) controls how many batches each worker stages ahead;
this plus `persistent_workers=True` whenever `num_workers > 0` is what overlaps fetch/tokenize with the
forward/backward pass rather than blocking on it.

### Bounded disk cache under streaming

`data.disk_cache_max_gb` (opt-in, default `null`, decimal GB) on top of `streaming: true` adds a ring-buffer-style
on-disk cache (`StreamingPackedDataset`) so repeated short runs against the same dataset/config don't re-fetch or
re-tokenize. Each DataLoader worker keeps its own manifest + shard files under `cache_dir`, replaying cached blocks
before continuing the live stream and flushing newly-packed blocks in `data.disk_cache_shard_size`-block shards
(default 100 — keep this well below a typical short run's block count, or nothing ever gets cached), evicting the
oldest shard first once the per-worker, per-split budget is exceeded.

Caveats. The cache directory can't be shared between two concurrently-running training processes (a lockfile makes
this fail fast rather than corrupt). Once a worker's raw partition is fully consumed once, later epochs (including
`train.py`'s `StopIteration`-based restart) silently replay only what fits in the cache rather than fetching new
data — a one-time warning is logged. Size `disk_cache_max_gb` to cover a full epoch, or skip disk-cache mode
entirely, for open-ended multi-epoch training over a dataset larger than the cache.

## SFT and DPO paths

`_format_sft_messages`/`_tokenize_sft_example`, `_tokenize_and_pack_sft`, `_tokenize_dpo_row`,
`_add_reference_logprobs` and `format_chat_prompt` live here too — see [post-training.md](post-training.md).
