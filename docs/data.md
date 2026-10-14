# `data.py` / `sft_data.py` / `dpo_data.py` — dataset loading, tokenization, packing

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
oldest shard first once the per-worker, per-split budget is exceeded. A resumed run skips the blocks it already
trained on instead of replaying them from the start; see [train.md](train.md#checkpoints-and-resume).

Two bugs here each duplicated or dropped data silently, so they're worth knowing about before you change this code:

- **Never skip the raw stream with `raw.skip()`.** Inside a DataLoader worker, HF `datasets` re-shards an
  `IterableDataset` across workers itself and *divides* any pending `skip(n)` by `num_workers`. `n_raw_consumed` is
  already a per-worker count, so `raw.skip(n_raw_consumed)` skipped only 1/`num_workers` of it, and every reuse of
  the cache re-tokenized and re-trained on most of what the cache held. The skip is now `itertools.islice` over the
  fully sharded per-worker stream. (The same re-sharding stacks on the explicit `raw.shard()`, so each of `N` workers
  reads 1/`N`² of the corpus's files and the run as a whole reaches only 1/`N` of them — plenty for fineweb, but a
  small corpus runs out sooner than you'd expect. Left as-is: changing it re-maps every existing cache.)
- **Flush only between tokenize batches.** `raw_consumed_since_flush` counts a whole tokenize batch at once, so a
  flush partway through one recorded raw examples whose blocks were never cached; the next run resumed the raw stream
  past them. The sub-block token tail left in `token_buffer` at a flush is still lost (under one block per flush).

Caveats. The cache directory can't be shared between two concurrently-running training processes (a lockfile makes
this fail fast rather than corrupt). Once a worker's raw partition is fully consumed once, later epochs (including
`train.py`'s `StopIteration`-based restart) silently replay only what fits in the cache rather than fetching new
data — a one-time warning is logged. Size `disk_cache_max_gb` to cover a full epoch, or skip disk-cache mode
entirely, for open-ended multi-epoch training over a dataset larger than the cache.

## Pretokenized data

`data.pretokenized_dir` (opt-in, default `null`) trains from flat token files on local disk, written beforehand by
`radiance-pretokenize --config <cfg>` (`pretokenize.py`), instead of any of the paths above. It exists because V1
streamed fineweb over the network for its whole run and a network outage ended it. The tool reads the same config's
`dataset`/`text_column`/`tokenizer`/`eval_split_size`, writes `val.bin` (the held-out documents — exactly the carve
above, so the same 2,000 documents V1 scored), one `train_NNN.bin` per worker (documents joined by EOS, the same
packing as everywhere else) and `meta.json` last, whose presence marks the directory complete. By default it writes
the tokens the config's run will draw (from `tokens_per_param` or `max_steps`) plus 1%; `--train-tokens` overrides.

- **Where the tokens come from.** Each worker reads a disjoint 1/P of the corpus's files, in a seeded shuffled
  order (`IterableDataset.shuffle` permutes files as well as buffering documents), so a partial pass samples the
  whole corpus rather than its first files — fineweb's are ordered by crawl dump. The held-out documents are
  removed from train by content hash, since the shuffled stream can reach them.
- **Interruption.** Each worker fsyncs its file and then records its position in `train_NNN.json`; re-running the
  same command truncates each file to that position and continues. The seeded shuffle replays the same order, so
  the result is byte-identical to an uninterrupted run (`tests/test_pretokenized.py` pins this). A position is a
  document count into one worker's shard, so it holds only under the partition it was recorded with: the first run
  writes that partition (worker count, quota, seed, tokenizer, ...) to `plan.json`, and a re-run takes its worker
  count from there whatever `--num-proc` says, and refuses any other change rather than append one partition's
  documents to another's files.
- **Workers hard-exit** (`os._exit`) once their output is durable. A worker stops at its quota partway through a live
  HF stream, and that stream's teardown is unbounded: in-flight parquet reads fail against closing sockets and
  retry with backoff (`'[Errno 9] Bad file descriptor' ... Retrying`). Measured at ~40 s per exit on a healthy
  connection; the first real run on a flaky one hung after all its data was written.
- **Reading.** `PretokenizedDataset` cuts `seq_len` blocks through `np.memmap` at read time (so one directory serves
  any `seq_len`), and `EpochPermutationSampler` draws a seeded permutation over *every* block per epoch. An epoch's
  order is a pure function of `(seed, epoch)`, so the data position is one integer and resume is exact — no block
  repeated, none skipped — where the streaming disk cache is exact only up to its shuffle buffer. The checkpoint
  records that integer in blocks together with the batch size it was drawn at, because an epoch is
  `floor(n / batch_size) * batch_size` blocks: a resume that picks a different batch size (`auto_batch_size`
  re-estimating against busier VRAM) maps through (epoch, offset) instead of multiplying a batch count by the new
  size, which would retrain or skip blocks. The loader checks
  `meta.json`'s tokenizer and EOS id against the config and refuses a mismatch.

`uint16` storage while the vocabulary fits (V2: ~18.5 GB for 9.2B tokens). `data.dataset_mix` is rejected alongside
it, since the directory holds one corpus.

## Validation is never shuffled

Every validation loader yields the same blocks in the same order on every eval. The streaming disk-cache path used
to run validation through the same shuffle buffer as train, and `evaluate()` reads only the first
`eval_max_batches` — so under `persistent_workers`, whose `random` state carries across evals, **each eval scored a
different random subset of the held-out set**. V1 scored ~400 of ~2,000 held-out blocks per eval, redrawn each time,
which is the ~0.02 wobble in its `val/loss` (and consistent with the ~0.019 stderr a *full* pass over those blocks
measures). `tests/test_run_observability.py` pins two consecutive evals to identical blocks.

## Dataset mix

`data.dataset_mix` (a path to a YAML file, default `null`) trains on a **weighted blend of several
corpora** at once instead of the single `data.dataset`. While set, `data.dataset` / `data.text_column`
are ignored; the mix file lists one `{dataset, [weight], [text_column]}` entry per corpus (see
`configs/dataset_mix.yaml` + `configs/tinystories_mix.yaml`). A relative path resolves against the
**config file's directory**, so the mix file sits next to the config. Every corpus shares the rest of
the data section — `tokenizer`, `seq_len`, `streaming`, `cache_dir`, `eval_split_size`, etc. Only the
corpus id, its `text_column`, and its `weight` differ per entry.

`weight` is a corpus's share of the training **tokens** (normalised across the mix), not a repetition
count. Each corpus is tokenized+packed **independently** — with its own per-corpus cache entry (keyed by
corpus / `text_column` / `tokenizer` / `seq_len` / `eval_split_size`, so changing one entry's weight or
adding a corpus never re-tokenizes the others) — and the packed `seq_len` blocks are then combined:

- **Non-streaming** — in-memory corpora are per-corpus shuffled and interleaved with
  `datasets.interleave_datasets(..., stopping_strategy="all_exhausted")`: a finite epoch in which every
  corpus is fully covered and each contributes ~its weight of the blocks.
- **Streaming** (with or without `disk_cache_max_gb`) — the live per-corpus block streams are interleaved
  by a small `_WeightedInterleave`: effectively infinite, each draw from a corpus chosen by weight, an
  exhausted corpus replayed, an empty one dropped. With `disk_cache_max_gb` each corpus wraps its own
  `StreamingPackedDataset`, and the bounded cache is partitioned across all `(corpus, split)` namespaces
  so the combined footprint stays within the budget.

The mixed **validation** split is the same mix restricted to the corpora that have one (a corpus with no
`validation` split and no `data.eval_split_size` carve simply doesn't contribute to `val/loss`; `val/loss`
is then measured on that subset — the same limitation the single-dataset path has). A corpus that packs to
zero blocks at the configured `seq_len` is skipped from the train mix with a warning; if *all* of them do,
the run raises.

## SFT and DPO paths

`_format_sft_messages`/`_tokenize_sft_example`, `_tokenize_and_pack_sft` and `format_chat_prompt` live in
`sft_data.py`; `_tokenize_dpo_row`, `_add_reference_logprobs` and the DPO cache/packing path live in
`dpo_data.py` — see [post-training.md](post-training.md).
