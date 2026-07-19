# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Radiance is an experimental LLM training framework. The current state is a minimal, from-scratch PyTorch training
pipeline: load a HuggingFace `user/dataset`-style dataset, tokenize it with an off-the-shelf HF tokenizer, and train
a configurable dense transformer on it, with W&B logging. This is meant to be a hackable base for trying non-standard
architectures/training ideas, not a production framework — prefer explicit, readable code over abstraction layers.

## Setup

No manual setup step — `uv run` creates/syncs `.venv` from `pyproject.toml`/`uv.lock` automatically on first use.

## Running training

```bash
WANDB_MODE=offline uv run radiance-train --config configs/tinystories.yaml
```

Drop `WANDB_MODE=offline` to log to your W&B account (`wandb.mode` in the config also controls this — `online`,
`offline`, or `disabled`). `configs/tinystories.yaml` is the reference config, tuned for a quick first run against
`roneneldan/TinyStories`. Copy it to start a new config for a different dataset/model size.

Real training runs should use the GPU (`train.device: auto`, the default, resolves to `cuda` when one's available —
see `resolve_device` in `config.py`). Don't drop to `train.device: cpu` for an actual run just because the GPU is
temporarily busy with another process; wait for it to free up or ask before doing a full run on CPU. CPU is fine only
for the tiny pipeline sanity-checks described below, which are explicitly meant to be cheap/throwaway, not for
anything whose numbers you intend to keep.

## Running inference

```bash
uv run radiance-generate --checkpoint checkpoints/tinystories/step_1000.pt --prompt "Once upon a time"
```

Loads the `Config` embedded in the checkpoint, rebuilds
the model and tokenizer from it, and autoregressively samples (`--temperature`, `--top-k`; `--temperature 0` for
greedy decoding). No KV cache — each step re-runs the full forward pass over the (truncated-to-`max_seq_len`) context,
which is fine at these model sizes but is the first thing to optimize if generation needs to get faster.

There is no test suite yet. To sanity-check changes to the model or data pipeline, run a tiny config (small
`seq_len`, `d_model`, `max_steps`) through `radiance.train` end-to-end on CPU before trusting a full run — see the
shapes/loss checks used during development for the pattern (construct a `Config`, build a `DenseTransformer`, run a
forward/backward pass on random token ids).

## Architecture

Everything lives under `src/radiance/`, driven entirely by a single YAML config (`radiance.config.Config`, loaded via
`load_config`). There are four modules and each maps to one stage of the pipeline:

- **`config.py`** — dataclass schema (`DataConfig`, `ModelConfig`, `TrainConfig`, `WandbConfig` nested in `Config`)
  and `load_config(path)`. This is the single source of truth for every tunable; a new hyperparameter should be added
  here first, then threaded through. Config values are plain dataclasses, not `OmegaConf`/Hydra — no CLI overrides or
  config composition, just one YAML file per run.
- **`data.py`** — `build_tokenizer(cfg)` loads an `AutoTokenizer`. `build_dataloaders(cfg, tokenizer)` calls
  `datasets.load_dataset(cfg.data.dataset)` (expects a HF `user/dataset` with `train`/`validation` splits), tokenizes,
  then **packs**: concatenates all tokenized examples (joined by EOS) into one long stream and chunks it into
  fixed-length `seq_len` blocks, discarding the remainder. This is standard causal-LM packing — sequences are *not*
  padded per-example, so `seq_len` and `model.max_seq_len` should generally match. The tokenized+packed result is
  cached to disk under `cfg.data.cache_dir` (`.gitignore`d), keyed by a hash of `dataset`/`tokenizer`/`text_column`/
  `seq_len` — subsequent runs with the same values load straight from disk instead of re-tokenizing. Changing any of
  those four fields produces a new cache entry automatically; set `cache_dir: null`/empty to disable caching.

  If the dataset has no `validation` split, set `data.eval_split_size` (default 0, disabled) to carve a deterministic
  slice of that many examples off the *front* of `train` to use as validation instead (same slice every run, so eval
  numbers stay comparable across runs); those examples are excluded from training. No-op whenever a real
  `validation` split already exists — `eval_split_size` only ever acts as a fallback.

  Setting `data.streaming: true` switches both splits to `datasets` streaming mode instead: `load_dataset(...,
  streaming=True)` + a shuffle-buffer (`data.shuffle_buffer_size`, default 1000, HF's own default) applied to the raw
  stream and again after packing, avoiding both the full download and the disk cache above (`cache_dir` is ignored
  unless `disk_cache_max_gb` is also set — see below). `DataLoader` `shuffle` is forced off for the streaming
  train loader (ordering comes from the shuffle buffer, not a sampler); with `num_workers > 0`, HF shards the stream
  across workers automatically but duplicates data across workers (with a warning) if the dataset doesn't have enough
  underlying file shards. `data.prefetch_factor` (default 2, applied to every `DataLoader`) controls how many batches
  each worker stages ahead of the training step — this is what overlaps fetch/tokenize with the forward/backward pass
  rather than blocking on it, along with `persistent_workers=True` whenever `num_workers > 0`.

  Setting `data.disk_cache_max_gb` (opt-in, default `null`, decimal GB i.e. 1 GB = 1_000_000_000 bytes) on top of
  `streaming: true` additionally enables a
  bounded, ring-buffer-style on-disk cache (`StreamingPackedDataset` in `data.py`) so repeated short runs against the
  same dataset/config don't re-fetch/re-tokenize data already streamed before: each DataLoader worker maintains its
  own manifest + shard files under `cache_dir`, replaying cached blocks before continuing the live stream, and
  flushing newly-packed blocks in `data.disk_cache_shard_size`-block shards (default 100 — keep this well below a
  typical short run's block count, or nothing ever gets cached) as it goes, evicting the oldest shard first once the
  (per-worker, per-split) budget derived from `disk_cache_max_gb` is exceeded. Caveats: the cache directory can't
  be shared between two concurrently-running training processes (a lockfile makes this fail fast rather than
  corrupt); once a worker's raw partition is fully consumed once, later epochs (including the `StopIteration`-based
  restart in `train.py`) silently replay only what fits in the cache rather than fetching new data — a one-time
  warning is logged when this happens. Size `disk_cache_max_gb` to cover a full epoch, or skip disk-cache mode
  entirely, for open-ended multi-epoch training over a dataset larger than the cache.
- **`model.py`** — `DenseTransformer`: token + learned positional embeddings, a stack of `n_layers` pre-norm
  `TransformerBlock`s, final LayerNorm, and a weight-tied LM head. Each block is `CausalSelfAttention` (uses
  `F.scaled_dot_product_attention` with `is_causal=True`, no manual mask construction) followed by `FeedForward`.
  Several `ModelConfig`/`TrainConfig` fields are stored as ratios rather than absolute values and expose the
  absolute quantity as a read-only derived property of the same name minus the ratio suffix, so the rest of the
  codebase (and `vars(cfg.model)`/`vars(cfg.train)` used for W&B logging) never needs to distinguish the two:
  `model.head_dim` (attention head size) implies `n_heads = d_model // head_dim`; `model.ffn_mult` (FFN expansion
  factor) implies `ffn_dim = round(d_model * ffn_mult)`; `train.warmup_ratio` (fraction of the run) implies
  `warmup_steps = round(max_steps * warmup_ratio)`. This keeps those quantities meaningful when sweeping `d_model`
  or `max_steps` instead of silently decoupling from them.
  `FeedForward`'s depth is configurable via `cfg.model.ffn_depth`: it stacks that many `Linear(ffn_dim) + GELU` hidden
  layers between the up- and down-projections, so `ffn_depth` controls MLP depth independently of `n_layers` (block
  count). This is the main axis intended for architecture experiments — new block/attention variants should follow
  the same `TransformerBlock`-shaped contract (`(batch, seq, d_model) -> (batch, seq, d_model)`) so they drop into
  `DenseTransformer` without changing the rest of the pipeline.

  The first block runs once; the remaining `n_layers - 1` blocks (`blocks[1:]`) form a shared-weight loop body that
  is re-run either a fixed `cfg.model.loop_count` times (default), or, when `cfg.model.use_router: true`, a learned
  number of times per token via `ACTRouter` — a small `LayerNorm + Linear(d_model, 1) + sigmoid` head implementing
  Adaptive Computation Time (Graves 2016). In router mode (`DenseTransformer._forward_act`), each token position
  accumulates its own halting probability across iterations and halts independently once that sum crosses
  `1 - cfg.model.halt_epsilon` or `cfg.model.max_loops` is reached; the loop's output is a probability-weighted sum
  of that token's per-iteration hidden states (not just the last one), and once a position halts its state is frozen
  and carried forward unchanged so later iterations' causal attention still sees a stable key/value for it. Because
  this is dense, fully-batched compute with no per-token gather/scatter, router mode does **not** save wall-clock
  compute over running `max_loops` iterations for every token — the adaptivity shows up in the loss signal
  (`ponder_cost`, see below) and in what gets accumulated into the output, not in runtime; that's the first thing to
  optimize if router mode needs to get faster. `forward()` returns `(logits, ponder_cost, mean_loop_depth)` in both
  modes — the latter two are zero scalar tensors when `use_router` is `False`, so callers have one contract either
  way. See `configs/tinystories_router.yaml` for a worked example.
- **`train.py`** — plain PyTorch training loop (no HF `Trainer`): AdamW + cosine-with-warmup LR schedule
  (`build_lr_scheduler`), manual loss computation (`compute_loss` shifts logits/labels by one position for standard
  causal LM loss), gradient clipping, periodic W&B logging (`train/loss`, `train/lm_loss`, `train/ponder_cost`,
  `train/mean_loop_depth`, `train/lr`, `val/loss`), periodic checkpointing to `cfg.train.output_dir` (raw
  `torch.save` of state dict + config), and periodic `evaluate()` against the validation split. The loop is
  step-based (`cfg.train.max_steps`), not epoch-based, and cycles the train `DataLoader` via manual `StopIteration`
  handling rather than epochs. Setting `cfg.train.tokens_per_param` (opt-in, default `null`) derives `max_steps`
  from model size instead of pinning it directly: once the model is built, `train()` overwrites `cfg.train.max_steps`
  with `round(tokens_per_param * raw_model.num_parameters() / (effective_batch_size * data.seq_len))` — e.g. `20`
  for a Chinchilla-optimal token budget — and prints/logs the resulting step count, so the same config keeps
  tracking the "right" number of steps as `model.*` fields (and therefore param count) change instead of needing
  `max_steps` hand-recomputed. `warmup_ratio` (see `TrainConfig`) is read as a live property off whatever
  `max_steps` ends up being, so warmup scales automatically along with it. See `configs/fineweb_500m.yaml` for a
  worked example; leave `tokens_per_param: null` and set `max_steps` directly for quick/pinned runs (e.g.
  `configs/tinystories.yaml`). `cfg.train.dtype` (`"fp32"`, `"fp16"`, or `"bf16"`, resolved via `resolve_dtype`)
  controls precision: the forward/loss pass runs under `torch.autocast` in that dtype while master weights and the
  optimizer state stay fp32; a `torch.amp.GradScaler` is enabled only for `fp16` (its narrow exponent range can
  underflow small gradients — `bf16` has fp32's exponent range so it needs no scaling). The training loss is
  `lm_loss + cfg.model.ponder_weight * ponder_cost` (the second term is always zero unless `use_router` is on);
  `evaluate()`'s `val/loss` stays pure LM loss (ponder cost discarded) so it's comparable across router and
  non-router runs.

- **`generate.py`** — `load_checkpoint(path, device)` reconstructs a `DenseTransformer` + tokenizer from a saved
  checkpoint (`torch.load(..., weights_only=False)`, since the checkpoint pickles the full `Config` object, not just
  tensors). `generate(...)` runs naive autoregressive sampling (temperature/top-k, no caching of past activations).
  Because there's no KV cache, the whole (truncated) context is recomputed every generation step — in router mode
  this means a given token's effective loop depth before halting can shift slightly across generation steps as
  surrounding context changes, which is expected, not a bug.

Entry points: `radiance.train:main` (`--config`) and `radiance.generate:main` (`--checkpoint`, `--prompt`, ...) —
`radiance-train` / `radiance-generate` console scripts after install.

## Extending

- New dataset: point `data.dataset` at any `user/dataset` HF dataset with `train`/`validation` splits and the right
  `text_column`; no code changes needed unless the schema differs. No `validation` split (e.g. `HuggingFaceFW/fineweb`,
  see `configs/fineweb_500m.yaml`): set `data.eval_split_size` to carve one off the front of `train` instead.
- Dataset too large to tokenize/cache up front: set `data.streaming: true` (see
  `configs/tinystories_streaming.yaml`) — trades a full local shuffle and disk cache for a streaming/shuffle-buffer
  approximation on both splits; no other config or code changes needed for a standard HF hub dataset. To also avoid
  re-fetching already-seen data across repeated short runs, additionally set `data.disk_cache_max_gb`.
- New model variant: add config fields to `ModelConfig`, then wire them into `model.py`. Keep the
  `TransformerBlock` I/O contract so `train.py` and `data.py` stay untouched. `ACTRouter` /
  `DenseTransformer._forward_act` (the learned per-token loop-halting mechanism, opt-in via `cfg.model.use_router`)
  is the reference example for a variant that changes `DenseTransformer.forward`'s control flow rather than just
  swapping in a different block.
- New training behavior (e.g. different scheduler, mixed precision): changes belong in `train.py`; keep the loop
  step-based and keep config-driven values in `TrainConfig` rather than hardcoding.
