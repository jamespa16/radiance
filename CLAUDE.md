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
- **`model.py`** — `DenseTransformer`: token + learned positional embeddings, a stack of `n_layers` pre-norm
  `TransformerBlock`s, final LayerNorm, and a weight-tied LM head. Each block is `CausalSelfAttention` (uses
  `F.scaled_dot_product_attention` with `is_causal=True`, no manual mask construction) followed by `FeedForward`.
  `FeedForward`'s depth is configurable via `cfg.model.ffn_depth`: it stacks that many `Linear(ffn_dim) + GELU` hidden
  layers between the up- and down-projections, so `ffn_depth` controls MLP depth independently of `n_layers` (block
  count). This is the main axis intended for architecture experiments — new block/attention variants should follow
  the same `TransformerBlock`-shaped contract (`(batch, seq, d_model) -> (batch, seq, d_model)`) so they drop into
  `DenseTransformer` without changing the rest of the pipeline.
- **`train.py`** — plain PyTorch training loop (no HF `Trainer`): AdamW + cosine-with-warmup LR schedule
  (`build_lr_scheduler`), manual loss computation (`compute_loss` shifts logits/labels by one position for standard
  causal LM loss), gradient clipping, periodic W&B logging (`train/loss`, `train/lr`, `val/loss`), periodic
  checkpointing to `cfg.train.output_dir` (raw `torch.save` of state dict + config), and periodic `evaluate()` against
  the validation split. The loop is step-based (`cfg.train.max_steps`), not epoch-based, and cycles the train
  `DataLoader` via manual `StopIteration` handling rather than epochs. `cfg.train.dtype` (`"fp32"`, `"fp16"`, or
  `"bf16"`, resolved via `resolve_dtype`) controls precision: the forward/loss pass runs under `torch.autocast` in
  that dtype while master weights and the optimizer state stay fp32; a `torch.amp.GradScaler` is enabled only for
  `fp16` (its narrow exponent range can underflow small gradients — `bf16` has fp32's exponent range so it needs no
  scaling).

- **`generate.py`** — `load_checkpoint(path, device)` reconstructs a `DenseTransformer` + tokenizer from a saved
  checkpoint (`torch.load(..., weights_only=False)`, since the checkpoint pickles the full `Config` object, not just
  tensors). `generate(...)` runs naive autoregressive sampling (temperature/top-k, no caching of past activations).

Entry points: `radiance.train:main` (`--config`) and `radiance.generate:main` (`--checkpoint`, `--prompt`, ...) —
`radiance-train` / `radiance-generate` console scripts after install.

## Extending

- New dataset: point `data.dataset` at any `user/dataset` HF dataset with `train`/`validation` splits and the right
  `text_column`; no code changes needed unless the schema differs.
- New model variant: add config fields to `ModelConfig`, then wire them into `model.py`. Keep the
  `TransformerBlock` I/O contract so `train.py` and `data.py` stay untouched.
- New training behavior (e.g. different scheduler, mixed precision): changes belong in `train.py`; keep the loop
  step-based and keep config-driven values in `TrainConfig` rather than hardcoding.
