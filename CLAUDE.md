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
greedy decoding). Uses a KV-cache (`model.py`'s `KVCache`/`DenseTransformer.new_kv_cache`): the prompt is prefilled
once, then each subsequent step forwards only the newly sampled token against cached past keys/values, instead of
re-running the full forward pass over the whole context every step. An overlong prompt is still truncated to the
trailing `max_seq_len` tokens once at the start, same as before; once generation would push the *total* sequence
length past `max_seq_len` it raises a clear error instead of silently sliding the context window forever.

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
- **`model.py`** — `padded_vocab_size(vocab_size, multiple)` rounds the tokenizer's vocab up to a multiple of
  `cfg.model.vocab_pad_multiple` (default `128`; `1` disables) before the model is built — `train.py` calls it
  instead of passing `len(tokenizer)` straight through. A vocab that isn't a multiple of 64/128 leaves the
  model's largest matmul (the `lm_head`) on a ragged tensor-core tile; the padding rows are unreachable by any
  tokenizer id, so this is behavior-preserving, and it's worth ~9% of step time with the gpt2 tokenizer
  (50257 -> 50304). `generate.py` reads the vocab width off the checkpoint rather than recomputing it (so
  checkpoints predating this still load) and masks the padding columns before sampling, since a sampled padding
  id would decode to nothing and corrupt the KV cache. Like `qk_norm`/`auto_batch_size`, this defaults *on*
  rather than following the file's usual opt-in-`False` convention.

  `DenseTransformer`: token + learned positional embeddings, a stack of `n_layers` pre-norm
  `TransformerBlock`s, final LayerNorm, and a weight-tied LM head. `_scale_residual_init` shrinks every
  projection that *writes into* the residual stream (`attn.out_proj`, `ffn.down_proj`, including each MoE
  expert's) by `1/sqrt(2 * blocks_executed)` after `_init_weights` — the standard GPT-2 depth-scaled init, but
  counting the blocks actually *executed* per forward (`1 + loop_multiplier * (n_layers - 1)`) rather than
  `n_layers`, since `blocks[1:]` is re-run `loop_count`/`max_loops` times and so performs the residual writes of
  a much deeper stack. Looping is exactly the regime where the unscaled init hurts most, because it multiplies
  effective depth without adding parameters. Each block is `CausalSelfAttention` (uses
  `F.scaled_dot_product_attention` with `is_causal=True`, no manual mask construction) followed by `FeedForward`.
  `CausalSelfAttention` supports GQA via `model.n_kv_heads` (default `None` = standard multi-head attention, one
  K/V head per query head; when set, `n_heads` must be evenly divisible by it — see `ModelConfig.n_kv_heads_resolved`
  — and `F.scaled_dot_product_attention` is called with `enable_gqa=True`, hence this project's `torch>=2.5` floor).
  It also applies `model.qk_norm` (an `RMSNorm` over each head's `head_dim`, applied to q and k before RoPE) for
  training stability across `blocks[1:]`'s weight-shared loop iterations (see below); unlike most toggles in this
  config, `qk_norm` defaults to `True` — a deliberate behavior change for every existing config, not the usual
  default-`False` opt-in convention (contrast `use_router: bool = False`).
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
  optimize if router mode needs to get faster. `forward()` returns
  `(logits, ponder_cost, mean_loop_depth, moe_aux_loss)` in every mode — the latter three are zero scalar tensors
  when the corresponding feature (`use_router` / `use_moe`) is off, so callers have one contract regardless of mode.
  See `configs/tinystories_router.yaml` for a worked example.

  `cfg.model.grad_checkpoint` (opt-in, default `False`) recomputes each block's activations during backward
  instead of storing them. It pays off disproportionately in this architecture: `blocks[1:]` is re-run
  `loop_count`/`max_loops` times per forward and *every* pass retains its own activations, so activation memory
  scales with the loop multiplier while parameter memory doesn't. Measured on `configs/fineweb_500m.yaml` at
  `seq_len=1024`: peak memory at micro-batch 4 drops 27.7 GB -> 11.2 GB, which is what lets micro-batch 16 fit at
  all (19.7 GB) where the unckeckpointed model OOMs above 4; throughput costs ~20-25% (18.6k -> 14.5k tok/s at
  batch 4). Gradients are bit-identical either way. It's training-only — `DenseTransformer.forward` gates it on
  `self.training and torch.is_grad_enabled() and kv_cache is None`, since recomputation under a KV cache would
  re-append to that cache. `activation_bytes_per_token` models the checkpointed regime too (one `d_model` tensor
  per block boundary plus the largest single block's transient recompute), so `auto_batch_size` spends the freed
  memory on a bigger micro-batch automatically.

  `cfg.model.use_moe` replaces `blocks[1:]`'s `FeedForward` with `MoEFeedForward` — `n_experts` parallel
  experts (`BatchedExperts`) plus an `MoERouter` (same `RMSNorm + Linear` shape as `ACTRouter`, but softmax over
  `n_experts` logits instead of a single sigmoid). Routing is Mixtral-style top-`k` (`cfg.model.moe_top_k`,
  default `2`): each token's router probabilities are renormalized over just its top-`k` selected experts, and
  the FFN output is their weighted sum. Dispatch is capacity-based, mirroring `_sparse_ffn_delta`'s
  gather/compute/scatter idiom (fixed `capacity = round(moe_capacity_factor * n_tokens * moe_top_k / n_experts)`
  per expert, `torch.topk` priority selection) but generalized to `n_experts` writers
  into one shared output buffer via `index_add` rather than `_sparse_ffn_delta`'s single-writer `index_copy` —
  with more than one expert able to write a nonzero value for the same token, `index_copy` would let a later
  expert's zero capacity-padding silently clobber an earlier expert's real output at that index. Tokens beyond
  an expert's capacity are dropped (zero contribution from that expert, standard Switch-Transformer policy);
  which tokens get dropped is decided by the router's own weight for that expert, so an over-capacity expert
  sheds the tokens it was least confident about. Both this and `_sparse_ffn_delta`'s ACT capacity selection are
  deterministic outside training mode — `_sparse_ffn_delta` keeps a random tiebreak while `ffn.training` (an
  unbiased choice among still-running positions) but falls back to sequence order in eval. Previously both drew
  `torch.rand` unconditionally, which made `val/loss` and even greedy decoding vary run-to-run for identical
  weights and inputs, defeating the point of comparing two configs' eval numbers. A
  load-balancing auxiliary loss (`n_experts * sum(f_i * P_i)`, the standard Switch-Transformer formulation)
  keeps routing from collapsing onto a few experts; it's `forward()`'s `moe_aux_loss` return value, weighted by
  `cfg.model.moe_aux_loss_weight` (mirrors `ponder_weight`'s role for ACT's ponder cost). Note that
  `_collect_moe_aux_loss` **sums** the per-layer aux losses, so a balanced model reports `moe_top_k * n_moe_layers`,
  not `moe_top_k` — e.g. ~8.0 for `configs/tinystories_moe.yaml`'s 4 MoE layers at `top_k=2`, which is the
  *healthy* value, not evidence of collapse. Measured per-layer aux at init is 2.01-2.10 against an ideal of 2.0,
  and the capacity drop rate at a realistic batch (16x512 tokens) is under 1% on average.

  All `n_experts` are computed in **one batched `baddbmm`**, not a Python loop over experts: `BatchedExperts`
  stores each projection as a single stacked `(n_experts, in, out)` tensor (the transpose of `nn.Linear`'s
  layout, so the forward is a plain `x @ W`), and the gather produces one `(n_experts, capacity, d_model)`
  tensor. Total FLOPs are constant in `n_experts` — `capacity` shrinks as experts are added — so this keeps
  step time roughly flat where the per-expert loop grew linearly with expert count. Measured fwd+bwd on one MoE
  layer (16x512 tokens, bf16): 4 experts 3.16 -> 2.32 ms, 8 experts 4.30 -> 2.15 ms, 16 experts 8.34 -> 3.42 ms,
  32 experts 24.0 -> 3.16 ms (7.6x). The forward is bit-identical to the old loop; only gradient accumulation
  order differs (~1e-4). `MoEFeedForward` registers a load-state-dict pre-hook that transposes and stacks the
  old per-expert `experts.{e}.{gate,up,down}_proj.*` keys, so MoE checkpoints predating this still load.

  `blocks[0]` always stays dense (it runs once per forward, not part of the recursive loop body); within
  `blocks[1:]`, `cfg.model.moe_dense_every` (opt-in) keeps every Nth block (1-indexed by position in the loop
  body) dense too, for interleaving MoE and dense layers. Because `blocks[1:]` is weight-shared but fed an
  evolving hidden state each loop iteration, a token's expert choice naturally changes across iterations as its
  representation evolves — this "recursion picks different experts per pass" behavior falls out for free from
  ordinary input-dependent routing, with no iteration-aware logic in `MoERouter`/`MoEFeedForward` itself.
  `MoEFeedForward.forward` is shape-agnostic over its leading dims, so it satisfies the same
  `(*, d_model) -> (*, d_model)` contract `FeedForward` does and composes with ACT's own `_sparse_ffn_delta`
  FFN-capacity sparsity (`cfg.model.act_ffn_capacity_ratio < 1.0`) with no special-casing in either mechanism —
  `_run_loop_body` calls `block.ffn` generically regardless of which FFN type it is. See
  `configs/tinystories_moe.yaml` for a worked example.
- **`train.py`** — plain PyTorch training loop (no HF `Trainer`): AdamW + cosine-with-warmup LR schedule
  (`build_lr_scheduler`), manual loss computation (`compute_loss`), gradient clipping, periodic W&B logging
  (`train/loss`, `train/lm_loss`, `train/ponder_cost`,
  `train/mean_loop_depth`, `train/lr`, `val/loss`), periodic checkpointing to `cfg.train.output_dir` (raw
  `torch.save` of state dict + config), and periodic `evaluate()` against the validation split.

  `compute_loss` applies the causal-LM one-position shift to the *labels* (padding them with `ignore_index`),
  not by slicing `logits[:, :-1]`. The slice is a non-contiguous view whose `.contiguous()`/`.view()` forces a
  full copy of the `(batch, seq, vocab_size)` logits — the largest activation in the model — on every forward;
  shifting labels instead makes `logits.view(-1, vocab_size)` a free reshape over exactly the same targets.
  Worth ~7% of step time at the `configs/tinystories.yaml` size.

  `build_param_groups` splits parameters into a weight-decayed group (`dim() >= 2`, i.e. the weight matrices,
  including the tied embedding) and a non-decayed group (RMSNorm gains and biases). Passing `model.parameters()`
  straight to `AdamW` would decay the 1-D scale/shift parameters too, which just fights the norm layers; decaying
  only the matrices is what GPT-2/Llama/nanoGPT do.

  `build_lr_scheduler`'s warmup ramps over `(step + 1)`, because `LambdaLR` evaluates the lambda at step 0 to set
  the LR for the *first* `optimizer.step()` — a plain `step / warmup_steps` ramp spends that entire first step at
  `lr=0`. The cosine then decays to `cfg.train.min_lr_ratio * lr` (default `0.1`) rather than to 0, since the tail
  of a run at a ~0 LR contributes nothing; set `min_lr_ratio: 0.0` for the old decay-to-zero behavior.

  Checkpoints are **resumable**: `save_checkpoint` writes the optimizer state, LR-scheduler state and
  `GradScaler` state alongside the weights/step/config, and `cfg.train.resume_from` (opt-in, default `null`)
  restores all of it. Set it to a checkpoint path, or to the literal `"auto"` to pick the highest-numbered
  `step_*.pt` in `output_dir` — so an interrupted run can be relaunched with its config unchanged. Without the
  optimizer moments a "resumed" run restarts AdamW from zero momentum at warmup LR, which shows up as a loss
  spike. An explicit `resume_from` path that doesn't exist raises rather than silently starting from scratch;
  `"auto"` against an empty `output_dir` is just a fresh run. What is *not* restored is the DataLoader position
  and RNG state, which trade off against each other: `train()` re-seeds off the resumed step so the loader draws
  a different shuffle order rather than replaying batches already trained on. A resumed run is therefore
  statistically equivalent to an uninterrupted one, not bit-identical — with `dropout: 0.0` it is bit-identical
  (verified: same weights, same AdamW moments, same LR sequence).

  `cfg.train.eval_max_batches` (default `50`) caps how many batches each `evaluate()` call consumes. Uncapped —
  the previous behavior, restored with `null` — every eval walks the *entire* validation split, which is
  unbounded for a streaming one and, at `configs/tinystories.yaml`'s settings, cost more wall-clock than all
  1000 training steps combined. A fixed batch count also keeps `val/loss` comparable across configs whose
  validation splits differ in size. The loop is
  step-based (`cfg.train.max_steps`), not epoch-based, and cycles the train `DataLoader` via manual `StopIteration`
  handling rather than epochs. Setting `cfg.train.tokens_per_param` (opt-in, default `null`) derives `max_steps`
  from model size instead of pinning it directly: once the model is built, `train()` overwrites `cfg.train.max_steps`
  with `round(tokens_per_param * raw_model.num_active_parameters() / (effective_batch_size * data.seq_len))` —
  e.g. `20` for a Chinchilla-optimal token budget — and prints/logs the resulting step count, so the same config
  keeps tracking the "right" number of steps as `model.*` fields (and therefore param count) change instead of
  needing `max_steps` hand-recomputed. `num_active_parameters()` differs from the flat `num_parameters()` only
  when `cfg.model.use_moe` is set, excluding each MoE layer's inactive (non-top-`k`) expert parameters — Chinchilla
  -style scaling assumes every parameter multiplies against every token, which is false for MoE, so using the flat
  count there would inflate `max_steps` by roughly `n_experts / moe_top_k`; `estimate_batch_size`'s optimizer-state
  sizing below still uses the flat count, since every expert parameter gets a gradient and optimizer state
  regardless of activation frequency. `warmup_ratio` (see `TrainConfig`) is read as a live property off whatever
  `max_steps` ends up being, so warmup scales automatically along with it. See `configs/fineweb_500m.yaml` for a
  worked example; leave `tokens_per_param: null` and set `max_steps` directly for quick/pinned runs (e.g.
  `configs/tinystories.yaml`). `cfg.train.grad_accum_steps` (default `1`, opt-in) accumulates gradients over that
  many `batch_size`-sized micro-batches before calling `optimizer.step()`/`scheduler.step()`, so the effective
  training batch (`effective_batch_size = batch_size * grad_accum_steps`) can exceed what fits in memory for a
  single forward/backward pass; each micro-batch's loss is divided by `grad_accum_steps` before `.backward()` so
  the accumulated gradient matches training on one `effective_batch_size`-sized batch, and `step`/W&B
  logging/`eval_every`/`save_every` all stay in accumulated-step units, unaffected by the setting. See
  `configs/fineweb_500m.yaml` for a worked example. `cfg.train.auto_batch_size` (CUDA-only) overwrites the
  configured `batch_size`/`grad_accum_steps` at startup with values computed from free VRAM and model size
  (`estimate_batch_size` in `train.py`) — a deliberately conservative closed-form estimate (params/gradients/
  optimizer state sized exactly, activation memory from a hand-derived, intentionally-overestimating per-token
  formula on `DenseTransformer.activation_bytes_per_token`) rather than an expensive live probe against the real
  model. It defaults to `True` — a deliberate behavior change for every existing config, not the usual
  default-`False` opt-in convention (contrast `use_router`/`use_moe`; see `qk_norm` for the other precedent of this)
  — since on CUDA it only ever makes the actual micro-batch size *safer* than a hand-picked one, never bigger, and
  it's what gates the OOM backoff described below; set it to `False` for a manually-chosen or W&B-swept
  `batch_size` to behave exactly as configured (e.g. a sweep that's already tuning `batch_size` itself). On
  CPU/MPS it's a no-op (prints a note and keeps the configured `batch_size`/`grad_accum_steps`), since
  `estimate_batch_size` only knows how to read free VRAM. `cfg.train.target_effective_batch_size` sets the desired
  `effective_batch_size` the estimate solves `grad_accum_steps` for; left at its default `None`, it falls back to
  whatever `effective_batch_size` the configured `batch_size`/`grad_accum_steps` already imply, so an existing
  config's effective batch size is preserved even as `auto_batch_size` re-splits it across `batch_size`/
  `grad_accum_steps` to fit VRAM — set it explicitly to target a different effective batch size instead.
  `cfg.train.vram_safety_margin` (default `0.5`) scales how much of the estimated budget is actually used.
  Because the estimate is approximate by construction, `auto_batch_size` also enables a
  two-tier OOM recovery escalation in the training loop, each tier attacking a different memory pool. Tier one: a
  CUDA OOM halves an internal `micro_chunk_size` (starts equal to `batch_size`, monotonically shrinks, floor `1`)
  that further splits each already-fetched micro-batch along the batch dimension and retries the same accumulated
  step, rather than ending the run — `batch_size`/`grad_accum_steps`/`effective_batch_size` (and therefore
  `tokens_per_param` accounting) are untouched by this, since backoff never rebuilds the DataLoader, only how many
  forward/backward calls it takes to process one already-fetched micro-batch; this only reduces activation memory,
  which scales with batch size. Tier two triggers once `micro_chunk_size` has already bottomed out at `1` and a CUDA
  OOM still hits (so shrinking the batch further can't help, since the remaining cost is fixed): the `AdamW`
  optimizer is swapped for `CPUOffloadAdamW` (`train.py`), which keeps `exp_avg`/`exp_avg_sq` in pinned CPU memory
  instead of on `device`, permanently freeing ~2x `num_params` fp32 bytes of VRAM for the rest of the run — model
  parameters, gradients, and activations all stay GPU-resident throughout, and only a grad copy (down) and the
  resulting update (up) cross PCIe, once per optimizer step rather than once per forward/backward, so this doesn't
  touch the forward path or invalidate `torch.compile`'s captured graph. `migrate_optimizer_to_cpu_offload`
  (`train.py`) preserves in-flight momentum by migrating each param's existing `exp_avg`/`exp_avg_sq`/`step` onto the
  new optimizer rather than resetting it, and the LR scheduler is rebuilt against the new optimizer with
  `last_epoch` set to the current step so the LR trajectory continues uninterrupted. Both tiers are sticky —
  `micro_chunk_size` only ever shrinks and CPU offload, once triggered, stays on for the rest of the run — and both
  are scoped to `auto_batch_size`: with it explicitly set to `False`, a CUDA OOM always ends the run cleanly as
  before, so a manually-chosen or W&B-swept `batch_size` behaves exactly as configured. The handler also resets
  the `GradScaler`'s per-optimizer bookkeeping via `update(get_scale())` before retrying: an OOM at or after
  `scaler.unscale_()` otherwise leaves the optimizer marked as already-unscaled, so the retry's own `unscale_`
  raises `"unscale_() has already been called on this optimizer since the last update()"` — an uncaught crash
  exactly when the backoff was supposed to save the run. Only reachable under `dtype: fp16`, since bf16/fp32
  run with the scaler disabled; `update(get_scale())` rather than a bare `update()` so the aborted step isn't
  mistaken for a successful one and used to grow the scale. See
  `configs/fineweb_500m.yaml`'s commented-out block for a worked example of overriding the defaults. `cfg.train.dtype` (`"fp32"`, `"fp16"`,
  or `"bf16"`, resolved via `resolve_dtype`)
  controls precision: the forward/loss pass runs under `torch.autocast` in that dtype while master weights and the
  optimizer state stay fp32; a `torch.amp.GradScaler` is enabled only for `fp16` (its narrow exponent range can
  underflow small gradients — `bf16` has fp32's exponent range so it needs no scaling). The training loss is
  `lm_loss + cfg.model.ponder_weight * ponder_cost + cfg.model.moe_aux_loss_weight * moe_aux_loss` (the second
  term is always zero unless `use_router` is on, the third unless `use_moe` is on); `evaluate()`'s `val/loss`
  stays pure LM loss (ponder cost and MoE aux loss both discarded) so it's comparable across configurations.

- **`generate.py`** — `load_checkpoint(path, device)` reconstructs a `DenseTransformer` + tokenizer from a saved
  checkpoint (`torch.load(..., weights_only=False)`, since the checkpoint pickles the full `Config` object, not just
  tensors). `generate(...)` runs autoregressive sampling (temperature/top-k) using a `KVCache`
  (`DenseTransformer.new_kv_cache()`): the (possibly truncated) prompt is prefilled once, then each later step
  forwards only the newly sampled token. Because `blocks[1:]` is a weight-shared loop body re-invoked `loop_count`
  (or, in ACT mode, `max_loops`) times per forward call — always the full iteration count regardless of per-token
  halting, since attention is unconditionally dense every iteration — a single K/V-per-layer cache isn't enough:
  the same block produces different K/V on each iteration since it's fed an evolving hidden state, so `KVCache` has
  one slot per (block, iteration) pair actually executed (`1 + loop_multiplier * (n_layers - 1)` slots), assigned
  implicitly by call order (`KVCache.begin_step()`/`write()`) rather than an explicit index threaded through every
  call. `RotaryEmbedding.forward` takes an `offset` so a cached token's RoPE position reflects its true absolute
  position rather than always starting from 0. If generation would push the total sequence length past
  `model.cfg.max_seq_len`, `generate()` raises a clear error rather than silently sliding the context window.

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
  swapping in a different block. See `configs/tinystories_gqa.yaml` for a worked example of GQA (`model.n_kv_heads`).
- MoE FFN: set `model.use_moe: true` (plus `model.moe_dense_every` to interleave dense blocks within `blocks[1:]`)
  — see `configs/tinystories_moe.yaml`. `MoEFeedForward` is the reference example for a variant that replaces a
  block's FFN sublayer wholesale while preserving `FeedForward`'s `(*, d_model) -> (*, d_model)` contract, which
  is what lets it compose with ACT's own `_sparse_ffn_delta` gather/scatter path with no changes to either.
- New training behavior (e.g. different scheduler, mixed precision): changes belong in `train.py`; keep the loop
  step-based and keep config-driven values in `TrainConfig` rather than hardcoding.
