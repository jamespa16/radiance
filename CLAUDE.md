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

## Running tests

```bash
uv run --group dev pytest                  # everything
uv run --group dev pytest -m "not slow"    # skip the compiled-GPU regression tests
```

The suite is built around **equivalence invariants** rather than golden files, because most features here are
supposed to be mathematically inert until configured: a feature that defaults on must produce bit-identical
logits to the same model with it off (`tests/test_inert_defaults.py`, `tests/test_loop_identity.py`), a cached
decode must match a full forward across every loop mode (`tests/test_kv_cache.py`), and so on. Those tests are
self-checking — there is nothing to regenerate when the model changes.

Two of them have already earned their keep and are worth knowing about:

- `test_no_dead_parameters` asserts every allocated parameter is reachable from the forward pass. It caught two
  parameters that were allocated but never differentiated (and so silently never trained).
- `tests/test_act_kv_invariance.py` pins down exactly what is and isn't invariant for a halted ACT position. The
  tempting-but-wrong version of that argument is very easy to re-derive; see the file's docstring before
  attempting to make ACT skip halted tokens.

`tests/test_compile.py` (marked `slow`, CUDA-only) covers two failures that eager tests structurally cannot see,
both of which cost a training run to find: document masking under `torch.compile` (`create_block_mask` must run
eagerly — `_doc_masks` is `@torch._dynamo.disable`d — or inductor fails to lower its data-dependent index
tensors), and stochastic loop depth under `mode="reduce-overhead"` (CUDA graphs assume a static execution path).
**If you change anything touching attention masking, the loop count, or compile settings, run the suite on a GPU
without `-m "not slow"`.** The eval path is worth particular attention there: `grad_checkpoint` is disabled under `eval()`,
so evaluation traces a *different* graph than training — the doc-masking bug compiled fine for training and blew
up at the first `evaluate()` call.

Still run a tiny config end-to-end (small `seq_len`, `d_model`, `max_steps`) through `radiance.train` on CPU
before trusting a full run — the suite covers the model, not the data pipeline or the training loop's plumbing.

## Architecture

Everything lives under `src/radiance/`, driven entirely by a single YAML config (`radiance.config.Config`, loaded via
`load_config`). Five modules; four map to a stage of the pipeline and one holds the optimizers:

- **`config.py`** — dataclass schema (`DataConfig`, `ModelConfig`, `TrainConfig`, `WandbConfig` nested in `Config`)
  and `load_config(path)`. This is the single source of truth for every tunable; a new hyperparameter should be added
  here first, then threaded through. Config values are plain dataclasses, not `OmegaConf`/Hydra — no CLI overrides or
  config composition, just one YAML file per run.

  **Defaults convention: features default *on*.** A new capability ships enabled unless there's a clear reason not
  to. What makes that safe is that the *parameters* default to an inert setting, so every existing config keeps
  training exactly as it did and the feature only engages once someone configures it or training moves its weights.
  In practice that means one of:

  - **zero-initialised**, so the term contributes exactly nothing at init and can only be learned into —
    `loop_input_injection`'s `W_inj`, the `IterLoRA` adapters' `B`, the router `iter_bias` tensors;
  - **identity-valued**, chosen so the arithmetic is exact — `value_residual`'s λ starts at exactly 1.0, and
    `attn_out_gate` is written `2 * sigmoid(zero_init)` precisely because a plain `sigmoid` cannot reach 1.0;
  - **range-collapsed**, where a `None` (or a unit scalar) resolves to the existing quantity —
    `loop_count_min/max` collapse to `loop_count`, `mup_base_d_model` resolves to `d_model` (making every muP
    correction exactly 1.0), `moe_expert_ffn_mult` resolves to `ffn_dim`, `hyper_conn_streams: 1` collapses the
    `n` hyper-connection streams back to a single residual stream.

  Four things deliberately do *not* default on, and the reasons are the template for future exceptions:
  `lr_schedule` stays `"cosine"` (WSD is an operational convenience, not a quality win, and switching it would
  silently reshape the LR trajectory of every config whose `lr` was tuned against cosine); `mtp_heads` stays `1`
  (each extra head materialises a full `(batch, seq, vocab_size)` logits tensor, so defaulting to 2 would quietly
  halve what `auto_batch_size` can fit); `hyper_conn_streams` stays `1` for the same reason plus a second one —
  it costs `n` times the residual stream's activation memory *and* 30-40% of step time in the looped regime,
  so "on" is not free even though it is inert; `loop_bptt_window` stays `None` (truncating the gradient is an
  approximation you reach for deliberately). The distinction is cost and reversibility, not novelty — a feature
  whose "on" state is free and inert defaults on; one that spends real memory or changes a tuned quantity doesn't.

  Five settings change results at their defaults, intentionally: `doc_attention_mask`, `optimizer: muon`,
  `z_loss_weight`, MoE's `moe_n_shared`/`moe_balance`, and `train.lr`. Each is a straightforward improvement
  rather than an experiment — see the A/B numbers recorded below.

  `train.lr` is the odd one out and the one to be careful with, because it is the only *tuned quantity* on that
  list rather than a feature flag — exactly the category the paragraph above says shouldn't change by default.
  It changed anyway because it had become simply wrong (`3.0e-4` -> `1.0e-2`; see `optim.py` below for the
  sweep). Two consequences worth knowing. It reaches only configs that **omit** `lr`, which today means the
  `sweep*`/`super*` ones — every worked example in `configs/` pins `lr: 3.0e-4` and so still trains at the old
  value until edited. And the 400-step baselines recorded under "Measured results" predate it, so reproducing
  them means pinning `lr: 3.0e-4` explicitly.
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
  id would decode to nothing and corrupt the KV cache. Defaults on, like everything else here — see the defaults
  convention under `config.py`.

  `forward()` returns a `ModelOutput` NamedTuple (`logits`, `ponder_cost`, `mean_loop_depth`, `moe_aux_loss`,
  `mtp_hidden`), not a bare tuple, so fields can be added without breaking every call site. Per-iteration state
  reaches the blocks on a `LoopContext` (which loop pass this is, the flex-attention `BlockMask`, the ACT capacity,
  the KV cache) — six separate features need to tell a block something about *which pass it is on*, and bundling
  that keeps `TransformerBlock`/`CausalSelfAttention`'s signatures stable. `LoopContext` deliberately holds only
  non-tensor state: tensors that participate in autograd (`v_first` for the value residual, the injection anchor)
  stay explicit positional arguments, because `torch.utils.checkpoint` only tracks tensors it receives
  positionally and one hidden in a dataclass would silently lose its recompute path under `grad_checkpoint`.

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
  training stability across `blocks[1:]`'s weight-shared loop iterations (see below).

  Two further attention refinements, both on by default and both exactly inert at initialisation:
  `model.value_residual` mixes each block's values with `blocks[0]`'s (`v = λ·v + (1−λ)·v_first`) via a learned
  per-block scalar initialised to exactly 1.0 — since `blocks[1:]` is a weight-shared loop, this gives every
  iteration direct access to the first block's values. The mix happens *before* the `kv_cache.write`, so cached
  values are already mixed and generation needs no extra cache slot (`blocks[0]` runs first within each decode
  step, producing that token's `v_first` in-step). `blocks[0]` deliberately has no `value_lambda` at all: it is
  the block that *produces* `v_first`, so a mix parameter there would be allocated, never differentiated, and
  never trained — `tests/test_optim.py::test_muon_step_updates_every_parameter` exists partly to catch that class
  of dead parameter. `model.attn_out_gate` applies a per-head gate to the attention output before `out_proj`,
  written `2 * sigmoid(Linear(x))` with the `Linear` zero-initialised in `_init_inert_gates` (which must run
  *after* `_init_weights`, the same ordering constraint `_scale_residual_init` has). The `2 *` is load-bearing:
  it makes the gate exactly `1.0` at init, where a plain `sigmoid` would start every head at 0.5 and halve the
  attention output.

  `model.doc_attention_mask` (default on) masks attention at packed-document boundaries. `data.py` concatenates
  many documents into each `seq_len` block joined by EOS, and plain causal attention reads straight across those
  joins, so the model spends capacity predicting one document's tokens from another's. Document ids are recovered
  **in-model** by `document_ids()` as an exclusive cumsum over EOS positions — no extra column in the packed
  dataset, and therefore no re-tokenizing and no cache invalidation for datasets already on disk. The mask is a
  `flex_attention` `BlockMask` built once per forward outside any compiled region and reused across every block
  and loop iteration, exactly as the RoPE cos/sin tables already are. Three fallbacks to plain SDPA, all silent
  and all necessary: during generation (a single prompt is one document), off CUDA (`flex_attention` is
  impractically slow there, and CPU sanity checks have to keep working), and at `head_dim < 16` (its compiled
  kernel rejects smaller heads with an inductor lowering error). `flex_attention` has **no attention-weight
  dropout**, so enabling this silently drops it — the model prints a one-time note when `dropout > 0`, and the
  new configs set `dropout: 0.0`. FFN residual dropout is unaffected. `model.loop_attn_windows` rides the same
  machinery to give each loop iteration its own receptive field (e.g. `[128, 128, 512, 512]` for local-then-global
  passes) — a knob that only exists *because* the architecture loops.

  `cfg.model.use_nsa` was an opt-in (default `False`) DeepSeek NSA-style learned block-sparse attention that
  has been **removed from the codebase**. It previously replaced plain dense/doc-masked attention with a
  simplified two-branch version: a coarse **compression** branch attending against mean-pooled blocks of raw
  K/V (cheap, always dense — `_nsa_compress`), and a fine **selection** branch attending only the top-`nsa_top_k_blocks`
  historical blocks a learned score picks per query *token*, plus that token's own local block (`_nsa_select_blocks`),
  combined by a per-token gate (`NSAGate`). Unlike `value_residual`/`attn_out_gate`, a learned key-selection
  mechanism has no zero/identity init that makes it equivalent to dense attention, so it followed the
  `use_moe`/`use_router`/`n_kv_heads` precedent — off by default, not a free inert addition.

  The feature is now removed: all NSA symbols (`NSAGate`, `NSACompressedCache`, `_nsa_*` helpers, `use_nsa`,
  `nsa_block_size`, `nsa_top_k_blocks`, `nsa_cache` in `LoopContext`, and the `_nsa_train_eval`/`_nsa_generate`
  branches in `CausalSelfAttention`) have been deleted from `src/radiance/model.py`, the corresponding config
  fields removed from `src/radiance/config.py`, and the related tests/configs removed. The removal was motivated
  by the A/B results below (quality regression vs. doc-masking, no clear compute win at TinyStories scale) and
  the maintenance burden of a second sparsity axis that conflicts with doc masking, loop windows, and ACT sparsity.

  Selection had been decided **per query token, not per query block**, even though every other block-sparse
  mechanism in this file (doc masking, `loop_attn_windows`) shares a decision across a `flex_attention` tile for
  efficiency. An earlier version shared the decision per query block to match tile granularity, and it was wrong:
  a block's selection would need to average in later tokens' scores that don't exist yet during incremental
  decoding, which is impossible without seeing the future. Since NSA had no dense fallback at inference (the model
  was *trained* attending only to selected blocks, so generation had to reproduce the identical computation, not
  an approximation of it — unlike `doc_attention_mask`'s generation fallback, which is exact only because a single
  prompt really is one document), train and decode had to compute the same thing, which meant per-token
  everywhere. The cost was real: most `flex_attention` tiles become "partial" (computed but masked) rather than
  cleanly skippable, since different query rows in the same tile can select different blocks.

  Previously, three things were incompatible with `use_nsa` and raised at `DenseTransformer.__init__` if combined:
  `doc_attention_mask` (mean-pooling raw K/V blocks has no document-boundary awareness, so a block straddling a
  packed-document join would blend two documents together *before* any attention mask is applied — masking the
  attention step can't undo that); `loop_attn_windows` (the selection branch's forced local block already covers the
  same recency need); and `act_capacity_ratio`/`act_ffn_capacity_ratio` below 1.0 (two independent sparsity axes).
  `cfg.model.nsa_block_size` (default `128`, matching `flex_attention`'s own default tile size) had to stay there
  unless a different value's kernel was confirmed to compile on your GPU — `create_block_mask`'s Triton kernel only
  accepts a `BLOCK_SIZE` that's a multiple of its own internal tile size, so a smaller value doesn't just run
  slower, it raises a `LoweringException` at the first compiled forward. Generation required its own machinery:
  `NSACompressedCache` mirrored `KVCache`'s implicit-call-order slot scheme, tracking each slot's finalized
  compressed blocks (from a small pre-RoPE pending buffer — `KVCache` itself only ever stores *post*-RoPE K/V,
  which can't be safely mean-pooled across positions) plus, implicitly, the real per-position K/V it needs for the
  selection branch's gather (already retained in full by the ordinary `KVCache`, since NSA's saving is in compute,
  not cache memory). See the git history for the removed `configs/tinystories_nsa.yaml` worked example, and the
  "Measured results" section below for why it wasn't (yet) a win at this scale.

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
  this is dense, fully-batched compute, router mode by default does **not** save wall-clock over running
  `max_loops` iterations for every token — the adaptivity shows up in the loss signal (`ponder_cost`, see below)
  and in what gets accumulated into the output, not in runtime.

  **`cfg.model.act_capacity_ratio` is what changes that.** Below 1.0, each interior ACT iteration computes only
  that fraction of each sequence's positions — the highest-priority still-running ones — through the *entire*
  loop-body stack. Everything else keeps its hidden state and serves its keys/values from a retained per-block
  store. `_run_loop_body_sparse` gathers the selected positions once, runs every block on that narrow tensor, and
  scatters back; `CausalSelfAttention.forward_sparse` computes Q/K/V only for them, scatters the fresh K/V into
  the retained full-length store, and attends the gathered queries against the whole thing.

  **Measured, and read the second table before quoting the first.** Pure fwd+bwd, bf16, `max_loops: 6`:

  | `act_capacity_ratio` | `d512 L8`, batch 8x512 | `d256 L6`, batch 16x512 |
  |---|---|---|
  | 1.0 (dense) | 105.6 ms | 71.3 ms |
  | 0.5 | 87.0 ms (1.21x) | 63.1 ms (1.13x) |
  | 0.25 | 73.4 ms (**1.44x**) | 60.1 ms (1.19x) |

  Peak memory at `d512 L8` falls 14.79 -> 12.23 -> 10.76 GB over the same ratios.

  The gain scales with how much of the step the loop body actually is — 93% of the FLOPs at
  `d_model: 512`, 83% at `256` — because the gather/scatter and the materialised mask are a roughly fixed
  overhead that amortises better on a wider model. **And the step is not the run.** A 400-step
  `configs/tinystories_router.yaml` A/B came out 192s dense vs 187s at ratio 0.25, barely 3%, because at that
  size 400 steps is only ~29s of model compute and the rest is compile, data loading and (always-dense) eval.
  Benchmark the step, not a short run, when judging this.

  Quality cost, over that same 400-step A/B at a pinned `batch_size: 16` (auto sizing would have handed the
  sparse arms a bigger batch and invalidated the comparison) — `val/loss` at step 400: **3.0735 dense, 3.0757 at
  ratio 0.5, 3.0728 at ratio 0.25.** Identical within noise, at every eval point. The approximation is cheap at
  this scale; whether that holds at longer horizons is untested.

  Three things about it are load-bearing:

  - **It is an approximation, and `tests/test_act_kv_invariance.py` says precisely why.** Reusing a halted
    position's K/V is exact only for the *first* block of the loop body: K and V are per-position projections of a
    block's input, and a halted position's input to `blocks[1]` is `frozen_x`, which by construction stops
    changing — but `blocks[1]`'s *output* there mixes in attention over still-running positions that keep
    evolving, so `blocks[2]`'s input and hence its K/V drift. Beware the misleading special case: if the halted
    set happens to be a causal *prefix*, halted positions never attend to a running one and every block looks
    invariant. Real halting is scattered. This is the same family of approximation `act_ffn_capacity_ratio`
    already made, extended from the FFN sublayer to the whole block.
  - **It is training-only**, gated like `grad_checkpoint` on `self.training and torch.is_grad_enabled() and
    kv_cache is None`. That is deliberate: cached decoding cannot use it (one token at a time, nothing to select
    among), so if a full forward used it while decoding didn't, `evaluate()` and `radiance-generate` would compute
    the same prompt two different ways. Confining it to training also keeps `val/loss` comparable across ratios,
    since eval always measures the full dense model.
  - **It refuses `grad_checkpoint`.** Recomputation during backward would write into the retained K/V store a
    second time and silently corrupt it. `DenseTransformer.__init__` raises rather than producing wrong gradients;
    sparsity already cuts activation memory substantially (see the table), so the two overlap anyway.

  Selection (`_act_select`) is per *sequence*, not over the flattened batch the way `_sparse_ffn_delta`'s is —
  attention needs each row's queries to belong to that row. Causality for the scattered queries is
  `true_position(query) >= key_position`, which `is_causal=True` cannot express, so `_sparse_attn_mask`
  materialises a `(batch, 1, capacity, seq_len)` boolean mask (folding in document masking when active, since a
  flex `BlockMask` assumes dense queries). That mask costs the fused causal kernel, which is why the measured
  speedup is ~70-80% of the theoretical `(2 + (max_loops - 2) * ratio) / max_loops`.

  `forward()` returns
  `(logits, ponder_cost, mean_loop_depth, moe_aux_loss)` in every mode — the latter three are zero scalar tensors
  when the corresponding feature (`use_router` / `use_moe`) is off, so callers have one contract regardless of mode.
  See `configs/tinystories_router.yaml` for a worked example.

  **Giving the loop body an identity.** A weight-shared loop body has no idea which iteration it is on — without
  help, iteration 5 is computationally indistinguishable from iteration 1 except through residual-stream norm
  drift. Four settings address that, all inert at their defaults and all worked through in
  `configs/tinystories_looped.yaml`:

  - `cfg.model.loop_iter_conditioning` (`"norm_gains"` default, `"lora"`, `"none"`). `"norm_gains"` gives each
    iteration its own `RMSNorm` gains — the adaLN trick — at ~`2 * loop_multiplier * d_model` parameters per
    block, plus a per-iteration bias on `ACTRouter` and `MoERouter`. The `MoERouter` one makes deliberate what
    previously fell out incidentally: a token's expert choice already drifted across iterations because the
    hidden state evolves, and now an iteration can learn a standing preference. `RMSNorm` keeps its original
    1-D `(d_model,)` gain shape at `n_variants == 1`, so an unconditioned model is bit-identical to one from
    before this existed, and a load-state-dict pre-hook broadcasts an old 1-D gain across variants so enabling
    conditioning on an existing run starts from exactly that run's weights. `"lora"` instead gives each
    iteration a rank-`loop_lora_rank` adapter on the fused QKV and FFN down projections (`IterLoRA`), with `B`
    zero-initialised.
  - `cfg.model.loop_input_injection` (default on) re-injects **`blocks[0]`'s output** — not the raw token
    embedding — at the start of every iteration after the first: `h = h + W_inj @ anchor`. `W_inj` is
    zero-initialised, so it is exactly a no-op at init at any loop count. Injecting the *prelude output* rather
    than the embedding matters for more than representation quality; see `loop_bptt_window` below.
  - `cfg.model.loop_count_min` / `loop_count_max` sample the loop count per training step. Both default to
    `None`, resolving to `loop_count` and collapsing the range to a point. Eval and generation always use the
    top of the range so their numbers stay deterministic. `radiance-generate --loops N` overrides it entirely,
    which is what lets a model trained across a range of depths spend *more* compute per token at inference
    than training used. `cfg.model.loop_multiplier` is the single source of truth for the worst-case count and
    is what `new_kv_cache`, `_scale_residual_init` and `activation_bytes_per_token` size themselves against.
    Note that stochastic depth forces `train.py` to compile **without CUDA graphs** (`mode=None` rather than
    `"reduce-overhead"`): a captured graph assumes a static execution path, and replaying a different loop count
    overwrites the previous graph's gradient tensors.
  - `cfg.model.loop_bptt_window` backpropagates through only the last N iterations, running earlier ones under
    `no_grad`, so activation memory becomes O(N) rather than O(loop_count). **It requires
    `loop_input_injection`, and `DenseTransformer.__init__` raises if you set one without the other.** The
    reason is subtle and was found the hard way: running early iterations under `no_grad` severs the graph
    *upstream* of the recurrence too, so `blocks[0]`'s only route to the loss disappears and it silently trains
    at its initial weights for the whole run while the loss still falls. Injecting `blocks[0]`'s output into the
    in-window iterations restores the path. (One-step cold start: `W_inj` starts at zero, so the anchor path has
    zero derivative on step 0; `W_inj` itself gets a gradient immediately, so from step 1 on `blocks[0]` trains
    normally. `tests/test_loop_identity.py` covers all of this.)

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

  **Hyper-connections: `n` residual streams instead of one.** Everything above gives the loop body an
  identity; `cfg.model.hyper_conn_streams` (default `1`) attacks the other half of the problem — the single
  accumulator all that depth writes into. At `n > 1` the hidden state becomes `(batch, seq, n, d_model)` and
  each sublayer's residual write becomes a `HyperConnection` read/write pair (Zhu et al., ICLR 2025,
  arXiv:2409.19606): `h_in = alpha_m^T H`, then `H' = alpha_r^T H + beta^T out`. So a sublayer learns *which*
  stream to read, how the streams mix, and how its output is distributed back across them. `blocks[0]` and the
  loop body both get them; `MTPHead`'s own `TransformerBlock` deliberately does not (it runs outside the
  recursion on the already-reduced hidden), which is what the `hyper` constructor flag is for.

  It is **exactly the plain residual network at init**, and the argument is worth keeping because it is what
  makes the feature safe: `alpha_m` starts one-hot, `alpha_r` the identity, `beta` all-ones, and the streams
  start as `n` copies of the same vector — so every stream receives the same `out` and they stay equal at every
  depth, with the one-hot read just picking one of `n` identical copies. Both matmuls are exact at those values.
  Verified bit-identical to the single-stream model at `n` = 2 and 4 across the whole loop-mode matrix, and to
  ~1 ulp at odd `n` (`tests/test_hyper_connections.py`). `cfg.model.hyper_conn_dynamic` (default on) adds
  input-conditioned corrections `coeff = static + s * tanh(norm(H) W)` with `W` zero-initialised, so it is
  bit-identical to the static version at init — the same inert-by-zero-init pattern as `W_inj` and `out_gate`.
  The three projections share one `dyn_proj` tensor so the forward is a single matmul.

  Three implementation details that are load-bearing, all found by measurement rather than reasoning:

  - **`_reduce_streams` averages; the paper sums.** With the streams identical at init, a sum hands `ln_f`
    exactly `n` times the single-stream hidden (verified bit-exact). `RMSNorm` is scale-invariant in exact
    arithmetic, so that ought to vanish — but `+ self.eps` sits inside the `rsqrt` and is *not* scale-invariant,
    which measured as a 1e-3 logit shift at `d_model: 32`. That would have surfaced in an `n=2` vs `n=4` A/B as
    an architecture effect when it was an epsilon artifact. Averaging keeps single-stream scale for every `n`,
    which is also why this repo does not need the paper's compensating "scale output-module init std by
    `sqrt(n)`" adjustment — applying it here would *break* the equal-to-a-residual-network property.
  - **The per-iteration stagger advances by one stream, not by the loop body's sublayer count.** The paper
    initialises layer `k`'s read to `e_{k mod n}`; the obvious generalisation to a weight-shared loop is to
    advance the variant by the body's true unrolled sublayer count, `2 * (n_layers - 1)`. That is always even,
    so at `n = 2` it is congruent to 0 mod `n` for *every* model (and at `n = 4` for every odd `n_layers`):
    every iteration reads the identical stream, the per-iteration routing silently collapses to one pattern,
    and nothing fails — the model just quietly gives up the reason to have per-iteration variants at all.
  - **`optim.py` excludes them from Muon and does not decay the static ones.** `alpha_r` is
    `(n_variants, n, n)` and `dyn_proj` is `(n_variants, d_model, 2 + n)` — both >= 2-D, so without the
    `"hyper"` entry in `_MUON_EXCLUDED_SUBSTRINGS` they fall straight through to Muon, whose
    fixed-spectral-norm update is exactly wrong for a one-hot read and an identity mix. `_is_hyper_static`
    additionally routes `beta`/`alpha_m`/`alpha_r` to the no-decay group: decaying them drags the model off
    precisely the initialisation that makes it a residual network. The paper splits the same way.

  **Cost.** Parameters are negligible (~0.06% of a `d_model: 256` model at `n=4`). Activation memory is `n` times
  the residual stream, which `activation_bytes_per_token` models so `auto_batch_size` accounts for it. Step time
  is the real price and it is *not* free the way the FLOP count suggests — the read/write are `O(n * d_model)`
  per token against a block's `O(12 * d_model^2)`, but at these widths they are memory-bound, so what matters is
  traffic, not arithmetic. Compiled, fwd+bwd, `n_layers: 4`, `loop_count: 6`, seq 512, bf16:

  | | `d_model` 256 (batch 16) | 512 (batch 8) | 1024 (batch 4) |
  |---|---|---|---|
  | `n=1` | 35.9 ms | 38.9 ms | 56.7 ms |
  | `n=2` | 46.7 ms (1.30x) | 51.0 ms (1.31x) | 69.0 ms (1.22x) |
  | `n=4` | 50.7 ms (1.41x) | 53.4 ms (1.37x) | 71.8 ms (1.27x) |

  The overhead falls with width, as the `n*d` vs `d^2` scaling predicts, but slowly. Unlooped
  (`loop_count: 1`) it is much smaller — 1.10x/1.14x at `d_model: 256` — because the loop body is what
  multiplies the number of residual writes. Measure compiled, not eager: inductor fuses the write's elementwise
  epilogue and the dynamic branch's norm/tanh, and eager overstates the cost by roughly 10 points. Three
  formulations were measured and the fast ones are in the code: a `matmul` read rather than an einsum
  contracting dim -2 (3.6x), a `movedim` pair around the write's matmul (1.3x), one fused `dyn_proj` matmul
  rather than three (3.9x), and an fp32 `rsqrt` reduction that never materialises an fp32 copy of the
  `n`-times-wider hidden (1.3x).

  Because the step is ~30-40% more expensive in the looped regime, a fixed-step A/B is *not* a fixed-compute
  comparison here: the recorded A/B pins steps, so it already flatters hyper-connections relative to what an
  equal-wall-clock comparison would show. See "Measured results" — it does not help even on the generous
  reading.

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

  `cfg.model.moe_balance` selects how load is balanced: `"aux_loss"` (the gradient term above), `"bias"`, or
  `"both"` (default). The bias term is a non-learned per-expert `expert_bias` buffer added to the routing logits
  **for top-k selection only** — the gating *weights* still come from the unbiased softmax, which is what makes it
  "loss-free": it steers which experts a token uses without distorting how much each contributes, so it never
  competes with the LM objective the way the aux loss does. It's updated by an explicit sign-based rule in
  `DenseTransformer.update_expert_bias()`, called from `train.py` **after** `optimizer.step()` rather than inside
  `forward` — it mutates a buffer with no gradient, so keeping it out of the graph avoids a `torch.compile` break
  on every micro-batch. Under `"bias"`, `_collect_moe_aux_loss` returns an exact zero rather than a weighted-down
  term, keeping it out of the graph entirely.

  `cfg.model.moe_n_shared` adds an always-on expert (DeepSeekMoE) whose output is added to every token
  unconditionally, with no routing and no capacity limit — it absorbs the computation every token needs so the
  routed experts can specialise instead of each re-learning the common case. It counts *in full* in
  `num_active_parameters()` (every token activates it), unlike routed experts which are discounted to `moe_top_k`.
  `cfg.model.moe_expert_ffn_mult` sets each routed expert's width as a fraction of `ffn_dim`, for fine-grained
  MoE: `n_experts: 32, moe_top_k: 8, moe_expert_ffn_mult: 0.25` activates the same parameter count per token as
  8 experts at `top_k` 2 but from 4x the combinations — affordable precisely because the batched `baddbmm`
  dispatch below keeps step time nearly flat in expert count.

  `cfg.model.moe_eval_full_capacity` (default on) sizes per-expert capacity to the *actual* per-expert load
  outside training mode, so no token is ever dropped at eval or generation. This fixes a real bug: capacity is
  otherwise `capacity_factor * n_tokens * top_k / n_experts`, which scales with how many tokens share the forward
  pass — so the same prompt scored alone and scored inside a batch dropped different tokens, and incremental
  decoding diverged from a full forward (`tests/test_kv_cache.py` covers this). Capacity limits exist to bound
  *training* throughput and memory; at inference they only discard computation.

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
- **`optim.py`** — optimizers and parameter-group construction, split out of `train.py` (which was carrying ~160
  lines unrelated to the training loop). `build_param_groups` lives here rather than in `train.py` because
  `migrate_optimizer_to_cpu_offload` needs it and importing it back would make the two modules circular.

  `cfg.train.optimizer` selects `"muon"` (default) or `"adamw"`. **Muon** replaces each hidden weight matrix's
  momentum update with its orthogonal polar factor, computed by a 5-step Newton–Schulz quintic iteration
  (`orthogonalize`) — every singular value is driven toward 1, so the step is spread evenly across directions
  instead of being dominated by the largest few, which is what lets it take a much larger learning rate.
  `orthogonalize` is written with batched `@`/`.mT`, so `BatchedExperts`' stacked `(n_experts, in, out)` weights
  are orthogonalised per expert with no reshaping — dim 0 is a free batch dimension. Note the iteration
  deliberately does *not* converge: the tuned coefficients settle into a band around [0.68, 1.13], which is the
  intended operating point (it needs the spread collapsed, not the values exact).

  `MuonWithAuxAdam` holds both algorithms in **one** Optimizer object with a per-group `algorithm` key, because
  everything downstream assumes a single optimizer — the `GradScaler`'s unscale_/step bookkeeping is keyed on it,
  `LambdaLR` drives its `param_groups`, `save_checkpoint` serialises its `state_dict`, and the OOM handler swaps
  it. `build_muon_param_groups` routes 2-D+ hidden weights to Muon and leaves the rest to AdamW: the tied
  `token_emb`/`lm_head` (per-token rows rather than a hidden linear map, and the largest tensor in the model, so
  Newton–Schulz on it would dominate step time), and the routers/gates (tiny tensors whose exact scale is
  load-bearing, where Muon's fixed-spectral-norm update is the wrong behaviour). Muon's group runs at
  `cfg.train.muon_lr` (default `0.02`, ~50x AdamW's) while the AdamW groups keep `cfg.train.lr` — a separate
  field precisely so every existing config's tuned `lr` still means what it did.

  **That last decision left `lr` badly mistuned, and it was the single largest quality bug measured in this
  repo.** Keeping `lr` fixed across the Muon switch preserved its *value* but silently changed its *job*: it was
  tuned when AdamW trained every tensor, and afterwards it governed only what Muon doesn't own — overwhelmingly
  the tied `token_emb`/`lm_head` matrix. At `3e-4` that embedding became the bottleneck of the whole model.
  Sweeping it on `configs/tinystories.yaml` (400 steps, batch pinned at 32, `auto_batch_size: false`) gives a
  clean bowl with its minimum **30-100x above the old default**:

  | `train.lr` | 3e-4 (old default) | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
  |---|---|---|---|---|---|---|
  | `val/loss` @400 | 2.9111 | 2.4460 | 2.1800 | 2.1117 | **2.1051** | 2.1520 |

  Run-to-run noise on this setup is ~0.002 (a re-run of the baseline gave 2.9091), so the ~0.8 spread is not
  close to a judgement call. `cfg.train.embed_lr` exists because of *where* that win comes from: holding
  everything else at `3e-4` and raising only the embedding recovers 0.754 of the 0.797, while raising only the
  norms/gates/routers recovers 0.416 — they overlap rather than add, but the embedding is plainly the dominant
  term. It defaults to `None`, resolving to `lr` (the range-collapsed inert default this file's convention asks
  for), so the embedding gets its own parameter group at exactly the LR it had when it shared AdamW's decayed
  group. Set it to tune the embedding independently of the routers and gates, whose exact scale is load-bearing
  in a way the embedding's is not — that is the whole reason it is a separate field rather than advice to raise
  `lr`.

  **muP** (`cfg.model.mup_base_d_model`) makes hyperparameters transfer across width. It defaults to `None`,
  resolving to `d_model`, so `mup_width_mult` is exactly `1.0` and every correction is an identity until a base
  is set. When it isn't 1: hidden-weight init std scales by `1/sqrt(m)`, and the output logit multiplier by
  `1/m`. Two subtleties. The tied embedding is initialised **last**, after `self.apply(_init_weights)`, because
  `apply` visits `token_emb` (an `nn.Embedding`) before `lm_head` (an `nn.Linear`) and they are the same tensor —
  so the hidden-weight branch would otherwise overwrite the embedding's std. And the attention logit scale stays
  `1/sqrt(head_dim)` rather than becoming `1/head_dim`: muP's `1/d` scale applies when the *head* dimension grows
  with width, but here `head_dim` is a fixed config constant and width grows by adding heads, so each q·k dot
  product sums a fixed number of terms and its variance doesn't grow with `d_model`.

  The third correction is the **per-tensor LR**, `1/m` on hidden weights and `Θ(1)` on everything else, and it
  lives in `build_param_groups` — i.e. only on the `optimizer: adamw` path. `build_muon_param_groups` carries a
  per-group `mup_lr_scale` that is `1.0` everywhere, deliberately rather than by omission: Muon's update is
  spectrally normalised and so already approximately width-invariant, and the tensors its auxiliary AdamW owns
  are exactly the ones muP leaves at `Θ(1)` anyway (the tied embedding, whose fan-in is a row lookup and so
  doesn't widen; 1-D gains and biases; and the routers/gates, which are hidden-like and would strictly want
  `1/m` but are a rounding error in both parameter count and gradient norm).

  One trap when the correction is live: with `embed_lr` unset — the default — the tied embedding sits in the
  *decayed* group and would ride its `1/m` scaling. `build_param_groups` therefore splits it into its own
  unscaled group as soon as `mup_width_mult != 1.0`, and both call sites (`build_optimizer` and
  `_adamw_to_cpu_offload`) must keep passing identical arguments so that extra group appears in both or neither
  — the tier-2 OOM migration zips the two optimizers' groups positionally.

  **muP is validated by coordinate check, not by assumption.** The claim is that activation scale is `Θ(1)` in
  width at *every* training step, which is the property that makes a tuned LR transfer. Measured on the real
  model through the real `build_optimizer`, identical data/seeds, widths 128→2048 (16x) at fixed `head_dim`,
  8 steps, fp32 — ratio of mean |activation| at `d=2048` vs `d=128`, where `1.00` is a pass:

  | arm | `attn_out` | `ffn_out` | `logits` |
  |---|---|---|---|
  | muon + muP (the default path) | 0.85 | 1.07 | 0.94 |
  | muon + muP, `loop_count: 4` | 0.95 | 0.92 | 0.99 |
  | muon, muP off | 1.81 | 6.45 | 3.48 |
  | muon, muP off, `loop_count: 4` | 2.53 | 4.28 | 7.69 |
  | adamw + muP | 1.03 | 1.00 | 0.49 |
  | adamw, muP off | 15.35 | 68.36 | 4.54 |

  The default path is flat to within ±8%, looped and unlooped, and muP is doing real work rather than being
  vacuously flat — turning it off moves logits 3.5x unlooped and 7.7x looped, and looping makes the
  unparameterised model *worse*, as multiplying residual writes into one accumulator should.

  The `adamw` row is post-fix. Before it, the missing `1/m` grew `attn_out` **16.7x across that 16x sweep —
  linearly in width** — and the reason it survived so long is the part worth remembering: **`val/loss` cannot
  see this.** `ln_f` is `RMSNorm` and therefore scale-invariant, so it launders the blown-up residual stream
  away immediately before the LM head; the logits ratio stayed at 1.04 while the network's interior was
  entirely width-dependent. A coordinate check catches it and a loss curve never will.

  Two caveats. `adamw + muP`'s logits still drift *down* ~2x over the 16x sweep (`m^-0.26` rather than flat) —
  far milder than the 4.5x growth without muP, and unchanged by the LR fix, so it's a separate readout-side
  effect on the tied-embedding path; the Muon path doesn't show it. And a coordinate check is necessary, not
  sufficient: the actual claim is that the *optimal LR* transfers, which needs an LR sweep at two widths
  showing the minimum land in the same place. That hasn't been run.

  `tests/test_optim.py::test_mup_keeps_adamw_activations_flat` is the regression test, a CPU-sized coordinate
  check. Its 16x width span and 20 steps were picked by measurement and are load-bearing: the drift accumulates
  with training, so a smaller version isn't a weaker test but a **vacuous** one — at 4x width over 6 steps the
  pre-fix code scored 1.12 and passed cleanly. At 16x over 20 steps it scores 9.71 against a fixed build's 0.99.
  The `mup_base_d_model=None` control matters as much as the assertion, for the same reason.

  Tier-2 OOM offload dispatches on optimizer type. For plain AdamW it swaps in `CPUOffloadAdamW` (a new object,
  so the caller rebuilds the scheduler); for `MuonWithAuxAdam` it flips that optimizer's AdamW groups to
  CPU-resident moments **in place** and returns the same object, so no scheduler rebuild is needed — callers test
  identity to decide. Only the AdamW groups are offloaded: Muon's state is a single momentum buffer (half
  AdamW's footprint) and its step is a matmul chain per parameter, so running it against CPU memory would cost
  far more than the VRAM it returns. In a Muon run the AdamW groups hold the embedding matrix anyway, typically
  the single largest tensor.

- **`train.py`** — plain PyTorch training loop (no HF `Trainer`): the optimizer from `optim.build_optimizer` plus
  a warmup + cosine-or-WSD LR schedule
  (`build_lr_scheduler`), manual loss computation (`compute_loss`), gradient clipping, periodic W&B logging
  (`train/loss`, `train/lm_loss`, `train/z_loss`, `train/mtp_loss`, `train/ponder_cost`,
  `train/mean_loop_depth`, `train/expert_bias_spread`, `train/lr`, `val/loss`) plus a matching stdout line,
  periodic checkpointing to `cfg.train.output_dir` (raw
  `torch.save` of state dict + config), and periodic `evaluate()` against the validation split. The stdout line
  matters more than it looks: W&B was previously the only place a loss ever appeared, so a run with
  `wandb.mode: disabled` (sweeps, CI, quick A/Bs) produced no visible signal at all.

  `compute_loss` applies the causal-LM one-position shift to the *labels* (padding them with `ignore_index`),
  not by slicing `logits[:, :-1]`. The slice is a non-contiguous view whose `.contiguous()`/`.view()` forces a
  full copy of the `(batch, seq, vocab_size)` logits — the largest activation in the model — on every forward;
  shifting labels instead makes `logits.view(-1, vocab_size)` a free reshape over exactly the same targets.
  Worth ~7% of step time at the `configs/tinystories.yaml` size. It returns `(lm_loss, z_loss)`: the second is
  the log-Z regulariser `mean(logsumexp(logits)^2)` (`cfg.model.z_loss_weight`, default `1e-4`), which keeps
  logit scale from drifting and matters more here than in a plain transformer because looping multiplies
  effective depth without adding parameters. It's applied to the *training* loss only — `evaluate()` discards it
  so `val/loss` stays a pure LM number, exactly as `ponder_cost` and `moe_aux_loss` already are. Note the
  reduction order: `logsumexp` over the vocab **first**, then mask, rather than `flat_logits[mask]`, which would
  copy nearly the whole logits tensor just to drop one row per sequence.

  `build_lr_scheduler`'s warmup ramps over `(step + 1)`, because `LambdaLR` evaluates the lambda at step 0 to set
  the LR for the *first* `optimizer.step()` — a plain `step / warmup_steps` ramp spends that entire first step at
  `lr=0`. It then decays to `cfg.train.min_lr_ratio * lr` (default `0.1`) rather than to 0, since the tail
  of a run at a ~0 LR contributes nothing; set `min_lr_ratio: 0.0` for the old decay-to-zero behavior.
  `cfg.train.lr_schedule` selects `"cosine"` (default) or `"wsd"` — warmup, hold at full LR, then decay only over
  the final `wsd_decay_ratio`. WSD's advantage is that its stable phase doesn't depend on `max_steps`, so a run
  can be extended or branched from a mid-training checkpoint without the earlier steps having been trained on a
  schedule shaped for a different horizon; cosine's shape is a function of `max_steps`, so changing it
  invalidates everything before the change. It stays off by default because it isn't a quality win and switching
  it would silently reshape every config whose `lr` was tuned against cosine.

  `compute_mtp_loss` handles the auxiliary multi-token-prediction heads (`cfg.model.mtp_heads`, default `1` =
  ordinary next-token prediction). Head *d* predicts the token *d+1* positions ahead, fusing the previous head's
  hidden state with the embedding of the token that head was predicting, running one `TransformerBlock`, and
  reusing the trunk's **shared** `lm_head` — which is what keeps a head's cost to one block rather than another
  `d_model x vocab_size` matrix. `forward()` returns the heads' *hidden states* rather than logits, so eval and
  generation (which skip the heads entirely) pay nothing and training projects one
  `(batch, seq, vocab_size)` tensor at a time. The heads are excluded from `num_active_parameters()`: they never
  run at inference, so counting them would inflate a `tokens_per_param`-derived `max_steps` — the same reasoning
  already applied to inactive MoE experts. See `configs/tinystories_mtp.yaml`.

  Checkpoints are **resumable**: `save_checkpoint` writes the optimizer state, LR-scheduler state and
  `GradScaler` state alongside the weights/step/config. Only one checkpoint per run is kept — each
  `save_every` save deletes the previous `step_*.pt` before writing the new one — so an output directory
  holds at most one `.pt` file. `cfg.train.resume_from` (opt-in, default `null`) restores all saved state.
  Set it to a checkpoint path, or to the literal `"auto"` to pick the single `step_*.pt` in `output_dir` —
  so an interrupted run can be relaunched with its config unchanged. Without the optimizer moments a
  "resumed" run restarts AdamW from zero momentum at warmup LR, which shows up as a loss spike. An explicit
  `resume_from` path that doesn't exist raises rather than silently starting from scratch; `"auto"` against
  an empty `output_dir` is just a fresh run. What is *not* restored is the DataLoader position and RNG state,
  which trade off against each other: `train()` re-seeds off the resumed step so the loader draws a different
  shuffle order rather than replaying batches already trained on. A resumed run is therefore statistically
  equivalent to an uninterrupted one, not bit-identical — with `dropout: 0.0` it is bit-identical (verified:
  same weights, same AdamW moments, same LR sequence).

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

  `--loops N` overrides the iteration count for the whole generation, sizing the KV cache to match
  (`new_kv_cache(loop_count=N)`). For a model trained with stochastic loop depth this is test-time compute
  scaling: the same weights can spend more per token at inference than any training step did. Per-iteration
  parameter banks (RMSNorm gains, router biases) clamp at their last entry rather than wrapping, so running
  deeper than training reuses the deepest learned parameters instead of cycling back to the shallow ones.
  `load_checkpoint` passes no `eos_id`, so document masking is simply off during generation — a single prompt is
  one document, and the mask would be all-ones anyway.

Entry points: `radiance.train:main` (`--config`) and `radiance.generate:main` (`--checkpoint`, `--prompt`,
`--loops`, ...) — `radiance-train` / `radiance-generate` console scripts after install.

## Measured results

A/B runs on `configs/tinystories.yaml` (400 steps, RTX 5090, `val/loss` at step 400), each changing one thing:

| change | baseline | with change |
|---|---|---|
| `optimizer: muon` vs `adamw` | 3.300 | **2.826** |
| `doc_attention_mask` on vs off | 2.808 | **2.774** |
| loop conditioning + input injection vs plain looping (`loop_count: 3`) | 2.900 | **2.861** |

Muon is the large one *of those three*: it passed AdamW's *final* loss at roughly half the steps (2.827 at step
200 vs AdamW's 3.300 at step 400). The other two are consistent rather than dramatic — better at every eval
point, not just the last. Re-run any of these with `wandb.mode: disabled`; the stdout logging makes the numbers
visible without W&B.

**Retuning `train.lr` dwarfs all of them, and it is a fix rather than a feature** — see the sweep table under
`optim.py` for why `3e-4` stopped being the right value the moment Muon took over the hidden weights. Confirmed
at a longer horizon (1200 steps, same config and pinning, so these are *not* comparable to the 400-step numbers
above):

| 1200 steps | step 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---|---|---|---|---|---|
| `lr: 3.0e-4` (old default) | 3.9008 | 2.8956 | 2.5929 | 2.4093 | 2.2926 | 2.2341 |
| `lr: 3.0e-4`, `embed_lr: 1.0e-2` | 2.7470 | 2.3321 | 2.1292 | 1.9872 | 1.8830 | **1.8290** |
| `lr: 1.0e-2` | 2.6801 | 2.2719 | 2.0797 | 1.9482 | 1.8500 | **1.7984** |

Better at every eval point, by a wide margin, in both retuned arms — and the retuned model reaches the old
default's *final* 1200-step loss in roughly 500 steps, so this is worth more than tripling the training budget.
Raising `lr` wholesale edges out raising `embed_lr` alone (1.7984 vs 1.8290), which is consistent with the
400-step decomposition: the embedding is ~93% of the win and the remaining norms/gates/routers supply the rest.

Two caveats worth keeping attached to these numbers. They were measured at one scale only (`d_model: 256`,
TinyStories) — muP (`mup_base_d_model`) is the intended mechanism for carrying a tuned `lr` across width, and
nothing here establishes that `1.0e-2` is right for `configs/fineweb_500m.yaml`. And the 400-step baseline here
(2.9111) does not match the 2.826 recorded in the table above, almost certainly because these runs pin
`batch_size: 32` with `auto_batch_size: false`; the deltas are sound because every arm shares those settings, but
the absolute numbers belong to their own series.

ACT sparsity is a throughput change rather than a quality one, so it's measured separately: on
`configs/tinystories_router.yaml` (400 steps, pinned `batch_size: 16` so the arms are comparable), `val/loss` at
step 400 was **3.0735** dense, **3.0757** at `act_capacity_ratio: 0.5` and **3.0728** at `0.25` — indistinguishable.
See the `act_capacity_ratio` discussion above for what it buys in time and memory, and why a short run's
wall-clock understates it.

`use_nsa` was A/B'd the same way (400 steps, RTX 5090, `d_model: 256`/`head_dim: 64`/`n_layers: 6`, pinned
`batch_size: 32`), specifically against the *real* default it would replace — `use_nsa` required
`doc_attention_mask: false`, so the fair comparison isn't NSA vs. a doc-masking-disabled baseline, it's NSA vs.
whatever a config actually runs with today. `val/loss` at step 400: **2.8540** dense + `doc_attention_mask: true`
(today's default), **2.8823** dense with `doc_attention_mask: false` (isolates the cost of losing doc-masking
alone), **2.9110** NSA. NSA is worse than both — worse than today's default by 0.057, and still worse than the
doc-masking-disabled baseline it's actually forced to compete against by 0.029. So `use_nsa` was never defaulted on:
the A/B doesn't support flipping it, and even setting quality aside, enabling it would have forced `doc_attention_mask`
off too (the two are mutually exclusive), silently giving up an already-proven win for one that isn't
yet real. Worth re-running at a longer `max_seq_len` before drawing a final conclusion — at 512 tokens and
`nsa_block_size: 128` there are only 4 blocks, which is little room for the selection branch's sparsity to pay
for its own approximation error, and this A/B measured quality only, not the compute saving that's NSA's other
selling point at genuinely long context. The feature has since been removed from the codebase (see above).

**Hyper-connections (`hyper_conn_streams`) do not pay for themselves at this scale, and the interesting result
is the learning rate rather than the architecture.** A/B on the looped shape they are aimed at — `d_model: 256`,
`n_layers: 4`, `loop_count: 6` (19 executed blocks), pinned `batch_size: 16`, `auto_batch_size: false`,
`lr: 1.0e-2`, 400 steps:

| `hyper_conn_streams` | `hyper_conn_lr` | step 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| 1 (baseline) | — | 3.3285 | 2.7206 | 2.4225 | **2.2821** |
| 2 | `1.0e-2` (= `lr`) | 4.1753 | 3.1816 | 2.8383 | **2.6744** |
| 2 | `1.0e-3` | 3.2490 | 2.7054 | 2.4195 | **2.2858** |
| 2 | `1.0e-4` | 3.2652 | 2.7172 | 2.4280 | **2.2881** |
| 4 | `1.0e-3` | 3.3389 | 2.7812 | 2.4928 | **2.3570** |
| 4 | `1.0e-4` | 3.2617 | 2.7293 | 2.4400 | **2.2964** |

Read the first two rows before anything else: sharing `lr` costs **0.39 val/loss**, far and away the largest
effect in the table and larger than any *feature* win recorded on this page. That is not hyper-connections being
bad, it is the `embed_lr` lesson repeating — `lr` reaches a grab-bag of tensors whose ideal step sizes differ by
orders of magnitude, and AdamW's update is ~`lr` per step almost regardless of gradient scale, so 400 steps at
`1e-2` moves a coefficient by O(1). For a *structural* coefficient (one-hot read, identity mix) an O(1) move
erases the routing rather than refining it. Hence `train.hyper_conn_lr`, and hence it defaults to `1e-3` rather
than to `None` — the usual range-collapsed default would have shipped the feature in its broken configuration.
Note also that the best LR falls as `n` rises (at `n=4`, `1e-4` beats `1e-3` by 0.06): more streams means more
coefficients, so more total drift at a given step size.

With that fixed, the honest verdict is **neutral-to-slightly-negative**: the best arm (`n=2`, `1e-3`) lands
0.004 *behind* the single-stream baseline, which is at the edge of this setup's ~0.002 noise floor, and `n=4` is
clearly behind. Since hyper-connections also cost 30-40% of step time in this regime (see the table under
`model.py`), a fixed-*compute* comparison is worse still than a fixed-step one. So `hyper_conn_streams` stays at
`1`, on the same terms `use_nsa` was removed: implemented, tested, documented, and not defaulted on because the
measurement doesn't support it.

Worth re-testing before treating this as settled, because the conditions the mechanism targets were only
partially met here. The paper's argument is about depth, and 19 executed blocks at `d_model: 256` over 400 steps
is a small instance of it; the `n*d` vs `d^2` scaling also means the step-time penalty shrinks with width, so a
wider, deeper, longer run is where the trade could plausibly flip. Nothing here rules that out — it rules out
turning it on by default today.

Two cautions when running your own A/B, both learned the hard way here. **Pin `batch_size` and set
`auto_batch_size: false`** — otherwise a change that reduces memory (sparsity, checkpointing) is silently handed a
larger batch, and the arms differ in two ways at once. And **check for `ending run early`** in the output: an
OOM-terminated arm still exits 0 and still prints a few evals, so it looks like a valid short run rather than a
failed one. `configs/tinystories_router.yaml` at the default `vram_safety_margin` does OOM on a 32 GB card —
`estimate_batch_size` is optimistic for router mode, where the loop body is re-run `max_loops` times. A third,
learned on this change: **give each arm the whole GPU.** Two of the first hyper-connection arms were invalidated
by a diagnostic run started alongside them, and the symptom was an OOM-shaped early exit in the *other* process,
not the one at fault.

**Startup compile cost**, measured at the same size on the first forward/backward with `mode=None`
(`d_model: 256`, `n_layers: 4`, `loop_count: 6`, batch 8 x 512):

| configuration | compile | warm step |
|---|---|---|
| fixed loop count, `doc_attention_mask: false` | 18.9s | 0.04s |
| fixed loop count, `doc_attention_mask: true` | 25.2s | 0.05s |
| fixed loop count, `+ grad_checkpoint` | 36.1s | 0.06s |

Document masking is cheap (~6s) and `grad_checkpoint` adds ~45%. **Stochastic loop depth is the expensive one**:
each distinct count in `[loop_count_min, loop_count_max]` is a separate dynamo graph costing roughly the
per-graph figure above, and it scales linearly — a 3-count range measured 78.2s, almost exactly 3x the 25.2s
single-graph case. A wide range, especially with `grad_checkpoint`, spends minutes before the first loss line
appears. That is compilation, not a hang; warm steps stay ~0.05s throughout. Keep the range narrow, or set
`train.compile: false`, while iterating on a config. (Beware measuring this back-to-back in one process —
inductor's FX graph cache makes later configurations look far cheaper than they are from cold.)

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
- Sparse attention: `model.use_nsa` has been removed from the codebase. See the "Architecture" section above
  for the removal rationale and git history for the former `configs/tinystories_nsa.yaml` example. Not currently a win
  at TinyStories scale (see "Measured results"); a candidate for a longer-context follow-up A/B if re-introduced.
- New training behavior (e.g. different scheduler, mixed precision): changes belong in `train.py`; keep the loop
  step-based and keep config-driven values in `TrainConfig` rather than hardcoding. A new *optimizer* belongs in
  `optim.py` — add it to `build_optimizer` and give it a `build_*_param_groups`; `MuonWithAuxAdam` is the
  reference for one that needs different treatment per parameter class.
- New per-iteration behavior for the loop body: put non-tensor state on `LoopContext` and read it in the block.
  `loop_iter_conditioning` is the reference example. Anything grad-carrying stays a positional argument (see
  `LoopContext`'s docstring), and remember to check `tests/test_loop_identity.py::test_no_dead_parameters` — a
  parameter bank that some loop mode never reaches trains at its init value forever, silently.
- Changing the residual stream itself (rather than what writes into it): `HyperConnection` /
  `cfg.model.hyper_conn_streams` is the reference example, and the one to read for what such a change touches.
  Widening the hidden state changes the *rank* of every tensor between sublayers, so the work is mostly in the
  places that reimplement the residual write or index it positionally — `TransformerBlock.forward`,
  `_run_loop_body`'s sparse closure, `_run_loop_body_sparse`'s gather/scatter, `_forward_act`'s halting
  broadcasts, and `activation_bytes_per_token`. Everything *below* the sublayer boundary (attention, FFN, MoE,
  the KV cache, `generate.py`) stays untouched, because the read hands them the same `(batch, seq, d_model)`
  tensor as before — preserve that and the blast radius stays small. Watch for blocks built outside the trunk:
  `MTPHead` constructs its own `TransformerBlock` and must keep the single-stream path.
- New default-on feature: make its parameters inert (zero-init, identity-valued, or range-collapsed — see the
  defaults convention under `config.py`), then add the pair of tests that pins it: bit-identical to the
  feature-off model at init, *and* demonstrably different once its weights move. `tests/test_inert_defaults.py`
  is the pattern. The second half matters as much as the first — an "inert" feature with no second test could be
  inert forever and nobody would notice. And check the *cost* before deciding "on": `hyper_conn_streams` is
  perfectly inert at `n > 1` and still defaults off, because inert is not the same as free.
