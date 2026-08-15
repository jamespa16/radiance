# `model.py` — the model

Everything architectural lives here. Related: [config.md](config.md) for the defaults convention,
[nvfp4.md](nvfp4.md) for 4-bit linears, [results.md](results.md) for every A/B referenced below.

## Contracts

`forward()` returns a `ModelOutput` NamedTuple (`logits`, `ponder_cost`, `mean_loop_depth`, `moe_aux_loss`,
`mtp_hidden`), not a bare tuple, so fields can be added without breaking every call site. The middle three are zero
scalar tensors when their feature is off, so callers have one contract regardless of mode.

Per-iteration state reaches the blocks on a `LoopContext` (which loop pass this is, the flex-attention `BlockMask`,
the ACT capacity, the KV cache) — six separate features need to tell a block *which pass it is on*, and bundling that
keeps `TransformerBlock`/`CausalSelfAttention`'s signatures stable. `LoopContext` deliberately holds only non-tensor
state: tensors that participate in autograd (`v_first`, the injection anchor) stay explicit positional arguments,
because `torch.utils.checkpoint` only tracks tensors it receives positionally and one hidden in a dataclass would
silently lose its recompute path under `grad_checkpoint`.

A new block/attention variant should keep the `TransformerBlock` I/O contract, `(batch, seq, d_model) -> (batch, seq,
d_model)`, so it drops into `DenseTransformer` without touching the rest of the pipeline.

## Structure

`DenseTransformer`: token + learned positional embeddings, `n_layers` pre-norm `TransformerBlock`s, final LayerNorm,
weight-tied LM head. Each block is `CausalSelfAttention` (`F.scaled_dot_product_attention` with `is_causal=True`, no
manual mask construction) followed by `FeedForward`.

`_scale_residual_init` shrinks every projection that *writes into* the residual stream (`attn.out_proj`,
`ffn.down_proj`, including each MoE expert's) by `1/sqrt(2 * blocks_executed)` after `_init_weights` — the standard
GPT-2 depth-scaled init, but counting the blocks actually *executed* per forward (`1 + loop_multiplier * (n_layers -
1)`) rather than `n_layers`, since `blocks[1:]` is re-run `loop_count`/`max_loops` times and so performs the residual
writes of a much deeper stack. Looping is exactly the regime where the unscaled init hurts most, because it
multiplies effective depth without adding parameters.

`FeedForward`'s depth is `cfg.model.ffn_depth`: that many `Linear(ffn_dim) + GELU` hidden layers between the up- and
down-projections, so MLP depth is controllable independently of block count.

### `padded_vocab_size`

`padded_vocab_size(vocab_size, multiple)` rounds the tokenizer's vocab up to a multiple of
`cfg.model.vocab_pad_multiple` (default `128`; `1` disables) before the model is built — `train.py` calls it instead
of passing `len(tokenizer)` straight through. A vocab that isn't a multiple of 64/128 leaves the model's largest
matmul (the `lm_head`) on a ragged tensor-core tile; the padding rows are unreachable by any tokenizer id, so this is
behavior-preserving, and it's worth ~9% of step time with the gpt2 tokenizer (50257 -> 50304). `generate.py` reads
the vocab width off the checkpoint rather than recomputing it (so older checkpoints still load) and masks the padding
columns before sampling, since a sampled padding id would decode to nothing and corrupt the KV cache.

### `activation_bytes_per_token`

Feeds `auto_batch_size`. It bills the `(batch, seq, vocab_size)` logits at `2 * activation_dtype_bytes + 4` (logits,
their gradient, one fp32 reduction buffer). It used to bill them at a flat fp32 x3 because `F.cross_entropy` sat on
autocast's fp32 list and upcast the whole tensor; `train.compute_loss` no longer goes through cross_entropy, so the
term now shrinks with the compute dtype like every other activation. Against the real measured peak the whole
estimate lands 1.12x high where the fp32 assumption had it 1.58x high — still on the intended over-estimating side,
but no longer spending that headroom on a smaller micro-batch than the model needs.

**Keep this in step with the loss**: it is the only place that models a term `model.py` doesn't own.

## Attention

`CausalSelfAttention` supports GQA via `model.n_kv_heads` (default `None` = standard multi-head; when set, `n_heads`
must be evenly divisible by it — `ModelConfig.n_kv_heads_resolved` — and SDPA is called with `enable_gqa=True`, hence
this project's `torch>=2.5` floor). `model.qk_norm` applies an `RMSNorm` over each head's `head_dim` to q and k
before RoPE, for stability across `blocks[1:]`'s weight-shared loop iterations. See `configs/tinystories_gqa.yaml`.

Two refinements, both on by default and both exactly inert at initialisation:

- **`model.value_residual`** mixes each block's values with `blocks[0]`'s (`v = λ·v + (1−λ)·v_first`) via a learned
  per-block scalar initialised to exactly 1.0 — since `blocks[1:]` is a weight-shared loop, this gives every
  iteration direct access to the first block's values. The mix happens *before* `kv_cache.write`, so cached values
  are already mixed and generation needs no extra cache slot (`blocks[0]` runs first within each decode step,
  producing that token's `v_first` in-step). `blocks[0]` deliberately has no `value_lambda`: it is the block that
  *produces* `v_first`, so a mix parameter there would be allocated, never differentiated, and never trained.
- **`model.attn_out_gate`** applies a per-head gate to the attention output before `out_proj`, written
  `2 * sigmoid(Linear(x))` with the `Linear` zero-initialised in `_init_inert_gates` (which must run *after*
  `_init_weights`, the same ordering constraint `_scale_residual_init` has). The `2 *` is load-bearing: it makes the
  gate exactly `1.0` at init, where a plain `sigmoid` would start every head at 0.5 and halve the attention output.

### Document masking (`model.doc_attention_mask`, default on)

`data.py` concatenates many documents into each `seq_len` block joined by EOS, and plain causal attention reads
straight across those joins, so the model spends capacity predicting one document's tokens from another's. Document
ids are recovered **in-model** by `document_ids()` as an exclusive cumsum over EOS positions — no extra column in the
packed dataset, so no re-tokenizing and no cache invalidation for datasets already on disk. The mask is a
`flex_attention` `BlockMask` built once per forward outside any compiled region and reused across every block and
loop iteration, exactly as the RoPE cos/sin tables are.

Three silent fallbacks to plain SDPA, all necessary: during generation (a single prompt is one document), off CUDA
(`flex_attention` is impractically slow there, and CPU sanity checks have to keep working), and at `head_dim < 16`
(its compiled kernel rejects smaller heads with an inductor lowering error).

`flex_attention` has **no attention-weight dropout**, so enabling this silently drops it — the model prints a
one-time note when `dropout > 0`, and the new configs set `dropout: 0.0`. FFN residual dropout is unaffected.

`model.loop_attn_windows` rides the same machinery to give each loop iteration its own receptive field (e.g.
`[128, 128, 512, 512]` for local-then-global passes) — a knob that exists *because* the architecture loops.

Document masking also rules out CUDA graphs; see [train.md](train.md)'s `resolve_compile_mode`.

### Differential Attention (`model.use_diff_attn`, opt-in)

Ye et al. 2024. Each head computes *two* softmax attention maps at half `head_dim` each and takes a learned
difference, `(A1 - λ·A2) @ V`, cancelling the common-mode "attention noise" the branches share — the trick a
differential amplifier uses. Structural attention variant with no zero/identity init reducing it to plain attention's
*computation*, so it follows the `use_moe`/`use_router`/`n_kv_heads` precedent (opt-in, evaluated by A/B) rather than
the default-on convention. **The A/B says yes** ([results.md](results.md)), but it costs 1.34x step time, so it is a
feature to reach for deliberately, not a free win.

The two branches are **not** extra projections. A normal head's Q/K is `head_dim` wide; a differential head splits
into `Q1`/`Q2` and `K1`/`K2` at `head_dim // 2` each, and two half-width heads sum back to exactly the width one
full-width head had — so `qkv_proj`'s output width (`d_model + 2 * kv_dim`) is identical either way and the reshape
is just a different chunking of the same tensor. RoPE's pairwise rotation already asserts `head_dim % 2 == 0`;
splitting needs that half-width to itself be even, so `use_diff_attn` requires **`head_dim % 4 == 0`** — checked as a
friendly `ValueError` at the very top of `DenseTransformer.__init__` (before any block is constructed) and again as a
defensive `assert` in `CausalSelfAttention.__init__`. The guard sits at the top of `__init__` because
`DenseTransformer.__init__` builds `blocks[0]` before it used to reach its own checks, so the assert fired first with
a worse message.

`λ` is reparameterized as `exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2) + λ_init(l)` (four learnable vectors of shape
`(head_dim // 2,)`, normal init std 0.1, one set per block — not per loop iteration; the shared loop body uses one
fixed `λ_init` for its structural position, as `value_lambda` has no per-iteration variant).
`λ_init(l) = 0.8 − 0.6·exp(−0.3·(l−1))`, keyed off `block_index` (already 0-indexed, so it *is* `l − 1`). Not a
dead-parameter risk despite the two `exp(dot)` terms nearly cancelling at init: the four vectors get real gradient
from step 0, the same one-step cold start `loop_input_injection`'s zero-init `W_inj` relies on.

A per-head RMSNorm (`diff_norm`, reusing this file's `RMSNorm` rather than the paper's non-parametric GroupNorm — a
deliberate deviation for consistency with `qk_norm`, not yet checked against the paper's version in isolation)
normalizes `A1 − λ·A2` before a fixed `(1 − λ_init)` rescale and the usual `attn_out_gate`/`out_proj` tail, which is
otherwise unaffected — both branches produce identically-shaped `(batch, n_heads, seq, head_dim)` tensors before that
point. `qk_norm`, when also enabled, applies the *same* `q_norm`/`k_norm` modules (now at width `head_dim // 2`) to
both branches rather than doubling to four. Value residual mixing is unaffected — `V` is never split.

**`KVCache` gained a second write path, `write3`, rather than a second slot per call.** Differential attention caches
`K1`, `K2` and one shared `V`, but this is still exactly one `CausalSelfAttention.forward` call per (block,
iteration) pair, so slot *count* is untouched; only what one slot holds changes. `write3` mirrors `write`'s
implicit-call-order/concat-on-refill logic. `tests/test_kv_cache.py`'s `MODES` gained `diff_attn`/`diff_attn_gqa`.

**A real `torch.compile` correctness bug was found and fixed here.** Two `flex_attention` calls per layer (one per
branch) sharing one reused `BlockMask`, whose Q/K both trace back to one `torch.split` of `qkv_proj`'s output,
silently diverged from eager once inductor traced both into one graph — no error, no warning, up to ~0.85 absolute
logit difference on a toy shape, isolated with a minimal repro independent of this codebase. `.contiguous()` and
`.clone()` did not fix it (and in some variants made *both* calls wrong instead of just the second); batching the two
into one wider call over a doubled head dimension was wrong too. The fix, `_diff_flex_attention`, is marked
`@torch._dynamo.disable`: forcing a graph break routes each call through `_flex_attention()`'s eager branch — a
separately compiled, cached `flex_attention` — instead of lowering it as part of the model's one big graph, which
measured bit-identical to eager (backward included). Plain attention is unaffected: one `flex_attention` call per
layer, already proven correct under compile.
`tests/test_compile.py::test_diff_attn_doc_masking_compiles_correctly` pins this by comparing *values*, not shapes or
crash-freedom, since the bug produced a perfectly-shaped, perfectly-finite, silently wrong tensor that every test in
`tests/test_diff_attention.py` passed the whole time. Re-verify after any PyTorch upgrade rather than assuming it is
permanent.

`act_capacity_ratio < 1.0` is incompatible with `use_diff_attn` and raises — `forward_sparse` has no differential
variant yet. `activation_bytes_per_token` bills differential attention at `17 * d_model` per token per block instead
of plain attention's `10 * d_model` — measured directly (peak allocated, fwd+bwd, bf16, diff on vs. off, delta
divided out by layer count/batch/seq/dtype width), not guessed: the extra `~7 * d_model` held at 6.99-7.00x across
`d_model` in `{512, 1024}` and across batch/seq shapes, though `d_model: 256` was noisier (7.99x) — trust the larger
measurements. See `configs/tinystories_diff_attn.yaml`.

### NSA (removed)

`cfg.model.use_nsa` was an opt-in DeepSeek NSA-style learned block-sparse attention: a coarse compression branch over
mean-pooled K/V blocks plus a fine selection branch over the top-`k` blocks a learned score picked per query *token*,
combined by a per-token gate. It has been **removed** — all NSA symbols, config fields, tests and configs are gone
from the tree. Reasons: it lost the A/B against today's default (see [results.md](results.md)), and it was a second
sparsity axis conflicting with doc masking, loop windows and ACT sparsity (it raised at `__init__` if combined with
any of them).

Two lessons worth keeping. Selection had to be per query *token*, not per query block as every other block-sparse
mechanism here is, because a block-level decision would need to average in later tokens' scores that don't exist yet
during incremental decoding — and NSA had no dense fallback at inference, since the model was *trained* attending
only to selected blocks, so decode had to reproduce the identical computation rather than an approximation of it
(contrast `doc_attention_mask`'s generation fallback, exact only because a single prompt really is one document).
That cost real efficiency: most `flex_attention` tiles become "partial" rather than cleanly skippable. And
`nsa_block_size` had to stay at `flex_attention`'s own tile size, because `create_block_mask`'s Triton kernel only
accepts a `BLOCK_SIZE` that is a multiple of its internal tile size — a smaller value raised a `LoweringException` at
the first compiled forward rather than merely running slower. See git history for the implementation and the removed
`configs/tinystories_nsa.yaml`.

## The loop

The first block runs once; the remaining `n_layers - 1` blocks (`blocks[1:]`) form a shared-weight loop body re-run
either a fixed `cfg.model.loop_count` times (default), or — with `cfg.model.use_router: true` — a learned number of
times per token via `ACTRouter`, a small `LayerNorm + Linear(d_model, 1) + sigmoid` head implementing Adaptive
Computation Time (Graves 2016).

In router mode (`_forward_act`), each token position accumulates its own halting probability across iterations and
halts independently once that sum crosses `1 - cfg.model.halt_epsilon` or `cfg.model.max_loops` is reached; the
loop's output is a probability-weighted sum of that token's per-iteration hidden states (not just the last one), and
once a position halts its state is frozen and carried forward unchanged so later iterations' causal attention still
sees a stable key/value for it. Because this is dense, fully-batched compute, router mode by default does **not**
save wall-clock over running `max_loops` iterations for every token — the adaptivity shows up in the loss signal
(`ponder_cost`) and in what gets accumulated, not in runtime. See `configs/tinystories_router.yaml`.

### `cfg.model.act_capacity_ratio` is what changes that

Below 1.0, each interior ACT iteration computes only that fraction of each sequence's positions — the
highest-priority still-running ones — through the *entire* loop-body stack. Everything else keeps its hidden state
and serves its keys/values from a retained per-block store. `_run_loop_body_sparse` gathers the selected positions
once, runs every block on that narrow tensor, and scatters back; `CausalSelfAttention.forward_sparse` computes Q/K/V
only for them, scatters the fresh K/V into the retained full-length store, and attends the gathered queries against
the whole thing.

Pure fwd+bwd, bf16, `max_loops: 6`:

| `act_capacity_ratio` | `d512 L8`, batch 8x512 | `d256 L6`, batch 16x512 |
|---|---|---|
| 1.0 (dense) | 105.6 ms | 71.3 ms |
| 0.5 | 87.0 ms (1.21x) | 63.1 ms (1.13x) |
| 0.25 | 73.4 ms (**1.44x**) | 60.1 ms (1.19x) |

Peak memory at `d512 L8` falls 14.79 -> 12.23 -> 10.76 GB over the same ratios. The gain scales with how much of the
step the loop body actually is — 93% of the FLOPs at `d_model: 512`, 83% at `256` — because the gather/scatter and
the materialised mask are a roughly fixed overhead that amortises better on a wider model.

**And the step is not the run.** A 400-step `configs/tinystories_router.yaml` A/B came out 192s dense vs 187s at
ratio 0.25, barely 3%, because at that size 400 steps is only ~29s of model compute and the rest is compile, data
loading and (always-dense) eval. Benchmark the step, not a short run. Quality cost over that same A/B at a pinned
`batch_size: 16`: `val/loss` **3.0735 dense, 3.0757 at 0.5, 3.0728 at 0.25** — identical within noise at every eval
point. Whether that holds at longer horizons is untested.

Three things about it are load-bearing:

- **It is an approximation, and `tests/test_act_kv_invariance.py` says precisely why.** Reusing a halted position's
  K/V is exact only for the *first* block of the loop body: K and V are per-position projections of a block's input,
  and a halted position's input to `blocks[1]` is `frozen_x`, which by construction stops changing — but
  `blocks[1]`'s *output* there mixes in attention over still-running positions that keep evolving, so `blocks[2]`'s
  input and hence its K/V drift. Beware the misleading special case: if the halted set happens to be a causal
  *prefix*, halted positions never attend to a running one and every block looks invariant. Real halting is
  scattered. Read that file's docstring before attempting to make ACT skip halted tokens.
- **It is training-only**, gated like `grad_checkpoint` on `self.training and torch.is_grad_enabled() and kv_cache is
  None`. Cached decoding cannot use it (one token at a time, nothing to select among), so if a full forward used it
  while decoding didn't, `evaluate()` and `radiance-generate` would compute the same prompt two different ways.
  Confining it to training also keeps `val/loss` comparable across ratios.
- **It refuses `grad_checkpoint`.** Recomputation during backward would write into the retained K/V store a second
  time and silently corrupt it, so `__init__` raises rather than producing wrong gradients. Sparsity already cuts
  activation memory substantially, so the two overlap anyway.

Selection (`_act_select`) is per *sequence*, not over the flattened batch the way `_sparse_ffn_delta`'s is —
attention needs each row's queries to belong to that row. Causality for the scattered queries is
`true_position(query) >= key_position`, which `is_causal=True` cannot express, so `_sparse_attn_mask` materialises a
`(batch, 1, capacity, seq_len)` boolean mask (folding in document masking when active, since a flex `BlockMask`
assumes dense queries). That mask costs the fused causal kernel, which is why the measured speedup is ~70-80% of the
theoretical `(2 + (max_loops - 2) * ratio) / max_loops`.

### Giving the loop body an identity

A weight-shared loop body has no idea which iteration it is on — without help, iteration 5 is computationally
indistinguishable from iteration 1 except through residual-stream norm drift. Four settings address that, all inert
at their defaults and all worked through in `configs/tinystories_looped.yaml`:

- **`loop_iter_conditioning`** (`"norm_gains"` default, `"lora"`, `"none"`). `"norm_gains"` gives each iteration its
  own `RMSNorm` gains — the adaLN trick — at ~`2 * loop_multiplier * d_model` parameters per block, plus a
  per-iteration bias on `ACTRouter` and `MoERouter`. The `MoERouter` one makes deliberate what previously fell out
  incidentally: a token's expert choice already drifted across iterations because the hidden state evolves, and now
  an iteration can learn a standing preference. `RMSNorm` keeps its original 1-D `(d_model,)` gain shape at
  `n_variants == 1`, so an unconditioned model is bit-identical to one from before this existed, and a
  load-state-dict pre-hook broadcasts an old 1-D gain across variants so enabling conditioning on an existing run
  starts from exactly that run's weights. `"lora"` instead gives each iteration a rank-`loop_lora_rank` adapter on
  the fused QKV and FFN down projections (`IterLoRA`), with `B` zero-initialised.
- **`loop_input_injection`** (default on) re-injects **`blocks[0]`'s output** — not the raw token embedding — at the
  start of every iteration after the first: `h = h + W_inj @ anchor`, with `W_inj` zero-initialised, so exactly a
  no-op at init at any loop count. Injecting the *prelude output* matters for more than representation quality; see
  `loop_bptt_window`.
- **`loop_count_min` / `loop_count_max`** sample the loop count per training step. Both default to `None`, resolving
  to `loop_count`. Eval and generation always use the top of the range so their numbers stay deterministic.
  `radiance-generate --loops N` overrides it entirely, which is what lets a model trained across a range of depths
  spend *more* compute per token at inference than training used. `cfg.model.loop_multiplier` is the single source of
  truth for the worst-case count and is what `new_kv_cache`, `_scale_residual_init` and
  `activation_bytes_per_token` size themselves against. Stochastic depth forces compiling **without CUDA graphs** —
  see [train.md](train.md).
- **`loop_bptt_window`** backpropagates through only the last N iterations, running earlier ones under `no_grad`, so
  activation memory becomes O(N) rather than O(loop_count). **It requires `loop_input_injection`, and `__init__`
  raises if you set one without the other.** Found the hard way: running early iterations under `no_grad` severs the
  graph *upstream* of the recurrence too, so `blocks[0]`'s only route to the loss disappears and it silently trains
  at its initial weights for the whole run while the loss still falls. Injecting `blocks[0]`'s output into the
  in-window iterations restores the path. (One-step cold start: `W_inj` starts at zero, so the anchor path has zero
  derivative on step 0; `W_inj` itself gets gradient immediately, so from step 1 on `blocks[0]` trains normally.
  `tests/test_loop_identity.py` covers all of this.)

### `grad_checkpoint` (opt-in)

Recomputes each block's activations during backward instead of storing them. It pays off disproportionately here:
`blocks[1:]` is re-run `loop_count`/`max_loops` times per forward and *every* pass retains its own activations, so
activation memory scales with the loop multiplier while parameter memory doesn't. Measured on
`configs/fineweb_500m.yaml` at `seq_len=1024`: peak at micro-batch 4 drops 27.7 GB -> 11.2 GB, which is what lets
micro-batch 16 fit at all (19.7 GB) where the uncheckpointed model OOMs above 4; throughput costs ~20-25% (18.6k ->
14.5k tok/s at batch 4). Gradients are bit-identical either way. Training-only, gated on `self.training and
torch.is_grad_enabled() and kv_cache is None`, since recomputation under a KV cache would re-append to that cache.
`activation_bytes_per_token` models the checkpointed regime too (one `d_model` tensor per block boundary plus the
largest single block's transient recompute), so `auto_batch_size` spends the freed memory automatically.

## Hyper-connections: `n` residual streams instead of one

`cfg.model.hyper_conn_streams` (default `1`) attacks the accumulator all that depth writes into. At `n > 1` the
hidden state becomes `(batch, seq, n, d_model)` and each sublayer's residual write becomes a `HyperConnection`
read/write pair (Zhu et al., ICLR 2025, arXiv:2409.19606): `h_in = alpha_m^T H`, then `H' = alpha_r^T H + beta^T out`.
So a sublayer learns *which* stream to read, how the streams mix, and how its output is distributed back.
`blocks[0]` and the loop body both get them; `MTPHead`'s own `TransformerBlock` deliberately does not (it runs
outside the recursion on the already-reduced hidden), which is what the `hyper` constructor flag is for.

It is **exactly the plain residual network at init**: `alpha_m` starts one-hot, `alpha_r` the identity, `beta`
all-ones, and the streams start as `n` copies of the same vector — so every stream receives the same `out` and they
stay equal at every depth, with the one-hot read just picking one of `n` identical copies. Both matmuls are exact at
those values. Verified bit-identical to the single-stream model at `n` = 2 and 4 across the whole loop-mode matrix,
and to ~1 ulp at odd `n` (`tests/test_hyper_connections.py`). `cfg.model.hyper_conn_dynamic` (default on) adds
input-conditioned corrections `coeff = static + s * tanh(norm(H) W)` with `W` zero-initialised, so bit-identical to
the static version at init. The three projections share one `dyn_proj` tensor so the forward is a single matmul.

Three implementation details are load-bearing, all found by measurement:

- **`_reduce_streams` averages; the paper sums.** With the streams identical at init, a sum hands `ln_f` exactly `n`
  times the single-stream hidden (verified bit-exact). `RMSNorm` is scale-invariant in exact arithmetic, so that
  ought to vanish — but `+ self.eps` sits inside the `rsqrt` and is *not* scale-invariant, which measured as a 1e-3
  logit shift at `d_model: 32`. That would have surfaced in an `n=2` vs `n=4` A/B as an architecture effect when it
  was an epsilon artifact. Averaging keeps single-stream scale for every `n`, which is also why this repo does not
  need the paper's compensating "scale output-module init std by `sqrt(n)`" adjustment — applying it here would
  *break* the equal-to-a-residual-network property.
- **The per-iteration stagger advances by one stream, not by the loop body's sublayer count.** The paper initialises
  layer `k`'s read to `e_{k mod n}`; the obvious generalisation to a weight-shared loop is to advance by the body's
  true unrolled sublayer count, `2 * (n_layers - 1)`. That is always even, so at `n = 2` it is congruent to 0 mod `n`
  for *every* model (and at `n = 4` for every odd `n_layers`): every iteration reads the identical stream, the
  per-iteration routing silently collapses to one pattern, and nothing fails — the model just quietly gives up the
  reason to have per-iteration variants at all.
- **`optim.py` excludes them from Muon and does not decay the static ones.** `alpha_r` is `(n_variants, n, n)` and
  `dyn_proj` is `(n_variants, d_model, 2 + n)` — both >= 2-D, so without the `"hyper"` entry in
  `_MUON_EXCLUDED_SUBSTRINGS` they fall straight through to Muon, whose fixed-spectral-norm update is exactly wrong
  for a one-hot read and an identity mix. `_is_hyper_static` additionally routes `beta`/`alpha_m`/`alpha_r` to the
  no-decay group: decaying them drags the model off precisely the initialisation that makes it a residual network.
  The paper splits the same way.

**Cost.** Parameters are negligible (~0.06% of a `d_model: 256` model at `n=4`). Activation memory is `n` times the
residual stream, which `activation_bytes_per_token` models. Step time is the real price and is *not* free the way the
FLOP count suggests — the read/write are `O(n * d_model)` per token against a block's `O(12 * d_model^2)`, but at
these widths they are memory-bound, so what matters is traffic, not arithmetic. Compiled, fwd+bwd, `n_layers: 4`,
`loop_count: 6`, seq 512, bf16:

| | `d_model` 256 (batch 16) | 512 (batch 8) | 1024 (batch 4) |
|---|---|---|---|
| `n=1` | 35.9 ms | 38.9 ms | 56.7 ms |
| `n=2` | 46.7 ms (1.30x) | 51.0 ms (1.31x) | 69.0 ms (1.22x) |
| `n=4` | 50.7 ms (1.41x) | 53.4 ms (1.37x) | 71.8 ms (1.27x) |

The overhead falls with width, as `n*d` vs `d^2` predicts, but slowly. Unlooped it is much smaller — 1.10x/1.14x at
`d_model: 256` — because the loop body is what multiplies the number of residual writes. **Measure compiled, not
eager**: inductor fuses the write's elementwise epilogue and the dynamic branch's norm/tanh, and eager overstates the
cost by roughly 10 points. Three formulations were measured and the fast ones are in the code: a `matmul` read rather
than an einsum contracting dim -2 (3.6x), a `movedim` pair around the write's matmul (1.3x), one fused `dyn_proj`
matmul rather than three (3.9x), and an fp32 `rsqrt` reduction that never materialises an fp32 copy of the
`n`-times-wider hidden (1.3x).

Because the step is ~30-40% more expensive in the looped regime, a fixed-step A/B is *not* a fixed-compute comparison
here — the recorded A/B pins steps, so it already flatters hyper-connections. It does not help even on that generous
reading; see [results.md](results.md).

## Mixture of Experts

`cfg.model.use_moe` replaces `blocks[1:]`'s `FeedForward` with `MoEFeedForward` — `n_experts` parallel experts
(`BatchedExperts`) plus an `MoERouter` (same `RMSNorm + Linear` shape as `ACTRouter`, but softmax over `n_experts`
logits instead of a single sigmoid). Routing is Mixtral-style top-`k` (`moe_top_k`, default `2`): each token's router
probabilities are renormalized over just its top-`k` experts, and the FFN output is their weighted sum. See
`configs/tinystories_moe.yaml`.

`blocks[0]` always stays dense (it runs once per forward, not part of the loop body); within `blocks[1:]`,
`moe_dense_every` (opt-in) keeps every Nth block (1-indexed by position in the loop body) dense too. Because
`blocks[1:]` is weight-shared but fed an evolving hidden state each iteration, a token's expert choice naturally
changes across iterations — that falls out of ordinary input-dependent routing, with no iteration-aware logic in
`MoERouter`/`MoEFeedForward`. `MoEFeedForward.forward` is shape-agnostic over its leading dims, so it satisfies the
same `(*, d_model) -> (*, d_model)` contract `FeedForward` does and composes with ACT's own `_sparse_ffn_delta` FFN
sparsity (`cfg.model.act_ffn_capacity_ratio < 1.0`) with no special-casing in either mechanism — `_run_loop_body`
calls `block.ffn` generically regardless of which FFN type it is.

### Dispatch

Capacity-based, mirroring `_sparse_ffn_delta`'s gather/compute/scatter idiom (fixed
`capacity = round(moe_capacity_factor * n_tokens * moe_top_k / n_experts)` per expert, `torch.topk` priority
selection) but generalized to `n_experts` writers into one shared output buffer via `index_add` rather than
`index_copy` — with more than one expert able to write a nonzero value for the same token, `index_copy` would let a
later expert's zero capacity-padding silently clobber an earlier expert's real output.

Tokens beyond an expert's capacity are dropped (standard Switch-Transformer policy); which tokens get dropped is
decided by the router's own weight for that expert, so an over-capacity expert sheds the tokens it was least
confident about. Both this and `_sparse_ffn_delta`'s ACT selection are deterministic outside training mode —
`_sparse_ffn_delta` keeps a random tiebreak while `ffn.training` (an unbiased choice among still-running positions)
but falls back to sequence order in eval. Previously both drew `torch.rand` unconditionally, which made `val/loss`
and even greedy decoding vary run-to-run for identical weights and inputs, defeating the point of comparing two
configs' eval numbers.

**All `n_experts` are computed in one batched `baddbmm`**, not a Python loop: `BatchedExperts` stores each projection
as a single stacked `(n_experts, in, out)` tensor (the transpose of `nn.Linear`'s layout, so the forward is a plain
`x @ W`), and the gather produces one `(n_experts, capacity, d_model)` tensor. Total FLOPs are constant in
`n_experts` — `capacity` shrinks as experts are added — so step time stays roughly flat where the per-expert loop
grew linearly. Measured fwd+bwd on one MoE layer (16x512 tokens, bf16): 4 experts 3.16 -> 2.32 ms, 8 experts 4.30 ->
2.15 ms, 16 experts 8.34 -> 3.42 ms, 32 experts 24.0 -> 3.16 ms (**7.6x**). The forward is bit-identical to the old
loop; only gradient accumulation order differs (~1e-4). A load-state-dict pre-hook transposes and stacks the old
per-expert `experts.{e}.{gate,up,down}_proj.*` keys, so older MoE checkpoints still load.

`moe_eval_full_capacity` (default on) sizes per-expert capacity to the *actual* per-expert load outside training
mode, so no token is ever dropped at eval or generation. This fixes a real bug: capacity otherwise scales with how
many tokens share the forward pass, so the same prompt scored alone and scored inside a batch dropped different
tokens, and incremental decoding diverged from a full forward (`tests/test_kv_cache.py` covers this). Capacity limits
exist to bound *training* throughput and memory; at inference they only discard computation.

### Load balancing

A load-balancing auxiliary loss (`n_experts * sum(f_i * P_i)`, the standard Switch-Transformer formulation) keeps
routing from collapsing onto a few experts; it is `forward()`'s `moe_aux_loss`, weighted by `moe_aux_loss_weight`
(mirroring `ponder_weight`'s role for ACT). Note `_collect_moe_aux_loss` **sums** the per-layer aux losses, so a
balanced model reports `moe_top_k * n_moe_layers`, not `moe_top_k` — e.g. ~8.0 for `configs/tinystories_moe.yaml`'s 4
MoE layers at `top_k=2`, which is the *healthy* value, not evidence of collapse. Measured per-layer aux at init is
2.01-2.10 against an ideal of 2.0, and the capacity drop rate at a realistic batch (16x512 tokens) is under 1%.

`moe_balance` selects how load is balanced: `"aux_loss"`, `"bias"`, or `"both"` (default). The bias term is a
non-learned per-expert `expert_bias` buffer added to the routing logits **for top-k selection only** — the gating
*weights* still come from the unbiased softmax, which is what makes it "loss-free": it steers which experts a token
uses without distorting how much each contributes, so it never competes with the LM objective the way the aux loss
does. It is updated by an explicit sign-based rule in `DenseTransformer.update_expert_bias()`, called from `train.py`
**after** `optimizer.step()` rather than inside `forward` — it mutates a buffer with no gradient, so keeping it out
of the graph avoids a `torch.compile` break on every micro-batch. Under `"bias"`, `_collect_moe_aux_loss` returns an
exact zero rather than a weighted-down term, keeping it out of the graph entirely.

`moe_balance_signal` selects *what* the bias rule drives toward uniformity: `"count"` (default, DeepSeek-V3's rule —
one per token routed to the expert) or `"weight"` (that token's gate weight). They disagree about an expert selected
often but weakly, and `"weight"` is the better-motivated proxy for the failure the bias exists to prevent — the
gradient reaching an expert's weights scales with the gate weight it was applied at, not with how many tokens listed
it, and the `expert_bias` mechanism manufactures exactly that disagreement itself. Because the update is sign-based
the two are interchangeable with no change to `moe_bias_update_rate`. It stays at `"count"` because **the A/B says
`"weight"` is worse** ([results.md](results.md)).

### Shared and fine-grained experts

`moe_n_shared` adds an always-on expert (DeepSeekMoE) whose output is added to every token unconditionally, with no
routing and no capacity limit — it absorbs the computation every token needs so the routed experts can specialise
instead of each re-learning the common case. It counts *in full* in `num_active_parameters()`, unlike routed experts
which are discounted to `moe_top_k`.

`moe_expert_ffn_mult` sets each routed expert's width as a fraction of `ffn_dim`, for fine-grained MoE:
`n_experts: 32, moe_top_k: 8, moe_expert_ffn_mult: 0.25` activates the same parameter count per token as 8 experts at
`top_k` 2 but from 4x the combinations — affordable precisely because the batched dispatch keeps step time nearly
flat in expert count.

### Counterfactual routing (`moe_counterfactual_weight`, default `0.0` = off)

Backprop gives the router a perfect utility signal for the experts a token *was* sent to and none at all for the
experts it was not: expert `e` contributes `w[t,e] * out[t,e]`, so `dL/dw[t,e] = <g_t, out[t,e]>`, but `out[t,e]`
never enters the graph for an unchosen `e`. The counterfactual is normally written off as unaffordable — and here it
is already paid for. Capacity is a fixed *shape*, so an under-loaded expert computes `capacity` rows regardless and
`valid` masks the surplus away: at `moe_capacity_factor: 1.25` roughly 20% of all MoE FLOPs go on running experts
over tokens they did not win, and discarding the answer.

Two changes harvest it. The dispatch tiebreak ranks *unassigned* tokens by their router probability instead of
leaving them tied at zero (where `topk` broke the tie by index, so surplus rows went to whichever tokens sat earliest
in the batch); the rows now land on the tokens each expert ranked just below its top-k. That part is unconditional
and free — those rows are multiplied by `valid`, so the forward output is identical either way. Then
`_counterfactual_probe_signal` turns them into router gradient: with `m_t = routed_t / sum_e w[t,e]` the mixture the
token actually got, the advantage of a probe is `adv[t,e] = -<g_t, out[t,e] - m_t>`, and the router should raise
`p[t,e]` where that is positive.

`g_t` exists only in backward, so this cannot be written as a forward loss. Rather than a custom `autograd.Function`
(which would break the graph under `torch.compile`), it uses the identity that `p - p.detach()` is exactly `0.0` with
unit gradient: adding `weight * (p - sg p) * sg(out[t,e] - m_t)` to the output leaves the forward **bit-identical**
while making `dL/dp[t,e] = -weight * adv[t,e]`. Three properties fall out free. It is scale-consistent with every
other gradient including under fp16's `GradScaler`, because it *is* an ordinary gradient rather than a
separately-scaled loss. `weight` has a natural unit — at `1.0` a probe's push is exactly as strong per unit of
utility as the true gradient on a chosen expert's gate weight, which measured as 22% of the router's own gradient
norm at `configs/tinystories.yaml`'s shape. And it is training-only, since a term that exists only to carry a
gradient is pure overhead where there is no backward.

**Verify the compiled behaviour, not just the eager behaviour, if this is ever changed**: a term that is identically
zero in the forward is exactly what an optimising compiler is entitled to delete, and if inductor folded it away it
would take the gradient with it and the feature would silently do nothing in every real run while passing every eager
test. Measured: the gradient survives (all 12 router tensors move), step time costs 0.4% and memory 0.01 GB. The
compiled forward differs from the feature-off model by 4.8e-07 — not the term being nonzero (eager is bit-identical)
but inductor scheduling a graph with extra nodes differently, which is *smaller* than the 8.3e-07 eager-vs-compiled
gap every compiled run already carries.

It defaults off because the A/B does not support turning it on — and the *diagnostic* matters more than the A/B; see
[results.md](results.md).

## Sequence log-probabilities

`sequence_logprob_sum(logits, input_ids, loss_mask)` gives a per-row `(batch,)` **sum** of log-probability at
`loss_mask==1` positions, using the same single-`logsumexp`-pass trick `train._nll_and_logz` uses but keeping the
batch dimension. It lives here, not in `train.py`, because of import direction: `data.py`'s DPO reference-logprob
precompute needs it and must not import `train.py`. `load_transformer_from_checkpoint(path, device, eos_id=None)`
moved here for the identical reason — `generate.load_checkpoint` is now a thin wrapper around it. See
[post-training.md](post-training.md).
