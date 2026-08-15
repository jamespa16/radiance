# `optim.py` — optimizers and parameter groups

Split out of `train.py` (which was carrying ~160 lines unrelated to the training loop). `build_param_groups` lives
here rather than in `train.py` because `migrate_optimizer_to_cpu_offload` needs it and importing it back would make
the two modules circular.

## Muon

`cfg.train.optimizer` selects `"muon"` (default) or `"adamw"`. **Muon** replaces each hidden weight matrix's momentum
update with its orthogonal polar factor, computed by a 5-step Newton–Schulz quintic iteration (`orthogonalize`) —
every singular value is driven toward 1, so the step is spread evenly across directions instead of being dominated by
the largest few, which is what lets it take a much larger learning rate. `orthogonalize` is written with batched
`@`/`.mT`, so `BatchedExperts`' stacked `(n_experts, in, out)` weights are orthogonalised per expert with no
reshaping — dim 0 is a free batch dimension. The iteration deliberately does *not* converge: the tuned coefficients
settle into a band around [0.68, 1.13], which is the intended operating point (it needs the spread collapsed, not the
values exact).

`MuonWithAuxAdam` holds both algorithms in **one** Optimizer object with a per-group `algorithm` key, because
everything downstream assumes a single optimizer — the `GradScaler`'s unscale_/step bookkeeping is keyed on it,
`LambdaLR` drives its `param_groups`, `save_checkpoint` serialises its `state_dict`, and the OOM handler swaps it.

`build_muon_param_groups` routes 2-D+ hidden weights to Muon and leaves the rest to AdamW: the tied
`token_emb`/`lm_head` (per-token rows rather than a hidden linear map, and the largest tensor in the model, so
Newton–Schulz on it would dominate step time), and the routers/gates (tiny tensors whose exact scale is load-bearing,
where Muon's fixed-spectral-norm update is the wrong behaviour). Muon's group runs at `cfg.train.muon_lr` (default
`0.02`, ~50x AdamW's) while the AdamW groups keep `cfg.train.lr` — a separate field precisely so every existing
config's tuned `lr` still means what it did.

### `_step_muon` batches by parameter shape

The arithmetic is per-parameter but issuing it per-parameter is what costs: Newton-Schulz is `ns_steps` iterations of
three matmuls, so a 42-tensor group is 630 matmuls, none large enough to saturate the GPU. A transformer has very few
distinct weight shapes (one per projection role, repeated per layer), so stacking same-shaped updates into a leading
batch dimension collapses that to 630 / (tensors per shape) launches. Momentum and weight decay use
`torch._foreach_*` for the same reason, as does `_step_adamw`'s update. Measured on `configs/tinystories.yaml`:
optimizer step **8.2 -> 4.4 ms**, of which the Muon groups are 6.7 -> 3.8 and the AdamW groups 1.4 -> 0.6.

This is a launch-overhead fix, so it pays where the tensors are small. It is numerically neutral, verified rather
than assumed: against an fp64 reference, batched and per-tensor orthogonalisation have **identical** error (~2.7e-2
either way — the bf16 iteration's own error, far larger than the difference between them) and are bit-identical for
most shapes. `tests/test_optim.py::test_orthogonalize_batches_over_leading_dims` and
`test_muon_batched_step_matches_per_parameter_step` pin that.

### The batching cap counts *matrices*, not tensors

`_MUON_MAX_STACK = 32`. Newton-Schulz's transient buffers (`A = X @ X.mT`, `B = b*A + c*(A@A)`, `B @ X`) scale with
the *stacked batch size*, not with the parameter count being orthogonalised, so an unbounded stack turns a throughput
win into an OOM on a model with many same-shaped matrices. That is exactly MoE: shape is keyed on `(in, out)`, so
every expert's projection in every MoE layer shares one shape. Before `muon_orthogonalize_reserve_bytes` (below),
`auto_batch_size` had no model of this cost whatsoever — it sized params/grads/optimizer state and activations, and
this is none of those — so it picked a batch that looked safe and the run died on the first optimizer step. Neither
OOM tier saves it: tier one shrinks the micro-batch, which this cost is independent of, and tier two offloads only
AdamW-owned tensors while the pressure is entirely Muon-side.

**The first version of this cap counted tensors, and it silently never fired.** `BatchedExperts` already stores each
projection role as *one* tensor shaped `(n_experts, in, out)`, so a 4-MoE-layer model has ~16 tensors of a given
shape no matter how many experts it has — `len(idxs) <= 32` is always true, nothing ever chunks, and stacking those
16 hands `orthogonalize` an effective batch of `16 * 48 = 768` matrices. A 48-expert model OOM'd on the first
optimizer step exactly as it had before the "fix". The correction operates one level down: unbind every Muon-owned
tensor into its individual `(in, out)` matrices first (a 2-D weight unbinds to itself; an `(n_experts, in, out)`
tensor unbinds to `n_experts` of them), group and chunk over *that* flat list, and scatter the updates back through
`(param index, expert index)`. Dense models are untouched — every tensor is already a single matrix, and the deepest
shipped config (`configs/fineweb_500m.yaml`, 22 layers) is under the cap.
`test_muon_step_chunks_within_batched_expert_tensors` pins the distinction with a single `(12, 10, 6)` tensor against
a patched cap of 4: one tensor, twelve matrices, so a tensor-count cap cannot see it.

**What it buys, and what it doesn't.** For a MoE-*dominant* model the fix is large: no router, no hyper-connections,
`d_model: 1024`, 4 MoE layers at `moe_expert_ffn_mult: 0.25` on a 32 GB card, `n_experts: 64` (1329M total, 322M
active) now fits at 27.1 GB where 32 experts had been the practical ceiling. For the three-feature config it did
essentially nothing: chunking bounds the *transient* buffer only, while the persistent params + grads + momentum
still scale linearly with `n_experts`, and in that config activation memory from `max_loops` and `hyper_conn_streams`
was the binding constraint anyway. `configs/fineweb_moe_hyper_router.yaml` therefore still ships `n_experts: 16`; its
header carries the full per-feature measurement table. **A fix aimed at one architecture's bottleneck can be a no-op
for another whose bottleneck is elsewhere — re-measure the config you actually intend to run.**

### `estimate_batch_size` now reserves for the transient orthogonalize() spike

`optim.muon_orthogonalize_reserve_bytes` closes the gap the previous section describes:
`estimate_batch_size` (`train.py`) calls it and subtracts the result from usable VRAM alongside the existing
grad/momentum reservation, when `train.optimizer == "muon"`. It re-derives the same per-shape matrix count
`_step_muon` computes (unbind every Muon-owned tensor into individual `(in, out)` matrices, group by shape), takes
the shape with the most matrices capped at `_MUON_MAX_STACK`, and multiplies by a measured
`_MUON_NS_BYTES_PER_ELEM = 12` — the peak `max_memory_allocated` delta of one isolated `orthogonalize()` call, per
fp32 grad element in the stacked input, measured directly (not derived from the `A`/`B`/`X` shapes analytically)
the same way `activation_bytes_per_token`'s diff-attention and logits terms are. Square `(in, out)` shapes measured
worst (12 bytes/elem, flat across batch 8-32 and shapes up to 1024x1024); rectangular shapes measured lower (9), so
12 is used uniformly to stay conservative rather than shape-specific.

**What this does and doesn't fix.** It only bounds the term this section is about — the transient batched-matmul
buffer inside one `orthogonalize()` call — which is exactly what `_MUON_MAX_STACK` chunking already bounds
per-step; the reserve just makes `estimate_batch_size` *see* that bound before committing to a batch size, instead
of the run discovering it empirically on the first optimizer step. It does not model the *persistent* params + grad
+ momentum footprint, which already has its own (optimizer-agnostic) term in `estimate_batch_size` and genuinely
does scale with `n_experts`; nor does it model activation memory from `max_loops`/`hyper_conn_streams`, which the
combined-config measurement above found to dominate in practice. `n_experts: 16` in
`configs/fineweb_moe_hyper_router.yaml` is still the right call for that config for the reasons given above — this
reserve changes what `auto_batch_size` knows, not what the binding constraint is for every config.

### Newton-Schulz cost makes effective batch a throughput knob

It is per optimizer step and independent of batch size. For an `(m, n)` weight it is ~`30 * min(m,n) * numel` FLOPs,
so per parameter it costs ~`30 * d_model` against fwd+bwd's ~`6 * tokens_per_step` — Muon dominates whenever
`tokens_per_step < 5 * d_model`. Not hypothetical: `configs/fineweb_500m.yaml` ships `batch_size: 4,
grad_accum_steps: 1` at `seq_len: 1024`, i.e. **4096 tokens per optimizer step against `d_model: 1280`**, and
measures 50% of step time in the optimizer.

| `micro x accum` | tokens/step | step | tok/s | optimizer share |
|---|---|---|---|---|
| 4 x 1 (as shipped) | 4,096 | 339 ms | 12,100 | 50% |
| 4 x 8 | 32,768 | 1459 ms | **22,500** | 12% |

**1.85x throughput for +0.4 GB**, and 4096 tokens/step was far too small an effective batch for a 500M model anyway.
Set `train.target_effective_batch_size` (the config carries it commented out) rather than raising `batch_size`, so
`auto_batch_size` keeps choosing the micro-batch. Left as-is deliberately: effective batch size reshapes
`tokens_per_param`'s derived `max_steps` and interacts with the tuned `lr`, so it is a run-shaping decision, not a
free win to apply silently.

## Parameter classification

**Both param-group builders classify norm gains by their owning module (`norm_gain_param_ids`), not by `param.dim() <
2`.** The rank rule was correct only while a norm gain was always `(d_model,)`. `loop_iter_conditioning:
"norm_gains"` — the *default* — gives `RMSNorm` a `(n_variants, d_model)` gain bank, one row per loop iteration,
which is still a per-channel scale but is 2-D. So the moment a config set `loop_count > 1`, every `ln1`/`ln2` gain
silently stopped being an AdamW no-decay tensor: on the Muon path it became a **Muon** tensor, where Newton-Schulz
drives its singular values to 1 (a norm gain *is* its scale, so that erases the conditioning rather than refining
it), and on the AdamW path it started being decayed toward zero from its 1.0 init. `configs/fineweb_500m.yaml` had 44
gain banks in the Muon group. Nothing raises, nothing shows up as a dead parameter, and `loop_count: 1` configs are
unaffected — which is why `configs/tinystories.yaml` never saw it and every looped config did.

**Measured, and it is a correctness fix rather than a quality win:** on a looped A/B (`d_model: 256`, `n_layers: 4`,
`loop_count: 6`, batch 16 pinned, `lr: 1.0e-2`, 1000 steps) `val/loss` at step 1000 was **1.9581 gains-to-Muon vs
1.9590 fixed** — indistinguishable, and the fixed arm was ahead at every earlier eval point (2.7821/2.3694/2.1681/
2.0284 vs 2.7861/2.3749/2.1699/2.0304). It is kept because it makes a tensor's optimizer independent of how many loop
variants it happens to hold, which is plainly what both builders' docstrings already intended. Worth re-testing at
depth before assuming it stays neutral.

When a new parameter class needs different treatment, classify it by **what owns it**, not by its rank. Adding a
leading dimension to something that was 1-D moves it across the `dim() < 2` boundary, which silently changes both its
optimizer and whether it is weight-decayed. `RMSNorm`'s per-iteration gains and `HyperConnection`'s coefficients have
both hit this.

## `train.lr`: the largest quality bug measured in this repo

The classification decision above left `lr` badly mistuned. Keeping `lr` fixed across the Muon switch preserved its
*value* but silently changed its *job*: it was tuned when AdamW trained every tensor, and afterwards it governed only
what Muon doesn't own — overwhelmingly the tied `token_emb`/`lm_head` matrix. At `3e-4` that embedding became the
bottleneck of the whole model. Sweeping it on `configs/tinystories.yaml` (400 steps, batch pinned at 32,
`auto_batch_size: false`) gives a clean bowl with its minimum **30-100x above the old default**:

| `train.lr` | 3e-4 (old default) | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|---|
| `val/loss` @400 | 2.9111 | 2.4460 | 2.1800 | 2.1117 | **2.1051** | 2.1520 |

Run-to-run noise on this setup is ~0.002 (a re-run of the baseline gave 2.9091), so the ~0.8 spread is not close to a
judgement call. See [results.md](results.md) for the 1200-step confirmation.

`cfg.train.embed_lr` exists because of *where* that win comes from: holding everything else at `3e-4` and raising
only the embedding recovers 0.754 of the 0.797, while raising only the norms/gates/routers recovers 0.416 — they
overlap rather than add, but the embedding is plainly the dominant term. It defaults to `None`, resolving to `lr`, so
the embedding gets its own parameter group at exactly the LR it had when it shared AdamW's decayed group. Set it to
tune the embedding independently of the routers and gates, whose exact scale is load-bearing in a way the embedding's
is not — that is the whole reason it is a separate field rather than advice to raise `lr`.

`cfg.train.hyper_conn_lr` exists for the same reason and defaults to `1e-3` rather than `None`, because the usual
range-collapsed default would have shipped hyper-connections in their broken configuration — sharing `lr` costs 0.39
`val/loss`. See [results.md](results.md).

## muP

`cfg.model.mup_base_d_model` makes hyperparameters transfer across width. It defaults to `None`, resolving to
`d_model`, so `mup_width_mult` is exactly `1.0` and every correction is an identity until a base is set. When it
isn't 1: hidden-weight init std scales by `1/sqrt(m)`, and the output logit multiplier by `1/m`.

Two subtleties in the init. The tied embedding is initialised **last**, after `self.apply(_init_weights)`, because
`apply` visits `token_emb` (an `nn.Embedding`) before `lm_head` (an `nn.Linear`) and they are the same tensor — so
the hidden-weight branch would otherwise overwrite the embedding's std. And the attention logit scale stays
`1/sqrt(head_dim)` rather than becoming `1/head_dim`: muP's `1/d` scale applies when the *head* dimension grows with
width, but here `head_dim` is a fixed config constant and width grows by adding heads, so each q·k dot product sums a
fixed number of terms and its variance doesn't grow with `d_model`.

The third correction is the **per-tensor LR**, `1/m` on hidden weights and `Θ(1)` on everything else, and it lives in
`build_param_groups` — i.e. only on the `optimizer: adamw` path. `build_muon_param_groups` carries a per-group
`mup_lr_scale` that is `1.0` everywhere, deliberately rather than by omission: Muon's update is spectrally normalised
and so already approximately width-invariant, and the tensors its auxiliary AdamW owns are exactly the ones muP
leaves at `Θ(1)` anyway (the tied embedding, whose fan-in is a row lookup and so doesn't widen; 1-D gains and biases;
and the routers/gates, which are hidden-like and would strictly want `1/m` but are a rounding error in both parameter
count and gradient norm).

One trap when the correction is live: with `embed_lr` unset — the default — the tied embedding sits in the *decayed*
group and would ride its `1/m` scaling. `build_param_groups` therefore splits it into its own unscaled group as soon
as `mup_width_mult != 1.0`, and both call sites (`build_optimizer` and `_adamw_to_cpu_offload`) must keep passing
identical arguments so that extra group appears in both or neither — the tier-2 OOM migration zips the two
optimizers' groups positionally.

### Validated by coordinate check, not by assumption

The claim is that activation scale is `Θ(1)` in width at *every* training step, which is the property that makes a
tuned LR transfer. Measured on the real model through the real `build_optimizer`, identical data/seeds, widths
128→2048 (16x) at fixed `head_dim`, 8 steps, fp32 — ratio of mean |activation| at `d=2048` vs `d=128`, where `1.00`
is a pass:

| arm | `attn_out` | `ffn_out` | `logits` |
|---|---|---|---|
| muon + muP (the default path) | 0.85 | 1.07 | 0.94 |
| muon + muP, `loop_count: 4` | 0.95 | 0.92 | 0.99 |
| muon, muP off | 1.81 | 6.45 | 3.48 |
| muon, muP off, `loop_count: 4` | 2.53 | 4.28 | 7.69 |
| adamw + muP | 1.03 | 1.00 | 0.49 |
| adamw, muP off | 15.35 | 68.36 | 4.54 |

The default path is flat to within ±8%, looped and unlooped, and muP is doing real work rather than being vacuously
flat — turning it off moves logits 3.5x unlooped and 7.7x looped, and looping makes the unparameterised model
*worse*, as multiplying residual writes into one accumulator should.

The `adamw` row is post-fix. Before it, the missing `1/m` grew `attn_out` **16.7x across that 16x sweep — linearly in
width** — and the reason it survived so long is the part worth remembering: **`val/loss` cannot see this.** `ln_f` is
`RMSNorm` and therefore scale-invariant, so it launders the blown-up residual stream away immediately before the LM
head; the logits ratio stayed at 1.04 while the network's interior was entirely width-dependent. A coordinate check
catches it and a loss curve never will.

Two caveats. `adamw + muP`'s logits still drift *down* ~2x over the 16x sweep (`m^-0.26` rather than flat) — far
milder than the 4.5x growth without muP, and unchanged by the LR fix, so it's a separate readout-side effect on the
tied-embedding path; the Muon path doesn't show it. And a coordinate check is necessary, not sufficient: the actual
claim is that the *optimal LR* transfers, which needs an LR sweep at two widths showing the minimum land in the same
place. That hasn't been run.

`tests/test_optim.py::test_mup_keeps_adamw_activations_flat` is the regression test, a CPU-sized coordinate check.
Its 16x width span and 20 steps were picked by measurement and are load-bearing: the drift accumulates with training,
so a smaller version isn't a weaker test but a **vacuous** one — at 4x width over 6 steps the pre-fix code scored 1.12
and passed cleanly. At 16x over 20 steps it scores 9.71 against a fixed build's 0.99. The `mup_base_d_model=None`
control matters as much as the assertion, for the same reason.

## CPU offload

Tier-2 OOM offload dispatches on optimizer type. For plain AdamW it swaps in `CPUOffloadAdamW` (a new object, so the
caller rebuilds the scheduler); for `MuonWithAuxAdam` it flips that optimizer's AdamW groups to CPU-resident moments
**in place** and returns the same object, so no scheduler rebuild is needed — callers test identity to decide. Only
the AdamW groups are offloaded: Muon's state is a single momentum buffer (half AdamW's footprint) and its step is a
matmul chain per parameter, so running it against CPU memory would cost far more than the VRAM it returns. In a Muon
run the AdamW groups hold the embedding matrix anyway, typically the single largest tensor.
