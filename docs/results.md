# Measured results

Every A/B run in this repo, and the methodology cautions that came out of them. Read
[the cautions](#cautions-when-running-an-ab) before running a new one.

## The core three

A/B on `configs/tinystories.yaml` (400 steps, RTX 5090, `val/loss` at step 400), each changing one thing:

| change | baseline | with change |
|---|---|---|
| `optimizer: muon` vs `adamw` | 3.300 | **2.826** |
| `doc_attention_mask` on vs off | 2.808 | **2.774** |
| loop conditioning + input injection vs plain looping (`loop_count: 3`) | 2.900 | **2.861** |

Muon is the large one *of those three*: it passed AdamW's *final* loss at roughly half the steps (2.827 at step 200
vs AdamW's 3.300 at step 400). The other two are consistent rather than dramatic — better at every eval point, not
just the last.

## Retuning `train.lr` dwarfs all of them

It is a fix rather than a feature — see [optim.md](optim.md) for why `3e-4` stopped being the right value the moment
Muon took over the hidden weights, and for the LR sweep. Confirmed at a longer horizon (1200 steps, same config and
pinning, so **not** comparable to the 400-step numbers above):

| 1200 steps | step 200 | 400 | 600 | 800 | 1000 | 1200 |
|---|---|---|---|---|---|---|
| `lr: 3.0e-4` (old default) | 3.9008 | 2.8956 | 2.5929 | 2.4093 | 2.2926 | 2.2341 |
| `lr: 3.0e-4`, `embed_lr: 1.0e-2` | 2.7470 | 2.3321 | 2.1292 | 1.9872 | 1.8830 | **1.8290** |
| `lr: 1.0e-2` | 2.6801 | 2.2719 | 2.0797 | 1.9482 | 1.8500 | **1.7984** |

Better at every eval point by a wide margin in both retuned arms — and the retuned model reaches the old default's
*final* 1200-step loss in roughly 500 steps, so this is worth more than tripling the training budget. Raising `lr`
wholesale edges out raising `embed_lr` alone, consistent with the 400-step decomposition: the embedding is ~93% of
the win and the remaining norms/gates/routers supply the rest.

Two caveats. Measured at one scale only (`d_model: 256`, TinyStories) — muP is the intended mechanism for carrying a
tuned `lr` across width, and nothing here establishes that `1.0e-2` is right for `configs/fineweb_500m.yaml`. And the
400-step baseline here (2.9111) does not match the 2.826 above, almost certainly because these runs pin
`batch_size: 32` with `auto_batch_size: false`; the deltas are sound because every arm shares those settings, but the
absolute numbers belong to their own series.

## The loop is dominated — depth and MoE are the frontier

The most consequential result on this page, because it is about the axis the rest of the framework is built to
serve: loop conditioning, input injection, BPTT windowing, ACT, ACT capacity sparsity and hyper-connections all
exist to make `blocks[1:]`'s weight-shared loop pay. **It does not pay.** Nothing had measured the loop against a
plain dense stack before this — the one loop-related A/B on record compared looping-with-conditioning against
looping-without, which presupposes the loop.

Five arms on TinyStories (`d_model: 256`, `head_dim: 64`, `ffn_mult: 4.0`, `ffn_depth: 3`, `seq_len: 512`), **each
arm LR-swept separately**, 3000 steps, effective batch 32 pinned. Full harness, logs and working notes in
[experiments/loop-vs-depth/](../experiments/loop-vs-depth/).

| arm | total | active | executed blocks | `val/loss` @3000 |
|---|---|---|---|---|
| A dense (`n_layers: 6`, `loop_count: 1`) | 31.8M | 31.8M | 6 | 1.6240 |
| B looped (`n_layers: 6`, `loop_count: 4`) | 31.9M | 31.9M | 21 | 1.5893 |
| **C deep (`n_layers: 21`, `loop_count: 1`)** | 79.1M | 79.1M | 21 | **1.5394** |
| D MoE + loop (`n_layers: 6`, `loop_count: 4`) | 73.5M | 31.5M | 21 | 1.5507 |
| E MoE dense (`n_layers: 6`, `loop_count: 1`) | 54.2M | 31.5M | 6 | 1.5752 |

B vs A is equal-parameter; B vs C is equal-FLOP (both 21 executed blocks). D and E are active-parameter-matched to
B (31.5M vs 31.9M, a 1.3% shortfall running *against* the MoE arms), with `moe_n_shared: 0` so no unconditional
FLOPs are added. Noise floor **0.008**, from running arm B three independent times — not the 0.002 quoted
elsewhere on this page, which belongs to a different configuration.

**The loop is dominated in both settings once cost is counted.** E beats B (1.5752 vs 1.5893) on **6 executed
blocks instead of 21** — a third of the compute — and in less wall time. C beats D (1.5394 vs 1.5507), also in less
wall time. The Pareto frontier is C (best quality) and E (cheapest); B and D are both inside it. The 2x2:

| | dense | looped | loop gain |
|---|---|---|---|
| no MoE | 1.6240 | 1.5893 | +0.0347 |
| MoE | 1.5752 | 1.5507 | +0.0245 |
| **MoE gain** | **+0.0488** | **+0.0386** | |

Both effects are real (3-6x noise) and **sub-additive** — combined A→D is +0.073 where the two individual gains sum
to +0.084. They supply overlapping capacity rather than complementary mechanisms. **MoE's +0.0488 at matched active
parameters is the largest architecture effect recorded on this page**, which is what makes `use_moe` the first
thing to reach for and the loop the first thing to drop.

Three methodological findings came out of this, and they matter more than the verdict:

1. **A 400-step verdict can invert.** At 400 steps the loop *lost* to dense by 0.013; by step 1000 it *led* by
   0.051; by 3000 the lead had decayed to 0.035 and was still shrinking, while C's lead over B grew from 0.032 to
   0.050. Depth pulls away from recursion the longer you train. **Most A/Bs on this page are 400-step runs** — that
   is not a hypothetical concern about them.
2. **Never compare two runs with different `max_steps` at the same step number.** `warmup_steps = round(max_steps *
   warmup_ratio)` and the cosine decays over `max_steps`, so a 1000-step run is fully annealed at step 1000 while a
   3000-step run is mid-schedule. Measured directly: arm D scored 1.7667 at step 1000 of a 1000-step run and 1.9401
   at step 1000 of a 3000-step run — **same config, same seed, a 0.17 gap that is purely schedule**. An earlier
   version of these notes reported MoE gains computed across exactly that mismatch; they were wrong by ~5x.
3. **The per-arm LR sweep was load-bearing, not diligence.** At `lr: 3e-3` arm B *leads* arm A by 0.010; at `1e-2`
   it trails. The arms' LR curves cross, so a single shared LR would have supported either conclusion.

Caveats: TinyStories at `d_model: 256`, one seed, and the per-arm LRs were tuned at 400 steps then applied at 3000
(longer runs often prefer slightly lower LRs). Arm B uses the repo's default loop configuration; a different
conditioning scheme is a separate question this does not address.

## ACT sparsity — neutral on quality, positive on throughput

A throughput change rather than a quality one, so it is measured separately. On `configs/tinystories_router.yaml`
(400 steps, pinned `batch_size: 16`), `val/loss` at step 400 was **3.0735** dense, **3.0757** at
`act_capacity_ratio: 0.5` and **3.0728** at `0.25` — indistinguishable. See [model.md](model.md) for what it buys in
time and memory, and why a short run's wall-clock understates it.

## NSA — negative, and since removed

A/B'd against the *real* default it would replace: `use_nsa` required `doc_attention_mask: false`, so the fair
comparison isn't NSA vs. a doc-masking-disabled baseline but NSA vs. what a config actually runs today. 400 steps,
RTX 5090, `d_model: 256`/`head_dim: 64`/`n_layers: 6`, pinned `batch_size: 32`. `val/loss` at step 400:

| arm | @400 |
|---|---|
| dense + `doc_attention_mask: true` (today's default) | **2.8540** |
| dense + `doc_attention_mask: false` (isolates the cost of losing doc masking) | 2.8823 |
| NSA | 2.9110 |

NSA is worse than both — worse than today's default by 0.057, and still worse than the doc-masking-disabled baseline
it is actually forced to compete against by 0.029. So it was never defaulted on: the A/B doesn't support it, and
enabling it would have forced `doc_attention_mask` off too, silently giving up an already-proven win for one that
isn't yet real.

Worth re-running at a longer `max_seq_len` before drawing a final conclusion — at 512 tokens and `nsa_block_size:
128` there are only 4 blocks, little room for the selection branch's sparsity to pay for its own approximation error
— and this A/B measured quality only, not the compute saving that is NSA's other selling point at genuinely long
context. The feature has since been removed; see [model.md](model.md).

## Differential Attention — the first architecture win

The first attention-mechanism change here that measured as a genuine win rather than a neutral-or-negative result to
record and move past. Two A/Bs on TinyStories, `d_model: 256`/`head_dim: 64`/`n_layers: 6`/`loop_count: 1` (dense, no
loop confound), pinned `batch_size: 32`, `auto_batch_size: false`, `lr: 1.0e-2`, `dropout: 0.0`, **3000 steps** —
deliberately not 400, since an arm can rank differently at step 400 than at step 1500+ once the LR schedule is far
enough into its decay. With a *second* baseline run at the same config first, to establish this setup's own noise
floor rather than borrow one measured elsewhere:

| arm | step 200 | 400 | 600 | 1000 | 1400 | 1800 | 2200 | 2600 | 3000 |
|---|---|---|---|---|---|---|---|---|---|
| dense baseline (run 1) | 2.7611 | 2.3744 | 2.2227 | 1.9959 | 1.8782 | 1.7884 | 1.7164 | 1.6574 | 1.6283 |
| dense baseline (run 2, re-run) | 2.7627 | 2.3824 | 2.2254 | 1.9944 | 1.8754 | 1.7866 | 1.7161 | 1.6568 | **1.6278** |
| dense + `use_diff_attn` | 2.8295 | 2.3830 | 2.2222 | 1.9848 | 1.8677 | 1.7798 | 1.7046 | 1.6492 | **1.6203** |
| MoE baseline (`n_experts: 8`, `top_k: 2`) | 2.6701 | 2.3178 | 2.1770 | 1.9495 | 1.8300 | 1.7385 | 1.6570 | 1.5976 | 1.5662 |
| MoE + `use_diff_attn` | 2.7462 | 2.3111 | 2.1676 | 1.9405 | 1.8192 | 1.7271 | 1.6473 | 1.5890 | **1.5572** |

The two baseline runs establish a noise floor around **0.0005-0.003** at these later steps. Differential attention
starts out clearly *worse* — step 200 is 0.067 behind, a real early-training disadvantage from the two branches
needing time to specialize — but from roughly step 600 on it is ahead at every remaining eval point by
**0.008-0.012**, 5-20x the noise floor, both in isolation and layered onto MoE. The two effects add rather than one
washing out the other.

Not free: compiled fwd+bwd step time on this shape measured **1.34x** (50.0 -> 66.8 ms), and
`activation_bytes_per_token` bills it at `17 * d_model` vs `10 * d_model`. So it is not a candidate for defaulting on
even though the quality result is positive — same category as `use_moe`.

**A fixed-*compute* verdict has not been checked**, stated plainly rather than left implicit. Every number above is
fixed-*step*, and the hyper-connections result below warns that a 1.3-1.4x step-time cost can plausibly erase a
fixed-step win at equal wall-clock; a rough extrapolation of the dense baseline's late-run improvement rate suggests
it is genuinely unclear which way this goes. Re-run at equal compute (give the baseline ~1.34x the steps) before
treating this as settled, and re-verify at `fineweb_500m.yaml` width.

## Hyper-connections — neutral-to-negative, and the LR is the real finding

A/B on the looped shape they are aimed at — `d_model: 256`, `n_layers: 4`, `loop_count: 6` (19 executed blocks),
pinned `batch_size: 16`, `auto_batch_size: false`, `lr: 1.0e-2`, 400 steps:

| `hyper_conn_streams` | `hyper_conn_lr` | step 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| 1 (baseline) | — | 3.3285 | 2.7206 | 2.4225 | **2.2821** |
| 2 | `1.0e-2` (= `lr`) | 4.1753 | 3.1816 | 2.8383 | **2.6744** |
| 2 | `1.0e-3` | 3.2490 | 2.7054 | 2.4195 | **2.2858** |
| 2 | `1.0e-4` | 3.2652 | 2.7172 | 2.4280 | **2.2881** |
| 4 | `1.0e-3` | 3.3389 | 2.7812 | 2.4928 | **2.3570** |
| 4 | `1.0e-4` | 3.2617 | 2.7293 | 2.4400 | **2.2964** |

Read the first two rows before anything else: sharing `lr` costs **0.39 val/loss**, far and away the largest effect
in the table and larger than any *feature* win recorded on this page. That is not hyper-connections being bad, it is
the `embed_lr` lesson repeating — `lr` reaches a grab-bag of tensors whose ideal step sizes differ by orders of
magnitude, and AdamW's update is ~`lr` per step almost regardless of gradient scale, so 400 steps at `1e-2` moves a
coefficient by O(1). For a *structural* coefficient (one-hot read, identity mix) an O(1) move erases the routing
rather than refining it. Hence `train.hyper_conn_lr`, and hence it defaults to `1e-3` rather than `None` — the usual
range-collapsed default would have shipped the feature in its broken configuration. Note also that the best LR falls
as `n` rises (at `n=4`, `1e-4` beats `1e-3` by 0.06): more streams means more coefficients, so more total drift at a
given step size.

With that fixed, the honest verdict is **neutral-to-slightly-negative**: the best arm (`n=2`, `1e-3`) lands 0.004
*behind* the single-stream baseline, at the edge of this setup's ~0.002 noise floor, and `n=4` is clearly behind.
Since hyper-connections also cost 30-40% of step time in this regime, a fixed-*compute* comparison is worse still. So
`hyper_conn_streams` stays at `1`, on the same terms `use_nsa` was removed: implemented, tested, documented, and not
defaulted on because the measurement doesn't support it.

Worth re-testing before treating this as settled, because the conditions the mechanism targets were only partially
met: the paper's argument is about depth, and 19 executed blocks at `d_model: 256` over 400 steps is a small instance
of it; the `n*d` vs `d^2` scaling also means the step-time penalty shrinks with width, so a wider, deeper, longer run
is where the trade could plausibly flip.

## Counterfactual routing and `moe_balance_signal` — negative, and the diagnostic is worth more than the A/B

Arms off `configs/tinystories.yaml`'s shape with MoE (`n_layers: 6`, `loop_count: 1`, `n_experts: 8`,
`moe_top_k: 2`, batch 32 pinned, `auto_batch_size: false`, `lr: 1.0e-2`, `dropout: 0.0`), `val/loss` at 1500 steps:

| arm | @1500 |
|---|---|
| baseline | 1.6810 |
| baseline, re-run (the noise floor) | 1.6823 |
| `moe_counterfactual_weight: 1.0` | **1.6804** |
| `moe_counterfactual_weight: 4.0` | 1.6947 |
| `moe_balance_signal: weight` | 1.7003 |
| both | 1.7107 |

Run-to-run noise here is 0.0013, so counterfactual routing at its natural scale is exactly neutral and everything
else is a real regression. Three follow-ups closed off the obvious escapes. It is **not a horizon effect**: at 6000
steps (4x) it is 1.4689 against the baseline's 1.4696, inside noise at all twelve eval points. It is **not a lack of
routing decisions to get right**: fine-grained MoE (`n_experts: 32`, `moe_top_k: 8`, `moe_expert_ffn_mult: 0.25`,
same active parameters) gives 1.7018 against 1.7026. And it is **not too weak to matter** — the deposited gradient
measures 22% of the router's own gradient norm at `weight: 1.0`, and raising it to 4.0 makes things monotonically
worse, which is what injecting *noise* looks like rather than what an ineffective term looks like.

What it actually is, from the diagnostic that should have been run first: compute the per-(taken expert, alternative
expert) mean advantage on two **disjoint** batches at identical weights, and correlate them. Real structure ("tokens
going to expert i would do better at expert j") reproduces across data; batch noise does not.

| step | 0 | 100 | 400 | 1000 | 2000 |
|---|---|---|---|---|---|
| corr(batch A, batch B) | 0.46 | 0.52 | -0.20 | 0.20 | 0.05 |
| best-alternative regret / realised utility | 2.39 | 1.97 | 1.58 | 1.72 | 1.86 |

In-sample regret stays large — the best unchosen expert looks ~1.8x better than what the token got — while
out-of-sample correlation collapses to zero within a few hundred steps. That is the signature of a quantity that is
real and unlearnable: `adv[t,e] = -<g_t, out[t,e] - m_t>` is a *first-order, single-batch* estimate, and once the
balancer has equalised the experts the only generic component (some expert being systematically better aligned at
init, which is what the 0.46 at step 0 measures) is gone, leaving per-batch noise. Following it is overfitting the
current micro-batch's gradient.

So the premise — that the discarded capacity-padding compute is a free counterfactual — is correct, and the
implementation is exact and free; the conclusion that a router should follow that counterfactual is what fails.
Reviving it means attacking the variance, not the plumbing: average the advantage over many batches before acting on
it, or aggregate it to a coarser unit (per expert-pair rather than per token) where the noise cancels. **Measure the
two-batch correlation before spending a run on any such variant** — it costs one forward/backward pair and would have
predicted every A/B above.

## NVFP4 — Amdahl-limited, not format-limited

Measured on `configs/fineweb_500m.yaml`'s shape (`d_model: 1280`, `n_layers: 22`, `loop_count: 2`, batch 4 x seq
1024, compiled, RTX 5090), with the step decomposed so the ceiling is visible rather than inferred:

| arm | step | tok/s | peak | fwd+bwd | optimizer | `refresh_fp4_weights` |
|---|---|---|---|---|---|---|
| bf16, `grad_accum 1` (as shipped) | 227.6 ms | 18.0k | 16.68 GB | 132.0 ms | 98.9 ms | — |
| nvfp4, `grad_accum 1` | 214.4 ms | 19.1k | 19.16 GB | **111.1 ms** | 99.0 ms | 7.2 ms |
| bf16, `grad_accum 8` | 1151.4 ms | 28.5k | 16.67 GB | 132.3 ms | 99.5 ms | — |
| nvfp4, `grad_accum 8`, conservative coverage | 994.6 ms | 32.9k | 19.16 GB | 112.0 ms | 99.5 ms | 7.3 ms |
| nvfp4, `grad_accum 8`, **full coverage** (shipped default) | **956.1 ms** | **34.3k** | 19.40 GB | **107.0 ms** | 99.5 ms | 7.3 ms |

Read the `fwd+bwd` column first: **FP4 buys 1.24x on the part it actually touches.** Everything else is that number
being diluted. The optimizer is a flat **99 ms per optimizer step regardless of arm** — Muon's Newton-Schulz, which
FP4 never touches and which [optim.md](optim.md) already flags as ~50% of step time at this config's 4096 tokens/step
— so at `grad_accum: 1` the total is only **1.06x**, and at `grad_accum: 8` it is **1.20x**, converging on the
fwd+bwd ceiling as the fixed cost amortises.

So: **1.06x at the shipped `grad_accum: 1` and 1.20x at the effective batch `fineweb_500m.yaml`'s own header already
recommends for the Muon reason.** `configs/fineweb_500m_nvfp4.yaml` therefore ships
`target_effective_batch_size: 32`. The two knobs compound rather than compete: raising the effective batch is worth
1.58x on its own for bf16 (18.0k -> 28.5k tok/s) and FP4 adds 1.20x on top.

**Coverage is worth about 4%**, which is why the `fp4_keep_bf16_*` knobs default to maximum: going from
`blocks[1:-1]` to every block plus `lm_head` moves fwd+bwd 112.0 -> 107.0 ms and the step 994.6 -> 956.1 ms. That
decision is made on throughput alone — the quality side is unmeasured; see [nvfp4.md](nvfp4.md) for the order to turn
coverage back.

Three things to carry forward before anyone reaches for this:

- **FP4 still costs ~2.7 GB *more* peak memory than bf16** (19.40 vs 16.67 GB), even with `fp4_save_activations` on,
  and `batch_size: 8` OOMs on a 32 GB card where bf16 fits. The weight caches are 0.48 GB of that; the rest is the
  backward briefly holding several quantized forms at once. `fp4_save_activations` *does* help — measured separately
  and controlled, it returns **~11% of peak** (7.73 -> 6.89 GB at batch 4, reproducible to ±0.001 GB) and is
  bit-identical — it just does not close the whole gap. **Do not measure this from a multi-arm sweep in one
  process:** the first attempt showed two configurations that were in fact identical differing by 3.8 GB, i.e. the
  peak column was allocator noise. Alternate the arms with repeats, or use one process per arm.
- **Per-linear the win is much larger than end-to-end** — 1.18-1.88x across `fineweb_500m`'s four projection shapes,
  2.35x at `d_model: 2048`, all three GEMMs included. Quote the end-to-end number unless the thing being sized is a
  linear.
- **`d_model: 256` measures 0.34x.** Do not evaluate this feature on TinyStories; it will report a loss, correctly,
  and tell you nothing.

**What is still bf16, and why it is not going away soon.** Attention's own two matmuls (`Q @ Kᵀ` and `attn @ V`) are
batched 4-D and `_scaled_mm_v2` is 2-D only, so they would need a per-head Python loop — the same launch-overhead
trap that rules out `BatchedExperts`. The routers and `out_gate` are excluded deliberately (tiny, and `out_gate`'s
zero-init is an *exact* identity). And Muon's Newton-Schulz is not a candidate despite being the largest remaining
term: the iteration is self-correcting enough that bf16 costs nothing there, but e2m1 has a **1-bit mantissa** and
would land outside the band the tuned coefficients assume. So ~1.24x on fwd+bwd is the ceiling for this architecture
without one of those changing.

**No quality A/B has been run.** A 60-step smoke run tracks bf16 closely (`val/loss` 6.3543 vs 6.3433, `d_model: 256`,
batch 16 pinned), which establishes that it trains rather than that it trains *as well*. When that A/B is run it must
be at **fixed compute, not fixed steps** — FP4 is supposed to change wall-clock, so a fixed-step comparison measures
only the quality cost and reports it as a pure loss — and with `batch_size` pinned in both arms, since the memory
difference would otherwise hand them different batches.

## Throughput work

Recorded separately from quality work, because it is supposed to change wall-clock and nothing else. A/B on
`configs/tinystories.yaml`'s shape (400 steps, pinned `batch_size: 32`, `auto_batch_size: false`, `lr: 1.0e-2`,
`dropout: 0.0`), single-pass loss + batched Muon vs the two-pass loss + per-parameter Muon:

| | wall clock | `val/loss` @400 | reserved VRAM |
|---|---|---|---|
| before | 55.3s | 2.1020 | 25.69 GB |
| after | **20.3s** | 2.1040 | **9.20 GB** |

**2.7x, with `val/loss` matching to within the ~0.002 noise floor** at every eval point (3.0469/2.4788/2.2285 vs
3.0612/2.4807/2.2294). The freed memory compounds through `auto_batch_size`, which now picks micro-batch 62 where it
picked 43. The loss change is by far the larger of the two and shrinks with width, so do not carry 2.7x across to a
wide model — the same comparison on `configs/fineweb_500m.yaml` (`d_model: 1280`) is 343.8 -> 338.7 ms/step and 24.5
-> 20.8 GB.

## Startup compile cost

Measured on the first forward/backward with `mode=None` (`d_model: 256`, `n_layers: 4`, `loop_count: 6`, batch 8 x
512):

| configuration | compile | warm step |
|---|---|---|
| fixed loop count, `doc_attention_mask: false` | 18.9s | 0.04s |
| fixed loop count, `doc_attention_mask: true` | 25.2s | 0.05s |
| fixed loop count, `+ grad_checkpoint` | 36.1s | 0.06s |

Document masking is cheap (~6s) and `grad_checkpoint` adds ~45%. **Stochastic loop depth is the expensive one**: each
distinct count in `[loop_count_min, loop_count_max]` is a separate dynamo graph costing roughly the per-graph figure
above, and it scales linearly — a 3-count range measured 78.2s, almost exactly 3x the 25.2s single-graph case. A wide
range, especially with `grad_checkpoint`, spends minutes before the first loss line appears. That is compilation, not
a hang; warm steps stay ~0.05s throughout. Keep the range narrow, or set `train.compile: false`, while iterating.

## Cautions when running an A/B

1. **Pin `batch_size` and set `auto_batch_size: false`.** Otherwise a change that reduces memory (sparsity,
   checkpointing) is silently handed a larger batch, and the arms differ in two ways at once.
2. **Check for `ending run early` in the output.** An OOM-terminated arm still exits 0 and still prints a few evals,
   so it looks like a valid short run rather than a failed one. `configs/tinystories_router.yaml` at the default
   `vram_safety_margin` does OOM on a 32 GB card — `estimate_batch_size` is optimistic for router mode, where the
   loop body is re-run `max_loops` times.
3. **Give each arm the whole GPU.** Two of the first hyper-connection arms were invalidated by a diagnostic run
   started alongside them, and the symptom was an OOM-shaped early exit in the *other* process, not the one at fault.
4. **Discard the first run of any config when timing.** Inductor's on-disk FX graph cache persists across processes:
   the first run of the looped A/B above took 71.8s and the second 38.9s for the *same* code and the same final loss
   — re-running the first arm warm gave 39.2s. That is a 1.8x phantom speedup available to whichever arm runs second.
   Always re-run the first arm at the end.
5. **Prefer a longer horizon.** An arm can rank differently at step 400 than at step 1500+ once the LR schedule is
   far enough into its decay. Establish the setup's own noise floor with a repeat baseline run rather than borrowing
   one measured elsewhere.
6. **Never compare two runs with different `max_steps` at the same step number.** `warmup_steps =
   round(max_steps * warmup_ratio)` and the cosine decays over `max_steps`, so a 1000-step run is fully annealed at
   step 1000 while a 3000-step run is only a third of the way through its schedule. Measured at 0.17 val/loss for
   the *same config and seed* — see [the loop-vs-depth section](#the-loop-is-dominated--depth-and-moe-are-the-frontier),
   where this invalidated a whole round of MoE conclusions by ~5x. Compare runs that share `max_steps`, or compare
   each run at its own endpoint. This is not implied by caution 5: a longer horizon fixes *when* you read, this
   fixes *what you may read it against*.
7. **Benchmark the step, not a short run**, when judging a throughput change — at TinyStories size, 400 steps is
   mostly compile, data loading and eval. And decompose the step before judging: FP4's end-to-end number was
   dominated by an optimizer cost it never touched.

### Historical note on wall-clock numbers

**Any wall-clock time recorded from a full `radiance-train` run on a doc-masked config before the CUDA-graph leak fix
is inflated by roughly 3x**, and OOM-shaped early exits on long looped runs were symptoms of it rather than of
genuine memory demand (see `resolve_compile_mode` in [train.md](train.md)). Quality numbers (`val/loss`) are
unaffected — the leak changed how the graph was executed, not what it computed. The standalone fwd+bwd benchmark
tables (ACT sparsity, hyper-connections, MoE dispatch) do not go through `train.py` and so were never exposed to it,
but none have been re-measured since; treat them as pre-fix figures until they are.
