# Loop-vs-depth-vs-MoE A/B (complete, 2026-08-11)

The harness and full working notes for the five-arm A/B that asks whether the weight-shared loop —
the axis most of this repo's machinery serves — actually pays. **It does not.** The headline result
is transcribed into [docs/results.md](../../docs/results.md); this directory is the working record
behind it: config generator, serial driver, per-phase logs, and the dead ends.

Read the notes below as a lab notebook, in order — several intermediate verdicts were **later
overturned** by longer runs, and that reversal is itself one of the findings. The final numbers are
under "FINAL: five-arm result at 3000 steps".

Two corrections to what the notes say inline, applied after the fact rather than rewritten in
place, so the reasoning stays legible:

1. **The memory leak described under Phase 2 is fixed.** It was CUDA graphs re-recording against a
   fresh `flex_attention` BlockMask every step; see `resolve_compile_mode` in
   [docs/train.md](../../docs/train.md). Every arm of the V1 sizing benchmark now holds reserved
   memory flat from the first measured step to the last, so the workaround these notes reach for
   (`grad_checkpoint: true`) is no longer needed for that reason.
2. **Use 0.008 as this setup's noise floor, not the 0.002 quoted elsewhere.** Arm B was run three
   independent times and its mid-run numbers spread by ~0.008. `docs/results.md`'s 0.002 belongs to
   a different configuration.

Phase 2c's wall-clock times are **not** comparable across arms — B and C ran with `grad_checkpoint`
as the leak workaround and A did not. Read quality from them, not throughput.

---

# Original resume notes (paused 2026-08-11, since completed)

Paused mid-Phase-1 by an environment failure, not a result. See "Why this stopped" below.

## The question

Does the weight-shared loop (`blocks[1:]` re-run `loop_count` times) buy anything over a plain
dense stack? Nothing in the repo measures this, even though the loop is what the framework is
built around — loop conditioning, input injection, BPTT windowing, ACT/router, ACT capacity
sparsity and hyper-connections all exist to serve it. The one loop-related A/B on record compares
looping-with-conditioning against looping-without.

## Design

Three arms off `configs/tinystories.yaml`'s reference shape (d_model 256, head_dim 64,
ffn_mult 4.0, ffn_depth 3, seq_len 512):

| arm | `n_layers` | `loop_count` | executed blocks | params | FLOPs |
|---|---|---|---|---|---|
| A dense | 6 | 1 | 6 | 31.79M | 1x |
| B looped | 6 | 4 | 21 | 31.87M (= A) | 3.5x |
| C deep | 21 | 1 | 21 | 79.09M | 3.5x (= B) |

- **B vs A** is equal-parameter: does recursion convert compute into quality without new params?
- **B vs C** is equal-FLOP: is recursion competitive with just being deep?

Neither comparison alone answers it. Note C is 2.49x A's *total* params, not 3.5x — the 12.9M
embedding is a fixed cost all three share; the 3.5x describes the block stack only. The equal-FLOP
framing is exact (both 21 executed blocks).

Phase 1 = per-arm LR sweep (400 steps, `lr` in {3e-3, 1e-2, 3e-2}) to de-confound. Phase 2 = the
real A/B at each arm's best LR, 3000 steps.

Phase 1 is not optional. CLAUDE.md's hyper-connection table shows an LR change moving val/loss
0.39 — roughly 10x any architecture delta ever measured in this repo. Judging B against A at one
shared LR would be unfalsifiable.

## Results so far

Arm A complete. Tracks the `lr` sweep already recorded in CLAUDE.md closely, which is the evidence
that this harness reproduces the repo's own numbers:

| `lr` | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|
| A_dense val/loss @400 | 2.1691 | 2.1044 | **2.0981** |
| CLAUDE.md's recorded sweep | 2.1800 | 2.1117 | 2.1051 |

(Small gap plausibly from `dropout: 0.0` here vs `0.03` there.)

Arms B and C: not yet measured. B_looped_lr0.003 crashed on the environment issue; B_looped_lr0.01
was killed mid-run when the sweep was stopped.

## Why this stopped

`nvidia-smi` fails with `Driver/library version mismatch. NVML library version: 610.57` — the
userspace NVML library was upgraded but the running kernel module is the old one. A reboot
reloads it.

The CUDA compute path does not touch NVML, which is why every arm-A run and the whole muP
coordinate-check series completed normally. PyTorch's `CUDACachingAllocator` calls `nvmlInit_v2`
when it needs host-level memory info, which it reaches under memory pressure — arm A (6 blocks)
never got there, arm B (21 executed blocks, ~3.5x activation memory) does, in the backward pass:

    RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at
    ".../c10/cuda/CUDACachingAllocator.cpp":1377

Environmental. Nothing in the muP fix or these configs is implicated.

## Resume

1. Confirm the driver is healthy: `nvidia-smi` should print a table.
2. Regenerate configs and rerun Phase 1 for **B and C only** (A is done):

       python gen.py --out phase1 --steps 400 --eval-every 100 \
                     --lrs 3e-3 1e-2 3e-2 --arms B_looped C_deep
       bash run.sh phase1

3. Pick each arm's best LR. **If B or C bottoms out at 3e-2 (the grid's top edge), extend upward
   before treating it as the optimum** — A's 3e-2 is defensible only because CLAUDE.md's wider
   sweep has 1e-1 at 2.1520, and nothing has measured 1e-1 for a 21-block model. Handing one arm a
   tuned LR and another a truncated one is the exact confound Phase 1 exists to remove.
4. Phase 2: same generator at `--steps 3000 --eval-every 250`, one config per arm at its best LR.

Budget from measured 400-step wall times: A ~65s/run, B and C ~4-5 min/run. Phase 1 remainder
~30 min, Phase 2 ~70 min.

## Hygiene baked into the configs (all from CLAUDE.md's hard-won list)

- `batch_size: 32` pinned with `auto_batch_size: false` — otherwise arm A, which uses far less
  memory, is silently handed a bigger batch and the arms differ in two ways at once.
- `run.sh` runs arms **serially**. Two hyper-connection arms were previously invalidated by a run
  started alongside them, and the symptom was an OOM-shaped early exit in the *innocent* process.
- `run.sh` greps every log for `ending run early`: an OOM-terminated arm still exits 0 and prints
  plausible evals, so it looks like a valid short run rather than a failure.
- `dropout: 0.0` (doc masking drops attention dropout anyway; keeps eval deterministic),
  `wandb: disabled` (stdout logging carries the numbers).

## Known limitation

TinyStories may not be able to answer this. It is deliberately simple, and the honest failure mode
is that 21 blocks of depth — recursive or real — buys nothing on it, giving three arms within
noise and a null result that reflects the dataset rather than the architecture. If Phase 1's B and
C come back bunched against A, move to fineweb before spending Phase 2's 70 minutes.

## Files

- `gen.py` — config generator (arm definitions live in `ARMS`)
- `run.sh` — serial driver with the OOM-detection grep
- `coord_check.py` — the muP coordinate check that found the AdamW bug (committed fix: 7e5337f)
- `phase1_logs_partial/` — arm A's three logs plus B's crash log

---

# Phase 1 RESULTS (complete, 2026-08-11)

Re-run in full at micro-batch 16 x grad_accum 2 (effective batch 32). The original micro-batch-32
configs OOM'd on arm B: 21 executed blocks peak near 30GB and die allocating the 1.65GB logits
tensor. Pre-reboot that pressure hit the broken NVML and surfaced as an allocator INTERNAL ASSERT,
which disguised a sizing bug as a pure driver fault. Arm A was re-run too rather than reusing its
micro-batch-32 numbers, so all arms share identical batch settings.

val/loss @ 400 steps:

| lr | 3e-3 | 1e-2 | 3e-2 | best |
|---|---|---|---|---|
| A_dense  (6 blocks, 31.8M)  | 2.1575 | 2.0859 | **2.0850** | 2.0850 |
| B_looped (21 exec, 31.9M)   | 2.1479 | **2.0983** | 2.1239 | 2.0983 |
| C_deep   (21 blocks, 79.1M) | 2.0914 | **2.0365** | 2.0400 | 2.0365 |

All nine runs valid (no OOM, no `ending run early`, all reached step 400). Every arm's optimum is
properly bracketed — B and C interior at 1e-2, A flat across 1e-2/3e-2 (0.0009 apart, inside the
~0.002 noise floor). No arm is reporting a truncated grid edge.

## Read

- **Equal-parameter (B vs A): -0.0133.** Spending 3.5x the FLOPs on recursion at fixed parameters
  is worse than not looping at all.
- **Equal-FLOP (B vs C): -0.0618**, ~30x the noise floor. For the same compute, real depth beats
  recursive depth decisively.
- **Control (C vs A): +0.0485.** TinyStories *does* reward 21 blocks of depth, so B's result is not
  "the dataset is too simple to need depth" — recursion is capturing almost none of what depth gives.

The LR sweep was load-bearing: at 3e-3 arm B *leads* arm A by 0.010, and the arms' LR curves cross.
A single shared LR would have supported either conclusion.

## Caveats

- 400 steps, TinyStories, d_model 256, one seed. Phase 2 (3000 steps) tests whether the ordering
  survives a longer horizon.
- LRs were tuned at 400 steps and applied at 3000. Longer runs often prefer slightly lower LRs, so
  Phase 2's per-arm LR is an extrapolation, not a re-tuned optimum.
- Arm A's best is a coin flip between 1e-2 (2.0859) and 3e-2 (2.0850). Phase 2 uses 3e-2 as the
  measured best; nothing turns on the choice.
- B uses the repo's default loop configuration (loop_iter_conditioning norm_gains,
  loop_input_injection on). A different conditioning scheme is a separate question this does not
  address.

# Phase 2 (running)

3000 steps, eval_every 250, each arm at its Phase-1 best LR: A@3e-2, B@1e-2, C@1e-2.
Expect ~10 min (A), ~19 min (B), ~29 min (C). Logs in phase2/logs/, driver log phase2_driver.log.

---

# Phase 2: partial results + an unresolved memory leak (2026-08-11)

## Results (all from eval points banked before the crashes -- these are valid)

| step | A_dense | B_looped | C_deep |
|---|---|---|---|
| 250 | 2.6005 | 2.6477 | 2.5620 |
| 500 | 2.2746 | 2.2510 | 2.2204 |
| 750 | 2.1158 | 2.0775 | 2.0380 |
| 1000 | 2.0109 | 1.9652 | 1.9217 |
| 1250 | 1.9338 | 1.8885 | 1.8434 |
| 1500 | 1.8712 | 1.8259 | - |
| 3000 | 1.6240 | - | - |

**Phase 1's 400-step verdict was horizon-limited and partly wrong.** B trails A at step 250, crosses
over by 500, and holds ~+0.045 from 750 onward. So:

- **B vs A (equal params): the loop WINS at longer horizons** (+0.045), having appeared to lose
  (-0.013) at 400 steps. It trains slower early -- plausible for a weight-shared block that must
  work at four different depths -- then pulls ahead.
- **B vs C (equal FLOPs): real depth still wins** by ~0.045 at step 1250.
- The gaps are near-identical: C beats B by 0.045, B beats A by 0.045. The loop sits exactly
  halfway between the shallow baseline and real depth.

**Implication for the rest of the repo:** most of CLAUDE.md's recorded A/Bs are 400-step runs at
this scale. This is direct evidence a 400-step verdict can invert. The hyper-connection result
(-0.004, judged neutral-to-slightly-negative) is the one most worth re-examining, since it was
measured in exactly the looped regime that turns out to need longer to pay off.

## The memory leak (UNRESOLVED -- separate from the A/B)

Symptom: 21-executed-block runs climb ~0.9 GB/min monotonically until they exhaust the 32GB card
around step ~1600. Killed B twice and C twice. Arm A survives only because it has headroom, so
this silently caps how long *any* run in this repo can go.

Ruled out by measurement:
- **Not fragmentation.** With PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True the failure changed
  from dozens of OOM warnings to `memory mapping failed ... free: 58MB` -- it consumed the whole
  card, rather than failing to find a contiguous block.
- **Not the model, optimizer, or doc masking.** `leak_probe.py` drives the model directly (no
  train.py, no DataLoader) for 300 steps: reserved memory is *dead flat* at 5.91-6.87 GB, in all
  four combinations of doc_attention_mask on/off x compile on/off.
- **Not evaluate().** 600-step runs with eval_every 100 vs 999999 both climb ~6 -> ~20 GB; the
  no-eval arm actually peaked *higher* (22.4 vs 20.4 GB).
- **Not desktop contention.** Discord (402MB) was on the GPU for one failure but is ~1% of the
  shortfall; the runs were heading for the ceiling regardless.

So it is somewhere in train.py's plumbing. The untested differences between the clean probe and a
real run are **the real DataLoader** and **grad_accum_steps: 2**. That is where to look next.

Workaround in use: `grad_checkpoint: true` (phase2c). Gradients are bit-identical, so arm A's
completed result stays comparable; it costs ~20-25% throughput and drops the starting footprint
from ~18 GB to ~13 GB, leaving enough headroom to outrun the growth over 3000 steps.

`leak_probe.py` is the tool -- extend it with a DataLoader and grad accumulation to bisect.

---

# Phase 2c: COMPLETE three-way at 3000 steps (2026-08-11)

B and C rerun with `grad_checkpoint: true` (bit-identical gradients; it also eliminated the memory
growth entirely -- see below). Arm A's number is from the original phase2 run.

| step | A_dense | B_looped | C_deep | B-A | C-B |
|---|---|---|---|---|---|
| 250 | 2.6005 | 2.6405 | 2.5671 | +0.040 | -0.073 |
| 500 | 2.2746 | 2.2494 | 2.2172 | -0.025 | -0.032 |
| 1000 | 2.0109 | 1.9602 | 1.9209 | **-0.051** | -0.039 |
| 1500 | 1.8712 | 1.8231 | 1.7789 | -0.048 | -0.044 |
| 2000 | 1.7589 | 1.7150 | 1.6686 | -0.044 | -0.046 |
| 2500 | 1.6707 | 1.6313 | 1.5840 | -0.039 | -0.047 |
| 3000 | **1.6240** | **1.5893** | **1.5394** | -0.035 | **-0.050** |

## Verdict

**Equal parameters (B vs A): the loop wins, but the win is eroding.** B trails at 250, crosses by
500, peaks at -0.051 near step 1000, then decays monotonically to -0.035 at 3000. On that trend the
advantage plausibly vanishes at longer horizons. It is a transient, not a stable property.

**Equal FLOPs (B vs C): real depth wins, and the gap is widening** -- -0.032 at 500 growing to
-0.050 at 3000.

Both trends point the same way: **depth pulls away from recursion the longer you train.** Note this
means neither the 400-step verdict (loop loses) nor the 1250-step snapshot (loop sits halfway) was
the whole story. Horizon matters more than any single reading.

Noise: B was run three independent times (phase2/2b/2c) and its step-1000/1250/1500 numbers spread
by ~0.008. So the 0.035 and 0.050 gaps are ~4-6x noise -- real but not enormous. Use 0.008, not the
0.002 figure quoted elsewhere in CLAUDE.md for a different configuration.

Wall time: A 626s, B 490s, C 561s. A's is inflated -- it ran without grad_checkpoint and so was
throttled by the memory growth (see below). Do not read step-time comparisons off these.

## The leak: partially characterised, still unfixed

`grad_checkpoint: true` did not merely add headroom -- **it eliminated the growth entirely** (flat
at 13217 MiB from startup through step 3000, where earlier runs hit 30+ GB by step 1600). Since
grad_checkpoint's whole effect is to not retain per-block activations, the thing accumulating is
**activation memory held across steps**. That also explains why only the 21-executed-block arms
died: they retain ~3.5x the activations.

It was also a large throughput tax, not just an OOM risk:

| B run | s/step |
|---|---|
| phase2 (leaking) | 0.39 |
| phase2b (leaking) | 0.43 |
| phase2c (grad_checkpoint) | **0.163** |

~2.5x. Any step-time figure in CLAUDE.md measured on a looped config without grad_checkpoint may be
measuring the leak rather than the architecture.

Ruled out: fragmentation, the model/optimizer/doc-masking (leak_probe.py is flat over 300 steps in
all four compile x doc_mask combinations), evaluate(), desktop contention. Remaining suspects: the
real DataLoader and grad_accum_steps. `leak_probe.py` is the tool -- add those two and bisect.

# Phase 3 (running): MoE arms

Does adding capacity via experts close the loop's 0.050 deficit against real depth? MoE adds
parameters without adding per-token FLOPs, which is exactly what B lacks versus C.

- **D_moeloop**: n_layers 6, loop_count 4, MoE -- 21 exec blocks, 73.5M total, 31.46M active
- **E_moedense**: n_layers 6, loop_count 1, MoE -- the control, separating "MoE helps generally"
  from "MoE helps *the loop*", the same role C played for B

MoE sizing: n_experts 8, top_k 2, moe_expert_ffn_mult 0.65, moe_n_shared 0. The 0.65 is measured,
not guessed: ffn_depth 3 makes an expert's two ffn_dim x ffn_dim layers scale quadratically with
width, so 0.5 gives an expert far smaller than half. Active lands at 31.46M vs B's 31.87M -- a 1.3%
shortfall that runs *against* the MoE arms, so a win cannot be blamed on extra compute.
moe_n_shared 0 (not the repo default 1) because an always-on expert adds unconditional FLOPs and
would break the match with B.

LR sweep at **1000 steps**, not 400 -- the horizon lesson above. Then 3000-step runs at the best LR.

---

# FINAL: five-arm result at 3000 steps (2026-08-11)

| arm | total | active | exec blocks | val@3000 | wall |
|---|---|---|---|---|---|
| A_dense | 31.8M | 31.8M | 6 | 1.6240 | 626s* |
| B_looped | 31.9M | 31.9M | 21 | 1.5893 | 490s |
| **C_deep** | 79.1M | 79.1M | 21 | **1.5394** | 561s |
| D_moeloop | 73.5M | 31.5M | 21 | 1.5507 | 796s |
| E_moedense | 54.2M | 31.5M | 6 | 1.5752 | 400s |

*A ran without grad_checkpoint and was throttled by the memory growth; its wall time is inflated.

## The 2x2

| | dense | looped | loop gain |
|---|---|---|---|
| no MoE | 1.6240 | 1.5893 | +0.0347 |
| MoE | 1.5752 | 1.5507 | +0.0245 |
| **MoE gain** | **+0.0488** | **+0.0386** | |

Both effects are real (~3-6x the 0.008 noise floor) and **sub-additive**: combined A->D is +0.073 where
the two individual gains sum to +0.084. MoE helps the dense arm more than the looped one, and the
loop helps the dense arm more than the MoE one -- they supply overlapping capacity, not
complementary mechanisms.

## Verdict on the loop

**The loop is dominated in both settings once cost is counted.**

- **E_moedense beats B_looped** (1.5752 vs 1.5893) using **6 executed blocks instead of 21** -- about
  a third of the compute -- and less wall time (400s vs 490s). Adding experts to a shallow model
  beats recursion outright, at a third of the cost.
- **C_deep beats D_moeloop** (1.5394 vs 1.5507) at less wall time (561s vs 796s). Real depth still
  wins on quality even against looped MoE.
- The Pareto frontier is **C_deep** (best quality) and **E_moedense** (cheapest). B and D are both
  dominated.

So at this scale, on TinyStories: recursion buys a genuine but modest gain (+0.035 dense, +0.025
with MoE) and never reaches the frontier. MoE is the better way to spend the same budget, and real
depth is the better way to spend more.

## METHODOLOGICAL WARNING -- cross-phase comparisons are invalid

`warmup_steps = round(max_steps * warmup_ratio)` and the cosine decays over `max_steps`, so **a
1000-step run and a 3000-step run are on different LR trajectories and are NOT comparable at the
same step number.** Measured directly: D at step 1000 scored 1.7667 in a 1000-step run (fully
annealed) and 1.9401 in a 3000-step run (mid-schedule) -- same config, same seed, a 0.17 gap that
is purely schedule.

An earlier version of these notes reported MoE gains of +0.19 to +0.23 computed across exactly that
mismatch. They were wrong by ~5x. Only compare runs that share `max_steps`, or compare at each
run's own endpoint.

The horizon lesson also stands independently: the loop's advantage over A peaked at +0.051 near step
1000 and decayed to +0.035 by 3000, while C's advantage over B grew from +0.032 to +0.050. Depth
pulls away from recursion the longer you train. Most of CLAUDE.md's A/Bs are 400-step runs; this is
direct evidence such a verdict can invert.
