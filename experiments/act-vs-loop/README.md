# Fixed loop vs learned depth (ACT), 2026-10-16

Does letting each token choose its loop depth (ACT, `use_router`) beat giving every token the same
depth? Four arms off loop-vs-depth's TinyStories shape, **16,384,000 tokens each** (1000 steps x
effective batch 32 x 512), every arm LR-swept, best LR repeated to n=3. `gen.py` generates every
config; `run.sh` (copied from loop-vs-depth) runs a directory serially.

| arm | executed blocks/token | best `lr` | `val/loss` @1000 (n=3) | mean | vs dense |
|---|---|---|---|---|---|
| dense (`loop_count: 1`) | 6 | 0.01 | 1.8160 / 1.8138 / 1.8162 | 1.8153 | — |
| **fixed `loop_count: 3`** | 16 | 0.03 | 1.7993 / 1.8034 / 1.7992 | **1.8006** | **−0.0147** |
| ACT `max_loops: 4`, capacity 0.5 | 16 (= loop3) | 0.03 | 1.8145 / 1.8146 / 1.8140 | 1.8144 | −0.0009 |
| ACT `max_loops: 4`, capacity 0.25 | 13.5 | 0.03 | 1.8192 / 1.8182 / 1.8202 | 1.8192 | +0.0039 |

Repeat spread (max − min at a fixed config) is 0.0006-0.0042, so the loop's −0.0147 is ~3.5x the
widest spread; ACT r0.5 vs dense is inside it; ACT r0.25 vs dense is ~2x it (marginal).

LR sweep (`val/loss` @1000):

| arm | 0.003 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|
| dense | 1.8595 | **1.8160** | 1.8187 | — |
| loop3 | 1.8287 | 1.8010 | **1.7993** | 1.8338 |
| act4_r50 | 1.8612 | 1.8206 | **1.8145** | 1.8370 |
| act4_r25 | 1.8600 | 1.8244 | **1.8192** | 1.8362 |

Three arms bottomed at 0.03, the original grid's top edge, so 0.1 was added; it is worse everywhere.

**Why ACT loses to the fixed loop: it learned to be shallow.** Measured on held-out batches with the
final checkpoints, `mean_loop_depth` is **1.85 / 4** (r0.5) and **1.69 / 4** (r0.25), `ponder_cost`
~2.0. Under the default `ponder_weight: 0.01` the router halts most tokens after the second
iteration, so ACT behaves like a ~1.8-iteration loop — and lands between dense (1) and loop3 (3), as
that would predict, while paying for 2 + 2 x capacity iterations of compute.

Throughput (steady-state, d256): dense 491k tok/s, loop3 282k, ACT r0.5 256k, ACT r0.25 243k.
Capacity 0.25 is *slower* than 0.5 at this width — the gather/scatter and materialised mask are a
fixed overhead that only amortises on wider models (docs/model.md).

Caveats: 1000 steps only (the loop's gain over dense *grew* from 1000 to 3000 steps in
loop-vs-depth, so 1000 likely understates loop3); one `ponder_weight` (a smaller one would let ACT
use more depth and is the obvious next arm); TinyStories at d256.

## Follow-up: `ponder_weight: 0.001` (10x below default)

Same harness, `lr` swept over {0.01, 0.03} (0.003 and 0.1 lost clearly above), best repeated to n=3.

| arm | `lr` 0.01 | `lr` 0.03 (n=3) | mean | vs dense | vs same arm at 0.01 | `mean_loop_depth` |
|---|---|---|---|---|---|---|
| ACT r0.5, `ponder_weight` 0.001 | 1.8222 | 1.8123 / 1.8139 / 1.8125 | 1.8129 | −0.0024 | −0.0015 | 2.00 / 4 (was 1.85) |
| ACT r0.25, `ponder_weight` 0.001 | 1.8245 | 1.8161 / 1.8160 / 1.8154 | 1.8158 | +0.0005 | −0.0034 | 2.02 / 4 (was 1.69) |

A small gain (clear for r0.25, marginal for r0.5), and the fixed loop still wins by 0.012-0.015 at equal or lower
compute. **The penalty was not what kept ACT shallow.** At 0.001 the ponder term is ~0.002 of loss, yet depth only
rose to ~2.0. Nor is it the init: `ACTRouter.proj.bias` starts at Graves' −1.0 (halting prob ~0.27/iteration, so
~4 iterations at init), so training actively pulls depth *down* to ~2. A plausible mechanism, not verified here:
later iterations get little output weight, so little gradient, so they stay worse, so the router keeps halting
early — a rich-get-richer collapse. Discriminating next arms: `ponder_weight: 0` (does depth still collapse with no
penalty at all?) and a fixed `loop_count: 2` control (is ACT just an expensive loop2?).
