# WSD vs cosine

**Question:** is `lr_schedule: wsd` a quality win, loss, or wash against the default `cosine`, at
matched steps?

**Verdict: a wash at each schedule's own best LR, and a loss at the LR you would actually have
used.** At the best peak LR for each schedule, WSD with a 30% decay is 0.0034 ahead (suggestive,
not decisive); with the default 20% decay it is level. But WSD is much more sensitive to peak LR
than cosine, so at the LR tuned for cosine it is behind by 0.006 (1x) to 0.022 (2x). `cosine`
stays the default; WSD remains an operational choice (extendable runs, `--decay-from` branches),
and adopting it means giving the decay ~30% of the run and re-sweeping the LR.

## Setup

- TinyStories, `ts16k` tokenizer (see [under-50m/](../under-50m/)), `d_model: 256`,
  `n_layers: 8`, `ffn_mult: 3.0`, `ffn_depth: 2`, `loop_count: 1` — **15.77M** params.
- `seq_len: 512`, effective batch 32 pinned (`batch_size: 16`, `grad_accum_steps: 2`,
  `auto_batch_size: false`), `dtype: bf16`, `dropout: 0.0`, `seed: 42`, `compile: true`.
- **4000 steps** for every arm, so every arm is read at its own fully-annealed endpoint and
  every arm shares `max_steps` (docs/results.md caution 6). `warmup_ratio: 0.04`,
  `min_lr_ratio: 0.1` (default) for both schedules.
- **Peak LR swept per schedule**: `lr` and `muon_lr` scaled together by a multiplier `m` from
  the base `1e-2` / `0.02`. One `LambdaLR` drives both the Muon and AdamW groups, so scaling one
  alone would change the ratio between them rather than the peak.
- WSD's decay is `1 - sqrt(progress)` from full LR to `min_lr_ratio`, over the final
  `wsd_decay_ratio` of the run.
- RTX 5090, one arm at a time (`run.sh`), ~114s per arm.

Run from the `v2-run-hardening` working tree at `c040097`. That tree adds `wsd_decay_steps` and
`wsd_decay_start_ratio` to `build_lr_scheduler`, but at their defaults (`None` / `1.0`, which
every arm here uses) the schedule is identical to `main`'s.

## Results — `val/loss` at step 4000

| peak LR `m` | cosine | WSD 10% | WSD 20% | WSD 30% |
|---|---|---|---|---|
| 0.25 | 1.6196 | | 1.6095 | |
| **0.5** | **1.6032** | 1.6181 | 1.6050 | **1.5998** |
| 1.0 | 1.6091 (n=3) | 1.6304 | 1.6147 (n=3) | 1.6093 |
| 2.0 | 1.6184 | | 1.6406 | |

**Noise floor**: three identical runs of each schedule at `m = 1` —
cosine 1.6084 / 1.6102 / 1.6088 (mean 1.6091, stdev 0.0009), WSD 20% 1.6154 / 1.6142 / 1.6146
(mean 1.6147, stdev 0.0006). About half the 0.0021 measured in under-50m for a 3x larger model;
the stdev of a *difference* of two single runs is ~0.0011, so rank single runs at ~0.003.

## What it shows

1. **Both schedules peak at `m ≈ 0.5`** — bracketed on both sides for both. So the base
   `lr: 1e-2` is slightly high for this shape (0.006 for cosine); that is this model's finding,
   not a general one.
2. **At each schedule's best LR, they tie.** WSD 20% is +0.0018 (noise). WSD 30% is −0.0034,
   ~3x the single-run difference stdev, but it is one run of each and 30% was chosen after
   seeing 20% — suggestive, not established.
3. **WSD is far more LR-sensitive.** Its mean LR over a run is higher than cosine's at the same
   peak, so it gains where the peak is too low (−0.010 at `m = 0.25`) and loses where it is too
   high (+0.006 at `m = 1`, +0.022 at `m = 2`). Switching a cosine-tuned config to WSD without
   re-sweeping lands on the losing side — which is the concrete version of docs/config.md's
   "changes a tuned quantity" reason.
4. **Decay length matters more than the schedule choice.** At `m = 0.5`: 10% → 1.6181,
   20% → 1.6050, 30% → 1.5998, monotone; same ordering at `m = 1`. 30% was the longest tried, so
   the WSD optimum may be longer still.
5. **The stable phase is far behind; the decay does the work.** At step 3000 (`m = 0.5`) WSD 20%
   is at 1.7595 against cosine's 1.6588, 0.10 behind, and recovers all of it in the last 800
   steps. Mid-run WSD `val/loss` is not a usable estimate of the finished model — read a WSD run
   through a `--decay-from` branch, not off its stable-phase evals.

## Not measured

- **Longer horizons and larger models.** WSD's reported advantages are usually at far longer
  runs than 4000 steps of a 15.8M model; nothing here says what happens on `configs/v2.yaml`.
- Repeats at `m = 0.5` — two each of cosine and WSD 30% would settle finding 2.
- Decay longer than 30%, decay shapes other than `1 - sqrt`, and `min_lr_ratio` other than 0.1.
- The operational case for WSD (one stable run, many `--decay-from` endpoints) — this compares
  single runs at a fixed horizon only.

## Files

- `gen.py` — writes the 12 configs in `main/`: the LR sweep (`a_*`), decay fraction at `m = 1`
  (`b_*`), and the noise-floor repeats (`c_*`).
- `followup/` — four configs added after the main grid showed both schedules preferring the
  lowest `m` tried: `m = 0.25` for both (`d_*`, to bracket the optimum) and WSD 10%/30% at
  `m = 0.5` (`e_*`). Each is a `main/` config with `lr`/`muon_lr`/`wsd_decay_ratio` edited.
- `run.sh` — serial driver (copied from under-50m), flags OOM early exits.
- `summarize.py` — `val/loss` per eval step for every log in a directory:
  `uv run python experiments/wsd-vs-cosine/summarize.py experiments/wsd-vs-cosine/main`.
- `main/logs/`, `followup/logs/`, `*_driver.log` — raw output. Checkpoints are gitignored.
