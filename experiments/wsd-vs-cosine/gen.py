"""Generate the main grid for the WSD-vs-cosine A/B on a small TinyStories model.

d256 / L8 / ffn_mult 3 with the ts16k tokenizer: 15.77M params. 4000 steps, effective batch
32 x 512 pinned (docs/results.md caution 1). Every arm shares max_steps, so arms are compared at
the same endpoint (caution 6).

`lr` was tuned against cosine, and WSD holds peak LR for most of the run, so a shared LR could
favour either schedule. Each schedule therefore gets its own peak-LR sweep: `lr` and `muon_lr`
scaled together by MULTS. Repeats at 1x give the setup's own noise floor (caution 5).

Usage:
    uv run python experiments/wsd-vs-cosine/gen.py --out experiments/wsd-vs-cosine/main
"""

from __future__ import annotations

import argparse
import pathlib

TEMPLATE = """\
run_name: {name}

data:
  dataset: roneneldan/TinyStories
  text_column: text
  tokenizer: .cache/radiance/tok/ts16k
  seq_len: 512
  num_workers: 4
  cache_dir: .cache/radiance/tokenized

model:
  d_model: 256
  head_dim: 64
  n_layers: 8
  loop_count: 1
  ffn_mult: 3.0
  ffn_depth: 2
  dropout: 0.0
  max_seq_len: 512

train:
  batch_size: 16
  grad_accum_steps: 2
  auto_batch_size: false
  lr: {lr}
  muon_lr: {muon_lr}
  weight_decay: 0.01
  warmup_ratio: 0.04
  lr_schedule: {schedule}
  wsd_decay_ratio: {decay}
  max_steps: {steps}
  grad_clip: 1.0
  log_every: 100
  eval_every: 500
  eval_max_batches: 50
  save_every: {steps}
  output_dir: {out}/ckpt/{name}
  seed: 42
  compile: true
  dtype: bf16

wandb:
  mode: disabled
"""

BASE_LR, BASE_MUON_LR = 1.0e-2, 0.02
MULTS = (0.5, 1.0, 2.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=4000)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms: list[tuple[str, str, float, float]] = []  # name, schedule, decay, mult
    for m in MULTS:
        arms.append((f"a_cos_m{m}", "cosine", 0.2, m))
        arms.append((f"a_wsd20_m{m}", "wsd", 0.2, m))
    arms += [("b_wsd10_m1.0", "wsd", 0.1, 1.0), ("b_wsd30_m1.0", "wsd", 0.3, 1.0)]
    for r in (1, 2):
        arms.append((f"c_cos_m1.0_r{r}", "cosine", 0.2, 1.0))
        arms.append((f"c_wsd20_m1.0_r{r}", "wsd", 0.2, 1.0))

    for name, schedule, decay, m in arms:
        (out / f"{name}.yaml").write_text(TEMPLATE.format(
            name=name, schedule=schedule, decay=decay, steps=args.steps, out=out,
            lr=BASE_LR * m, muon_lr=BASE_MUON_LR * m,
        ))
    print(f"wrote {len(arms)} configs to {out}")


if __name__ == "__main__":
    main()
