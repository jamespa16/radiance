"""Generate the fixed-loop vs learned-depth (ACT) A/B configs.

Four arms off experiments/loop-vs-depth's TinyStories shape (d_model 256, head_dim 64, n_layers 6,
ffn_mult 4.0, ffn_depth 3, seq_len 512), every one trained on exactly the same tokens:
1000 steps x effective batch 32 x 512 = 16,384,000.

    dense      loop_count 1                          ->  6   executed blocks/token
    loop3      loop_count 3                          -> 16
    act4_r50   use_router, max_loops 4, capacity 0.5 -> 16   (1 + 5 x (2 + 2 x 0.5)): = loop3's compute
    act4_r25   use_router, max_loops 4, capacity 0.25-> 13.5 (1 + 5 x (2 + 2 x 0.25))

ACT's first and last iterations are always dense; act_capacity_ratio thins only the interior ones.

Hygiene, as in loop-vs-depth: batch pinned (auto_batch_size false) at micro 16 x accum 2 for every
arm, so the deepest arm fits without grad_checkpoint (which ACT capacity < 1 refuses anyway) and no
arm is handed a different batch; dropout 0; wandb disabled; every arm shares max_steps, so warmup
and the cosine schedule are identical. Each arm is LR-swept separately — the loop-vs-depth sweep
found the arms' LR curves cross.
"""

from __future__ import annotations

import argparse
import pathlib

ARMS = {
    "dense": "  loop_count: 1\n",
    "loop3": "  loop_count: 3\n",
    "act4_r50": "  use_router: true\n  max_loops: 4\n  act_capacity_ratio: 0.5\n",
    "act4_r25": "  use_router: true\n  max_loops: 4\n  act_capacity_ratio: 0.25\n",
}

TEMPLATE = """\
run_name: {name}

data:
  dataset: roneneldan/TinyStories
  text_column: text
  tokenizer: gpt2
  seq_len: 512
  num_workers: 4
  cache_dir: .cache/radiance/tokenized

model:
  d_model: 256
  head_dim: 64
  n_layers: 6
{arm}  ffn_mult: 4.0
  ffn_depth: 3
  dropout: 0.0
  max_seq_len: 512

train:
  batch_size: 16
  grad_accum_steps: 2
  auto_batch_size: false
  lr: {lr}
  muon_lr: 0.02
  weight_decay: 0.01
  warmup_ratio: 0.2
  max_steps: 1000
  grad_clip: 1.0
  log_every: 50
  eval_every: 100
  save_every: 1000000
  output_dir: {ckpt}/{name}
  seed: {seed}
  device: auto
  compile: true
  dtype: bf16

wandb:
  project: radiance
  entity: null
  mode: disabled
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--ckpt", required=True, help="checkpoint dir (scratch; models are not kept)")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--lrs", nargs="+", default=["0.003", "0.01", "0.03"])
    parser.add_argument("--repeat", type=int, default=1, help="runs per (arm, lr); repeats get suffix _r2, _r3")
    parser.add_argument("--first-repeat", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for lr in args.lrs:
            for rep in range(args.first_repeat, args.first_repeat + args.repeat):
                name = f"{arm}_lr{lr}" + (f"_r{rep}" if rep > 1 else "")
                # Same seed on repeats: the noise worth measuring is GPU nondeterminism at a fixed
                # config (docs/results.md's noise floors are measured the same way).
                (args.out / f"{name}.yaml").write_text(
                    TEMPLATE.format(name=name, arm=ARMS[arm], lr=lr, ckpt=args.ckpt, seed=42)
                )


if __name__ == "__main__":
    main()
