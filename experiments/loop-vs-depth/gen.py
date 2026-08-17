"""Generate loop-vs-depth A/B configs.

Three arms off configs/tinystories.yaml's reference shape:

    A  dense    n_layers 6,  loop_count 1  ->  6 executed blocks, 1x params, 1x FLOPs
    B  looped   n_layers 6,  loop_count 4  -> 21 executed blocks, 1x params, 3.5x FLOPs
    C  deep     n_layers 21, loop_count 1  -> 21 executed blocks, 3.5x params, 3.5x FLOPs

B vs A is equal-parameter, B vs C is equal-FLOP. The A/B hygiene CLAUDE.md asks for is baked in:
batch_size pinned with auto_batch_size false (so an arm that uses less memory is not silently
handed a bigger batch), dropout 0 (doc masking drops attention dropout anyway, and it keeps eval
deterministic), and wandb disabled (stdout logging carries the numbers).
"""

from __future__ import annotations

import argparse
import pathlib

# MoE sizing is chosen so the routed arms match B/A on *active* parameters (i.e. FLOPs per token)
# while carrying C-like total capacity. Note ffn_depth 3 makes an expert's two ffn_dim x ffn_dim
# hidden layers scale quadratically with width, so moe_expert_ffn_mult 0.5 gives an expert far
# smaller than half -- 0.65 is what actually lands on B's active count (31.46M vs 31.87M, a 1.3%
# shortfall that runs *against* the MoE arms, so a win cannot be attributed to extra compute).
# moe_n_shared is 0 rather than the repo default 1 on purpose: an always-on shared expert adds
# unconditional FLOPs and would break the match with B.
_MOE = """  use_moe: true
  n_experts: 8
  moe_top_k: 2
  moe_expert_ffn_mult: 0.65
  moe_n_shared: 0
"""

ARMS = {
    "A_dense": dict(n_layers=6, loop_count=1, extra=""),
    "B_looped": dict(n_layers=6, loop_count=4, extra=""),
    "C_deep": dict(n_layers=21, loop_count=1, extra=""),
    "D_moeloop": dict(n_layers=6, loop_count=4, extra=_MOE),
    "E_moedense": dict(n_layers=6, loop_count=1, extra=_MOE),
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
  n_layers: {n_layers}
  loop_count: {loop_count}
  ffn_mult: 4.0
  ffn_depth: 3
  dropout: 0.0
  max_seq_len: 512
{extra}
train:
  # Micro-batch halved with grad_accum making up the difference, so effective_batch_size stays 32 --
  # the value arm A and CLAUDE.md's own lr sweep were measured at. Arm B (21 executed blocks) peaks
  # near 30GB at micro-batch 32 and OOMs allocating the 1.65GB logits tensor. Activation memory
  # scales with the micro-batch, not the effective batch, so this buys ~2x headroom for free.
  batch_size: {batch_size}
  grad_accum_steps: {grad_accum_steps}
  auto_batch_size: false
  lr: {lr}
  muon_lr: 0.02
  weight_decay: 0.01
  warmup_ratio: 0.2
  max_steps: {max_steps}
  grad_clip: 1.0
  log_every: 25
  eval_every: {eval_every}
  save_every: 1000000
  output_dir: {out_dir}
  seed: 42
  device: auto
  compile: true
  dtype: bf16

wandb:
  project: radiance
  entity: null
  mode: disabled
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eval-every", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lrs", type=float, nargs="+", required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for arm in args.arms:
        for lr in args.lrs:
            name = f"{arm}_lr{lr:g}"
            path = out / f"{name}.yaml"
            path.write_text(TEMPLATE.format(
                name=name, lr=f"{lr:g}", max_steps=args.steps, eval_every=args.eval_every,
                batch_size=args.batch_size, grad_accum_steps=args.grad_accum, out_dir=f"{out}/ckpt/{name}", **ARMS[arm]))
            written.append(str(path))
    print("\n".join(written))


if __name__ == "__main__":
    main()
