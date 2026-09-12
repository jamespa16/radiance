"""Generate configs for the "best model under 50M parameters" study.

The constraint is a **total** parameter cap, which is a different question from every A/B on
docs/results.md: those match *active* parameters (or nothing at all) and let total params float.
Under a total cap the embedding stops being a fixed background cost and becomes the thing you are
bidding against — 12.9M of the budget at `d_model: 256` with the gpt2 tokenizer, 19.3M at 384 —
so widening the model spends the budget twice, once on the blocks and once on the embedding.

Every arm is sized with `radiance-size` to land in a narrow band just under 50M so no arm wins by
being bigger. Hygiene follows docs/results.md's cautions: `batch_size` pinned with
`auto_batch_size: false`, `dropout: 0.0`, wandb disabled, and a per-arm LR sweep before any
architecture comparison (caution 3 — the arms' LR curves cross, so one shared LR can support
either conclusion).

Usage:
    uv run python experiments/under-50m/gen.py --stage shape --steps 6000 --out experiments/under-50m/shape
"""

from __future__ import annotations

import argparse
import pathlib

TEMPLATE = """\
run_name: {name}

data:
  dataset: roneneldan/TinyStories
  text_column: text
  tokenizer: {tokenizer}
  seq_len: 512
  num_workers: 4
  cache_dir: .cache/radiance/tokenized

model:
  d_model: {d_model}
  head_dim: 64
  n_layers: {n_layers}
  loop_count: 1
  ffn_mult: {ffn_mult}
  ffn_depth: 2
  dropout: 0.0
  max_seq_len: 512
{extra}
train:
  # Effective batch 32 x 512 = 16384 tokens/step, pinned so an arm that uses less memory is not
  # silently handed a bigger batch (docs/results.md caution 1). Micro-batch 16 keeps the widest
  # and deepest arms on the same setting as the cheapest.
  batch_size: 16
  grad_accum_steps: 2
  auto_batch_size: false
  lr: {lr}
  muon_lr: 0.02
  weight_decay: 0.01
  warmup_ratio: 0.04
  max_steps: {steps}
  grad_clip: 1.0
  log_every: 100
  eval_every: {eval_every}
  eval_max_batches: 50
  # Must divide max_steps: train.py saves only on `step % save_every == 0` and has no
  # end-of-training save, so a save_every larger than max_steps silently produces no checkpoint
  # at all -- and the vocab arms have to be scored from a checkpoint by bits_per_char.py.
  save_every: {save_every}
  output_dir: {out_dir}/ckpt/{name}
  seed: {seed}
  compile: true
  dtype: bf16

wandb:
  mode: disabled
"""

# Shapes chosen off `radiance-size`, all landing in 46.1-49.1M total with the gpt2 tokenizer.
# They span the width/depth trade at a fixed budget: d256_L24 is the narrowest and deepest the
# cap allows, d512_L4 the widest and shallowest.
SHAPES = {
    "d256_L24_fm3": dict(d_model=256, n_layers=24, ffn_mult=3.0),  # 47.61M
    "d256_L16_fm4": dict(d_model=256, n_layers=16, ffn_mult=4.0),  # 46.53M
    "d320_L14_fm3": dict(d_model=320, n_layers=14, ffn_mult=3.0),  # 47.73M
    "d320_L10_fm4": dict(d_model=320, n_layers=10, ffn_mult=4.0),  # 48.94M
    "d384_L9_fm3": dict(d_model=384, n_layers=9, ffn_mult=3.0),  # 48.59M
    "d384_L6_fm4": dict(d_model=384, n_layers=6, ffn_mult=4.0),  # 47.69M
    "d448_L6_fm3": dict(d_model=448, n_layers=6, ffn_mult=3.0),  # 49.09M
    "d448_L4_fm4": dict(d_model=448, n_layers=4, ffn_mult=4.0),  # 48.27M
    "d512_L4_fm3": dict(d_model=512, n_layers=4, ffn_mult=3.0),  # 48.87M
    # --- depth extension ---
    # The first round's winner (d256_L24_fm3) was the deepest shape in the grid above, so the grid
    # measured its own edge rather than the optimum -- the same failure the `lr` stage hit at 3e-2.
    # Depth is bought here by narrowing the FFN rather than the residual stream, since ffn_mult
    # trades against n_layers much more cheaply than d_model does (attention and the embedding both
    # scale with d_model, and the embedding is already a third of the budget).
    "d256_L29_fm2.5": dict(d_model=256, n_layers=29, ffn_mult=2.5),  # 46.75M
    "d256_L37_fm2": dict(d_model=256, n_layers=37, ffn_mult=2.0),  # 46.99M
    "d256_L48_fm1.5": dict(d_model=256, n_layers=48, ffn_mult=1.5),  # 46.89M
    "d320_L17_fm2.5": dict(d_model=320, n_layers=17, ffn_mult=2.5),  # 47.11M
    "d320_L22_fm2": dict(d_model=320, n_layers=22, ffn_mult=2.0),  # 47.77M
    "d192_L46_fm3": dict(d_model=192, n_layers=46, ffn_mult=3.0),  # 47.14M
}


# Each shape's own best `lr`, measured by the `lr` stage at 1500 steps over
# {3e-3, 1e-2, 3e-2, 1e-1} -- see README.md for the full table. Recorded here rather than passed
# on the command line so the shape sweep cannot accidentally be run at one shared LR, which
# caution 3 shows can support either verdict: the within-arm LR spread here reaches 0.055 against
# a between-arm shape effect of 0.021.
#
# Note the optima split by FFN width rather than by depth: every `fm3` arm wants 1e-2 and every
# `fm4` arm wants 3e-2.
BEST_LR = {
    "d256_L24_fm3": 1.0e-2,
    "d256_L16_fm4": 3.0e-2,
    "d320_L14_fm3": 1.0e-2,
    "d320_L10_fm4": 3.0e-2,
    "d384_L9_fm3": 3.0e-2,
    "d384_L6_fm4": 3.0e-2,
    "d448_L6_fm3": 1.0e-2,
    "d448_L4_fm4": 3.0e-2,
    "d512_L4_fm3": 1.0e-2,
    "d256_L29_fm2.5": 1.0e-2,
    "d256_L37_fm2": 1.0e-2,
    "d256_L48_fm1.5": 1.0e-2,
    "d320_L17_fm2.5": 1.0e-2,
    "d320_L22_fm2": 1.0e-2,
    "d192_L46_fm3": 1.0e-2,
}


# Vocabulary arms. Each spends the budget a smaller tokenizer frees (13.5M at d320 going 50k ->
# 8k) on depth or width rather than banking it, so every arm still fills the 50M cap and the
# comparison stays "best use of 50M". The gpt2 baseline is the dense base from the feature stage.
#
# These are scored in **bits/char** by bits_per_char.py, never in val/loss: a smaller vocabulary
# cuts the same story into more, individually easier tokens, so its per-token cross-entropy falls
# for reasons unrelated to model quality. See that script's docstring.
VOCAB_ARMS: dict[str, dict] = {
    "ts8k_d320_L32": dict(  # 48.69M -- +10 layers over gpt2's L22 at the same budget
        tokenizer=".cache/radiance/tok/ts8k", d_model=320, n_layers=32, ffn_mult=2.0,
    ),
    "ts8k_d384_L22": dict(  # 48.73M -- a cheap embedding makes width affordable again
        tokenizer=".cache/radiance/tok/ts8k", d_model=384, n_layers=22, ffn_mult=2.0,
    ),
    "ts16k_d320_L30": dict(  # 48.43M
        tokenizer=".cache/radiance/tok/ts16k", d_model=320, n_layers=30, ffn_mult=2.0,
    ),
}


# Feature arms, layered onto the shape sweep's winner (`d320_L22_fm2`, 47.77M, 22 executed
# blocks). Each arm is a *self-contained* shape rather than a flag dropped onto the base, because
# under a total cap a flag that costs parameters has to buy them back from somewhere, and leaving
# that implicit is how an arm ends up winning on size. Every arm is sized with `radiance-size`;
# the comment on each is `total / active`.
#
# Where an arm frees parameters (GQA) the freed budget is spent on depth rather than banked, so
# the comparison stays "best use of 50M" rather than "same model, slightly smaller".
FEATURES: dict[str, dict] = {
    # Costs 768 parameters (two per-head lambda vectors) -- free under a cap. docs/results.md
    # records it as a genuine win but declines to default it on because it costs 1.34x step time;
    # a parameter cap does not charge for step time, so the trade it lost on there does not apply.
    "diff_attn": dict(extra="  use_diff_attn: true\n"),  # 47.77M / 47.77M

    # The loop, asked the question a parameter cap actually poses. docs/results.md's verdict --
    # dominated -- is an equal-FLOP one against depth, and the honest re-test keeps FLOPs matched
    # while letting the loop spend what weight-sharing saves. Both arms execute ~22 blocks, the
    # same as the dense base, but reach d448/d512 where the dense base can only afford d320. So
    # this is not "loop vs depth" but "at equal compute and equal parameters, does weight-sharing
    # buy enough extra *width* to pay for the sharing?"
    "loop_w448": dict(  # 45.30M / 45.30M, 22 executed blocks
        d_model=448, n_layers=8, ffn_mult=2.0,
        extra="  loop_count: 3\n",
    ),
    "loop_w512": dict(  # 48.12M / 48.12M, 21 executed blocks
        d_model=512, n_layers=6, ffn_mult=2.0,
        extra="  loop_count: 4\n",
    ),
    # And the other loop question, which only a *parameter* cap makes interesting: the cap is on
    # parameters, not compute, so simply buying more compute at the same parameter count is legal.
    # 43 executed blocks against the base's 22, for +0.11M parameters.
    "loop2_deep": dict(extra="  loop_count: 2\n"),  # 47.88M / 47.88M, 43 executed blocks

    # Multi-query attention (d320 gives 5 heads, so n_kv_heads must be 1 or 5 -- there is no
    # halfway option). It frees 3.6M, spent here on L22 -> L24 so the arm still fills the budget.
    "mqa_L24": dict(n_layers=24, extra="  n_kv_heads: 1\n"),  # 46.70M / 46.70M

    # MoE is docs/results.md's largest architecture effect -- but measured at matched *active*
    # parameters, with total params left to grow 1.7x. A total cap removes exactly that freedom:
    # `use_moe: true` on the base shape costs 192M, 4x the budget, so the experts must be bought
    # back out of depth and width. Every MoE shape that fits lands at 25-28M *active* against the
    # dense arms' 47.8M, so the cap converts MoE's capacity win into an active-parameter deficit.
    # That is the thing being tested, and it is why this arm is not expected to reproduce the
    # result it comes from.
    "moe_d320_L8": dict(  # 45.30M / 28.05M
        d_model=320, n_layers=8, ffn_mult=4.0,
        extra="  use_moe: true\n  n_experts: 8\n  moe_top_k: 2\n"
        "  moe_n_shared: 0\n  moe_expert_ffn_mult: 0.25\n",
    ),
    "moe_d256_L14": dict(  # 45.82M / 25.30M
        d_model=256, n_layers=14, ffn_mult=4.0,
        extra="  use_moe: true\n  n_experts: 8\n  moe_top_k: 2\n"
        "  moe_n_shared: 0\n  moe_expert_ffn_mult: 0.25\n",
    ),

    # Multi-token prediction: a training-time auxiliary objective. num_active_parameters()
    # discounts the extra head, but the cap here is on *total* parameters -- what a checkpoint
    # actually contains -- so the head is paid for out of depth (L22 -> L21) rather than waved
    # through on the grounds that inference never runs it.
    "mtp2_L21": dict(n_layers=21, extra="  mtp_heads: 2\n"),  # 47.97M / 46.33M
}


def write(
    out_dir: pathlib.Path, name: str, *, steps: int, lr: float, seed: int = 42,
    extra: str = "", tokenizer: str = "gpt2", **shape,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.yaml").write_text(
        TEMPLATE.format(
            name=name,
            steps=steps,
            eval_every=max(steps // 10, 1),
            save_every=steps,
            tokenizer=tokenizer,
            lr=lr,
            seed=seed,
            extra=extra,
            out_dir=out_dir,
            **shape,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["lr", "shape", "feature", "vocab"])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=None, help="override the per-shape swept BEST_LR")
    p.add_argument("--repeat", default=None, help="shape to re-run at seed 43 for a noise floor")
    p.add_argument("--base-shape", default="d384_L6_fm4", help="shape the feature stage layers onto")
    p.add_argument("--features", nargs="+", default=None, help="subset of FEATURES to emit")
    p.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="subset of SHAPES to emit; default all",
    )
    args = p.parse_args()
    out = pathlib.Path(args.out)
    names = args.shapes or list(SHAPES)

    if args.stage == "lr":
        # Caution 3: the per-arm LR sweep is load-bearing, not diligence. Run it on the extremes
        # of the width/depth span -- if the optimum does not move between them it does not move
        # in the middle either, and one shared LR is defensible for the shape sweep.
        # 1e-1 is in the grid because the first pass found the wide-shallow arm's optimum at the
        # grid's own top edge (3e-2), which is not a measurement of the optimum but of the grid.
        # An arm whose best LR sits on a boundary is under-tuned, and under-tuning the wide arms
        # is exactly how a width/depth sweep talks itself into "depth wins".
        for name in names:
            for lr in (3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1):
                write(out, f"{name}_lr{lr:g}", steps=args.steps, lr=lr, **SHAPES[name])
    elif args.stage == "shape":
        for name in names:
            write(out, name, steps=args.steps, lr=args.lr or BEST_LR[name], **SHAPES[name])
        # A repeat of one arm at a different seed, to establish this setup's own noise floor
        # rather than borrow the 0.008 measured on loop-vs-depth's different configuration
        # (caution 5). Without it the top arms, which sit 0.002 apart at 1500 steps, cannot be
        # ranked at all.
        if args.repeat:
            write(
                out, f"{args.repeat}_seed43", steps=args.steps,
                lr=args.lr or BEST_LR[args.repeat], seed=43, **SHAPES[args.repeat],
            )
    elif args.stage == "feature":
        base = SHAPES[args.base_shape]
        # The baseline is re-run inside this stage rather than carried over from the shape sweep,
        # so every arm on the page shares a max_steps and a schedule (caution 6).
        write(out, "base", steps=args.steps, lr=args.lr or BEST_LR[args.base_shape], **base)
        for name, spec in FEATURES.items():
            if args.features and name not in args.features:
                continue
            shape = {**base, **{k: v for k, v in spec.items() if k != "extra"}}
            n_heads = shape["d_model"] // 64
            extra = spec["extra"].format(n_kv_heads=max(n_heads // 2, 1))
            write(out, name, steps=args.steps, lr=args.lr or BEST_LR[args.base_shape], extra=extra, **shape)

    if args.stage == "vocab":
        for name, spec in VOCAB_ARMS.items():
            spec = dict(spec)
            write(
                out, name, steps=args.steps, lr=args.lr or 1.0e-2,
                tokenizer=spec.pop("tokenizer"), extra=spec.pop("extra", ""), **spec,
            )

    print(f"wrote {len(list(out.glob('*.yaml')))} configs to {out}")


if __name__ == "__main__":
    main()
