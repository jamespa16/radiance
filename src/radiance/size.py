"""Report a config's parameter count without training it.

Sizing a model is otherwise a training run: `radiance-train` is the only thing that builds a
`DenseTransformer`, and it does so after loading a tokenizer and a dataset. That is a slow and
occasionally destructive way to answer "does this shape fit in my budget?", which is a question
worth asking dozens of times in a row while designing an A/B under a parameter cap.

So this builds the model on the `meta` device — no allocation, no data, no GPU — and prints the
same three numbers a run would report, plus the split that actually drives shape decisions:

    total     `num_parameters()`, every allocated parameter
    active    `num_active_parameters()`, what multiplies against a token (see transformer.py)
    embed     the tied token embedding + learned positional embedding
    blocks    total - embed, i.e. what the shape knobs actually buy

**Read `embed` first when working under a total-parameter cap.** It is a fixed tax paid before any
architecture decision: at `d_model: 256` with the gpt2 tokenizer it is 12.9M, over half of the
repo default's 25.5M, and it grows linearly with width while the block stack grows quadratically.
`--vocab-size` prices a smaller tokenizer against that tax without needing to train one first.

Usage:

    radiance-size --config configs/v1.yaml
    radiance-size --config configs/v1.yaml --set model.d_model=512 model.n_layers=10
    radiance-size --set model.d_model=384 --vocab-size 8192

`--set` takes `dotted.key=value` pairs parsed as YAML (so `true`, `null`, `4.0` and `8` land as
the right type), applied on top of `--config`, or on top of the dataclass defaults when no
`--config` is given. Repeating the flag or passing several pairs to one flag both work, which
makes the shape sweep a shell loop rather than a pile of throwaway YAML files.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from radiance.config import Config, load_config
from radiance.model import DenseTransformer, padded_vocab_size

# Resolving this needs the tokenizer, which needs a network fetch or a warm HF cache -- far too
# slow for a tool whose whole point is being fast enough to call in a loop. gpt2 is the repo
# default everywhere (configs/default.yaml, every config in configs/), and --vocab-size overrides
# it for anything else, so the constant buys speed without hiding a decision.
GPT2_VOCAB_SIZE = 50257


def _apply_overrides(raw: dict, assignments: list[str]) -> dict:
    """Apply `dotted.key=value` assignments to a raw config dict, parsing values as YAML."""
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"--set expects dotted.key=value, got {assignment!r}")
        key, _, value = assignment.partition("=")
        section, _, field = key.strip().partition(".")
        if not field:
            raise ValueError(f"--set key must name a section, e.g. model.d_model, got {key!r}")
        raw.setdefault(section, {})[field] = yaml.safe_load(value)
    return raw


def size_config(cfg: Config, vocab_size: int) -> dict[str, int]:
    """Build `cfg`'s model on the meta device and return its parameter breakdown."""
    padded = padded_vocab_size(vocab_size, cfg.model.vocab_pad_multiple)
    # meta means shapes without storage: no allocation, so a 500M-parameter shape prices as fast
    # as a 5M one, and nothing touches the GPU that a real run might be using.
    with torch.device("meta"):
        model = DenseTransformer(cfg.model, vocab_size=padded)
    embed = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "token_emb" in name or "pos_emb" in name or "lm_head" in name
    )
    total = model.num_parameters()
    return {
        "total": total,
        "active": model.num_active_parameters(),
        "embed": embed,
        "blocks": total - embed,
        "vocab": padded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report a config's parameter count without training it.")
    parser.add_argument("--config", help="config YAML to size; omit to size the dataclass defaults")
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="dotted.key=value overrides applied on top of --config, parsed as YAML",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=GPT2_VOCAB_SIZE,
        help=f"tokenizer vocabulary size before padding (default {GPT2_VOCAB_SIZE}, gpt2)",
    )
    args = parser.parse_args()

    if args.set:
        # Round-trip through the raw dict so overrides go through load_config's own validation
        # rather than being poked onto an already-resolved Config, where a cross-field check
        # (dtype/fp4_linear, dataset/dataset_mix) would never see them.
        raw = yaml.safe_load(open(args.config)) or {} if args.config else {}
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            yaml.safe_dump(_apply_overrides(raw, args.set), f)
            f.flush()
            cfg = load_config(f.name)
    elif args.config:
        cfg = load_config(args.config)
    else:
        cfg = Config()

    counts = size_config(cfg, args.vocab_size)
    m = cfg.model
    print(
        f"d_model={m.d_model} n_layers={m.n_layers} loop_count={m.loop_count} "
        f"ffn_mult={m.ffn_mult} ffn_depth={m.ffn_depth} use_moe={m.use_moe} vocab={counts['vocab']:,}"
    )
    for label in ("total", "active", "embed", "blocks"):
        print(f"  {label:7s} {counts[label] / 1e6:8.2f}M  ({counts[label]:,})")


if __name__ == "__main__":
    main()
