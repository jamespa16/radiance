"""Export a training checkpoint's weights to safetensors, for consumers outside this repo.

A `radiance-train` checkpoint is a pickled dict (`torch.save`) holding the optimizer moments, the
scheduler, the step, and a **pickled `Config` object**. That is exactly right for resuming a run
and exactly wrong for handing the model to anything else: unpickling it executes arbitrary code
and needs `radiance` importable just to read the tensor shapes, so a checkpoint can only be opened
by the repo that wrote it.

This writes the parts that are actually a *model* into a directory anyone can read:

    model.safetensors   the weights, zero-copy mmap-able, no pickle
    config.json         the full Config as plain JSON, plus step / vocab_size / parameter counts

Deliberately not exported: optimizer moments, scheduler and scaler state. They are several times
the size of the weights, mean nothing outside this codebase's parameter groups, and an export is
not a resume point — keep the `.pt` for that.

Two things need care, and both are handled here rather than left to the caller:

**Weight tying.** `token_emb.weight` and `lm_head.weight` are the same tensor (transformer.py:266),
and safetensors refuses to serialise two names sharing storage — it stores raw buffers, so the
alias would silently become two independent copies on load, doubling the file and letting a
consumer train them apart. The duplicate name is dropped and recorded in the file's metadata under
`shared_tensors`, so `load_export` can restore it and a foreign consumer can see the tie exists.

**dtype.** `--dtype` casts floating-point tensors on the way out (integer buffers such as MoE's
expert counters are left alone). A `native_bf16` run is already bf16 on disk; this matters for
exporting an fp32 checkpoint at half the size, and the choice is recorded in config.json.

Usage:

    radiance-export --checkpoint checkpoints/tinystories/step_1000.pt --output export/tinystories
    radiance-export --checkpoint ... --output ... --dtype bf16 --tokenizer
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from radiance.config import (
    Config,
    DataConfig,
    DPOConfig,
    ModelConfig,
    SFTConfig,
    TrainConfig,
    WandbConfig,
)
from radiance.model import DenseTransformer, cast_params_to_native_bf16, checkpoint_vocab_size

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

# safetensors metadata values must be strings, so structured entries are JSON-encoded into one.
EXPORT_FORMAT_VERSION = "1"

DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _dedupe_shared(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Split a state dict into (tensors to write, alias name -> name it aliases).

    Keyed on `data_ptr()` rather than `is` because tying assigns the same `nn.Parameter` object,
    but a checkpoint round-trips through `torch.save`/`torch.load`, which rebuilds tied entries as
    distinct Python objects over one shared storage. The *first* name wins, and state dicts
    preserve module registration order, so `token_emb.weight` (registered at transformer.py:216)
    is what survives and `lm_head.weight` becomes the alias — the useful way round, since the
    embedding is the name a foreign consumer looks for.
    """
    kept: dict[str, torch.Tensor] = {}
    by_storage: dict[tuple[int, int], str] = {}
    aliases: dict[str, str] = {}
    for name, tensor in state.items():
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if key in by_storage:
            aliases[name] = by_storage[key]
            continue
        by_storage[key] = name
        kept[name] = tensor
    return kept, aliases


def _cast(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Cast floating-point tensors to `dtype`; leave everything else alone.

    A state dict is not uniformly float: MoE's `expert_bias` is float but its load counters and
    any bookkeeping buffers are not, and casting an integer buffer to bf16 would corrupt counts
    that happen to exceed bf16's 8-bit mantissa. `.contiguous()` runs regardless of dtype because
    safetensors stores a flat buffer and refuses a non-contiguous tensor.
    """
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor.contiguous()


def _summarize_dtype(tensors: dict[str, torch.Tensor]) -> str:
    """The one dtype every exported tensor shares, or "mixed"."""
    dtypes = {str(t.dtype).removeprefix("torch.") for t in tensors.values()}
    return dtypes.pop() if len(dtypes) == 1 else "mixed"


def _config_to_json(cfg: Config) -> dict[str, Any]:
    """`Config` as plain JSON-able data.

    `asdict` alone is not enough: the schema holds `list[int] | None` and `float('inf')`-free
    numerics that JSON handles, but also `torch.dtype`-adjacent strings and `Path`-like values
    from resolved fields (`data.dataset_mix` is stored resolved by `load_config`). Anything JSON
    cannot represent is stringified rather than dropped, so the export stays readable instead of
    silently losing a field.
    """

    def coerce(value):
        if isinstance(value, dict):
            return {k: coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [coerce(v) for v in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    return coerce(asdict(cfg))


def _config_from_json(raw: dict[str, Any]) -> Config:
    """Rebuild a `Config` from `_config_to_json`'s output.

    Unknown keys are ignored and missing ones fall back to the field's current default, which is
    the same schema-drift contract `load_config` gives a YAML file and `Config.__setstate__` gives
    an old pickle: an export written by a different revision of this repo must still load, with
    any field that revision didn't have at today's default.
    """
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "wandb": WandbConfig,
        "sft": SFTConfig,
        "dpo": DPOConfig,
    }
    kwargs: dict[str, Any] = {"run_name": raw.get("run_name", Config.run_name)}
    for section, cls in sections.items():
        known = {f.name for f in fields(cls)}
        kwargs[section] = cls(**{k: v for k, v in raw.get(section, {}).items() if k in known})
    return Config(**kwargs)


def export_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    dtype: torch.dtype | None = None,
    tokenizer: bool = False,
) -> dict[str, Any]:
    """Write `checkpoint`'s weights + config into `output_dir` as safetensors. Returns the summary
    the CLI prints (also the body of `config.json`, minus the config itself).

    `map_location="cpu"` throughout: an export is a file-format conversion, and there is no reason
    for it to contend for a GPU that a training run is probably using.
    """
    output_dir = Path(output_dir)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    state: dict[str, torch.Tensor] = ckpt["model"]

    tensors, aliases = _dedupe_shared(state)
    tensors = {name: _cast(t, dtype) for name, t in tensors.items()}

    summary = {
        "format_version": EXPORT_FORMAT_VERSION,
        "step": ckpt.get("step"),
        "run_name": cfg.run_name,
        "vocab_size": checkpoint_vocab_size(ckpt),
        # Counted over the deduped tensors, so a tied embedding is counted once — the same
        # convention as DenseTransformer.num_parameters() and radiance-size's `total`.
        "total_parameters": sum(t.numel() for t in tensors.values()),
        # "mixed" rather than the first tensor's dtype: without --dtype the export inherits
        # whatever the checkpoint held, and an fp4_linear run's state dict genuinely is not one
        # dtype. Naming the first tensor's would be a plausible-looking lie.
        "dtype": _summarize_dtype(tensors),
        "shared_tensors": aliases,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output_dir / WEIGHTS_NAME,
        # "format": "pt" is what safetensors' own framework detection reads; the rest duplicates
        # config.json so the weights file alone is still self-describing (a consumer mmap-ing it
        # without the directory can still recover the tie and the step it came from).
        metadata={"format": "pt", **{k: json.dumps(v) for k, v in summary.items()}},
    )
    with open(output_dir / CONFIG_NAME, "w") as f:
        json.dump({**summary, "config": _config_to_json(cfg)}, f, indent=2)
        f.write("\n")

    if tokenizer:
        # Opt-in, and last: build_tokenizer hits the HF hub (or a warm cache), so the weights and
        # config are already on disk before anything can fail on a network error. Imported here
        # rather than at module scope to keep the offline path free of the datasets/transformers
        # import chain.
        from radiance.data import build_tokenizer

        build_tokenizer(cfg).save_pretrained(output_dir)

    return summary


def load_export(output_dir: str | Path, device: str = "cpu") -> tuple[DenseTransformer, Config]:
    """Rebuild a `DenseTransformer` + `Config` from an export directory — the inverse of
    `export_checkpoint`, and what makes the export verifiable rather than merely written.

    Mirrors `model.load_transformer_from_checkpoint`, including its `native_bf16` ordering
    subtlety: the model's storage dtype is set *before* `load_state_dict`, because `copy_` keeps
    the destination's dtype and would otherwise upcast a bf16 export back to fp32.
    """
    output_dir = Path(output_dir)
    with open(output_dir / CONFIG_NAME) as f:
        meta = json.load(f)
    cfg = _config_from_json(meta["config"])

    state = load_file(str(output_dir / WEIGHTS_NAME), device=device)
    # Re-tie before load_state_dict so it can stay strict: the alias was dropped at write time,
    # and a strict load would otherwise report lm_head.weight as a missing key.
    for alias, source in meta.get("shared_tensors", {}).items():
        state[alias] = state[source]

    model = DenseTransformer(cfg.model, vocab_size=meta["vocab_size"])
    if cfg.train.native_bf16:
        cast_params_to_native_bf16(model)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint's weights to safetensors.")
    parser.add_argument("--checkpoint", required=True, help="path to a radiance-train .pt checkpoint")
    parser.add_argument("--output", required=True, help="directory to write model.safetensors + config.json into")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default=None,
        help="cast floating-point tensors on export (default: keep the checkpoint's own dtype)",
    )
    parser.add_argument(
        "--tokenizer",
        action="store_true",
        help="also save the config's HF tokenizer into the output directory (needs the hub or a warm cache)",
    )
    args = parser.parse_args()

    summary = export_checkpoint(
        args.checkpoint,
        args.output,
        dtype=DTYPES[args.dtype] if args.dtype else None,
        tokenizer=args.tokenizer,
    )
    out = Path(args.output)
    print(f"wrote {out / WEIGHTS_NAME} and {out / CONFIG_NAME}")
    print(
        f"  run={summary['run_name']} step={summary['step']} "
        f"vocab={summary['vocab_size']:,} dtype={summary['dtype']}"
    )
    print(f"  params  {summary['total_parameters'] / 1e6:8.2f}M  ({summary['total_parameters']:,})")
    for alias, source in summary["shared_tensors"].items():
        print(f"  tied    {alias} -> {source} (stored once)")


if __name__ == "__main__":
    main()
