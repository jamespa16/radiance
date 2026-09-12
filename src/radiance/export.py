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

Reading one back is not a separate tool: `read_checkpoint` dispatches on the path, so an export
directory is accepted anywhere a `.pt` is — `radiance-generate`, `radiance-serve`, `radiance-eval`,
`dpo.reference_checkpoint`, `train.init_from`. `train.resume_from` is the one exception and refuses
an export by name, since there are no optimizer moments in one to resume from.

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
    radiance-generate --checkpoint export/tinystories --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # model/ imports this module (see read_checkpoint), so the import is deferred
    from radiance.model import DenseTransformer

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
    # Deferred: the model package imports this module for read_checkpoint, so nothing here may
    # import it at module scope. Only the two functions that genuinely need a model do.
    from radiance.model import checkpoint_vocab_size

    output_dir = Path(output_dir)
    ckpt = read_checkpoint(checkpoint, map_location="cpu")
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


def is_export(path: str | Path) -> bool:
    """Whether `path` names an export directory (as opposed to a `.pt` checkpoint).

    Accepts the directory itself or the `model.safetensors` inside it, because both are things a
    person plausibly types after `ls`. Both files must be present: a directory holding only
    weights is not loadable (there is no config to rebuild the model from), and saying so at the
    dispatch point gives a better error than a FileNotFoundError from inside the loader.
    """
    path = Path(path)
    if path.is_file() and path.name == WEIGHTS_NAME:
        path = path.parent
    return path.is_dir() and (path / CONFIG_NAME).is_file() and (path / WEIGHTS_NAME).is_file()


def read_export(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Read an export directory into the same dict shape `save_checkpoint` writes.

    Returning a checkpoint-shaped dict rather than a model is what lets every existing consumer
    (`load_transformer_from_checkpoint`, `load_pretrained_weights`, `checkpoint_param_bytes`)
    accept an export by changing which loader they call and nothing else — the alternative, an
    export-specific branch at each site, would have four places to keep in step with the schema.

    The tie dropped at export time is restored here, as the *same tensor* under both names, so the
    dict is indistinguishable from a freshly-loaded `.pt` — including for `checkpoint_param_bytes`,
    which sums over the state dict and would otherwise price an export differently from the
    checkpoint it came from.
    """
    path = Path(path)
    if path.is_file() and path.name == WEIGHTS_NAME:
        path = path.parent
    with open(path / CONFIG_NAME) as f:
        meta = json.load(f)
    state = load_file(str(path / WEIGHTS_NAME), device=map_location)
    for alias, source in meta.get("shared_tensors", {}).items():
        state[alias] = state[source]
    return {"model": state, "config": _config_from_json(meta["config"]), "step": meta.get("step")}


def read_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load either a `.pt` checkpoint or an export directory, as a checkpoint-shaped dict.

    The single entry point every weight-loading path in the codebase goes through, so "anywhere a
    checkpoint is accepted, an export is accepted too" is one dispatch rather than a property each
    call site has to remember to preserve. What comes back from an export has no `optimizer`,
    `scheduler` or `scaler` key — an export carries weights only — which is why `train.resume_from`
    rejects one outright (checkpointing.find_resume_checkpoint) instead of resuming from a
    checkpoint with silently missing moments.
    """
    if is_export(path):
        return read_export(path, map_location=map_location)
    return torch.load(str(path), map_location=map_location, weights_only=False)


def load_export(output_dir: str | Path, device: str = "cpu") -> tuple["DenseTransformer", Config]:
    """Rebuild a `DenseTransformer` + `Config` from an export directory — the inverse of
    `export_checkpoint`.

    Thin by design: `load_transformer_from_checkpoint` accepts an export directory anywhere it
    accepts a `.pt`, so this is a named shortcut for readers who think in terms of the export
    rather than a second reconstruction path that could drift from it.
    """
    from radiance.model import load_transformer_from_checkpoint

    return load_transformer_from_checkpoint(str(output_dir), device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint's weights to safetensors.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to a radiance-train .pt checkpoint (or an existing export, to re-export it)",
    )
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
