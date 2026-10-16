"""radiance-posttrain: a whole post-training chain — SFT, DPO, RLAIF rounds, judge eval — from one YAML.

    radiance-posttrain --config configs/post_training/pipeline.yaml --set base=<annealed base.pt>

A pipeline file lists stages, each naming one of the ordinary stage configs (sft.yaml,
dpo_ultrafeedback.yaml, ...) and the stage whose final checkpoint it starts `from`. This module
only does the hand-off docs/post-training.md otherwise does by hand: it builds each stage's `--set`
overrides (train.init_from, train.output_dir, run_name) and runs it. A stage with a `pairs:` block
is an on-policy RLAIF round: `radiance-rlaif pairs` from its starting policy first, then DPO on the
pairs. An `eval:` block runs `radiance-rlaif eval` over finished stages at the end.

Each step is its own `radiance-train` / `radiance-rlaif` subprocess rather than a call into
train(): a stage then starts with a clean CUDA allocator, no torch.compile caches holding the last
stage's graphs, and fresh W&B state — the same process a hand-run chain would get.

Re-running the same command continues the chain. A stage whose output_dir already holds a
checkpoint at its max_steps is skipped; an interrupted one resumes (train.resume_from: auto); an
RLAIF round whose pairs manifest exists does not resample. A stage that *ends* short of max_steps
(train() stops early on an unrecoverable OOM) stops the pipeline, because the next stage would
otherwise silently fine-tune from a half-trained checkpoint.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from radiance.checkpointing import _step_checkpoints
from radiance.config import load_config_with_overrides
from radiance.sft_data import RLAIF_MANIFEST_NAME

_PIPELINE_KEYS = {"base", "output_dir", "run_name", "set", "stages", "eval"}
_STAGE_KEYS = {"name", "config", "from", "set", "pairs"}
_EVAL_KEYS = {"baseline", "compare", "args"}
# Pipeline fields `--set` can assign directly; any other key is a stage override (see load_pipeline).
_CLI_PIPELINE_FIELDS = {"base", "output_dir", "run_name"}
# Flags the pipeline itself passes to radiance-rlaif, so a `pairs:`/`eval.args` block can't.
_RLAIF_RESERVED = {"checkpoint", "output"}


@dataclass
class Stage:
    name: str
    config: Path
    source: str  # "base", or an earlier stage's name
    overrides: list[str]
    pairs: dict | None  # radiance-rlaif pairs flags; set makes the stage an RLAIF round


@dataclass
class EvalSpec:
    baseline: str
    compare: list[str]
    args: dict


@dataclass
class Pipeline:
    base: str | None
    output_dir: Path
    run_name: str
    overrides: list[str]  # applied to every stage, before the stage's own
    stages: list[Stage]
    eval: EvalSpec | None

    def stage_dir(self, name: str) -> Path:
        return self.output_dir / name


def _check_keys(where: str, raw: dict, allowed: set[str]) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; expected some of {sorted(allowed)}")


def _overrides_list(where: str, value) -> list[str]:
    value = value or []
    if not isinstance(value, list) or not all(isinstance(v, str) and "=" in v for v in value):
        raise ValueError(f"{where}: `set` must be a list of 'section.field=value' strings, got {value!r}")
    return list(value)


def load_pipeline(path: str, assignments: list[str] = ()) -> Pipeline:
    """Parse and validate a pipeline YAML, with `--set` assignments applied first.

    An assignment to `base`, `output_dir` or `run_name` sets that pipeline field. One whose first
    component is a stage name (`sft.train.max_steps=10`) is appended to that stage's overrides.
    Anything else (`wandb.mode=offline`) is appended to the overrides every stage gets.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    _check_keys(path, raw, _PIPELINE_KEYS)
    raw_stages = raw.get("stages") or []
    if not raw_stages:
        raise ValueError(f"{path}: `stages` is empty")
    stage_names = [s.get("name") for s in raw_stages if isinstance(s, dict)]

    global_overrides = _overrides_list(path, raw.get("set"))
    stage_cli: dict[str, list[str]] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"--set expects key=value, got {assignment!r}")
        key, _, value = assignment.partition("=")
        head, _, rest = key.strip().partition(".")
        if head in _CLI_PIPELINE_FIELDS and not rest:
            raw[head] = value  # verbatim: these are all paths/names
        elif head in stage_names and rest:
            stage_cli.setdefault(head, []).append(f"{rest}={value}")
        else:
            global_overrides.append(assignment)

    config_dir = Path(path).parent
    stages: list[Stage] = []
    for i, s in enumerate(raw_stages):
        where = f"{path}: stages[{i}]"
        if not isinstance(s, dict):
            raise ValueError(f"{where}: expected a mapping, got {s!r}")
        _check_keys(where, s, _STAGE_KEYS)
        for required in ("name", "config", "from"):
            if not s.get(required):
                raise ValueError(f"{where}: `{required}` is required")
        name = s["name"]
        if name == "base" or name in (st.name for st in stages):
            raise ValueError(f"{where}: stage name {name!r} is reserved or already used")
        if s["from"] != "base" and s["from"] not in (st.name for st in stages):
            raise ValueError(f"{where}: `from: {s['from']}` must be `base` or an earlier stage's name")
        config = Path(s["config"])
        if not config.is_absolute():
            config = config_dir / config  # relative to the pipeline file, like data.dataset_mix
        if not config.exists():
            raise ValueError(f"{where}: config {config} does not exist")
        pairs = s.get("pairs")
        if pairs is not None:
            if not isinstance(pairs, dict):
                raise ValueError(f"{where}: `pairs` must be a mapping of radiance-rlaif pairs flags (or {{}})")
            if _RLAIF_RESERVED & set(pairs):
                raise ValueError(f"{where}: `pairs` cannot set {sorted(_RLAIF_RESERVED)}; the pipeline does")
        stages.append(
            Stage(
                name=name,
                config=config,
                source=s["from"],
                overrides=_overrides_list(where, s.get("set")) + stage_cli.get(name, []),
                pairs=pairs,
            )
        )

    eval_spec = None
    if raw.get("eval"):
        e = raw["eval"]
        _check_keys(f"{path}: eval", e, _EVAL_KEYS)
        compare = e.get("compare") or []
        for ref in [e.get("baseline"), *compare]:
            if ref not in stage_names:
                raise ValueError(f"{path}: eval refers to {ref!r}, which is not a stage name")
        if not compare:
            raise ValueError(f"{path}: eval.compare is empty")
        args = e.get("args") or {}
        if _RLAIF_RESERVED & set(args):
            raise ValueError(f"{path}: eval.args cannot set {sorted(_RLAIF_RESERVED)}; the pipeline does")
        eval_spec = EvalSpec(baseline=e["baseline"], compare=list(compare), args=args)

    return Pipeline(
        base=raw.get("base"),
        output_dir=Path(raw.get("output_dir") or "checkpoints/post_training"),
        run_name=raw.get("run_name") or "post-training",
        overrides=global_overrides,
        stages=stages,
        eval=eval_spec,
    )


def _flags(options: dict) -> list[str]:
    """{max_prompts: 8000} -> ["--max-prompts", "8000"], radiance-rlaif's own argparse does the typing."""
    out = []
    for key, value in options.items():
        out += [f"--{key.replace('_', '-')}", str(value)]
    return out


def stage_overrides(pipeline: Pipeline, stage: Stage, init_from: Path) -> list[str]:
    """Every `--set` a stage runs with: the pipeline's, then the stage's own, then the hand-off.

    The hand-off goes last so it wins (apply_overrides is sequential). It is the pipeline's job,
    not a stage's: a stage `set` naming its own output_dir would break skip/resume detection.
    """
    stage_dir = pipeline.stage_dir(stage.name)
    overrides = [
        *pipeline.overrides,
        *stage.overrides,
        f"train.init_from={init_from}",
        f"train.output_dir={stage_dir}",
        "train.resume_from=auto",
        f"run_name={pipeline.run_name}-{stage.name}",
    ]
    if stage.pairs is not None:
        overrides.append(f"dpo.dataset={stage_dir / 'pairs'}")
    return overrides


def finished_checkpoint(stage_dir: Path, max_steps: int) -> Path | None:
    """The stage's final checkpoint, if it reached max_steps; None if it is absent or short."""
    found = _step_checkpoints(stage_dir) if stage_dir.exists() else []
    if found and found[-1][0] >= max_steps:
        return found[-1][1]
    return None


def _run(argv: list[str]) -> None:
    """One step as a subprocess (see the module docstring for why not in-process)."""
    print(f"[radiance] posttrain: $ {' '.join(argv[2:])}", flush=True)
    result = subprocess.run([sys.executable, *argv])
    if result.returncode != 0:
        raise SystemExit(f"[radiance] posttrain: step failed with exit code {result.returncode}; fix and re-run to continue")


def run_pipeline(pipeline: Pipeline, dry_run: bool = False, hud: bool = True) -> dict[str, Path]:
    """Run (or, with dry_run, print the plan for) every stage, then the eval. Returns each finished
    stage's final checkpoint, keyed by stage name."""
    if pipeline.base is None:
        raise ValueError("pipeline `base` is unset; pass --set base=<annealed base checkpoint>")
    if not Path(pipeline.base).exists() and not dry_run:
        raise FileNotFoundError(f"base checkpoint {pipeline.base} does not exist")
    finished: dict[str, Path] = {"base": Path(pipeline.base)}

    for stage in pipeline.stages:
        stage_dir = pipeline.stage_dir(stage.name)
        label = f"stage {stage.name!r} ({stage.config.name}, from {stage.source})"
        if stage.source not in finished:
            # Only reachable in a dry run: a real run stops at the first unfinished stage.
            print(f"[radiance] posttrain: {label}: waits on {stage.source!r}")
            continue
        init_from = finished[stage.source]
        if dry_run and not init_from.exists():
            # inherit_model_config reads init_from's config, so a missing base can't be planned past.
            print(f"[radiance] posttrain: {label}: base checkpoint {init_from} does not exist")
            continue
        overrides = stage_overrides(pipeline, stage, init_from)
        max_steps = load_config_with_overrides(str(stage.config), overrides).train.max_steps

        done = finished_checkpoint(stage_dir, max_steps)
        if done is not None:
            print(f"[radiance] posttrain: {label}: already finished at {done}, skipping")
            finished[stage.name] = done
            continue
        if dry_run:
            print(f"[radiance] posttrain: {label}: would run {max_steps} steps into {stage_dir}")
            continue

        if stage.pairs is not None:
            pairs_dir = stage_dir / "pairs"
            if (pairs_dir / RLAIF_MANIFEST_NAME).exists():
                print(f"[radiance] posttrain: {label}: reusing pairs in {pairs_dir}")
            else:
                # The manifest is written last, so a directory without one is a crashed `pairs` run.
                if pairs_dir.exists():
                    print(f"[radiance] posttrain: removing incomplete pairs directory {pairs_dir}")
                    shutil.rmtree(pairs_dir)
                _run(["-m", "radiance.rlaif", "pairs", "--checkpoint", str(init_from),
                      "--output", str(pairs_dir), *_flags(stage.pairs)])

        argv = ["-m", "radiance.train", "--config", str(stage.config)]
        for o in overrides:
            argv += ["--set", o]
        if not hud:
            argv.append("--no-hud")
        _run(argv)

        done = finished_checkpoint(stage_dir, max_steps)
        if done is None:
            raise SystemExit(
                f"[radiance] posttrain: {label} exited without a checkpoint at step {max_steps} in "
                f"{stage_dir} (an early stop, e.g. OOM). Not continuing from a half-trained stage; "
                "fix the cause and re-run to resume it."
            )
        finished[stage.name] = done

    if pipeline.eval is not None:
        report = pipeline.output_dir / "eval.json"
        names = [pipeline.eval.baseline, *pipeline.eval.compare]
        if report.exists():
            print(f"[radiance] posttrain: eval report {report} exists, skipping (delete it to re-score)")
        elif not all(n in finished for n in names):
            print(f"[radiance] posttrain: eval of {names} waits on unfinished stages")
        elif not dry_run:
            argv = ["-m", "radiance.rlaif", "eval", "--output", str(report)]
            for n in names:
                argv += ["--checkpoint", str(finished[n])]
            _run(argv + _flags(pipeline.eval.args))

    if not dry_run:
        print("[radiance] posttrain: done")
        for name, path in finished.items():
            if name != "base":
                print(f"  {name}: {path}")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a post-training pipeline (SFT -> DPO / RLAIF -> eval) from one YAML.")
    parser.add_argument("--config", required=True, help="pipeline YAML, e.g. configs/post_training/pipeline.yaml")
    parser.add_argument(
        "--set",
        nargs="+",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="base=/output_dir=/run_name= set the pipeline field; <stage>.section.field=value overrides one "
        "stage; any other section.field=value overrides every stage (e.g. wandb.mode=offline)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would run and what is already done")
    parser.add_argument("--no-hud", action="store_true", help="pass --no-hud to each radiance-train stage")
    args = parser.parse_args()
    run_pipeline(load_pipeline(args.config, args.set), dry_run=args.dry_run, hud=not args.no_hud)


if __name__ == "__main__":
    main()
