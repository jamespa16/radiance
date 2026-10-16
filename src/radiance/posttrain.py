"""radiance-posttrain: a whole post-training chain — SFT, DPO, RLAIF rounds, judge eval — from one YAML.

    radiance-posttrain --config configs/post_training/pipelines/default.yaml --set base=<annealed base.pt>

A pipeline file lists stages, each naming one of the ordinary stage configs (sft.yaml,
dpo_ultrafeedback.yaml, ...) and the stage whose final checkpoint it starts `from`. This module
only does the hand-off docs/post-training.md otherwise does by hand: it builds each stage's `--set`
overrides (train.init_from, train.output_dir, run_name) and runs it. A stage with a `pairs:` block
is an on-policy RLAIF round: `radiance-rlaif pairs` from its starting policy first, then DPO on the
pairs. An `eval:` block runs `radiance-rlaif eval` over finished stages at the end.

Each step is its own `radiance-train` / `radiance-rlaif` subprocess rather than a call into
train(): a stage then starts with a clean CUDA allocator, no torch.compile caches holding the last
stage's graphs, and fresh W&B state — the same process a hand-run chain would get.

Re-running the same command continues the chain. A stage whose latest checkpoint reached the
max_steps it was trained with (as recorded in that checkpoint, so a tokens_per_param-derived count
is the one compared) is skipped; an interrupted one resumes (train.resume_from: auto); an RLAIF
round whose pairs manifest exists does not resample. A stage that *ends* short (train() stops early
on an unrecoverable OOM) stops the pipeline, because the next stage would otherwise silently
fine-tune from a half-trained checkpoint.

"Continues" is only safe while the inputs are unchanged, so each stage directory records what it
was started from (STAGE_INPUTS_NAME: the resolved config, the source checkpoint's identity, the
`pairs` flags). A re-run whose inputs differ — a smoke run's `--set sft.train.max_steps=100`, a new
base, a retrained upstream stage — refuses that stage rather than resuming or reusing it; the fix
is to delete the directory or pick another output_dir. The eval is cheap and only reads, so it
re-scores instead whenever the checkpoints it would compare are not the ones it last scored.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from radiance.checkpointing import _step_checkpoints
from radiance.config import apply_overrides, load_config_with_overrides
from radiance.export import read_checkpoint_config
from radiance.sft_data import RLAIF_MANIFEST_NAME

_PIPELINE_KEYS = {"base", "output_dir", "run_name", "set", "stages", "eval"}
_STAGE_KEYS = {"name", "config", "from", "set", "pairs"}
_EVAL_KEYS = {"baseline", "compare", "args"}
# Pipeline fields `--set` can assign directly (see load_pipeline for how the rest are routed).
_CLI_PIPELINE_FIELDS = {"base", "output_dir", "run_name"}
# Flags the pipeline itself passes to radiance-rlaif, so a `pairs:`/`eval.args` block can't.
_RLAIF_RESERVED = {"checkpoint", "output"}
# Config sections that don't change what a stage trains, so editing them doesn't make it stale.
_INPUTS_IGNORED = {"run_name", "wandb"}
STAGE_INPUTS_NAME = "posttrain_inputs.json"
EVAL_INPUTS_NAME = "eval_inputs.json"


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


def _check_keys(where: str, raw, allowed: set[str]) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping, got {raw!r}")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; expected some of {sorted(allowed)}")


def _overrides_list(where: str, value) -> list[str]:
    value = value or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where}: `set` must be a list of 'section.field=value' strings, got {value!r}")
    return list(value)


def _flags(options: dict) -> list[str]:
    """{max_prompts: 8000} -> ["--max-prompts", "8000"]; radiance-rlaif's own argparse does the typing.

    Every radiance-rlaif option takes one scalar value, so anything else is refused rather than
    str()'d into an argv that parses: `subset: null` would otherwise arrive as the dataset config
    name "None"."""
    out = []
    for key, value in options.items():
        if value is None:
            raise ValueError(f"radiance-rlaif option {key!r} is null; leave it out to get radiance-rlaif's default")
        if isinstance(value, (bool, list, dict)):
            raise ValueError(f"radiance-rlaif option {key!r} must be a single number or string, got {value!r}")
        out += [f"--{key.replace('_', '-')}", str(value)]
    return out


def _check_rlaif_flags(where: str, command: str, options: dict) -> None:
    """Parse `options` the way the real subprocess will, so a typo or a bad value fails at load
    time instead of after the stages before it have trained."""
    from radiance.rlaif import build_parser

    if _RLAIF_RESERVED & set(options):
        raise ValueError(f"{where}: cannot set {sorted(_RLAIF_RESERVED)}; the pipeline does")
    argv = [command, "--checkpoint", "x", "--output", "x", *_flags(options)]
    try:
        build_parser().parse_args(argv)
    except SystemExit:  # argparse has already printed why to stderr
        raise ValueError(f"{where}: radiance-rlaif {command} rejects {options!r} (see the usage error above)") from None


def load_pipeline(path: str, assignments: list[str] = ()) -> Pipeline:
    """Parse and validate a pipeline YAML, with `--set` assignments applied first.

    Routing is by the key's depth, which is unambiguous because a Config key has at most two
    components (`section.field`): `base`, `output_dir` and `run_name` set that pipeline field; a
    three-component key `<stage>.section.field=value` is appended to that stage's overrides; any
    other key (`wandb.mode=offline`, `sft.seq_len=1024`) goes to every stage. Routing on the first
    component alone would let the shipped stage named `sft` swallow every `sft.*` override.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    _check_keys(path, raw, _PIPELINE_KEYS)
    raw_stages = raw.get("stages") or []
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"{path}: `stages` must be a non-empty list")
    stage_names = [s.get("name") for s in raw_stages if isinstance(s, dict)]

    global_overrides = _overrides_list(path, raw.get("set"))
    stage_cli: dict[str, list[str]] = {}
    for assignment in assignments:
        key, eq, value = assignment.partition("=")
        parts = key.strip().split(".")
        if eq and len(parts) == 1 and parts[0] in _CLI_PIPELINE_FIELDS:
            raw[parts[0]] = value  # verbatim: these are all paths/names
        elif eq and len(parts) >= 3:
            if parts[0] not in stage_names:
                raise ValueError(f"--set {key!r}: no stage named {parts[0]!r} (stages: {stage_names})")
            stage_cli.setdefault(parts[0], []).append(f"{'.'.join(parts[1:])}={value}")
        else:
            global_overrides.append(assignment)  # malformed ones are rejected with the rest below

    config_dir = Path(path).parent
    stages: list[Stage] = []
    for i, s in enumerate(raw_stages):
        where = f"{path}: stages[{i}]"
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
            _check_rlaif_flags(f"{where}: pairs", "pairs", pairs)
        overrides = _overrides_list(where, s.get("set")) + stage_cli.get(name, [])
        # The same check radiance-train applies, run now: a bad key would otherwise surface only
        # when this stage starts, after every stage before it has trained.
        try:
            apply_overrides({}, global_overrides + overrides)
        except ValueError as e:
            raise ValueError(f"{where} ({name}): {e}") from None
        stages.append(Stage(name=name, config=config, source=s["from"], overrides=overrides, pairs=pairs))

    eval_spec = None
    if raw.get("eval"):
        e = raw["eval"]
        _check_keys(f"{path}: eval", e, _EVAL_KEYS)
        compare = e.get("compare") or []
        if not isinstance(compare, list) or not compare:
            raise ValueError(f"{path}: eval.compare must be a non-empty list of stage names")
        for ref in [e.get("baseline"), *compare]:
            if ref not in stage_names:
                raise ValueError(f"{path}: eval refers to {ref!r}, which is not a stage name")
        args = e.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"{path}: eval.args must be a mapping of radiance-rlaif eval flags")
        _check_rlaif_flags(f"{path}: eval.args", "eval", args)
        eval_spec = EvalSpec(baseline=e["baseline"], compare=list(compare), args=args)

    return Pipeline(
        base=raw.get("base"),
        output_dir=Path(raw.get("output_dir") or "checkpoints/post_training"),
        run_name=raw.get("run_name") or "post-training",
        overrides=global_overrides,
        stages=stages,
        eval=eval_spec,
    )


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


def checkpoint_identity(path: Path) -> dict:
    """Which checkpoint file this is, cheaply: path, size and mtime. A retrained stage rewrites its
    final checkpoint at the same path, so the path alone would not notice. An export directory is
    identified by its files."""
    path = Path(path).resolve()
    files = sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else [path]
    return {"path": str(path), "files": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files]}


def stage_inputs(cfg, init_from: Path, pairs: dict | None) -> dict:
    """What a stage's output is a function of. JSON round-tripped so it compares equal to what
    a re-run reads back from STAGE_INPUTS_NAME."""
    config = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _INPUTS_IGNORED}
    inputs = {"source": checkpoint_identity(init_from), "config": config, "pairs": pairs}
    return json.loads(json.dumps(inputs, default=str))


def _differences(recorded, current, prefix: str = "") -> list[str]:
    if isinstance(recorded, dict) and isinstance(current, dict):
        keys = set(recorded) | set(current)
        if prefix.startswith("config"):
            # A Config field only one side has was added or removed by a code change, not by the
            # user; features default to inert (docs/config.md), so it doesn't change what trains.
            keys = set(recorded) & set(current)
        out = []
        for k in sorted(keys, key=str):
            out += _differences(recorded.get(k), current.get(k), f"{prefix}.{k}" if prefix else str(k))
        return out
    return [] if recorded == current else [f"{prefix or 'value'}: {recorded!r} -> {current!r}"]


def stale_reason(stage_dir: Path, inputs: dict) -> str | None:
    """Why the stage's existing outputs can't be resumed or reused under `inputs`; None if they can
    (or there are none). Directories with outputs but no record are refused too: nothing says what
    they were trained from."""
    has_outputs = stage_dir.exists() and (_step_checkpoints(stage_dir) or (stage_dir / "pairs").exists())
    record = stage_dir / STAGE_INPUTS_NAME
    if not record.exists():
        return f"{stage_dir} holds outputs but no {STAGE_INPUTS_NAME} recording what they came from" if has_outputs else None
    diff = _differences(json.loads(record.read_text()), inputs)
    if not diff:
        return None
    shown = "; ".join(diff[:5]) + (f"; and {len(diff) - 5} more" if len(diff) > 5 else "")
    return f"{stage_dir} was started from different inputs ({shown})"


def finished_checkpoint(stage_dir: Path) -> Path | None:
    """The stage's final checkpoint, if it reached the max_steps it was trained with; None if it
    is absent or short. Read from the checkpoint rather than the stage config: train() derives
    max_steps at run time under train.tokens_per_param and saves the derived value."""
    found = _step_checkpoints(stage_dir) if stage_dir.exists() else []
    if found and found[-1][0] >= read_checkpoint_config(found[-1][1]).train.max_steps:
        return found[-1][1]
    return None


def _run(argv: list[str]) -> None:
    """One step as a subprocess (see the module docstring for why not in-process)."""
    print(f"[radiance] posttrain: $ {' '.join(argv[2:])}", flush=True)
    result = subprocess.run([sys.executable, *argv])
    if result.returncode != 0:
        raise SystemExit(f"[radiance] posttrain: step failed with exit code {result.returncode}; fix and re-run to continue")


def _plan(cfg, stage_dir: Path) -> str:
    """The dry-run line for a stage that would run: fresh or resumed, and how far."""
    target = (
        f"tokens_per_param={cfg.train.tokens_per_param} steps (derived at run time)"
        if cfg.train.tokens_per_param is not None
        else f"{cfg.train.max_steps} steps"
    )
    found = _step_checkpoints(stage_dir) if stage_dir.exists() else []
    if found:
        return f"would resume from step {found[-1][0]} to {target} in {stage_dir}"
    return f"would run {target} into {stage_dir}"


def _run_eval(pipeline: Pipeline, finished: dict[str, Path], dry_run: bool) -> None:
    report = pipeline.output_dir / "eval.json"
    record = pipeline.output_dir / EVAL_INPUTS_NAME
    names = [pipeline.eval.baseline, *pipeline.eval.compare]
    if not all(n in finished for n in names):
        print(f"[radiance] posttrain: eval of {names} waits on unfinished stages")
        return
    inputs = {"checkpoints": [[n, checkpoint_identity(finished[n])] for n in names], "args": pipeline.eval.args}
    inputs = json.loads(json.dumps(inputs))
    if report.exists() and record.exists() and json.loads(record.read_text()) == inputs:
        print(f"[radiance] posttrain: eval report {report} is current, skipping")
        return
    if report.exists():
        print(f"[radiance] posttrain: eval report {report} scored other checkpoints or args; re-scoring")
    if dry_run:
        print(f"[radiance] posttrain: would score {names} into {report}")
        return
    record.unlink(missing_ok=True)  # written only once the new report is, below
    argv = ["-m", "radiance.rlaif", "eval", "--output", str(report)]
    for n in names:
        argv += ["--checkpoint", str(finished[n])]
    _run(argv + _flags(pipeline.eval.args))
    record.write_text(json.dumps(inputs, indent=2))


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
        cfg = load_config_with_overrides(str(stage.config), overrides)
        inputs = stage_inputs(cfg, init_from, stage.pairs)

        stale = stale_reason(stage_dir, inputs)
        if stale is not None:
            message = (
                f"[radiance] posttrain: {label}: {stale}. Not resuming or reusing it; delete {stage_dir} "
                "(or set a different output_dir) to retrain this stage from its current inputs."
            )
            if dry_run:
                print(message)
                continue
            raise SystemExit(message)

        done = finished_checkpoint(stage_dir)
        if done is not None:
            print(f"[radiance] posttrain: {label}: already finished at {done}, skipping")
            finished[stage.name] = done
            continue
        if dry_run:
            print(f"[radiance] posttrain: {label}: {_plan(cfg, stage_dir)}")
            continue

        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / STAGE_INPUTS_NAME).write_text(json.dumps(inputs, indent=2))

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

        done = finished_checkpoint(stage_dir)
        if done is None:
            raise SystemExit(
                f"[radiance] posttrain: {label} exited without reaching its max_steps in {stage_dir} "
                "(an early stop, e.g. OOM). Not continuing from a half-trained stage; fix the cause "
                "and re-run to resume it."
            )
        finished[stage.name] = done

    if pipeline.eval is not None:
        _run_eval(pipeline, finished, dry_run)

    if not dry_run:
        print("[radiance] posttrain: done")
        for name, path in finished.items():
            if name != "base":
                print(f"  {name}: {path}")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a post-training pipeline (SFT -> DPO / RLAIF -> eval) from one YAML.")
    parser.add_argument("--config", required=True, help="pipeline YAML, e.g. configs/post_training/pipelines/default.yaml")
    parser.add_argument(
        "--set",
        nargs="+",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="base=/output_dir=/run_name= set the pipeline field; <stage>.section.field=value overrides one "
        "stage; section.field=value overrides every stage (e.g. wandb.mode=offline)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would run and what is already done")
    parser.add_argument("--no-hud", action="store_true", help="pass --no-hud to each radiance-train stage")
    args = parser.parse_args()
    run_pipeline(load_pipeline(args.config, args.set), dry_run=args.dry_run, hud=not args.no_hud)


if __name__ == "__main__":
    main()
