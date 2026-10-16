"""radiance-posttrain: pipeline parsing, the stage hand-off, and skip/resume/stop behavior.

The steps themselves are subprocesses, faked here by `_FakeRunner`, which writes what each real step
would leave on disk. Everything else — loading the shipped stage configs through
inherit_model_config against a real base checkpoint — is the real code path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
import yaml

from radiance import posttrain
from radiance.checkpointing import save_checkpoint
from radiance.config import Config, DataConfig, ModelConfig, TrainConfig, load_config_with_overrides
from radiance.model import DenseTransformer
from radiance.optim import build_optimizer
from radiance.sft_data import RLAIF_MANIFEST_NAME
from radiance.train import build_lr_scheduler

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "post_training" / "pipelines" / "default.yaml"


def _base(tmp_path) -> Path:
    model_cfg = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=64)
    cfg = Config(data=DataConfig(tokenizer="my-tok"), model=model_cfg, train=TrainConfig(device="cpu", compile=False))
    model = DenseTransformer(model_cfg, vocab_size=64)
    opt = build_optimizer(model, cfg, "cpu")
    path = tmp_path / "base.pt"
    save_checkpoint(path, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    return path


def _write_checkpoint(base: Path, path: Path, cfg: Config) -> None:
    """The base's weights under `cfg`, as train() would save them: the config a checkpoint embeds
    is the one it was trained with, max_steps included."""
    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    ckpt["config"] = cfg
    torch.save(ckpt, path)


class _FakeRunner:
    """Stands in for posttrain._run: records argv, and leaves behind what the real step would.
    `short` stages stop halfway; `derived` maps a stage to the max_steps train() would derive
    (tokens_per_param) in place of the config's."""

    def __init__(self, base: Path, short: set[str] = frozenset(), derived: dict[str, int] | None = None):
        self.base, self.short, self.derived, self.calls = base, short, derived or {}, []

    def __call__(self, argv):
        self.calls.append(argv)
        module, rest = argv[1], argv[2:]
        if module == "radiance.rlaif" and rest[0] == "pairs":
            out = Path(rest[rest.index("--output") + 1])
            out.mkdir(parents=True)
            (out / RLAIF_MANIFEST_NAME).write_text("{}")
        elif module == "radiance.rlaif":
            Path(rest[rest.index("--output") + 1]).write_text("{}")
        else:
            sets = [rest[i + 1] for i, a in enumerate(rest) if a == "--set"]
            cfg = load_config_with_overrides(rest[rest.index("--config") + 1], sets)
            out = Path(cfg.train.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            cfg.train.max_steps = self.derived.get(out.name, cfg.train.max_steps)
            step = cfg.train.max_steps // 2 if out.name in self.short else cfg.train.max_steps
            _write_checkpoint(self.base, out / f"step_{step}.pt", cfg)

    def stages_trained(self):
        return [
            Path(next(a.split("=", 1)[1] for a in c if a.startswith("train.output_dir="))).name
            for c in self.calls
            if c[1] == "radiance.train"
        ]


def test_shipped_pipeline_loads_and_routes_cli_overrides():
    p = posttrain.load_pipeline(
        str(SHIPPED),
        ["base=/x/base.pt", "wandb.mode=offline", "sft.train.max_steps=7", "run_name=r2", "sft.seq_len=128"],
    )
    assert p.base == "/x/base.pt" and p.run_name == "r2"
    assert [s.name for s in p.stages] == ["sft", "dpo_ultrafeedback", "rlaif_round1"]
    assert all(s.config.exists() for s in p.stages)
    # Two components is a Config key even when it starts with a stage's name: `sft` the section.
    assert p.overrides == ["wandb.mode=offline", "sft.seq_len=128"]
    assert p.stages[0].overrides == ["train.max_steps=7"]
    assert p.stages[1].overrides == []
    assert p.stages[2].pairs["max_prompts"] == 8000
    assert p.eval.baseline == "sft" and p.eval.compare == ["dpo_ultrafeedback", "rlaif_round1"]


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"stages": []}, "empty"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "b"}]}, "earlier stage"),
        ({"stages": [{"name": "base", "config": "sft.yaml", "from": "base"}]}, "reserved"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "lr": 1}]}, "unknown key"),
        ({"stages": [{"name": "a", "config": "nope.yaml", "from": "base"}]}, "does not exist"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "pairs": {"output": "x"}}]}, "cannot set"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "pairs": {"subset": None}}]}, "null"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "pairs": {"seed": [1]}}]}, "single"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "pairs": {"max_promts": 8}}]}, "rejects"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "set": ["trian.lr=1"]}]}, "unknown section"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "set": ["train.lrr=1"]}]}, "no field"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base", "set": ["=1"]}]}, "set"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base"}], "eval": ["a"]}, "mapping"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base"}], "eval": "a"}, "mapping"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base"}], "eval": {"baseline": "a", "compare": "a"}}, "list"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base"}], "eval": {"baseline": "a", "compare": ["a"], "args": {"judge_batch_size": "x"}}}, "rejects"),
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base"}], "eval": {"baseline": "a", "compare": ["z"]}}, "not a stage"),
    ],
)
def test_pipeline_validation(tmp_path, raw, match):
    shutil.copy(SHIPPED.parent.parent / "sft.yaml", tmp_path / "sft.yaml")
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        posttrain.load_pipeline(str(path))


def test_chain_hands_each_stage_its_source_and_resumes_by_skipping(tmp_path, monkeypatch):
    base = _base(tmp_path)
    out = tmp_path / "out"
    runner = _FakeRunner(base)
    monkeypatch.setattr(posttrain, "_run", runner)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"])

    finished = posttrain.run_pipeline(p)
    assert runner.stages_trained() == ["sft", "dpo_ultrafeedback", "rlaif_round1"]
    assert finished["sft"] == out / "sft" / "step_3000.pt"

    train_calls = [c for c in runner.calls if c[1] == "radiance.train"]
    assert f"train.init_from={base}" in train_calls[0]
    assert f"train.init_from={finished['sft']}" in train_calls[1]
    assert f"dpo.dataset={out / 'rlaif_round1' / 'pairs'}" in train_calls[2]
    pairs_call = next(c for c in runner.calls if c[2:4] == ["pairs", "--checkpoint"])
    assert pairs_call[4] == str(finished["sft"])  # the round samples from its starting policy
    eval_call = runner.calls[-1]
    assert eval_call[2] == "eval"
    assert [eval_call[i + 1] for i, a in enumerate(eval_call) if a == "--checkpoint"] == [
        str(finished[n]) for n in ("sft", "dpo_ultrafeedback", "rlaif_round1")
    ]

    # A second run finds every stage finished and the report written: nothing runs.
    runner.calls.clear()
    assert posttrain.run_pipeline(p) == finished
    assert runner.calls == []


def test_a_stage_that_ends_short_stops_the_pipeline(tmp_path, monkeypatch):
    base = _base(tmp_path)
    runner = _FakeRunner(base, short={"sft"})
    monkeypatch.setattr(posttrain, "_run", runner)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={tmp_path / 'out'}"])
    with pytest.raises(SystemExit, match="half-trained"):
        posttrain.run_pipeline(p)
    assert runner.stages_trained() == ["sft"]


def _finished_run(tmp_path, monkeypatch, extra=()):
    """A completed pipeline under tmp_path/out, and the runner that ran it (calls cleared)."""
    base = _base(tmp_path)
    runner = _FakeRunner(base)
    monkeypatch.setattr(posttrain, "_run", runner)
    out = tmp_path / "out"
    posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}", *extra]))
    runner.calls.clear()
    return base, out, runner


def test_cli_set_with_three_components_needs_a_real_stage():
    with pytest.raises(ValueError, match="no stage named 'sfx'"):
        posttrain.load_pipeline(str(SHIPPED), ["sfx.train.max_steps=1"])
    with pytest.raises(ValueError, match="unknown section"):
        posttrain.load_pipeline(str(SHIPPED), ["sft.trian.max_steps=1"])  # caught at load, not at stage 1


def test_incomplete_pairs_directory_is_resampled(tmp_path, monkeypatch):
    base, out, runner = _finished_run(tmp_path, monkeypatch)
    pairs = out / "rlaif_round1" / "pairs"
    (pairs / RLAIF_MANIFEST_NAME).unlink()  # a crashed `pairs`: no manifest
    (pairs / "junk").write_text("x")
    for ckpt in (out / "rlaif_round1").glob("step_*.pt"):
        ckpt.unlink()
    posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"]))
    assert runner.stages_trained() == ["rlaif_round1"]
    assert not (pairs / "junk").exists()


def test_finished_is_judged_by_the_max_steps_the_checkpoint_was_trained_with(tmp_path, monkeypatch):
    # tokens_per_param: train() derives max_steps at run time, so the config's 3000 never lands.
    base = _base(tmp_path)
    out = tmp_path / "out"
    runner = _FakeRunner(base, derived={"sft": 37})
    monkeypatch.setattr(posttrain, "_run", runner)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"])
    assert posttrain.run_pipeline(p)["sft"] == out / "sft" / "step_37.pt"
    # ...and a derived count above the config's is not mistaken for finished when cut short.
    runner = _FakeRunner(base, short={"sft"}, derived={"sft": 5000})
    monkeypatch.setattr(posttrain, "_run", runner)
    with pytest.raises(SystemExit, match="half-trained"):
        posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={tmp_path / 'o2'}"]))


def test_changed_overrides_refuse_to_resume_a_stage(tmp_path, monkeypatch):
    # A smoke run's checkpoints must not become the full run's resume point.
    base, out, runner = _finished_run(tmp_path, monkeypatch, ["sft.train.max_steps=100"])
    assert (out / "sft" / "step_100.pt").exists()
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"])
    with pytest.raises(SystemExit, match=r"different inputs \(config\.train\.max_steps: 100 -> 3000\)"):
        posttrain.run_pipeline(p)
    assert runner.calls == []
    # wandb and run_name don't change what trains, so they don't make a stage stale.
    posttrain.run_pipeline(posttrain.load_pipeline(
        str(SHIPPED), [f"base={base}", f"output_dir={out}", "sft.train.max_steps=100", "wandb.mode=disabled", "run_name=x"]
    ))
    assert runner.calls == []


def test_a_new_base_refuses_stages_trained_from_the_old_one(tmp_path, monkeypatch):
    base, out, runner = _finished_run(tmp_path, monkeypatch)
    new_base = tmp_path / "new_base.pt"
    shutil.copy(base, new_base)
    with pytest.raises(SystemExit, match="different inputs .*source.path"):
        posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={new_base}", f"output_dir={out}"]))
    assert runner.calls == []


def test_changed_pairs_flags_refuse_to_reuse_the_pairs(tmp_path, monkeypatch):
    base, out, runner = _finished_run(tmp_path, monkeypatch)
    raw = yaml.safe_load(SHIPPED.read_text())
    raw["stages"][2]["pairs"]["max_prompts"] = 16
    for s in raw["stages"]:
        s["config"] = str((SHIPPED.parent / s["config"]).resolve())
    edited = tmp_path / "edited.yaml"
    edited.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit, match="pairs.max_prompts: 8000 -> 16"):
        posttrain.run_pipeline(posttrain.load_pipeline(str(edited), [f"base={base}", f"output_dir={out}"]))


def test_retraining_upstream_refuses_downstream_and_rescores_eval(tmp_path, monkeypatch):
    base, out, runner = _finished_run(tmp_path, monkeypatch)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"])
    # A retrained leaf stage: it trains again, and the eval re-scores rather than reporting old numbers.
    shutil.rmtree(out / "dpo_ultrafeedback")
    posttrain.run_pipeline(p)
    assert runner.stages_trained() == ["dpo_ultrafeedback"]
    assert runner.calls[-1][2] == "eval"
    # A retrained stage others start from: they refuse, instead of keeping weights from the old one.
    runner.calls.clear()
    shutil.rmtree(out / "sft")
    with pytest.raises(SystemExit, match="dpo_ultrafeedback.*different inputs \\(source"):
        posttrain.run_pipeline(p)
    assert runner.stages_trained() == ["sft"]


def test_outputs_without_a_record_are_refused(tmp_path, monkeypatch):
    base = _base(tmp_path)
    out = tmp_path / "out"
    (out / "sft").mkdir(parents=True)
    shutil.copy(base, out / "sft" / "step_3000.pt")
    monkeypatch.setattr(posttrain, "_run", _FakeRunner(base))
    with pytest.raises(SystemExit, match=posttrain.STAGE_INPUTS_NAME):
        posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"]))


def test_dry_run_runs_nothing(tmp_path, monkeypatch, capsys):
    base = _base(tmp_path)
    runner = _FakeRunner(base)
    monkeypatch.setattr(posttrain, "_run", runner)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={tmp_path / 'out'}"])
    posttrain.run_pipeline(p, dry_run=True)
    assert runner.calls == []
    text = capsys.readouterr().out
    assert "would run 3000 steps" in text and "waits on 'sft'" in text


def test_dry_run_reports_a_resume_and_a_stale_stage(tmp_path, monkeypatch, capsys):
    base = _base(tmp_path)
    out = tmp_path / "out"
    runner = _FakeRunner(base, short={"sft"})
    monkeypatch.setattr(posttrain, "_run", runner)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"])
    with pytest.raises(SystemExit):
        posttrain.run_pipeline(p)
    capsys.readouterr()
    posttrain.run_pipeline(p, dry_run=True)
    assert "would resume from step 1500 to 3000 steps" in capsys.readouterr().out
    posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}", "sft.train.lr=1e-5"]), dry_run=True)
    assert "different inputs (config.train.lr" in capsys.readouterr().out
