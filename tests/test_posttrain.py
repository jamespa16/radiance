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

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "post_training" / "pipeline.yaml"


def _base(tmp_path) -> Path:
    model_cfg = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=64)
    cfg = Config(data=DataConfig(tokenizer="my-tok"), model=model_cfg, train=TrainConfig(device="cpu", compile=False))
    model = DenseTransformer(model_cfg, vocab_size=64)
    opt = build_optimizer(model, cfg, "cpu")
    path = tmp_path / "base.pt"
    save_checkpoint(path, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    return path


class _FakeRunner:
    """Stands in for posttrain._run: records argv, and leaves behind what the real step would."""

    def __init__(self, base: Path, short: set[str] = frozenset()):
        self.base, self.short, self.calls = base, short, []

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
            step = cfg.train.max_steps // 2 if out.name in self.short else cfg.train.max_steps
            shutil.copy(self.base, out / f"step_{step}.pt")

    def stages_trained(self):
        return [
            Path(next(a.split("=", 1)[1] for a in c if a.startswith("train.output_dir="))).name
            for c in self.calls
            if c[1] == "radiance.train"
        ]


def test_shipped_pipeline_loads_and_routes_cli_overrides():
    p = posttrain.load_pipeline(
        str(SHIPPED), ["base=/x/base.pt", "wandb.mode=offline", "sft.train.max_steps=7", "run_name=r2"]
    )
    assert p.base == "/x/base.pt" and p.run_name == "r2"
    assert [s.name for s in p.stages] == ["sft", "dpo_ultrafeedback", "rlaif_round1"]
    assert all(s.config.exists() for s in p.stages)
    assert p.overrides == ["wandb.mode=offline"]
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
        ({"stages": [{"name": "a", "config": "sft.yaml", "from": "base"}], "eval": {"baseline": "a", "compare": ["z"]}}, "not a stage"),
    ],
)
def test_pipeline_validation(tmp_path, raw, match):
    shutil.copy(SHIPPED.parent / "sft.yaml", tmp_path / "sft.yaml")
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


def test_incomplete_pairs_directory_is_resampled(tmp_path, monkeypatch):
    base = _base(tmp_path)
    out = tmp_path / "out"
    for name in ("sft", "dpo_ultrafeedback"):  # already done
        (out / name).mkdir(parents=True)
    shutil.copy(base, out / "sft" / "step_3000.pt")
    shutil.copy(base, out / "dpo_ultrafeedback" / "step_1000.pt")
    (out / "rlaif_round1" / "pairs").mkdir(parents=True)  # a crashed `pairs`: no manifest
    (out / "rlaif_round1" / "pairs" / "junk").write_text("x")
    runner = _FakeRunner(base)
    monkeypatch.setattr(posttrain, "_run", runner)
    posttrain.run_pipeline(posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={out}"]))
    assert runner.stages_trained() == ["rlaif_round1"]
    assert not (out / "rlaif_round1" / "pairs" / "junk").exists()


def test_dry_run_runs_nothing(tmp_path, monkeypatch, capsys):
    base = _base(tmp_path)
    runner = _FakeRunner(base)
    monkeypatch.setattr(posttrain, "_run", runner)
    p = posttrain.load_pipeline(str(SHIPPED), [f"base={base}", f"output_dir={tmp_path / 'out'}"])
    posttrain.run_pipeline(p, dry_run=True)
    assert runner.calls == []
    text = capsys.readouterr().out
    assert "would run 3000 steps" in text and "waits on 'sft'" in text
