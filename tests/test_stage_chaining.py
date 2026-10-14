"""Chaining post-training stages: train.inherit_model_config and the `--set` overrides
radiance-train now shares with radiance-size."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from radiance.checkpointing import save_checkpoint
from radiance.config import Config, DataConfig, ModelConfig, TrainConfig, load_config, load_config_with_overrides
from radiance.export import export_checkpoint, read_checkpoint_config
from radiance.losses import validate_post_training_config
from radiance.model import DenseTransformer
from radiance.optim import build_optimizer
from radiance.train import build_lr_scheduler


def _save(tmp_path, model_cfg: ModelConfig, tokenizer: str = "my-tok"):
    cfg = Config(data=DataConfig(tokenizer=tokenizer), model=model_cfg, train=TrainConfig(device="cpu", compile=False))
    model = DenseTransformer(model_cfg, vocab_size=64)
    opt = build_optimizer(model, cfg, "cpu")
    path = tmp_path / "base.pt"
    save_checkpoint(path, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    return path


def _write(tmp_path, raw: dict) -> str:
    path = tmp_path / "stage.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_inherit_model_config_takes_the_checkpoint_model_with_explicit_keys_on_top(tmp_path):
    # loop_count and qk_norm are behavioral fields load_pretrained_weights does not check — the
    # ones a hand-copied model block silently loses.
    source = ModelConfig(d_model=32, head_dim=8, n_layers=3, loop_count=2, qk_norm=False, max_seq_len=64)
    ckpt = _save(tmp_path, source)
    cfg = load_config(
        _write(
            tmp_path,
            {"train": {"init_from": str(ckpt), "inherit_model_config": True}, "model": {"dropout": 0.25}},
        )
    )
    assert (cfg.model.d_model, cfg.model.n_layers, cfg.model.loop_count, cfg.model.qk_norm) == (32, 3, 2, False)
    assert cfg.model.dropout == 0.25  # the explicit key wins
    assert cfg.data.tokenizer == "my-tok"


def test_inherit_respects_an_explicit_tokenizer_and_reads_exports(tmp_path):
    ckpt = _save(tmp_path, ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=32))
    export_dir = tmp_path / "export"
    export_checkpoint(str(ckpt), str(export_dir))
    assert read_checkpoint_config(export_dir).model.d_model == 32
    cfg = load_config(
        _write(
            tmp_path,
            {
                "data": {"tokenizer": "explicit"},
                "train": {"init_from": str(export_dir), "inherit_model_config": True},
            },
        )
    )
    assert cfg.model.d_model == 32 and cfg.data.tokenizer == "explicit"


def test_run_time_model_fields_are_not_inherited(tmp_path):
    # How the source run trained — its numeric format (fp4_* comes from train.dtype: nvfp4), loss
    # weights, regularisation, and doc mask — is not architecture: a stage takes its own.
    source = ModelConfig(
        d_model=32, head_dim=8, n_layers=2, max_seq_len=32,
        fp4_linear=True, dropout=0.0, z_loss_weight=0.0, doc_attention_mask=False, grad_checkpoint=True,
    )
    cfg = Config(data=DataConfig(tokenizer="t"), model=source, train=TrainConfig(device="cpu", compile=False))
    model = DenseTransformer(ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=32), vocab_size=64)
    opt = build_optimizer(model, cfg, "cpu")
    ckpt = tmp_path / "base.pt"
    save_checkpoint(ckpt, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    raw = {"train": {"init_from": str(ckpt), "inherit_model_config": True, "dtype": "bf16"}, "sft": {"enabled": True}}
    inherited = load_config(_write(tmp_path, raw)).model
    default = ModelConfig()
    assert inherited.d_model == 32 and inherited.n_layers == 2
    for name in ("fp4_linear", "dropout", "z_loss_weight", "doc_attention_mask", "grad_checkpoint"):
        assert getattr(inherited, name) == getattr(default, name), name
    # ...unless the stage spells one out.
    raw["model"] = {"dropout": 0.0}
    assert load_config(_write(tmp_path, raw)).model.dropout == 0.0


def test_a_resumed_stage_inherits_from_its_resume_checkpoint(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    resumed = _save(tmp_path, ModelConfig(d_model=32, head_dim=8, n_layers=3, max_seq_len=32))
    resumed.rename(out / "step_5.pt")
    raw = {
        # init_from is gone (rotated away): a resume must not need it.
        "train": {"init_from": str(tmp_path / "gone.pt"), "inherit_model_config": True,
                  "resume_from": "auto", "output_dir": str(out)},
    }
    assert load_config(_write(tmp_path, raw)).model.n_layers == 3
    # A fresh run (nothing to resume yet) still reads init_from.
    raw["train"]["output_dir"] = str(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        load_config(_write(tmp_path, raw))


def test_inherit_requires_init_from(tmp_path):
    with pytest.raises(ValueError, match="requires train.init_from"):
        load_config(_write(tmp_path, {"train": {"inherit_model_config": True}}))


_STAGES = sorted((Path(__file__).parent.parent / "configs" / "post_training").glob("*.yaml"))


@pytest.mark.parametrize("stage", _STAGES, ids=lambda p: p.stem)
def test_post_training_stage_configs_load_as_documented(stage, tmp_path):
    """Each stage config, loaded the way its header says to run it (init_from via --set), resolves
    to a valid post-training config on top of the checkpoint's own architecture."""
    source = ModelConfig(d_model=32, head_dim=8, n_layers=3, max_seq_len=1024)
    ckpt = _save(tmp_path, source, tokenizer="gpt2")
    overrides = [f"train.init_from={ckpt}"]
    if "rlaif" in stage.stem:
        overrides.append(f"dpo.dataset={tmp_path / 'round1'}")
    cfg = load_config_with_overrides(str(stage), overrides)
    assert cfg.sft.enabled != cfg.dpo.enabled
    assert cfg.model.d_model == 32 and cfg.data.tokenizer == "gpt2"
    assert cfg.resolved_seq_len <= cfg.model.max_seq_len
    validate_post_training_config(cfg)
    if cfg.dpo.enabled:
        assert cfg.resolved_reference_checkpoint == str(ckpt)
        # The inherited default (0.1 here) must be overridden: see losses.note_dpo_dropout.
        assert cfg.model.dropout == 0.0
    with pytest.raises(ValueError, match="requires train.init_from"):
        load_config(str(stage))


def test_overrides_are_typed_and_validated(tmp_path):
    path = _write(tmp_path, {"train": {"max_steps": 10}, "dpo": {"enabled": True}})
    cfg = load_config_with_overrides(path, ["train.max_steps=200", "dpo.beta=0.05", "train.init_from=a.pt", "dpo.eval_split=null"])
    assert cfg.train.max_steps == 200 and cfg.dpo.beta == 0.05 and cfg.dpo.eval_split is None
    assert cfg.resolved_reference_checkpoint == "a.pt"
    assert load_config_with_overrides(path, []).train.max_steps == 10
    with pytest.raises(ValueError, match="dotted.key=value"):
        load_config_with_overrides(path, ["train.max_steps"])
    # Overrides go through load_config's own cross-field checks.
    with pytest.raises(ValueError, match="contradict"):
        load_config_with_overrides(path, ["train.dtype=nvfp4", "model.fp4_linear=false"])
    # No temporary files left beside the config.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stage.yaml"]


def test_string_fields_take_override_text_verbatim(tmp_path):
    path = _write(tmp_path, {"dpo": {"enabled": True}})
    cfg = load_config_with_overrides(
        path, ["dpo.dataset=2026-10-14", "train.init_from=a #b.pt", "dpo.eval_split=null", "dpo.beta=0.05"]
    )
    assert cfg.dpo.dataset == "2026-10-14" and cfg.train.init_from == "a #b.pt"
    assert cfg.dpo.eval_split is None and cfg.dpo.beta == 0.05


def test_float_fields_take_dotless_scientific_notation(tmp_path):
    # PyYAML (YAML 1.1) reads `1e-5` as a string; a float field must still land a float.
    path = _write(tmp_path, {"train": {"max_steps": 10}})
    cfg = load_config_with_overrides(path, ["train.lr=1e-5", "train.weight_decay=1e-2", "model.moe_expert_ffn_mult=null"])
    assert cfg.train.lr == 1e-5 and isinstance(cfg.train.lr, float)
    assert cfg.train.weight_decay == 0.01 and cfg.model.moe_expert_ffn_mult is None
    with pytest.raises(ValueError, match="expects a number"):
        load_config_with_overrides(path, ["train.lr=fast"])


def test_set_is_repeatable(monkeypatch, capsys):
    from radiance import size

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["radiance-size", *argv])
        size.main()
        return capsys.readouterr().out

    both = run("--set", "model.d_model=64", "model.n_layers=3")
    assert run("--set", "model.d_model=64", "--set", "model.n_layers=3") == both
    assert run("--set", "model.n_layers=3") != both


def test_overrides_fill_an_empty_section_and_need_no_writable_config_dir(tmp_path):
    path = _write(tmp_path, {"sft": None, "train": {"max_steps": 10}})
    tmp_path.chmod(0o500)
    try:
        cfg = load_config_with_overrides(path, ["sft.enabled=true", "train.max_steps=3"])
    finally:
        tmp_path.chmod(0o700)
    assert cfg.sft.enabled and cfg.train.max_steps == 3


def test_set_overrides_a_top_level_field(tmp_path):
    # Each chained RLAIF round needs its own run_name; it is the one top-level Config field.
    path = _write(tmp_path, {"run_name": "round1", "train": {"max_steps": 10}})
    cfg = load_config_with_overrides(path, ["run_name=rlaif_round2"])
    assert cfg.run_name == "rlaif_round2"
    for bad in ("model=3", "no_such_field=1"):
        with pytest.raises(ValueError, match="section field"):
            load_config_with_overrides(path, [bad])
