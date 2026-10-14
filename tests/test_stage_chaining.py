"""Chaining post-training stages: train.inherit_model_config and the `--set` overrides
radiance-train now shares with radiance-size."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from radiance.checkpointing import save_checkpoint
from radiance.config import Config, DataConfig, DPOConfig, ModelConfig, TrainConfig, load_config, load_config_with_overrides
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


def test_a_dpo_runs_resolved_doc_mask_is_not_inherited(tmp_path):
    # resolve_dpo_doc_mask saves doc_attention_mask=False into every DPO checkpoint; an SFT stage
    # started from one must not inherit that, or its packed conversations attend to each other.
    model_cfg = ModelConfig(d_model=32, head_dim=8, n_layers=2, max_seq_len=32, doc_attention_mask=False)
    cfg = Config(data=DataConfig(tokenizer="t"), model=model_cfg, train=TrainConfig(device="cpu", compile=False))
    cfg.dpo.enabled = True
    model = DenseTransformer(model_cfg, vocab_size=64)
    opt = build_optimizer(model, cfg, "cpu")
    ckpt = tmp_path / "dpo.pt"
    save_checkpoint(ckpt, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    raw = {"train": {"init_from": str(ckpt), "inherit_model_config": True}, "sft": {"enabled": True}}
    assert load_config(_write(tmp_path, raw)).model.doc_attention_mask is True
    # ...while a non-DPO source's explicit choice is inherited as-is.
    cfg.dpo.enabled = False
    save_checkpoint(ckpt, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    assert load_config(_write(tmp_path, raw)).model.doc_attention_mask is False
    # ...and so is a DPO source's, once it recorded that the architecture itself had the mask off.
    cfg.dpo.enabled = True
    cfg.dpo.doc_attention_mask_before_resolve = False
    save_checkpoint(ckpt, model, opt, build_lr_scheduler(opt, cfg), torch.amp.GradScaler("cpu", enabled=False), 0, cfg)
    assert load_config(_write(tmp_path, raw)).model.doc_attention_mask is False


def test_resolve_dpo_doc_mask_records_what_it_overrode():
    from radiance.losses import resolve_dpo_doc_mask

    cfg = Config(dpo=DPOConfig(enabled=True))
    resolve_dpo_doc_mask(cfg)
    assert cfg.model.doc_attention_mask is False and cfg.dpo.doc_attention_mask_before_resolve is True


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


def test_overrides_fill_an_empty_section_and_need_no_writable_config_dir(tmp_path):
    path = _write(tmp_path, {"sft": None, "train": {"max_steps": 10}})
    tmp_path.chmod(0o500)
    try:
        cfg = load_config_with_overrides(path, ["sft.enabled=true", "train.max_steps=3"])
    finally:
        tmp_path.chmod(0o700)
    assert cfg.sft.enabled and cfg.train.max_steps == 3
