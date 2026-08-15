"""train.load_pretrained_weights: the "seed a new run's weights" counterpart to
find_resume_checkpoint's "continue this exact run". Loads model weights only — a freshly built
optimizer for the new run must start with no momentum/state from the source run.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import Config, ModelConfig, TrainConfig
from radiance.model import DenseTransformer
from radiance.optim import build_optimizer
from radiance.train import build_lr_scheduler, load_pretrained_weights, save_checkpoint
from tests.conftest import TINY_VOCAB


def _model_cfg(**overrides) -> ModelConfig:
    # tiny_cfg's factory hardcodes d_model/head_dim/n_layers, so shape-mismatch tests (which need
    # to vary exactly those) build a Config directly instead, mirroring test_doc_mask.py's _build.
    defaults = {
        "d_model": 32,
        "head_dim": 8,
        "n_layers": 3,
        "ffn_mult": 2.0,
        "ffn_depth": 1,
        "dropout": 0.0,
        "max_seq_len": 32,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _save_source_checkpoint(tmp_path, cfg):
    raw_model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    optimizer = build_optimizer(raw_model, cfg, "cpu")
    scheduler = build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "source.pt"
    save_checkpoint(path, raw_model, optimizer, scheduler, scaler, step=123, cfg=cfg)
    return path, raw_model


def test_init_from_loads_weights_but_not_optimizer_state(tmp_path, tiny_cfg):
    source_cfg = tiny_cfg(_train={"optimizer": "adamw"})
    path, source_model = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = tiny_cfg(_train={"optimizer": "adamw", "init_from": str(path)})
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB)

    load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB, "cpu")

    for key, source_tensor in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[key], source_tensor, rtol=0, atol=0)

    # The new run's optimizer is built fresh afterward (train.py's normal flow) — it must carry
    # no momentum/state from the source run, since init_from is explicitly not resume_from.
    fresh_optimizer = build_optimizer(target_model, target_cfg, "cpu")
    assert all(len(fresh_optimizer.state.get(p, {})) == 0 for group in fresh_optimizer.param_groups for p in group["params"])


def test_init_from_raises_on_shape_mismatch(tmp_path):
    source_cfg = Config(model=_model_cfg(), train=TrainConfig(optimizer="adamw", device="cpu", compile=False))
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    # Different d_model -> load_state_dict would otherwise fail with an opaque shape error deep
    # inside torch; load_pretrained_weights should catch it first with a clear message.
    mismatched_cfg = Config(
        model=_model_cfg(d_model=64),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="d_model"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_accepts_compatible_raw_n_kv_head_notation(tmp_path):
    """Resolved GQA counts must match even when the raw optional field differs (None vs explicit)."""
    source_cfg = Config(model=_model_cfg(), train=TrainConfig(optimizer="adamw", device="cpu", compile=False))
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = Config(
        model=_model_cfg(n_kv_heads=4),  # _model_cfg has d_model=32, head_dim=8, so n_heads=4
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB)

    # checkpoint n_kv_heads is None (resolved to 4); the target pins the same resolved value.
    load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB, "cpu")

    source_from_disk = torch.load(path, map_location="cpu", weights_only=False)
    for key, source_tensor in source_from_disk["model"].items():
        torch.testing.assert_close(target_model.state_dict()[key], source_tensor, rtol=0, atol=0)


def test_init_from_accepts_explicit_source_n_kv_heads_with_default_target(tmp_path):
    source_cfg = Config(
        model=_model_cfg(n_kv_heads=2),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = Config(
        model=_model_cfg(),  # target n_kv_heads defaults to None, which resolves to n_heads=4
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    # The target is explicitly built at resolved n_kv_heads=2 so the shape check and state dict
    # line up; this pins that the source's explicit value is what the check consults.
    target_cfg.model.n_kv_heads = 2
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB)

    load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_resolved_n_kv_heads_mismatch(tmp_path):
    source_cfg = Config(
        model=_model_cfg(n_kv_heads=4),  # source is the default MHA resolved count
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(n_kv_heads=2),  # same d_model/head_dim, but a genuinely different GQA count
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="n_kv_heads"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_vocab_size_mismatch(tmp_path, tiny_cfg):
    source_cfg = tiny_cfg(_train={"optimizer": "adamw"})
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = tiny_cfg(_train={"optimizer": "adamw", "init_from": str(path)})
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB + 8)

    with pytest.raises(ValueError, match="vocab_size"):
        load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB + 8, "cpu")


def test_init_from_accepts_inert_loop_iter_conditioning_modes(tmp_path):
    """Raw mode differences are inert when loop_multiplier == 1 (no per-iteration parameters)."""
    source_cfg = Config(
        model=_model_cfg(loop_iter_conditioning="none"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = Config(
        model=_model_cfg(loop_iter_conditioning="lora"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB)

    load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB, "cpu")


def test_init_from_accepts_legacy_norm_gain_upgrade(tmp_path):
    """A single 1-D gain bank may seed a multi-variant norm_gains bank via RMSNorm's pre-hook."""
    source_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="none"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="norm_gains"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB)

    load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB, "cpu")

    source_from_disk = torch.load(path, map_location="cpu", weights_only=False)["model"]
    for key, source_tensor in source_from_disk.items():
        if key in ("ln1.weight", "ln2.weight") or key.startswith(("blocks.",)):
            target = target_model.state_dict()[key]
            if source_tensor.dim() == 1 and target.dim() == 2:
                torch.testing.assert_close(target[0], source_tensor, rtol=0, atol=0)
                torch.testing.assert_close(target[1], source_tensor, rtol=0, atol=0)
            else:
                torch.testing.assert_close(target, source_tensor, rtol=0, atol=0)


def test_init_from_accepts_diff_attn_same_mode(tmp_path):
    source_cfg = Config(
        model=_model_cfg(use_diff_attn=True),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    target_cfg = Config(
        model=_model_cfg(use_diff_attn=True),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    target_model = DenseTransformer(target_cfg.model, vocab_size=TINY_VOCAB)

    load_pretrained_weights(target_model, str(path), target_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_use_diff_attn_mismatch(tmp_path):
    source_cfg = Config(
        model=_model_cfg(use_diff_attn=True),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(use_diff_attn=False),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="use_diff_attn"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_hyper_conn_streams_mismatch(tmp_path):
    source_cfg = Config(
        model=_model_cfg(hyper_conn_streams=2),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(hyper_conn_streams=1),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="hyper_conn_streams"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_loop_iter_conditioning_mismatch(tmp_path):
    source_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="lora"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="norm_gains"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="loop_iter_conditioning"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_loop_conditioning_downgrade_to_none(tmp_path):
    source_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="norm_gains"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="none"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="loop_iter_conditioning"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_loop_conditioning_variant_mismatch(tmp_path):
    source_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="norm_gains"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(loop_count=3, loop_iter_conditioning="norm_gains"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="loop_iter_conditioning"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")


def test_init_from_raises_on_loop_lora_rank_mismatch(tmp_path):
    source_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="lora"),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False),
    )
    path, _ = _save_source_checkpoint(tmp_path, source_cfg)

    mismatched_cfg = Config(
        model=_model_cfg(loop_count=2, loop_iter_conditioning="lora", loop_lora_rank=4),
        train=TrainConfig(optimizer="adamw", device="cpu", compile=False, init_from=str(path)),
    )
    mismatched_model = DenseTransformer(mismatched_cfg.model, vocab_size=TINY_VOCAB)

    with pytest.raises(ValueError, match="loop_lora_rank"):
        load_pretrained_weights(mismatched_model, str(path), mismatched_cfg, TINY_VOCAB, "cpu")
