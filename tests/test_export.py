"""The export is only worth anything if it round-trips.

`export_checkpoint` drops a tensor (the tied `lm_head.weight`) and may cast every other one, so
the failure mode it has to be pinned against is a *quiet* one: a file that loads fine and produces
subtly different logits. Every test here therefore compares a model rebuilt from the export
against the model the checkpoint came from, rather than checking that files appeared.
"""

from __future__ import annotations

import json

import pytest
import torch

from radiance.config import Config
from radiance.export import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    _config_from_json,
    _config_to_json,
    export_checkpoint,
    load_export,
)
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


def _write_checkpoint(tmp_path, cfg: Config, step: int = 7):
    """A minimal stand-in for save_checkpoint's output: the exporter reads only these keys, and
    building the real optimizer/scheduler here would test train.py rather than the export."""
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    path = tmp_path / "step_7.pt"
    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, path)
    return path, model


def test_round_trip_matches_the_source_model(tmp_path, tiny_cfg, tiny_ids):
    cfg = tiny_cfg()
    path, model = _write_checkpoint(tmp_path, cfg)

    export_checkpoint(path, tmp_path / "export")
    loaded, loaded_cfg = load_export(tmp_path / "export")

    ids = tiny_ids()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, loaded(ids).logits)
    assert loaded_cfg.model == cfg.model


def test_tied_embedding_is_stored_once_and_restored(tmp_path, tiny_cfg):
    cfg = tiny_cfg()
    path, _ = _write_checkpoint(tmp_path, cfg)

    summary = export_checkpoint(path, tmp_path / "export")

    # The alias is dropped from the file (safetensors would otherwise write the embedding twice,
    # and a consumer would get two independently trainable copies of one tied matrix)...
    assert summary["shared_tensors"] == {"lm_head.weight": "token_emb.weight"}
    from safetensors.torch import load_file

    assert "lm_head.weight" not in load_file(tmp_path / "export" / WEIGHTS_NAME)

    # ...and load_export puts it back as the *same* tensor, not a copy — the tie has to survive
    # the round trip, or a resumed/fine-tuned export would train the two apart.
    loaded, _ = load_export(tmp_path / "export")
    assert loaded.lm_head.weight is loaded.token_emb.weight


def test_bf16_export_halves_the_weights_and_stays_close(tmp_path, tiny_cfg, tiny_ids):
    cfg = tiny_cfg()
    path, model = _write_checkpoint(tmp_path, cfg)

    export_checkpoint(path, tmp_path / "fp32")
    export_checkpoint(path, tmp_path / "bf16", dtype=torch.bfloat16)

    fp32_bytes = (tmp_path / "fp32" / WEIGHTS_NAME).stat().st_size
    bf16_bytes = (tmp_path / "bf16" / WEIGHTS_NAME).stat().st_size
    assert bf16_bytes < fp32_bytes * 0.6  # not exactly half: the JSON header is dtype-independent

    loaded, _ = load_export(tmp_path / "bf16")
    ids = tiny_ids()
    model.eval()
    with torch.no_grad():
        # Rounding to bf16 is the only difference, so this is a precision bound, not an
        # equivalence: a real mismatch (a permuted or half-loaded tensor) misses it by miles.
        assert torch.allclose(model(ids).logits, loaded(ids).logits.float(), atol=0.05)


def test_native_bf16_checkpoint_round_trips_without_upcasting(tmp_path, tiny_cfg, tiny_ids):
    """A native_bf16 run stores bf16 parameters, and load_state_dict keeps the *destination*
    dtype — so load_export must set the model's dtype before loading, exactly as
    load_transformer_from_checkpoint does, or the export silently comes back as fp32."""
    cfg = tiny_cfg(_train=dict(native_bf16=True))
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    from radiance.model import cast_params_to_native_bf16

    cast_params_to_native_bf16(model)
    path = tmp_path / "step_7.pt"
    torch.save({"model": model.state_dict(), "step": 7, "config": cfg}, path)

    export_checkpoint(path, tmp_path / "export")
    loaded, _ = load_export(tmp_path / "export")

    assert loaded.token_emb.weight.dtype == torch.bfloat16
    model.eval()
    ids = tiny_ids()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, loaded(ids).logits)


def test_moe_checkpoint_round_trips(tmp_path, tiny_cfg, tiny_ids):
    """MoE is where a state dict stops being uniformly float: expert_bias and the balancing
    counters ride along beside the weights, and casting the wrong one corrupts routing."""
    cfg = tiny_cfg(use_moe=True, n_experts=4, moe_top_k=2)
    path, model = _write_checkpoint(tmp_path, cfg)

    export_checkpoint(path, tmp_path / "export", dtype=torch.bfloat16)
    loaded, _ = load_export(tmp_path / "export")

    source = model.state_dict()
    for name, tensor in loaded.state_dict().items():
        if not tensor.is_floating_point():
            assert torch.equal(tensor, source[name]), f"{name} was cast but is not floating point"


def test_config_json_is_plain_readable_json(tmp_path, tiny_cfg):
    """The whole point of the export is that a consumer needs neither pickle nor `radiance` on the
    path to read it, so config.json must survive a bare json.load with no custom decoding."""
    cfg = tiny_cfg()
    path, _ = _write_checkpoint(tmp_path, cfg)

    export_checkpoint(path, tmp_path / "export")
    with open(tmp_path / "export" / CONFIG_NAME) as f:
        meta = json.load(f)

    assert meta["step"] == 7
    assert meta["vocab_size"] == TINY_VOCAB
    assert meta["config"]["model"]["d_model"] == cfg.model.d_model
    assert meta["total_parameters"] == sum(
        p.numel() for p in DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).parameters()
    )


def test_config_json_tolerates_schema_drift():
    """An export written by another revision of this repo must still load — the same contract
    load_config gives an omitted YAML key and Config.__setstate__ gives an old pickle."""
    raw = _config_to_json(Config())
    raw["model"]["a_field_from_the_future"] = 123
    del raw["model"]["dropout"]

    cfg = _config_from_json(raw)

    assert cfg.model.dropout == Config().model.dropout
    assert not hasattr(cfg.model, "a_field_from_the_future")


def test_optimizer_state_is_not_exported(tmp_path, tiny_cfg):
    """An export is a model, not a resume point. Optimizer moments are several times the size of
    the weights and meaningless outside this repo's parameter groups; if they ever start being
    written, that should be a decision rather than an accident."""
    cfg = tiny_cfg()
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.zeros(1, 4, dtype=torch.long)).logits.sum().backward()
    opt.step()
    path = tmp_path / "step_7.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": 7, "config": cfg}, path
    )

    export_checkpoint(path, tmp_path / "export")

    from safetensors.torch import load_file

    names = set(load_file(tmp_path / "export" / WEIGHTS_NAME))
    assert names == set(model.state_dict()) - {"lm_head.weight"}


def test_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_checkpoint(tmp_path / "nope.pt", tmp_path / "export")


# --- Importing an export back into the pipeline -------------------------------------------------
# An export nothing can read is a write-only format. These pin the other half: every path that
# takes a checkpoint takes an export directory too, and the one path that can't says so.


def test_load_transformer_from_checkpoint_accepts_an_export(tmp_path, tiny_cfg, tiny_ids):
    """The single loader behind generate, serve, eval and DPO's reference model. If it takes an
    export, all four do — which is why the dispatch lives there and not at each CLI."""
    cfg = tiny_cfg()
    path, model = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "export")

    from radiance.model import load_transformer_from_checkpoint

    loaded, loaded_cfg = load_transformer_from_checkpoint(str(tmp_path / "export"), "cpu")

    ids = tiny_ids()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, loaded(ids).logits)
    assert loaded_cfg.model == cfg.model


def test_export_is_found_by_directory_or_by_weights_file(tmp_path, tiny_cfg):
    cfg = tiny_cfg()
    path, _ = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "export")

    from radiance.export import is_export

    assert is_export(tmp_path / "export")
    assert is_export(tmp_path / "export" / WEIGHTS_NAME)  # what you get from tab-completion
    assert not is_export(path)  # a .pt is a checkpoint, not an export
    assert not is_export(tmp_path / "nowhere")
    # Weights without a config cannot rebuild a model; failing the check here gives a better error
    # than a FileNotFoundError from inside the loader.
    (tmp_path / "export" / CONFIG_NAME).unlink()
    assert not is_export(tmp_path / "export")


def test_init_from_seeds_a_run_from_an_export(tmp_path, tiny_cfg, tiny_ids):
    """train.init_from needs weights and nothing else — exactly what an export has."""
    from radiance.checkpointing import load_pretrained_weights

    cfg = tiny_cfg()
    path, source = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "export")

    fresh = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    load_pretrained_weights(fresh, str(tmp_path / "export"), cfg, TINY_VOCAB, "cpu")

    fresh.eval()
    source.eval()
    ids = tiny_ids()
    with torch.no_grad():
        assert torch.equal(source(ids).logits, fresh(ids).logits)


def test_init_from_an_export_still_checks_the_model_shape(tmp_path, tiny_cfg):
    """The shape-mismatch guard exists so a wrong checkpoint names the field that disagrees rather
    than surfacing as a tensor-shape RuntimeError from inside torch. It must not be bypassed by
    coming in through the export path."""
    from radiance.checkpointing import load_pretrained_weights

    path, _ = _write_checkpoint(tmp_path, tiny_cfg())
    export_checkpoint(path, tmp_path / "export")

    other = tiny_cfg(use_diff_attn=True)
    with pytest.raises(ValueError, match="use_diff_attn"):
        load_pretrained_weights(
            DenseTransformer(other.model, vocab_size=TINY_VOCAB),
            str(tmp_path / "export"),
            other,
            TINY_VOCAB,
            "cpu",
        )


def test_resume_from_an_export_is_rejected_by_name(tmp_path, tiny_cfg):
    """An export has no optimizer moments, so "resuming" from one restarts AdamW from zero
    momentum at warmup LR — the exact loss spike save_checkpoint exists to prevent. Silently
    allowing it would be far worse than refusing."""
    from radiance.checkpointing import find_resume_checkpoint

    cfg = tiny_cfg()
    path, _ = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "export")

    cfg.train.resume_from = str(tmp_path / "export")
    with pytest.raises(ValueError, match="init_from"):
        find_resume_checkpoint(cfg)


def test_half_precision_export_is_not_silently_upcast_on_load(tmp_path, tiny_cfg):
    """`--dtype bf16` on a checkpoint whose run was *not* native_bf16: the config says fp32, the
    weights say bf16. Trusting the config would build an fp32 model and let load_state_dict's
    copy_ upcast the export straight back to the size it was exported to avoid."""
    cfg = tiny_cfg()
    assert not cfg.train.native_bf16
    path, _ = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "export", dtype=torch.bfloat16)

    loaded, _ = load_export(tmp_path / "export")

    assert loaded.token_emb.weight.dtype == torch.bfloat16


def test_param_bytes_agree_between_a_checkpoint_and_its_export(tmp_path, tiny_cfg):
    """serve/batching size VRAM off this before deciding to load a model, so an export must price
    the same as the checkpoint it came from — which is only true because read_export restores the
    dropped tie instead of leaving the state dict one tensor short."""
    from radiance.model import checkpoint_param_bytes

    cfg = tiny_cfg()
    path, _ = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "export")

    assert checkpoint_param_bytes(str(tmp_path / "export")) == checkpoint_param_bytes(str(path))


def test_an_export_can_be_re_exported(tmp_path, tiny_cfg):
    """read_checkpoint is what export_checkpoint reads with too, so `--dtype` on an existing
    export works without a special case — and the result must still round-trip."""
    cfg = tiny_cfg()
    path, _ = _write_checkpoint(tmp_path, cfg)
    export_checkpoint(path, tmp_path / "fp32")

    summary = export_checkpoint(tmp_path / "fp32", tmp_path / "bf16", dtype=torch.bfloat16)

    assert summary["dtype"] == "bfloat16"
    assert summary["step"] == 7
    assert summary["shared_tensors"] == {"lm_head.weight": "token_emb.weight"}
    loaded, _ = load_export(tmp_path / "bf16")
    assert loaded.lm_head.weight is loaded.token_emb.weight
