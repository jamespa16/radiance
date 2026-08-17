"""train.native_bf16: params/grads/optimizer moments stored in bf16 instead of fp32 masters.

See docs/optim.md's "Native bf16 storage" section and TrainConfig.native_bf16's docstring.
"""

from __future__ import annotations

import pytest
import torch

from radiance.batching import param_state_dtype_bytes
from radiance.config import Config, ModelConfig, TrainConfig, _apply_dtype_sugar
from radiance.model import DenseTransformer
from radiance.optim import MuonWithAuxAdam, build_optimizer
from tests.conftest import TINY_VOCAB


def _cast_to_native_bf16(model: DenseTransformer) -> None:
    """Mirrors train.py's cast exactly, so the test exercises the same operation train() runs."""
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)


def _model_and_cfg(optimizer: str = "muon", **model_kwargs):
    cfg = Config(
        model=ModelConfig(
            d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
            dropout=0.0, max_seq_len=32, **model_kwargs,
        ),
        train=TrainConfig(
            optimizer=optimizer, dtype="bf16", native_bf16=True,
            auto_batch_size=False, device="cpu", compile=False,
        ),
    )
    torch.manual_seed(0)
    return DenseTransformer(cfg.model, vocab_size=TINY_VOCAB), cfg


# --- config validation -----------------------------------------------------------------------


def test_native_bf16_requires_bf16_dtype():
    with pytest.raises(ValueError, match="native_bf16 requires train.dtype"):
        _apply_dtype_sugar(Config(train=TrainConfig(native_bf16=True, dtype="fp32")))
    with pytest.raises(ValueError, match="native_bf16 requires train.dtype"):
        _apply_dtype_sugar(Config(train=TrainConfig(native_bf16=True, dtype="fp16")))
    _apply_dtype_sugar(Config(train=TrainConfig(native_bf16=True, dtype="bf16")))  # does not raise


def test_native_bf16_incompatible_with_fp4_linear():
    with pytest.raises(ValueError, match="incompatible with model.fp4_linear"):
        _apply_dtype_sugar(Config(
            model=ModelConfig(fp4_linear=True),
            train=TrainConfig(native_bf16=True, dtype="bf16"),
        ))


def test_native_bf16_defaults_off():
    """The inert case: a config that never sets native_bf16 gets fp32 storage exactly as before."""
    assert TrainConfig().native_bf16 is False
    assert param_state_dtype_bytes(Config()) == 4


# --- VRAM accounting --------------------------------------------------------------------------


def test_param_state_dtype_bytes_halves_under_native_bf16():
    assert param_state_dtype_bytes(Config(train=TrainConfig(dtype="bf16", native_bf16=True))) == 2
    assert param_state_dtype_bytes(Config(train=TrainConfig(dtype="bf16", native_bf16=False))) == 4


# --- casting behavior -------------------------------------------------------------------------


def test_cast_leaves_buffers_at_fp32():
    """Only nn.Parameters are cast — RoPE's cos/sin cache and (under MoE) expert_bias carry no
    optimizer state, so casting them would cost precision for no memory win."""
    model, _ = _model_and_cfg()
    _cast_to_native_bf16(model)

    for p in model.parameters():
        assert p.dtype == torch.bfloat16
    assert model.rope.cos_cached.dtype == torch.float32
    assert model.rope.sin_cached.dtype == torch.float32


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_grads_and_optimizer_state_follow_param_dtype(optimizer):
    """The whole point: with native_bf16 on, grad and every optimizer-moment buffer end up bf16 —
    not just the parameters — which is what actually halves the static VRAM footprint."""
    model, cfg = _model_and_cfg(optimizer=optimizer)
    _cast_to_native_bf16(model)
    opt = build_optimizer(model, cfg, "cpu")

    model(torch.randint(0, TINY_VOCAB, (2, 8))).logits.float().square().mean().backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.dtype == torch.bfloat16
    opt.step()

    for p in model.parameters():
        assert p.dtype == torch.bfloat16  # the step must not silently upcast the parameter

    if isinstance(opt, MuonWithAuxAdam):
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                if group["algorithm"] == "muon":
                    assert state["momentum_buffer"].dtype == torch.bfloat16
                else:
                    assert state["exp_avg"].dtype == torch.bfloat16
                    assert state["exp_avg_sq"].dtype == torch.bfloat16
    else:
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                assert state["exp_avg"].dtype == torch.bfloat16
                assert state["exp_avg_sq"].dtype == torch.bfloat16


def test_fp32_run_still_gets_fp32_optimizer_state():
    """Regression guard for the _new_state_like fix: an ordinary (non-native_bf16) run's optimizer
    state must stay fp32 exactly as before — the dtype now follows the parameter, and every
    parameter here still is fp32."""
    cfg = Config(
        model=ModelConfig(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                           dropout=0.0, max_seq_len=32),
        train=TrainConfig(optimizer="muon", auto_batch_size=False, device="cpu", compile=False),
    )
    torch.manual_seed(0)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    opt = build_optimizer(model, cfg, "cpu")
    model(torch.randint(0, TINY_VOCAB, (2, 8))).logits.square().mean().backward()
    opt.step()

    for group in opt.param_groups:
        if group["algorithm"] != "adamw":
            continue
        for p in group["params"]:
            state = opt.state[p]
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32
