"""estimate_batch_size and tokens_per_step must size batches using the resolved per-row token width,
not cfg.data.seq_len directly, so sft.seq_len / dpo.seq_len overrides reach auto_batch_size and
tokens_per_param's derived max_steps."""

from __future__ import annotations

import pytest
import torch

from radiance.config import (
    Config,
    DataConfig,
    DPOConfig,
    ModelConfig,
    SFTConfig,
    TrainConfig,
)
from radiance.train import estimate_batch_size


class _FakeModel:
    def num_parameters(self) -> int:
        return 1000

    def activation_bytes_per_token(self, activation_dtype_bytes: int) -> int:
        return 400 * activation_dtype_bytes


def _estimate_cfg(sft: SFTConfig | None = None, dpo: DPOConfig | None = None) -> Config:
    return Config(
        data=DataConfig(seq_len=512),
        model=ModelConfig(),
        train=TrainConfig(
            batch_size=2,
            grad_accum_steps=1,
            auto_batch_size=False,
            compile=False,
            device="cpu",
            dtype="bf16",
            target_effective_batch_size=4,
            vram_safety_margin=1.0,
        ),
        sft=sft or SFTConfig(),
        dpo=dpo or DPOConfig(),
    )


def _estimate_with_fixed_vram(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    # _FakeModel reports 1000 params (3 * 1000 * 4 fp32 bytes) and 400 bytes/token under bf16
    # (activation_dtype_bytes=2). Free bytes are pinned so the usable budget is exactly 1024 tokens.
    free_bytes = 3 * 1000 * 4 + 1024 * 800
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, 1_000_000_000))
    return estimate_batch_size(_FakeModel(), cfg, "cuda", "cuda")


def test_resolved_row_tokens_pretrain():
    cfg = _estimate_cfg()
    assert cfg.resolved_seq_len == 512
    assert cfg.train_width_multiplier == 1
    assert cfg.resolved_row_tokens == 512


def test_resolved_row_tokens_sft_override():
    cfg = _estimate_cfg(sft=SFTConfig(enabled=True, dataset="ignored", seq_len=256))
    assert cfg.resolved_seq_len == 256
    assert cfg.resolved_row_tokens == 256


def test_resolved_row_tokens_dpo_override():
    cfg = _estimate_cfg(dpo=DPOConfig(enabled=True, dataset="ignored", seq_len=256, reference_checkpoint="ckpt"))
    assert cfg.resolved_seq_len == 256
    assert cfg.train_width_multiplier == 2
    assert cfg.resolved_row_tokens == 512


def test_estimate_batch_size_pretrain_uses_data_seq_len(monkeypatch):
    assert _estimate_with_fixed_vram(_estimate_cfg(), monkeypatch) == (2, 2)


def test_estimate_batch_size_sft_seq_len_override(monkeypatch):
    cfg = _estimate_cfg(sft=SFTConfig(enabled=True, dataset="ignored", seq_len=256))
    assert _estimate_with_fixed_vram(cfg, monkeypatch) == (4, 1)


def test_estimate_batch_size_dpo_seq_len_override(monkeypatch):
    cfg = _estimate_cfg(dpo=DPOConfig(enabled=True, dataset="ignored", seq_len=256, reference_checkpoint="ckpt"))
    # 1024 usable tokens / (256 seq_len * 2 rows/pair) = 2 pairs, not 4.
    assert _estimate_with_fixed_vram(cfg, monkeypatch) == (2, 2)


def test_estimate_batch_size_dpo_uses_pair_width_without_override(monkeypatch):
    cfg = _estimate_cfg(dpo=DPOConfig(enabled=True, dataset="ignored", reference_checkpoint="ckpt"))
    # 1024 usable tokens / (512 data.seq_len * 2 rows/pair) = 1 pair.
    assert _estimate_with_fixed_vram(cfg, monkeypatch) == (1, 4)


def test_estimate_batch_size_reserve_bytes_shrinks_the_budget(monkeypatch):
    # reserve_bytes holds back VRAM for a model that hasn't loaded yet (a DPO cache-miss's
    # reference checkpoint) - it must reduce the usable token budget exactly like
    # not_yet_allocated_bytes does, not get silently ignored.
    cfg = _estimate_cfg()
    free_bytes = 3 * 1000 * 4 + 1024 * 800  # same fixture as _estimate_with_fixed_vram: 1024 usable tokens
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, 1_000_000_000))

    without_reserve = estimate_batch_size(_FakeModel(), cfg, "cuda", "cuda")
    # Reserve half the usable budget's worth of bytes (512 tokens * 800 bytes/token): usable tokens
    # drop from 1024 to 512, halving batch_size from 2 to 1.
    with_reserve = estimate_batch_size(_FakeModel(), cfg, "cuda", "cuda", reserve_bytes=512 * 800)

    assert without_reserve == (2, 2)
    assert with_reserve == (1, 4)
