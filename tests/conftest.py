"""Shared fixtures.

Everything here is deliberately CPU-sized and fp32: these tests assert *equivalences* (feature-off
vs feature-on-but-inert, dense vs sparse, cached vs uncached), not quality, so the model only needs
to be large enough to exercise every code path — a few thousand parameters is plenty and keeps the
whole suite fast enough to run on every change.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import Config, DataConfig, ModelConfig, TrainConfig


TINY_VOCAB = 64


@pytest.fixture
def tiny_cfg():
    """Factory for a small Config. Pass model/data/train overrides as kwargs:

        cfg = tiny_cfg(loop_count=3, use_moe=True)          # model fields
        cfg = tiny_cfg(seq_len=16, _train=dict(lr=1e-3))    # data / train fields
    """

    def _make(_data: dict | None = None, _train: dict | None = None, **model_kwargs) -> Config:
        model = ModelConfig(
            d_model=32,
            head_dim=8,
            n_layers=3,
            ffn_mult=2.0,
            ffn_depth=1,
            dropout=0.0,  # deterministic forwards; dropout is tested separately where it matters
            max_seq_len=32,
            **model_kwargs,
        )
        data = DataConfig(seq_len=16, num_workers=0, **(_data or {}))
        train = TrainConfig(
            batch_size=2,
            auto_batch_size=False,  # CPU-only tests must not try to read free VRAM
            compile=False,
            device="cpu",
            **(_train or {}),
        )
        return Config(data=data, model=model, train=train)

    return _make


@pytest.fixture
def tiny_ids():
    """Deterministic (batch, seq) token ids inside TINY_VOCAB."""

    def _make(batch: int = 2, seq: int = 16, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(0, TINY_VOCAB, (batch, seq), generator=gen)

    return _make


@pytest.fixture(autouse=True)
def _deterministic():
    """Every test starts from the same RNG state, so `build_model`-style helpers that construct two
    models and compare them get identical initialisations without each test re-seeding by hand."""
    torch.manual_seed(0)
