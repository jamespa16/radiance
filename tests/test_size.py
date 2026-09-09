"""`radiance-size` must agree with a real build, or it is worse than useless.

The tool reports parameter counts from a model constructed on the `meta` device — shapes without
storage. That is the whole reason it is fast enough to call in a loop while designing an A/B under
a parameter cap, and also the reason it could silently drift: a parameter allocated through a path
that behaves differently under `meta` would go uncounted and the tool would quietly under-report
the budget it exists to police.
"""

import torch

from radiance.config import Config, ModelConfig
from radiance.model import DenseTransformer
from radiance.size import size_config

VOCAB = 4096


def _real_counts(model_cfg: ModelConfig) -> tuple[int, int]:
    model = DenseTransformer(model_cfg, vocab_size=VOCAB)
    return model.num_parameters(), model.num_active_parameters()


def _meta_counts(model_cfg: ModelConfig) -> tuple[int, int]:
    counts = size_config(Config(model=model_cfg), VOCAB)
    return counts["total"], counts["active"]


def test_meta_sizing_matches_real_build_dense():
    cfg = ModelConfig(d_model=64, head_dim=32, n_layers=3, vocab_pad_multiple=1)
    assert _meta_counts(cfg) == _real_counts(cfg)


def test_meta_sizing_matches_real_build_moe():
    # MoE is the case where total and active diverge, so it exercises the branch of
    # num_active_parameters() that a dense shape leaves dead.
    cfg = ModelConfig(
        d_model=64, head_dim=32, n_layers=3, vocab_pad_multiple=1,
        use_moe=True, n_experts=4, moe_top_k=2,
    )
    total, active = _meta_counts(cfg)
    assert (total, active) == _real_counts(cfg)
    assert active < total


def test_meta_sizing_matches_real_build_looped():
    cfg = ModelConfig(d_model=64, head_dim=32, n_layers=3, loop_count=4, vocab_pad_multiple=1)
    assert _meta_counts(cfg) == _real_counts(cfg)


def test_looping_costs_far_less_than_a_block_per_iteration():
    """Looping is *nearly* free in parameters, not exactly free -- and the tool must show that.

    `blocks[1:]` is weight-shared across iterations, but `loop_iter_conditioning: norm_gains`
    allocates a per-iteration RMSNorm gain bank and `loop_input_injection` a `W_inj`, so the count
    does move with `loop_count`. Under a total-parameter cap that distinction is the entire reason
    looping is worth re-examining, so it is pinned rather than left to be rediscovered: each extra
    iteration must cost far less than the block it re-runs.
    """
    kw = dict(d_model=64, head_dim=32, vocab_pad_multiple=1)
    flat = size_config(Config(model=ModelConfig(n_layers=3, loop_count=1, **kw)), VOCAB)["total"]
    looped = size_config(Config(model=ModelConfig(n_layers=3, loop_count=4, **kw)), VOCAB)["total"]
    deeper = size_config(Config(model=ModelConfig(n_layers=9, loop_count=1, **kw)), VOCAB)["total"]

    per_iteration = (looped - flat) / 3
    per_real_block = (deeper - flat) / 6
    assert 0 < per_iteration < 0.2 * per_real_block


def test_embed_plus_blocks_is_total():
    cfg = ModelConfig(d_model=64, head_dim=32, n_layers=3, vocab_pad_multiple=1)
    counts = size_config(Config(model=cfg), VOCAB)
    assert counts["embed"] + counts["blocks"] == counts["total"]


def test_vocab_padding_is_applied():
    cfg = ModelConfig(d_model=64, head_dim=32, n_layers=2, vocab_pad_multiple=128)
    assert size_config(Config(model=cfg), 4097)["vocab"] == 4224
