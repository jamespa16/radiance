"""Differential Attention (cfg.model.use_diff_attn, Ye et al. 2024): two half-head_dim softmax
attention maps combined as (A1 - lambda*A2) @ V.

Unlike qk_norm/value_residual/attn_out_gate this has no zero/identity init that reduces it to the
same *computation* as standard attention (splitting head_dim is structural), so there is no
bit-identical-at-init test here the way test_hyper_connections.py has one — it is opt-in and
evaluated by A/B like use_moe/use_router/n_kv_heads. What's tested instead: it doesn't silently
break anything (shapes, gradients, KV-cache correctness — see the added entries in
test_kv_cache.py), the two new incompatibility guards actually fire, the new lambda parameters
land in the right optimizer group, and the lambda_init(layer) formula itself is right (a
sign/off-by-one bug there wouldn't show up as a crash, just as a quietly worse run).
"""

from __future__ import annotations

import math

import pytest
import torch

from radiance.config import Config, ModelConfig, TrainConfig
from radiance.model import DenseTransformer
from radiance.optim import build_muon_param_groups, build_param_groups
from tests.conftest import TINY_VOCAB


# Mirrors test_kv_cache.py/test_hyper_connections.py's loop-mode matrix, minus the ACT-sparse
# entries (act_capacity_ratio/act_ffn_capacity_ratio < 1.0) — use_diff_attn is validated against
# those combinations in DenseTransformer.__init__ and covered separately below.
MODES = {
    "dense": dict(loop_count=1),
    "looped": dict(loop_count=3),
    "gqa": dict(n_kv_heads=2, loop_count=2),
    "moe": dict(use_moe=True, n_experts=4, loop_count=2),
    "moe_looped": dict(use_moe=True, n_experts=4, loop_count=3),
    "router": dict(use_router=True, max_loops=3),
}


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    overrides.setdefault("n_layers", 3)
    cfg = ModelConfig(
        d_model=32, head_dim=8, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=32, use_diff_attn=True, **overrides,
    )
    torch.manual_seed(seed)
    return DenseTransformer(cfg, vocab_size=TINY_VOCAB)


@pytest.mark.parametrize("mode", list(MODES))
def test_shapes_and_no_dead_parameters(mode):
    model = _build(**MODES[mode]).train()
    ids = torch.randint(0, TINY_VOCAB, (2, 12))

    out = model(ids)
    assert out.logits.shape == (2, 12, TINY_VOCAB)

    loss = out.logits.float().square().mean() + out.ponder_cost + out.moe_aux_loss
    loss.backward()

    dead = [name for name, p in model.named_parameters() if p.grad is None]
    assert not dead, f"{mode}: dead parameters {dead}"


def test_head_dim_not_divisible_by_four_raises():
    # 40 % 8 == 0 (valid n_heads) but 8 % 4 == 0 is what's actually required; use a head_dim that
    # passes the pre-existing %2 assert but fails the new %4 one.
    cfg = ModelConfig(d_model=40, head_dim=10, use_diff_attn=True)
    with pytest.raises(ValueError, match="multiple of 4"):
        DenseTransformer(cfg, vocab_size=TINY_VOCAB)


def test_incompatible_with_act_capacity_sparsity_raises():
    cfg = ModelConfig(
        d_model=32, head_dim=8, use_diff_attn=True, use_router=True, act_capacity_ratio=0.5
    )
    with pytest.raises(ValueError, match="act_capacity_ratio"):
        DenseTransformer(cfg, vocab_size=TINY_VOCAB)


def test_diff_attn_off_is_completely_unaffected():
    """use_diff_attn: false (the default) must not change anything about a plain model — this is
    the one bit-identical guarantee this feature does give, just not at the attention-math level:
    the *code path* for a non-diff model is untouched by this feature's existence."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1, dropout=0.0, max_seq_len=32)
    model = DenseTransformer(cfg, vocab_size=TINY_VOCAB).eval()
    ids = torch.randint(0, TINY_VOCAB, (2, 12))
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (2, 12, TINY_VOCAB)
    assert not model.blocks[0].attn.use_diff_attn


def test_lambda_vectors_are_not_routed_to_muon():
    cfg = Config(
        model=ModelConfig(
            d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
            dropout=0.0, max_seq_len=32, use_diff_attn=True,
        ),
        train=TrainConfig(optimizer="muon", auto_batch_size=False, device="cpu", compile=False),
    )
    torch.manual_seed(0)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)

    groups = build_muon_param_groups(model, cfg)
    muon_params = {id(p) for g in groups if g["algorithm"] == "muon" for p in g["params"]}
    no_decay_params = {
        id(p) for g in groups if g["algorithm"] == "adamw" and g.get("weight_decay", 0) == 0 for p in g["params"]
    }

    for block in model.blocks:
        for name in ("diff_lambda_q1", "diff_lambda_k1", "diff_lambda_q2", "diff_lambda_k2"):
            p = getattr(block.attn, name)
            assert id(p) not in muon_params, f"{name} incorrectly routed to Muon"
            assert id(p) in no_decay_params, f"{name} not routed to the no-decay group"
        # The projection width is unchanged by use_diff_attn, so qkv_proj is still a real hidden
        # matrix and should still go to Muon exactly as it does in a non-diff model.
        assert id(block.attn.qkv_proj.weight) in muon_params

    # Same classification (dim() < 2 -> no-decay) must hold on the plain AdamW path too, since
    # build_param_groups has no _MUON_EXCLUDED_SUBSTRINGS check at all — it relies purely on
    # param.dim().
    adamw_groups = build_param_groups(model, cfg.train.weight_decay)
    decayed_params = {id(p) for g in adamw_groups if g.get("weight_decay", 0) != 0 for p in g["params"]}
    for block in model.blocks:
        for name in ("diff_lambda_q1", "diff_lambda_k1", "diff_lambda_q2", "diff_lambda_k2"):
            assert id(getattr(block.attn, name)) not in decayed_params


def test_activation_estimate_grows_with_diff_attn():
    """auto_batch_size sizes the batch off activation_bytes_per_token; if that doesn't account for
    the second attention branch it hands a diff-attn model the same micro-batch as a plain one and
    the run OOMs on a real step, same failure mode as test_hyper_connections.py's analogous test.

    Ratio band calibrated against a direct peak-memory measurement (fwd+bwd, bf16, diff on vs off,
    delta averaged over several d_model/n_layers/batch/seq shapes): the extra cost is consistently
    ~7 * d_model per token per block on top of plain attention's ~10 * d_model, which at
    tinystories.yaml's shape (FFN and logits dominate at ffn_mult=4/ffn_depth=3) works out to ~1.04x
    overall — small, because attention is a minority of the per-token cost at this shape, not
    because the measurement found a small effect."""
    def estimate(use_diff_attn):
        cfg = ModelConfig(d_model=256, head_dim=64, n_layers=6, ffn_mult=4.0, ffn_depth=3,
                           dropout=0.0, max_seq_len=512, use_diff_attn=use_diff_attn)
        torch.manual_seed(0)
        return DenseTransformer(cfg, vocab_size=50304).activation_bytes_per_token(2)

    off, on = estimate(False), estimate(True)
    assert on > off
    assert 1.02 < on / off < 1.10


def test_lambda_init_matches_paper_formula():
    """lambda_init(l) = 0.8 - 0.6*exp(-0.3*(l-1)), keyed off each block's structural position
    (block_index, already 0-indexed so it *is* l-1). A sign or off-by-one error here wouldn't
    crash anything — it would just quietly change how strongly each layer starts cancelling —
    so pin the exact values rather than trusting the formula by inspection."""
    model = _build(n_layers=4, loop_count=1)
    for block_index, block in enumerate(model.blocks):
        expected = 0.8 - 0.6 * math.exp(-0.3 * block_index)
        assert block.attn.diff_lambda_init == pytest.approx(expected)
        assert block.attn.diff_out_scale == pytest.approx(1.0 - expected)

    # Known closed-form value at block_index=0: exp(0) == 1, so this simplifies exactly, not just
    # approximately, to 0.8 - 0.6 = 0.2 — catches a formula bug the general check above might not
    # if both the bug and the fix agree at other points.
    assert model.blocks[0].attn.diff_lambda_init == pytest.approx(0.2)
    # Monotonically increasing toward 0.8 with depth, per the paper's justification (deeper layers
    # start with a stronger noise-cancelling term).
    inits = [b.attn.diff_lambda_init for b in model.blocks]
    assert inits == sorted(inits)
    assert all(v < 0.8 for v in inits)
