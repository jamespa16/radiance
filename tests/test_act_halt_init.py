"""ACT router initial halting bias (model.act_halt_bias_init) and tokens_per_param_basis."""

import math

import pytest
import torch

from radiance.config import ModelConfig, load_config_with_overrides
from radiance.model import DenseTransformer
from radiance.model.act import initial_halt_bias
from radiance.pretokenize import run_tokens
from radiance.size import size_config


def _init_depth(**overrides) -> float:
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, head_dim=32, n_layers=3, use_router=True, max_seq_len=64, **overrides)
    model = DenseTransformer(cfg, vocab_size=256).eval()
    with torch.no_grad():
        return float(model(torch.randint(0, 256, (4, 64))).mean_loop_depth)


@pytest.mark.parametrize("max_loops", [2, 3, 4, 5, 6, 8])
def test_derived_bias_halts_a_constant_prob_token_at_half_depth(max_loops):
    cfg = ModelConfig(use_router=True, max_loops=max_loops)
    p = 1.0 / (1.0 + math.exp(-initial_halt_bias(cfg)))
    t = 1.0 - cfg.halt_epsilon
    halts_at = next(n for n in range(1, max_loops + 1) if n * p >= t or n == max_loops)
    assert halts_at == (max_loops + 1) // 2


def test_fresh_model_starts_at_half_depth():
    # Includes the router's per-token weight noise, not just the bias. max_loops 4 is the
    # showcase's: logit-centring leaves ~2.3 nats of margin either side, so every token is at 2.
    assert _init_depth(max_loops=4) == 2.0


def test_explicit_bias_overrides_the_derived_one():
    cfg = ModelConfig(use_router=True, max_loops=4, act_halt_bias_init=-1.0)
    assert initial_halt_bias(cfg) == -1.0
    model = DenseTransformer(
        ModelConfig(d_model=64, head_dim=32, n_layers=3, use_router=True, max_loops=4, act_halt_bias_init=-1.0),
        vocab_size=256,
    )
    assert torch.all(model.router.proj.bias == -1.0)
    # Graves' -1.0 is the old behaviour: at max_loops 4 nearly nobody halts before the cap.
    assert _init_depth(max_loops=4, act_halt_bias_init=-1.0) > 3.9


_MOE = ["model.use_moe=true", "model.n_experts=8", "model.moe_top_k=2", "train.tokens_per_param=20",
        "model.d_model=128", "model.head_dim=32", "model.n_layers=3"]


def test_tokens_per_param_basis_defaults_to_total_and_active_is_smaller():
    total_cfg = load_config_with_overrides(None, _MOE)
    active_cfg = load_config_with_overrides(None, _MOE + ["train.tokens_per_param_basis=active"])
    assert total_cfg.train.tokens_per_param_basis == "total"
    sizes = size_config(total_cfg, 1000)
    assert sizes["active"] < sizes["total"]
    step = total_cfg.train.effective_batch_size * total_cfg.resolved_row_tokens
    assert run_tokens(total_cfg, 1000) == max(1, round(20 * sizes["total"] / step)) * step
    assert run_tokens(active_cfg, 1000) == max(1, round(20 * sizes["active"] / step)) * step


def test_tokens_per_param_basis_rejects_unknown_values():
    with pytest.raises(ValueError, match="tokens_per_param_basis"):
        load_config_with_overrides(None, ["train.tokens_per_param_basis=unique"])
