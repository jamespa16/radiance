"""ACT router initial halting bias (model.act_halt_bias_init) and tokens_per_param_basis."""

import collections
import math

import pytest
import torch
import torch.nn.functional as F

from radiance.config import ModelConfig, load_config_with_overrides
from radiance.model import DenseTransformer
from radiance.model.act import initial_halt_bias
from radiance.pretokenize import run_tokens
from radiance.size import size_config


def _target_pass(max_loops: int) -> int:
    # initial_halt_bias's documented target, restated so the tests pin it independently.
    return max(2, max_loops // 2)


def _small(d_model: int = 64, **overrides) -> DenseTransformer:
    cfg = ModelConfig(d_model=d_model, head_dim=32, n_layers=3, use_router=True, max_seq_len=64, **overrides)
    return DenseTransformer(cfg, vocab_size=256)


def _init_depth(**overrides) -> float:
    torch.manual_seed(0)
    model = _small(**overrides).eval()
    with torch.no_grad():
        return float(model(torch.randint(0, 256, (4, 64))).mean_loop_depth)


def _init_depth_histogram(d_model: int, max_loops: int) -> dict[int, float]:
    """Fraction of a fresh model's 512 tokens halting at each pass, router noise included.

    Halting is replayed from the router's recorded per-pass probabilities with _forward_act's
    rule, and the replay's mean is checked against the forward's own mean_loop_depth, so a change
    to that rule fails here instead of silently measuring something else."""
    torch.manual_seed(0)
    model = _small(d_model, max_loops=max_loops).eval()
    probs = []
    hook = model.router.register_forward_hook(lambda module, args, out: probs.append(out))
    with torch.no_grad():
        out = model(torch.randint(0, 256, (8, 64)))
    hook.remove()
    threshold = 1.0 - model.cfg.halt_epsilon
    cum_prob = torch.zeros_like(probs[0])
    running = torch.ones_like(probs[0], dtype=torch.bool)
    depth = torch.zeros_like(probs[0])
    for n, p in enumerate(probs, start=1):
        halts = running & ((cum_prob + p >= threshold) | (n == max_loops))
        depth += running.float()
        cum_prob = torch.where(running & ~halts, cum_prob + p, cum_prob)
        running &= ~halts
    assert depth.mean().item() == pytest.approx(out.mean_loop_depth.item())
    counts = collections.Counter(int(d) for d in depth.flatten())
    return {k: counts[k] / depth.numel() for k in sorted(counts)}


@pytest.mark.parametrize("max_loops", range(2, 9))
def test_derived_bias_halts_a_constant_prob_token_at_the_target_pass(max_loops):
    cfg = ModelConfig(use_router=True, max_loops=max_loops)
    p = 1.0 / (1.0 + math.exp(-initial_halt_bias(cfg)))
    t = 1.0 - cfg.halt_epsilon
    halts_at = next(n for n in range(1, max_loops + 1) if n * p >= t or n == max_loops)
    assert halts_at == _target_pass(max_loops)


@pytest.mark.parametrize("max_loops", range(2, 9))
def test_fresh_router_gets_a_gradient(max_loops):
    # A token halting at pass 1 gets weight 1 - 0 = 1, a constant, so if every token did the
    # router would get exactly zero gradient and depth would never move — which is what the
    # previous derivation did at max_loops 2.
    torch.manual_seed(0)
    model = _small(max_loops=max_loops).train()
    ids = torch.randint(0, 256, (2, 32))
    out = model(ids)
    loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)), ids[:, 1:].reshape(-1))
    (loss + model.cfg.ponder_weight * out.ponder_cost).backward()
    assert model.router.proj.weight.grad.abs().max() > 0
    assert model.router.proj.bias.grad.abs().max() > 0


@pytest.mark.parametrize("d_model", [64, 1024])
@pytest.mark.parametrize("max_loops", range(2, 9))
def test_fresh_model_starts_near_half_depth(d_model, max_loops):
    # Includes the router's per-token weight noise (sd ~0.02 * sqrt(d_model) in logits), not just
    # the bias. These are the claims initial_halt_bias, ModelConfig and docs/model.md make.
    hist = _init_depth_histogram(d_model, max_loops)
    k = _target_pass(max_loops)
    assert min(hist) >= 2  # never pass 1, at any width
    if max_loops <= 5:
        # k == 2: ~2.3 nats of margin either side, wider than the noise even at d_model 1024.
        assert hist == {2: 1.0}
    elif d_model == 64:
        # k >= 3: the interval is 0.34 (k 3) / 0.20 (k 4) nats wide either side, comparable to the
        # noise, so tokens spill onto neighbouring passes — but at this width only by one.
        assert max(hist, key=hist.get) == k
        assert min(hist) >= k - 1 and max(hist) <= k + 1
    else:
        # At d_model 1024 the noise (sd ~0.6) exceeds the margin: tokens spread over passes 2-7
        # and the mean sits below k (measured 2.77 at k 3, 3.57 at k 4).
        mean = sum(depth * frac for depth, frac in hist.items())
        assert k - 0.5 <= mean <= k


@pytest.mark.parametrize("halt_epsilon", [0.0, 1.0, 1.5, -0.1])
@pytest.mark.parametrize("bias", [None, -1.0])
def test_halt_epsilon_outside_unit_interval_is_rejected(halt_epsilon, bias):
    cfg = ModelConfig(use_router=True, max_loops=4, halt_epsilon=halt_epsilon, act_halt_bias_init=bias)
    with pytest.raises(ValueError, match="halt_epsilon"):
        initial_halt_bias(cfg)
    with pytest.raises(ValueError, match="halt_epsilon"):
        _small(max_loops=4, halt_epsilon=halt_epsilon, act_halt_bias_init=bias)


def test_router_with_a_single_pass_is_rejected():
    with pytest.raises(ValueError, match="max_loops"):
        _small(max_loops=1)


def test_explicit_bias_overrides_the_derived_one():
    cfg = ModelConfig(use_router=True, max_loops=4, act_halt_bias_init=-1.0)
    assert initial_halt_bias(cfg) == -1.0
    model = _small(max_loops=4, act_halt_bias_init=-1.0)
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
