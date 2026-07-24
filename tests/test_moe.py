"""MoE routing: capacity, load balancing, the shared expert, and fine-grained experts."""

from __future__ import annotations

import pytest
import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer, MoEFeedForward
from tests.conftest import TINY_VOCAB


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    fields = dict(
        d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1, dropout=0.0,
        max_seq_len=64, use_moe=True, n_experts=4, moe_top_k=2,
    )
    fields.update(overrides)
    torch.manual_seed(seed)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB)


def _moe_layers(model: DenseTransformer) -> list[MoEFeedForward]:
    return [b.ffn for b in model.blocks if isinstance(b.ffn, MoEFeedForward)]


# --- capacity ---------------------------------------------------------------------------------


def test_eval_capacity_makes_output_independent_of_batching():
    """A token's output must not depend on how many other tokens shared its forward pass.

    Under the training capacity formula it did: capacity scales with n_tokens, so the same
    sequence scored alone and scored inside a batch dropped different tokens. That made val/loss
    batch-size-dependent and made incremental decoding diverge from a full forward.
    """
    model = _build(loop_count=2).eval()
    torch.manual_seed(1)
    seq = torch.randint(0, TINY_VOCAB, (1, 16))
    padding = torch.randint(0, TINY_VOCAB, (3, 16))

    with torch.no_grad():
        alone = model(seq).logits
        batched = model(torch.cat([seq, padding])).logits[:1]

    torch.testing.assert_close(alone, batched, rtol=1e-5, atol=1e-5)


def test_training_still_uses_bounded_capacity():
    """Capacity limits must stay in training mode — they're what bounds the gather's size, and so
    what keeps step time and memory independent of how unbalanced routing happens to be."""
    model = _build(loop_count=2).train()
    layer = _moe_layers(model)[0]
    n_tokens = 4 * 16
    expected = round(layer.cfg.moe_capacity_factor * n_tokens * layer.cfg.moe_top_k / layer.cfg.n_experts)

    from radiance.model import _moe_capacity

    assert _moe_capacity(layer.cfg, n_tokens) == expected


def test_drop_rate_is_low_at_a_realistic_batch():
    """Documented in CLAUDE.md as under 1% on average; a regression here means routing collapsed."""
    model = _build(loop_count=2).train()
    layer = _moe_layers(model)[0]
    torch.manual_seed(0)
    x = torch.randn(16, 64, 32)

    probs = layer.router(x.reshape(-1, 32))
    assigned = torch.zeros_like(probs).scatter_(1, probs.topk(2, dim=-1).indices, 1.0)
    capacity = round(layer.cfg.moe_capacity_factor * x[:, :, 0].numel() * 2 / 4)
    overflow = (assigned.sum(dim=0) - capacity).clamp(min=0).sum()

    assert overflow / assigned.sum() < 0.01


# --- load balancing ---------------------------------------------------------------------------


def test_aux_loss_is_near_ideal_at_init():
    """The Switch formulation reports n_experts * sum(f_i * P_i), which is 1.0 per layer under
    perfect balance and n_experts/... in the collapsed case. CLAUDE.md records 2.01-2.10 per layer
    at top_k=2 against an ideal of 2.0."""
    model = _build(loop_count=1).train()
    torch.manual_seed(0)
    model(torch.randint(0, TINY_VOCAB, (8, 32))).logits.sum().backward()

    for layer in _moe_layers(model):
        per_layer = layer.aux_loss_accum / layer._aux_loss_calls
        assert 1.8 < per_layer.item() < 2.4, f"per-layer aux loss {per_layer.item()} far from ideal 2.0"


def test_expert_bias_starts_at_zero_and_is_not_a_parameter():
    """It's updated by an explicit rule, never by gradient — that's what makes it 'loss-free'."""
    model = _build()
    layer = _moe_layers(model)[0]
    torch.testing.assert_close(layer.expert_bias, torch.zeros(4))
    assert not isinstance(layer.expert_bias, torch.nn.Parameter)
    assert "expert_bias" not in dict(model.named_parameters())


def test_expert_bias_moves_toward_underloaded_experts():
    model = _build(loop_count=1).train()
    layer = _moe_layers(model)[0]

    # Expert 0 heavily loaded, expert 3 idle.
    layer.expert_load = torch.tensor([100.0, 10.0, 10.0, 0.0])
    layer.update_expert_bias()

    assert layer.expert_bias[0] < 0, "overloaded expert's bias must decrease"
    assert layer.expert_bias[3] > 0, "idle expert's bias must increase"
    assert layer.expert_load.sum() == 0, "load counter must reset after the update"


def test_bias_only_balancing_drops_the_aux_loss_term():
    """With 'bias', load is balanced without a gradient term, so the aux loss shouldn't be applied
    at all — it's returned as an exact zero rather than weighted down."""
    model = _build(loop_count=2, moe_balance="bias").train()
    out = model(torch.randint(0, TINY_VOCAB, (2, 16)))
    assert out.moe_aux_loss.item() == 0.0

    both = _build(loop_count=2, moe_balance="both").train()
    assert both(torch.randint(0, TINY_VOCAB, (2, 16))).moe_aux_loss.item() > 0.0


def test_bias_affects_selection_but_not_gate_weights():
    """The separation that makes the bias loss-free: it steers *which* experts a token uses without
    distorting *how much* each contributes."""
    model = _build(loop_count=1, moe_balance="bias").eval()
    layer = _moe_layers(model)[0]
    torch.manual_seed(0)
    x = torch.randn(4, 32)

    with torch.no_grad():
        before = layer(x)
        layer.expert_bias[0] = 10.0  # force expert 0 into every token's top-k
        after = layer(x)

    assert (before - after).abs().max() > 1e-5


# --- shared expert ------------------------------------------------------------------------------


def test_shared_expert_processes_every_token():
    """Unconditional and uncapped: unlike a routed expert it can never drop a token."""
    model = _build(loop_count=2, moe_n_shared=1).eval()
    layer = _moe_layers(model)[0]
    assert layer.shared_expert is not None

    torch.manual_seed(0)
    x = torch.randn(4, 32)
    with torch.no_grad():
        with_shared = layer(x)
        layer.shared_expert = None
        without = layer(x)
    # Every token's output must change, not just the routed ones.
    assert ((with_shared - without).abs().amax(-1) > 1e-6).all()


def test_shared_expert_counts_fully_in_active_parameters():
    """Chinchilla-style sizing counts parameters that multiply against every token. Routed experts
    are discounted to top_k; the shared expert is not, because every token activates it."""
    without = _build(loop_count=2, moe_n_shared=0)
    with_shared = _build(loop_count=2, moe_n_shared=1)

    shared_params = sum(
        sum(p.numel() for p in layer.shared_expert.parameters())
        for layer in _moe_layers(with_shared)
    )
    delta = with_shared.num_active_parameters() - without.num_active_parameters()
    assert delta == shared_params


def test_routed_experts_are_still_discounted():
    """num_active_parameters must count only moe_top_k experts' worth of routed FFN parameters."""
    model = _build(loop_count=2, n_experts=8, moe_top_k=2, moe_n_shared=0)
    inactive = sum(
        layer.per_expert_parameter_count() * (8 - 2) for layer in _moe_layers(model)
    )
    assert model.num_parameters() - model.num_active_parameters() == inactive


# --- fine-grained experts -----------------------------------------------------------------------


def test_expert_width_defaults_to_ffn_dim():
    cfg = ModelConfig(d_model=32, ffn_mult=2.0, use_moe=True)
    assert cfg.moe_expert_dim == cfg.ffn_dim


def test_fine_grained_experts_keep_active_width_constant():
    """4x the experts at 1/4 the width and 4x top_k activates the same parameter count per token,
    but from far more expert combinations."""
    coarse = ModelConfig(d_model=64, ffn_mult=4.0, use_moe=True, n_experts=8, moe_top_k=2)
    fine = ModelConfig(
        d_model=64, ffn_mult=4.0, use_moe=True, n_experts=32, moe_top_k=8, moe_expert_ffn_mult=0.25
    )
    assert fine.moe_expert_dim * fine.moe_top_k == coarse.moe_expert_dim * coarse.moe_top_k


def test_fine_grained_model_builds_and_runs(tiny_ids):
    model = _build(loop_count=2, n_experts=16, moe_top_k=4, moe_expert_ffn_mult=0.25).train()
    out = model(tiny_ids(batch=2, seq=16))
    assert out.logits.shape == (2, 16, TINY_VOCAB)
    out.logits.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


# --- legacy checkpoints -------------------------------------------------------------------------


def test_legacy_per_expert_checkpoint_still_loads():
    """MoE checkpoints predating the batched baddbmm layout store experts.{e}.{gate,up,down}_proj.*
    as separate nn.Linear weights; the pre-hook transposes and stacks them."""
    model = _build(loop_count=2, ffn_depth=1)
    layer = _moe_layers(model)[0]
    n_experts, d_model, ffn_dim = 4, 32, layer.cfg.moe_expert_dim

    state = {k: v.clone() for k, v in model.state_dict().items()}
    prefix = next(k for k in state if k.endswith("experts.gate_w")).removesuffix("gate_w")
    for name in ("gate_w", "up_w", "down_w", "gate_b", "up_b", "down_b"):
        state.pop(prefix + name)
    for e in range(n_experts):
        for old, shape in (("gate_proj", (ffn_dim, d_model)), ("up_proj", (ffn_dim, d_model)),
                           ("down_proj", (d_model, ffn_dim))):
            state[f"{prefix}{e}.{old}.weight"] = torch.randn(*shape)
            state[f"{prefix}{e}.{old}.bias"] = torch.randn(shape[0])

    model.load_state_dict(state, strict=False)  # must not raise
    assert layer.experts.gate_w.shape == (n_experts, d_model, ffn_dim)
