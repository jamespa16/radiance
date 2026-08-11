"""Per-iteration conditioning, input injection, stochastic loop depth and truncated BPTT.

These are the features that give the weight-shared loop body a sense of *which pass it is on*.
The recurring risk with all of them is allocating a parameter bank that some configuration never
reaches — so `test_no_dead_parameters` is parameterised over the whole loop-mode matrix rather
than testing one config.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    cfg = ModelConfig(
        d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=32, **overrides,
    )
    torch.manual_seed(seed)
    return DenseTransformer(cfg, vocab_size=TINY_VOCAB)


LOOP_MODES = {
    "single": dict(loop_count=1),
    "looped": dict(loop_count=4),
    "lora": dict(loop_count=4, loop_iter_conditioning="lora"),
    "no_conditioning": dict(loop_count=4, loop_iter_conditioning="none"),
    "stochastic": dict(loop_count=4, loop_count_min=2, loop_count_max=5),
    "bptt": dict(loop_count=4, loop_bptt_window=2),
    "router": dict(use_router=True, max_loops=4),
    "moe": dict(loop_count=3, use_moe=True, n_experts=4),
    "nsa": dict(loop_count=3, use_nsa=True, nsa_block_size=4, nsa_top_k_blocks=2, doc_attention_mask=False),
    "hyper": dict(loop_count=3, hyper_conn_streams=4),
}


@pytest.mark.parametrize("mode", list(LOOP_MODES))
def test_no_dead_parameters(tiny_ids, mode):
    """Every allocated parameter must be reachable from the forward pass.

    A parameter that isn't costs optimizer state and never trains, and the failure is silent — it
    just sits at its initial value forever. This check has already caught two: blocks[0]'s
    value_lambda (blocks[0] produces v_first, never consumes it) and input_injection at
    loop_count=1 (injection starts at the second iteration, which never happens).
    """
    model = _build(**LOOP_MODES[mode]).train()
    model(tiny_ids(batch=2, seq=16)).logits.square().mean().backward()

    dead = [name for name, p in model.named_parameters() if p.grad is None]
    assert not dead, f"{mode}: parameters allocated but unreachable from forward: {dead}"


# --- iteration conditioning ------------------------------------------------------------------


def test_norm_gains_are_per_iteration_when_looping():
    model = _build(loop_count=4)
    assert model.blocks[1].ln1.n_variants == 4
    assert model.blocks[1].ln1.weight.shape == (4, 32)
    # blocks[0] runs once per forward, so it has nothing to condition on.
    assert model.blocks[0].ln1.n_variants == 1
    assert model.blocks[0].ln1.weight.shape == (32,)


def test_norm_gain_shape_is_unchanged_without_looping():
    """The 1-D shape must survive at loop_count=1 so existing checkpoints keep loading."""
    model = _build(loop_count=1)
    for block in model.blocks:
        assert block.ln1.weight.shape == (32,)


def test_legacy_1d_gain_broadcasts_into_variants():
    """Enabling conditioning on an existing run should start from exactly that run's weights."""
    flat = _build(loop_count=1)
    conditioned = _build(loop_count=4)

    state = flat.state_dict()
    conditioned.load_state_dict(state, strict=False)

    trained_gain = state["blocks.1.ln1.weight"]
    for variant in range(4):
        torch.testing.assert_close(conditioned.blocks[1].ln1.weight[variant], trained_gain)


def test_conditioning_is_inert_at_init(tiny_ids):
    """All gains start at 1.0 and all router/LoRA biases at 0, so a conditioned model matches an
    unconditioned one until training separates the variants."""
    ids = tiny_ids(batch=2, seq=16)
    off = _build(loop_count=4, loop_iter_conditioning="none").eval()
    for arm in ("norm_gains", "lora"):
        on = _build(loop_count=4, loop_iter_conditioning=arm).eval()
        # Load unconditioned -> conditioned, the direction the upgrade actually happens in: the
        # 1-D gains broadcast across variants via RMSNorm's pre-hook, and the conditioned model's
        # extra tensors (iter_bias, LoRA A/B) keep their inert initialisation.
        on.load_state_dict(off.state_dict(), strict=False)
        with torch.no_grad():
            torch.testing.assert_close(on(ids).logits, off(ids).logits, rtol=0, atol=0)


def test_conditioning_changes_output_once_variants_diverge(tiny_ids):
    ids = tiny_ids(batch=2, seq=16)
    model = _build(loop_count=4).eval()
    with torch.no_grad():
        before = model(ids).logits
        model.blocks[1].ln1.weight[2].mul_(1.5)  # make iteration 3 behave differently
        after = model(ids).logits
    assert (before - after).abs().max() > 1e-5


def test_router_iteration_bias_exists_and_is_zero(tiny_ids):
    act = _build(use_router=True, max_loops=4)
    torch.testing.assert_close(act.router.iter_bias, torch.zeros(4))

    moe = _build(loop_count=3, use_moe=True, n_experts=4)
    router = next(b.ffn.router for b in moe.blocks[1:] if hasattr(b.ffn, "router"))
    torch.testing.assert_close(router.iter_bias, torch.zeros(3, 4))


# --- input injection -------------------------------------------------------------------------


def test_input_injection_is_inert_at_init(tiny_ids):
    """Zero-initialised W_inj means h + W_inj @ x0 == h exactly, at any loop count."""
    ids = tiny_ids(batch=2, seq=16)
    on = _build(loop_count=4, loop_input_injection=True).eval()
    off = _build(loop_count=4, loop_input_injection=False).eval()
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k in off.state_dict()})

    assert on.input_injection is not None
    torch.testing.assert_close(on.input_injection.weight, torch.zeros(32, 32))
    with torch.no_grad():
        torch.testing.assert_close(on(ids).logits, off(ids).logits, rtol=0, atol=0)


def test_input_injection_changes_output_once_trained(tiny_ids):
    ids = tiny_ids(batch=2, seq=16)
    model = _build(loop_count=4).eval()
    with torch.no_grad():
        before = model(ids).logits
        torch.nn.init.normal_(model.input_injection.weight, std=0.02)
        after = model(ids).logits
    assert (before - after).abs().max() > 1e-5


def test_act_injection_preserves_frozen_state(tiny_ids):
    """In ACT mode, injecting into an already-halted position would change the key/value that
    still-running positions attend to — the frozen-state invariant the whole mechanism rests on.
    Verified indirectly: with halt_epsilon forcing immediate halting, output must not depend on
    the injection weights at all."""
    ids = tiny_ids(batch=2, seq=16)
    model = _build(use_router=True, max_loops=4, halt_epsilon=0.99).eval()
    with torch.no_grad():
        before = model(ids).logits
        torch.nn.init.normal_(model.input_injection.weight, std=0.5)
        after = model(ids).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)


# --- stochastic depth and BPTT ---------------------------------------------------------------


def test_stochastic_depth_is_inert_when_range_unset():
    model = _build(loop_count=3).train()
    assert {model.resolved_loop_count() for _ in range(20)} == {3}


def test_stochastic_depth_samples_in_range_while_training():
    model = _build(loop_count=4, loop_count_min=2, loop_count_max=5)
    model.train()
    sampled = {model.resolved_loop_count() for _ in range(200)}
    assert sampled == {2, 3, 4, 5}


def test_stochastic_depth_is_deterministic_outside_training():
    """Eval and generation must not have loss or sampled text move run to run."""
    model = _build(loop_count=4, loop_count_min=2, loop_count_max=5).eval()
    assert {model.resolved_loop_count() for _ in range(20)} == {5}


def test_loop_multiplier_sizes_worst_case():
    """KV-cache slots and depth-scaled init must use the top of the range, not loop_count."""
    cfg = ModelConfig(loop_count=4, loop_count_min=2, loop_count_max=7)
    assert cfg.loop_multiplier == 7
    assert ModelConfig(use_router=True, max_loops=6, loop_count=2).loop_multiplier == 6
    assert ModelConfig(loop_count=3).loop_multiplier == 3


def test_loop_count_override_runs_more_iterations(tiny_ids):
    """Test-time compute scaling: more loop passes than the config's default, same weights."""
    ids = tiny_ids(batch=1, seq=8)
    model = _build(loop_count=2, loop_count_min=1, loop_count_max=4).eval()
    with torch.no_grad():
        default = model(ids).logits
        deeper = model(ids, loop_count=8).logits
    assert (default - deeper).abs().max() > 1e-5

    cache = model.new_kv_cache(loop_count=8)
    with torch.no_grad():
        model(ids, kv_cache=cache, loop_count=8)
    assert cache._cursor == len(cache._k)


def test_bptt_window_still_produces_gradients(tiny_ids):
    """Truncating backprop must narrow *which iterations* carry gradient, not break training."""
    model = _build(loop_count=4, loop_bptt_window=2).train()
    model(tiny_ids(batch=2, seq=16)).logits.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_bptt_keeps_first_block_trainable(tiny_ids):
    """The subtle failure mode of truncated BPTT here: running early iterations under no_grad cuts
    blocks[0]'s only path to the loss, so it silently never trains while the loss still falls.
    Input injection (of blocks[0]'s *output*) is what reconnects it.

    Note the one-step cold start: W_inj is zero-initialised, so at step 0 the anchor path exists
    but has zero derivative. W_inj itself gets a nonzero gradient immediately (its own gradient is
    grad_out (x) anchor, which doesn't vanish), so from step 1 on blocks[0] trains normally. Here
    that first step is simulated by perturbing W_inj.
    """
    ids = tiny_ids(batch=2, seq=16)

    model = _build(loop_count=4, loop_bptt_window=2).train()
    model(ids).logits.square().mean().backward()
    # Reachable from the loss at init (a severed graph would leave these as None)...
    assert all(p.grad is not None for p in model.blocks[0].parameters())
    # ...and W_inj is what bootstraps the path, so it must move on the very first step.
    assert model.input_injection.weight.grad.abs().sum() > 0

    model.zero_grad()
    torch.nn.init.normal_(model.input_injection.weight, std=0.02)  # stand in for one step
    model(ids).logits.square().mean().backward()
    for name, p in model.blocks[0].named_parameters():
        assert p.grad.abs().sum() > 0, f"blocks.0.{name} still frozen after W_inj became nonzero"


def test_bptt_without_injection_is_rejected():
    """Rather than silently training a model whose first block is frozen."""
    with pytest.raises(ValueError, match="loop_bptt_window"):
        _build(loop_count=4, loop_bptt_window=2, loop_input_injection=False)


def test_bptt_window_does_not_change_the_forward(tiny_ids):
    """It is a backward-pass memory lever; the forward numbers must be untouched."""
    ids = tiny_ids(batch=2, seq=16)
    windowed = _build(loop_count=4, loop_bptt_window=2).eval()
    full = _build(loop_count=4).eval()
    with torch.no_grad():
        torch.testing.assert_close(windowed(ids).logits, full(ids).logits, rtol=0, atol=0)
