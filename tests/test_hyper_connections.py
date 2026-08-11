"""Hyper-connections (cfg.hyper_conn_streams): n parallel residual streams instead of one.

Three properties carry the whole feature, and each one has a way of failing silently:

* At `hyper_conn_streams: 1` nothing exists — no parameters, no extra tensor dimension, the same
  code path as before. That is what makes the default safe.
* At n > 1 the model *starts* as the plain residual network, because the streams begin identical
  and every sublayer writes the same output into all of them. Break this and every existing config
  changes its init, silently.
* The streams must actually be able to diverge. The tempting symmetric init (`alpha_m = 1/n`
  instead of the staggered one-hot) satisfies both properties above and is still completely
  broken: every connection coefficient then receives an identical gradient forever, so n streams
  behave as one for the life of the run and the only symptom is an A/B that comes back flat.
  test_static_init_staggers_across_blocks_and_variants is what pins that down.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import Config, ModelConfig, TrainConfig
from radiance.model import DenseTransformer
from radiance.optim import build_muon_param_groups, build_param_groups
from tests.conftest import TINY_VOCAB


# The loop-mode matrix, mirroring tests/test_loop_identity.py: hyper-connections replace the
# residual write in *every* one of these paths, including the two that reimplement it (ACT's
# capacity-based sparse loop body and the FFN-only sparse dispatch).
MODES = {
    "dense": dict(loop_count=1),
    "looped": dict(loop_count=3),
    "no_conditioning": dict(loop_count=4, loop_iter_conditioning="none"),
    "lora": dict(loop_count=4, loop_iter_conditioning="lora"),
    "stochastic": dict(loop_count=4, loop_count_min=2, loop_count_max=5),
    "bptt": dict(loop_count=4, loop_bptt_window=2),
    "router": dict(use_router=True, max_loops=4),
    "router_block_sparse": dict(use_router=True, max_loops=4, act_capacity_ratio=0.5),
    "router_ffn_sparse": dict(use_router=True, max_loops=4, act_ffn_capacity_ratio=0.5),
    "moe": dict(use_moe=True, n_experts=4, loop_count=2),
    "gqa": dict(n_kv_heads=2, loop_count=2),
    "grad_checkpoint": dict(loop_count=2, grad_checkpoint=True),
    "mtp": dict(mtp_heads=3, loop_count=2),
    "nsa": dict(loop_count=3, use_nsa=True, nsa_block_size=4, nsa_top_k_blocks=2,
                doc_attention_mask=False),
}


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    cfg = ModelConfig(
        d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=32, **overrides,
    )
    torch.manual_seed(seed)
    return DenseTransformer(cfg, vocab_size=TINY_VOCAB)


def _logits(model: DenseTransformer, ids: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(1234)  # same dropout/tiebreak stream on both sides of a comparison
    return model(ids).logits


def _share_weights(src: DenseTransformer, dst: DenseTransformer) -> None:
    """Copy every parameter the two models have in common, leaving the rest at its own init."""
    dst.load_state_dict(
        {k: v for k, v in src.state_dict().items() if k in dst.state_dict()}, strict=False
    )


def _hyper_names(model: DenseTransformer) -> list[str]:
    return [name for name, _ in model.named_parameters() if "hyper" in name]


def test_single_stream_allocates_nothing():
    """The default must be free, not merely inert: no parameters, no optimizer state, no branch."""
    model = _build(loop_count=3)
    assert _hyper_names(model) == []
    assert all(b.hyper_attn is None and b.hyper_ffn is None for b in model.blocks)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("streams", [2, 4])
def test_residual_init_matches_the_single_stream_model(tiny_ids, mode, streams):
    """At init, n streams compute exactly what one stream did.

    alpha_m starts one-hot, alpha_r the identity and beta all-ones, so every stream receives the
    same sublayer output and they stay equal at every depth — the read just picks one of n
    identical copies. Both matmuls are exact at those values (multiply by 1.0, sum exact zeros),
    and _reduce_streams averages rather than sums specifically so the reduced hidden keeps
    single-stream scale; hence this is bit-identical at power-of-two n rather than merely close.
    See test_odd_stream_counts_are_close_but_not_bit_exact for the one caveat.
    """
    ids = tiny_ids(batch=2, seq=16)
    one = _build(**MODES[mode]).train()
    many = _build(hyper_conn_streams=streams, **MODES[mode]).train()
    _share_weights(one, many)

    torch.testing.assert_close(_logits(one, ids), _logits(many, ids), rtol=0, atol=0)


def test_odd_stream_counts_are_close_but_not_bit_exact(tiny_ids):
    """n=3 rounds where n=2/4 do not, and it is the reduction's mean that does it.

    Averaging three identical floats is exact in real arithmetic but `(z + z) + z` is not
    representable for every z, so the equivalence above degrades from bit-exact to ~1 ulp. Worth
    pinning so nobody reads the zero-tolerance test above as a claim about every n.
    """
    ids = tiny_ids(batch=2, seq=16)
    one = _build(loop_count=3).train()
    many = _build(hyper_conn_streams=3, loop_count=3).train()
    _share_weights(one, many)

    torch.testing.assert_close(_logits(one, ids), _logits(many, ids), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", list(MODES))
def test_dynamic_hyper_connections_are_inert_at_init(tiny_ids, mode):
    """The dynamic corrections are zero-initialised, and tanh(0) is exactly 0."""
    ids = tiny_ids(batch=2, seq=16)
    static = _build(hyper_conn_streams=4, hyper_conn_dynamic=False, **MODES[mode]).train()
    dynamic = _build(hyper_conn_streams=4, hyper_conn_dynamic=True, **MODES[mode]).train()
    _share_weights(static, dynamic)

    torch.testing.assert_close(_logits(static, ids), _logits(dynamic, ids), rtol=0, atol=0)


def test_streams_diverge_once_static_weights_move(tiny_ids):
    """The other half of the inert-default contract: it must stop being a no-op when trained.

    Checked at the streams themselves, not just the logits — identical streams are the precise
    condition under which hyper-connections are a rename of the residual stream.
    """
    ids = tiny_ids(batch=2, seq=16)
    model = _build(hyper_conn_streams=4, loop_count=3).eval()

    with torch.no_grad():
        hidden = model._expand_streams(model.token_emb(ids))
        cos, sin = model.rope(ids.size(1))
        from radiance.model import LoopContext

        at_init, _, _ = model.blocks[1](hidden, cos, sin, LoopContext(iteration=1))
        assert (at_init - at_init[..., :1, :]).abs().max() == 0.0

        before = model(ids).logits
        for name, param in model.named_parameters():
            if "hyper" in name and "dyn" not in name:
                param.add_(torch.randn_like(param) * 0.3)
        trained, _, _ = model.blocks[1](hidden, cos, sin, LoopContext(iteration=1))
        after = model(ids).logits

    assert (trained - trained[..., :1, :]).abs().max() > 1e-5, "streams stayed identical"
    assert (before - after).abs().max() > 1e-5, "logits unchanged after the weights moved"


def test_dynamic_projections_reach_the_loss_immediately(tiny_ids):
    """Zero-init is a plateau if nothing gradients through it. tanh'(0) = 1, so the projections
    themselves get a real gradient on the very first step."""
    model = _build(hyper_conn_streams=4, loop_count=3).train()
    model(tiny_ids(batch=2, seq=16)).logits.square().mean().backward()

    dyn = [(n, p) for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] == "dyn_proj"]
    assert dyn, "dynamic hyper-connection projections were not built"
    assert all(p.grad is not None and p.grad.abs().max() > 0 for _, p in dyn)


def test_dynamic_scales_cold_start_for_exactly_one_step(tiny_ids):
    """The `s` in `s * tanh(norm(H) @ W)` has zero gradient while W is still zero, since tanh(0)
    is exactly 0 — the same one-step cold start loop_input_injection's W_inj has, and benign for
    the same reason: W moves on step 0, so s trains from step 1 on. Pinned because a parameter
    that stays at its init forever is the exact failure test_no_dead_parameters exists for, and
    `grad == 0` slips past that check while `grad is None` does not.
    """
    ids = tiny_ids(batch=2, seq=16)
    model = _build(hyper_conn_streams=4, loop_count=3).train()
    scales = [p for n, p in model.named_parameters() if "dyn_scale" in n]
    assert scales

    model(ids).logits.square().mean().backward()
    assert all(p.grad.abs().max() == 0 for p in scales), "expected a step-0 cold start"

    with torch.no_grad():  # stand in for the optimizer step that moves W off zero
        for name, param in model.named_parameters():
            if name.rsplit(".", 1)[-1] == "dyn_proj":
                param.add_(torch.randn_like(param) * 0.02)
    model.zero_grad()
    model(ids).logits.square().mean().backward()
    assert all(p.grad.abs().max() > 0 for p in scales), "cold start never ended"


@pytest.mark.parametrize("streams", [2, 3, 4])
def test_static_init_staggers_across_blocks_and_variants(streams):
    """Each sublayer reads a different stream, and each loop iteration reads a different one again.

    This is the property that lets the streams diverge at all (see the module docstring), and the
    per-variant half is what gives a weight-shared loop body a genuinely different routing per
    pass rather than replaying one.

    Parametrized over stream counts on purpose. Advancing the variant by the loop body's true
    unrolled sublayer count — the obvious reading of the paper's `k mod n`, and what this was
    written as first — aliases to a single stream at streams=2 for *every* model, because that
    count is always even. Nothing fails when it does; the feature just quietly stops doing the
    thing it is here to do.
    """
    model = _build(hyper_conn_streams=streams, loop_count=6)

    for block in model.blocks:
        for unit in (block.hyper_attn, block.hyper_ffn):
            assert unit.alpha_m.sum(dim=-1).eq(1.0).all(), "alpha_m must start one-hot"
            torch.testing.assert_close(
                unit.alpha_r, torch.eye(streams).expand_as(unit.alpha_r), rtol=0, atol=0
            )
            assert unit.beta.eq(1.0).all()

    # Adjacent sublayers of the same block read different streams...
    first = model.blocks[1]
    assert first.hyper_attn.alpha_m[0].argmax() != first.hyper_ffn.alpha_m[0].argmax()
    # ...and every stream is read by some iteration rather than the schedule collapsing.
    reads = first.hyper_attn.alpha_m.argmax(dim=-1)
    assert reads.unique().numel() == streams, f"schedule collapsed: {reads.tolist()}"


def test_variant_selection_clamps_past_the_trained_depth():
    """radiance-generate --loops N can run deeper than training did; reuse the deepest routing
    rather than wrapping back to the shallowest. Mirrors RMSNorm's per-variant gain selection."""
    model = _build(hyper_conn_streams=4, loop_count=3)
    unit = model.blocks[1].hyper_attn
    hidden = torch.randn(1, 4, 4, 32)

    last = unit.read(hidden, unit.n_variants - 1)[0]
    beyond = unit.read(hidden, unit.n_variants + 10)[0]
    torch.testing.assert_close(last, beyond, rtol=0, atol=0)


def _groups(model, optimizer="muon"):
    cfg = Config()
    cfg.train = TrainConfig(optimizer=optimizer)
    if optimizer == "muon":
        return build_muon_param_groups(model, cfg)
    return build_param_groups(
        model,
        cfg.train.weight_decay,
        embed_lr=cfg.train.embed_lr_resolved,
        hyper_conn_lr=cfg.train.hyper_conn_lr,
    )


def test_hyper_params_get_their_own_learning_rate():
    """The coefficients are structural (one-hot read, identity mix) and AdamW's step is ~lr
    regardless of gradient scale, so at train.lr's post-Muon 1e-2 a few hundred steps erase the
    structure rather than refine it. Measured: 2.697 vs 2.110 val/loss. They get their own group."""
    cfg = Config()
    cfg.train = TrainConfig(lr=1.0e-2, hyper_conn_lr=1.0e-3)
    model = _build(hyper_conn_streams=4, loop_count=3)

    for groups in (build_muon_param_groups(model, cfg),
                   build_param_groups(model, cfg.train.weight_decay,
                                      embed_lr=cfg.train.embed_lr_resolved,
                                      hyper_conn_lr=cfg.train.hyper_conn_lr)):
        hyper_ids = {id(p) for n, p in model.named_parameters() if "hyper" in n}
        for group in groups:
            if any(id(p) in hyper_ids for p in group["params"]):
                assert all(id(p) in hyper_ids for p in group["params"]), "group mixes hyper params"
                assert group["lr"] == 1.0e-3


def test_hyper_params_are_not_routed_to_muon():
    """alpha_r is (variants, n, n) and dyn_proj is (variants, d_model, 2 + n) — both >= 2-D, so without
    the "hyper" exclusion they fall straight through to Muon, whose fixed-spectral-norm update is
    exactly wrong for a one-hot read and an identity depth mix."""
    model = _build(hyper_conn_streams=4, loop_count=3)
    groups = _groups(model)
    muon_ids = {id(p) for g in groups if g["algorithm"] == "muon" for p in g["params"]}

    hyper = [p for n, p in model.named_parameters() if "hyper" in n]
    assert hyper
    assert not any(id(p) in muon_ids for p in hyper)


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_static_hyper_params_get_no_weight_decay(optimizer):
    """Decaying a one-hot read and an identity depth mix would drag the model off exactly the
    initialisation that makes it a residual network. The paper splits the same way: it decays only
    the dynamic projections."""
    model = _build(hyper_conn_streams=4, loop_count=3)
    groups = _groups(model, optimizer)
    decayed = {id(p) for g in groups if g["weight_decay"] != 0.0 for p in g["params"]}

    for name, param in model.named_parameters():
        if "hyper" not in name:
            continue
        # Only the dynamic projection is decayed: the static coefficients are structural, and
        # dyn_scale_* are 1-D scale parameters the ordinary dim() < 2 rule would spare anyway.
        should_decay = name.rsplit(".", 1)[-1] == "dyn_proj"
        assert (id(param) in decayed) == should_decay, name


def test_every_parameter_still_lands_in_exactly_one_group():
    model = _build(hyper_conn_streams=4, loop_count=3, use_moe=True, n_experts=4)
    seen = [id(p) for g in _groups(model) for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())
    assert len(seen) == len(set(seen))


def test_activation_estimate_grows_with_streams():
    """auto_batch_size defaults on, so an unupdated estimate would hand n=4 the same micro-batch
    as n=1 and OOM at the first step."""
    one = _build(loop_count=3).activation_bytes_per_token(2)
    four = _build(hyper_conn_streams=4, loop_count=3).activation_bytes_per_token(2)
    assert four > one


@pytest.mark.parametrize("mode", list(MODES))
def test_no_dead_parameters(tiny_ids, mode):
    """Every allocated parameter must be reachable from the forward pass — see
    tests/test_loop_identity.py::test_no_dead_parameters for why this keeps earning its place."""
    model = _build(hyper_conn_streams=4, **MODES[mode]).train()
    out = model(tiny_ids(batch=2, seq=16))
    loss = out.logits.square().mean()
    for hidden in out.mtp_hidden or ():
        loss = loss + hidden.square().mean()
    loss.backward()

    dead = [name for name, p in model.named_parameters() if p.grad is None]
    assert not dead, f"{mode}: parameters allocated but unreachable from forward: {dead}"


@pytest.mark.parametrize("mode", ["dense", "looped", "router", "moe", "gqa"])
def test_generation_matches_a_full_forward(tiny_ids, mode):
    """Hyper-connections live between sublayers and never touch K/V, so cached decoding must stay
    exactly equivalent to a full forward — the same contract tests/test_kv_cache.py pins."""
    ids = tiny_ids(batch=1, seq=12)
    model = _build(hyper_conn_streams=4, **MODES[mode]).eval()

    with torch.no_grad():
        full = model(ids).logits
        cache = model.new_kv_cache()
        prefill = model(ids[:, :-1], kv_cache=cache).logits
        step = model(ids[:, -1:], kv_cache=cache).logits

    torch.testing.assert_close(prefill[:, -1], full[:, -2], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(step[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)
