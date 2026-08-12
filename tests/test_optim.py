"""Muon's Newton-Schulz iteration, parameter routing, muP scaling, and the CPU-offload OOM tier."""

from __future__ import annotations

import pytest
import torch

import radiance.optim as optim_module
from radiance.config import Config, ModelConfig, TrainConfig
from radiance.model import DenseTransformer
from radiance.optim import (
    MuonWithAuxAdam,
    build_muon_param_groups,
    build_optimizer,
    build_param_groups,
    migrate_optimizer_to_cpu_offload,
    orthogonalize,
)
from tests.conftest import TINY_VOCAB


# --- Newton-Schulz -------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32), (64, 16), (16, 64)])
def test_orthogonalize_drives_singular_values_toward_one(shape):
    """The point of Muon: every singular value of the update ends up near 1, so the step is spread
    evenly across directions instead of being dominated by the largest few."""
    torch.manual_seed(0)
    g = torch.randn(*shape)
    s_before = torch.linalg.svdvals(g.float())
    s_after = torch.linalg.svdvals(orthogonalize(g).float())

    # Wide spread before (a random Gaussian's singular values span a Marchenko-Pastur bulk)...
    assert s_before.max() / s_before.min() > 2.0
    # ...and banded around 1 after. Deliberately a band, not an equality: the tuned quintic
    # coefficients trade exact convergence for speed and settle into roughly [0.68, 1.13] after 5
    # steps rather than approaching 1. That is the intended operating point — the optimizer needs
    # the *spread* collapsed, not the values exact.
    assert 0.6 < s_after.min() and s_after.max() < 1.4
    assert s_after.max() / s_after.min() < 2.0
    assert s_after.max() / s_after.min() < s_before.max() / s_before.min()


def test_orthogonalize_is_batched_over_leading_dims():
    """BatchedExperts stores weights as (n_experts, in, out); Muon must handle that shape directly,
    orthogonalising each expert independently rather than treating the stack as one matrix."""
    torch.manual_seed(0)
    stacked = torch.randn(4, 16, 32)
    batched = orthogonalize(stacked)
    assert batched.shape == stacked.shape

    for e in range(stacked.size(0)):
        torch.testing.assert_close(batched[e], orthogonalize(stacked[e]), rtol=1e-2, atol=1e-2)


# --- parameter routing ---------------------------------------------------------------------


def _model_and_cfg(**model_kwargs):
    cfg = Config(
        model=ModelConfig(
            d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
            dropout=0.0, max_seq_len=32, **model_kwargs,
        ),
        train=TrainConfig(optimizer="muon", auto_batch_size=False, device="cpu", compile=False),
    )
    torch.manual_seed(0)
    return DenseTransformer(cfg.model, vocab_size=TINY_VOCAB), cfg


def test_embedding_and_norms_are_not_routed_to_muon():
    """The tied embedding is per-token rows, not a hidden linear map, and is the largest tensor in
    the model — orthogonalising it is both meaningless and the dominant cost."""
    model, cfg = _model_and_cfg()
    groups = build_muon_param_groups(model, cfg)
    muon_params = {id(p) for g in groups if g["algorithm"] == "muon" for p in g["params"]}

    assert id(model.token_emb.weight) not in muon_params
    assert id(model.lm_head.weight) not in muon_params  # same tied tensor
    for block in model.blocks:
        assert id(block.ln1.weight) not in muon_params
        assert id(block.attn.out_gate.weight) not in muon_params
        # ...but the real hidden matrices are:
        assert id(block.attn.qkv_proj.weight) in muon_params
        assert id(block.ffn.down_proj.weight) in muon_params


def test_every_parameter_lands_in_exactly_one_group():
    """A parameter silently dropped from every group would simply never train."""
    model, cfg = _model_and_cfg(use_moe=True, n_experts=4)
    groups = build_muon_param_groups(model, cfg)

    routed = [id(p) for g in groups for p in g["params"]]
    assert len(routed) == len(set(routed)), "a parameter appears in more than one group"
    assert set(routed) == {id(p) for p in model.parameters() if p.requires_grad}


def test_moe_expert_stacks_are_routed_to_muon():
    """BatchedExperts' 3-D weights are hidden matrices and should get Muon, not fall through to
    AdamW just because they aren't 2-D."""
    model, cfg = _model_and_cfg(use_moe=True, n_experts=4)
    groups = build_muon_param_groups(model, cfg)
    muon_params = {id(p) for g in groups if g["algorithm"] == "muon" for p in g["params"]}

    moe = next(b.ffn for b in model.blocks[1:] if hasattr(b.ffn, "experts"))
    assert id(moe.experts.gate_w) in muon_params
    assert id(moe.experts.down_w) in muon_params
    assert id(moe.router.proj.weight) not in muon_params  # routers stay on AdamW



def test_muon_group_uses_muon_lr():
    """Muon needs a ~50x larger LR than AdamW; the two must not share one field."""
    model, cfg = _model_and_cfg()
    cfg.train.lr, cfg.train.muon_lr = 3e-4, 0.02
    groups = build_muon_param_groups(model, cfg)

    assert next(g["lr"] for g in groups if g["algorithm"] == "muon") == 0.02
    assert all(g["lr"] == 3e-4 for g in groups if g["algorithm"] == "adamw")


def test_embed_lr_is_inert_when_unset():
    """cfg.train.embed_lr defaults to None, which resolves to `lr` — so the embedding gets its own
    group but at exactly the LR it had when it shared AdamW's decayed group. Every existing config
    therefore trains identically."""
    model, cfg = _model_and_cfg()
    cfg.train.lr, cfg.train.embed_lr = 3e-4, None
    groups = build_muon_param_groups(model, cfg)

    assert cfg.train.embed_lr_resolved == 3e-4
    assert all(g["lr"] == 3e-4 for g in groups if g["algorithm"] == "adamw")


def test_embed_lr_applies_to_the_embedding_alone():
    """Set, it must move the tied token_emb/lm_head matrix and nothing else: the routers, gates and
    norm gains AdamW also owns keep `lr`, since their exact scale is load-bearing in a way the
    embedding's is not."""
    model, cfg = _model_and_cfg()
    cfg.train.lr, cfg.train.embed_lr = 3e-4, 0.05
    groups = build_muon_param_groups(model, cfg)

    embed_groups = [g for g in groups if any(p is model.token_emb.weight for p in g["params"])]
    assert len(embed_groups) == 1, "the tied matrix must live in exactly one group"
    assert embed_groups[0]["lr"] == 0.05
    assert embed_groups[0]["algorithm"] == "adamw"  # never orthogonalised
    # It is alone there — the gates/routers/norms did not come along for the larger step.
    assert [id(p) for p in embed_groups[0]["params"]] == [id(model.token_emb.weight)]
    for group in groups:
        if group is embed_groups[0]:
            continue
        assert group["lr"] == (0.02 if group["algorithm"] == "muon" else 3e-4)


def test_embed_lr_actually_changes_the_embedding_update():
    """The other half of the inert-default pair: prove the knob does something. Two otherwise
    identical models stepped on identical gradients must diverge in the embedding and nowhere
    else."""
    results = {}
    for embed_lr in (None, 0.05):
        model, cfg = _model_and_cfg()
        cfg.train.lr, cfg.train.embed_lr = 3e-4, embed_lr
        optimizer = build_optimizer(model, cfg, "cpu")
        torch.manual_seed(0)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        results[embed_lr] = (
            model.token_emb.weight.detach().clone(),
            model.blocks[1].ln1.weight.detach().clone(),
        )

    embed_default, norm_default = results[None]
    embed_raised, norm_raised = results[0.05]
    assert not torch.allclose(embed_default, embed_raised), "embed_lr did not move the embedding"
    torch.testing.assert_close(norm_default, norm_raised)  # ...and moved nothing else


# --- optimizer behaviour -------------------------------------------------------------------


def test_muon_step_updates_every_parameter():
    """Every parameter must receive a gradient and move.

    This doubles as a dead-parameter check: a parameter that is allocated but never reaches the
    forward graph gets no gradient, trains forever at its init value, and still costs optimizer
    state. blocks[0].attn.value_lambda was exactly that until blocks[0] stopped creating it (it is
    the block that *produces* v_first, so it is always called with v_first=None).
    """
    model, cfg = _model_and_cfg()
    optimizer = build_optimizer(model, cfg, "cpu")
    assert isinstance(optimizer, MuonWithAuxAdam)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model(torch.randint(0, TINY_VOCAB, (2, 8))).logits.square().mean().backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient (unreachable from the forward pass)"

    optimizer.step()
    for name, p in model.named_parameters():
        assert not torch.equal(before[name], p), f"{name} was not updated"


def test_first_block_has_no_value_lambda():
    """blocks[0] produces v_first rather than consuming it, so a mix parameter there would be dead
    weight — allocated, never differentiated, never trained."""
    model, _ = _model_and_cfg()
    assert not hasattr(model.blocks[0].attn, "value_lambda")
    assert all(hasattr(b.attn, "value_lambda") for b in model.blocks[1:])


def test_adamw_optimizer_still_selectable():
    model, cfg = _model_and_cfg()
    cfg.train.optimizer = "adamw"
    assert isinstance(build_optimizer(model, cfg, "cpu"), torch.optim.AdamW)


def test_unknown_optimizer_raises():
    model, cfg = _model_and_cfg()
    cfg.train.optimizer = "lion"
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(model, cfg, "cpu")


def test_cpu_offload_tier_mutates_muon_in_place():
    """MuonWithAuxAdam offloads its AdamW groups in place and returns itself, which is what lets
    train() skip rebuilding the LR scheduler. Muon's own momentum stays resident: Newton-Schulz is
    a matmul chain per step and would cost far more against CPU memory than the VRAM it frees."""
    model, cfg = _model_and_cfg()
    optimizer = build_optimizer(model, cfg, "cpu")
    model(torch.randint(0, TINY_VOCAB, (2, 8))).logits.square().mean().backward()
    optimizer.step()

    migrated = migrate_optimizer_to_cpu_offload(optimizer, model, cfg, "cpu")

    assert migrated is optimizer
    for group in migrated.param_groups:
        assert group["offload"] is (group["algorithm"] == "adamw")
    optimizer.step()  # still steps cleanly with offloaded state


# --- muP ------------------------------------------------------------------------------------


def test_mup_is_exactly_inert_when_base_is_unset(tiny_ids):
    """mup_base_d_model=None resolves to d_model, so width_mult is 1.0 and every correction is an
    identity — the property that lets muP default on without touching existing configs."""
    assert ModelConfig(d_model=256).mup_width_mult == 1.0

    ids = tiny_ids(batch=2, seq=16)
    torch.manual_seed(0)
    a = DenseTransformer(ModelConfig(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0,
                                     ffn_depth=1, dropout=0.0, max_seq_len=32), TINY_VOCAB).eval()
    torch.manual_seed(0)
    b = DenseTransformer(ModelConfig(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                                     dropout=0.0, max_seq_len=32, mup_base_d_model=32), TINY_VOCAB).eval()

    assert b.mup_output_mult == 1.0
    with torch.no_grad():
        torch.testing.assert_close(a(ids).logits, b(ids).logits, rtol=0, atol=0)


def test_mup_scales_init_and_output_at_wider_width():
    """At 4x the base width: hidden init std halves (1/sqrt(4)) and the output multiplier is 1/4,
    while the embedding's std is unchanged — its fan-in is a row lookup, so it doesn't widen."""
    torch.manual_seed(0)
    wide = DenseTransformer(
        ModelConfig(d_model=128, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                    dropout=0.0, max_seq_len=32, mup_base_d_model=32),
        vocab_size=TINY_VOCAB,
    )

    assert wide.cfg.mup_width_mult == 4.0
    assert wide.mup_output_mult == pytest.approx(0.25)
    assert wide.blocks[0].attn.qkv_proj.weight.std().item() == pytest.approx(0.01, rel=0.15)
    assert wide.token_emb.weight.std().item() == pytest.approx(0.02, rel=0.15)


def _mup_model(d_model: int, base: int | None, n_layers: int = 2) -> DenseTransformer:
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=d_model, head_dim=16, n_layers=n_layers, ffn_mult=2.0, ffn_depth=1,
                      dropout=0.0, max_seq_len=32, vocab_pad_multiple=1, mup_base_d_model=base)
    return DenseTransformer(cfg, vocab_size=TINY_VOCAB)


def test_mup_scales_adamw_hidden_lr_only():
    """muP's per-tensor LR correction for Adam is 1/m on the hidden weights and Theta(1) on
    everything else. Under optimizer="adamw" plain AdamW owns the hidden weights, so this is the
    path where the correction is load-bearing — see test_mup_keeps_adamw_activations_flat for what
    its absence actually did."""
    model = _mup_model(d_model=128, base=32)
    assert model.cfg.mup_width_mult == 4.0

    cfg = Config()
    cfg.model, cfg.train = model.cfg, TrainConfig(lr=3.0e-4, optimizer="adamw", device="cpu")
    groups = build_param_groups(model, 0.1, cfg.train.embed_lr, cfg.train.hyper_conn_lr,
                                lr=cfg.train.lr, mup_width_mult=cfg.model.mup_width_mult)

    hidden, no_decay, embed = groups[0], groups[1], groups[2]
    assert hidden["lr"] == pytest.approx(3.0e-4 / 4.0)   # 1/m
    assert "lr" not in no_decay                          # 1-D gains/biases inherit AdamW's own lr
    assert embed["lr"] == pytest.approx(3.0e-4)          # fan-in 1, so it does not widen

    # The tied embedding must be in that unscaled group even though embed_lr was left at its
    # default of None -- otherwise it lands in `decay` and rides the 1/m correction.
    embed_ids = {id(p) for n, p in model.named_parameters() if "token_emb" in n or "lm_head" in n}
    assert {id(p) for p in embed["params"]} == embed_ids
    assert not any(id(p) in embed_ids for p in hidden["params"])


def test_mup_leaves_param_groups_untouched_when_inert():
    """mup_base_d_model unset resolves to d_model, so width_mult is 1.0 and no group gains an `lr`
    override at all -- the range-collapsed inert default this repo's convention asks for."""
    model = _mup_model(d_model=128, base=None)
    assert model.cfg.mup_width_mult == 1.0

    plain = build_param_groups(model, 0.1)
    with_mup = build_param_groups(model, 0.1, lr=3.0e-4, mup_width_mult=1.0)
    assert [sorted(g.keys()) for g in plain] == [sorted(g.keys()) for g in with_mup]
    assert not any("lr" in g for g in with_mup)


def test_mup_needs_lr_to_scale_against():
    with pytest.raises(ValueError, match="muP"):
        build_param_groups(_mup_model(d_model=128, base=32), 0.1, mup_width_mult=4.0)


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_mup_keeps_adamw_activations_flat(optimizer):
    """A coordinate check, shrunk to CPU size: muP's claim is that activation scale is Theta(1) in
    width at *every* training step, which is what makes a tuned LR transfer across width. So the
    same quantity measured at 4x width should barely move.

    This is the test that would have caught the missing 1/m correction in build_param_groups. Note
    what would *not* have: the loss, or the logits. ln_f is RMSNorm and therefore scale-invariant,
    so it launders a blown-up residual stream away right before the LM head -- with the correction
    missing, hidden activations grew ~linearly in d_model while the logits stayed flat.

    The mup_base_d_model=None arm is the control, and it matters as much as the assertion: without
    it this test could pass because muP works, or because the setup is too small/short to move any
    activation at all. Under SP the same measurement must visibly blow up.

    The 16x width span and 20 steps are both load-bearing and were picked by measurement, not
    taste. The drift accumulates with *training*, so a shorter or narrower version is not merely a
    weaker test but a vacuous one: at 4x width over 6 steps the pre-fix code scored 1.12 and would
    have sailed through. At 16x over 20 steps it scores 9.71 against this fixed build's 0.99.
    """
    def coord_size(d_model: int, base: int | None) -> float:
        model = _mup_model(d_model, base)
        cfg = Config()
        cfg.model, cfg.train = model.cfg, TrainConfig(lr=3.0e-4, optimizer=optimizer, device="cpu")
        opt = build_optimizer(model, cfg, "cpu")

        seen = []
        handles = [b.attn.out_proj.register_forward_hook(
            lambda _m, _i, out: seen.append(out.detach().abs().mean().item())) for b in model.blocks]

        gen = torch.Generator().manual_seed(1234)
        for _ in range(20):
            seen.clear()
            ids = torch.randint(0, TINY_VOCAB, (4, 32), generator=gen)
            logits = model(ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, TINY_VOCAB), ids[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        for h in handles:
            h.remove()
        return sum(seen) / len(seen)

    mup_ratio = coord_size(512, base=32) / coord_size(32, base=32)
    sp_ratio = coord_size(512, base=None) / coord_size(32, base=None)

    assert mup_ratio < 1.5, f"muP activation scale grew {mup_ratio:.2f}x over 16x width"
    assert sp_ratio > 3.0, f"control did not drift ({sp_ratio:.2f}x) — test may be vacuous"


def test_orthogonalize_batches_over_leading_dims():
    """Stacking same-shaped updates into a leading batch dim must not change the result.

    This is what _step_muon's shape-grouping relies on: Newton-Schulz is `ns_steps` iterations of
    three matmuls per tensor, so a group of 42 weights costs 630 individually-launched matmuls,
    none big enough to saturate the GPU. `@`/`.mT` batch over leading dims and matrix_norm is taken
    with keepdim, so each matrix is still normalised and iterated independently.

    The tolerance is loose on purpose: the iteration runs in bf16 by design (it is self-correcting
    and does not converge to machine-precision orthogonality), so batched and per-tensor matmuls
    pick different reduction orders. Measured against an fp64 reference the two are *equally* far
    from exact — ~2.7e-2 either way — so this asserts they agree far better than the method's own
    error, not that they are bit-identical.
    """
    torch.manual_seed(0)
    mats = [torch.randn(24, 16) for _ in range(5)]

    separate = torch.stack([orthogonalize(m) for m in mats])
    batched = orthogonalize(torch.stack(mats))

    torch.testing.assert_close(batched, separate, rtol=2e-2, atol=2e-2)


def test_muon_batched_step_matches_per_parameter_step():
    """A Muon step over N same-shaped tensors must match N one-tensor Muon steps.

    _step_muon groups a param group by shape and orthogonalises each shape as one batched tensor.
    Grouping changes which tensors share a kernel launch, and must change nothing else — not the
    momentum buffers, not the sqrt(fan-out/fan-in) scale (which depends only on shape), and not the
    weight decay.
    """
    torch.manual_seed(0)
    shared = [torch.randn(12, 8) for _ in range(4)]
    grads = [torch.randn(12, 8) * 0.1 for _ in range(4)]

    grouped = [p.detach().clone().requires_grad_(True) for p in shared]
    per_param = [p.detach().clone().requires_grad_(True) for p in shared]
    for p, q, g in zip(grouped, per_param, grads):
        p.grad, q.grad = g.clone(), g.clone()

    kwargs = dict(algorithm="muon", lr=0.02, weight_decay=0.01)
    batched_opt = MuonWithAuxAdam([{"params": grouped, **kwargs}])
    # One group per parameter: each has a single shape, so the batching path is never taken.
    single_opts = [MuonWithAuxAdam([{"params": [q], **kwargs}]) for q in per_param]

    for _ in range(3):
        batched_opt.step()
        for o in single_opts:
            o.step()

    for i, (p, q) in enumerate(zip(grouped, per_param)):
        torch.testing.assert_close(p, q, rtol=2e-2, atol=1e-3, msg=f"param {i} diverged")


def test_muon_mixed_shapes_in_one_group():
    """Shape-grouping must handle a group holding several distinct shapes, including 3-D.

    BatchedExperts' stacked (n_experts, in, out) weights are 3-D, so a group can mix ranks; each
    shape becomes its own batch and a shape with only one tensor must skip stacking entirely.
    """
    torch.manual_seed(1)
    params = [
        torch.randn(10, 6, requires_grad=True),
        torch.randn(10, 6, requires_grad=True),
        torch.randn(6, 10, requires_grad=True),  # only one of this shape
        torch.randn(3, 8, 4, requires_grad=True),  # 3-D, stacked-experts layout
        torch.randn(3, 8, 4, requires_grad=True),
    ]
    for p in params:
        p.grad = torch.randn_like(p) * 0.1
    before = [p.detach().clone() for p in params]

    MuonWithAuxAdam([{"params": params, "algorithm": "muon", "lr": 0.02, "weight_decay": 0.0}]).step()

    for i, (p, b) in enumerate(zip(params, before)):
        assert torch.isfinite(p).all(), f"param {i} went non-finite"
        assert not torch.equal(p, b), f"param {i} was not updated"


def test_muon_step_chunks_large_same_shape_groups(monkeypatch):
    """A same-shaped group larger than _MUON_MAX_STACK must be split into multiple orthogonalize()
    calls (bounding the transient Newton-Schulz buffer size — see _step_muon's docstring for the
    MoE OOM this guards against) while still matching an unbounded single-stack step exactly.

    Patches _MUON_MAX_STACK down to 2 so a 5-tensor group exercises multiple full chunks plus a
    remainder chunk without needing a real MoE-sized parameter count.
    """
    torch.manual_seed(2)
    shared = [torch.randn(10, 6) for _ in range(5)]
    grads = [torch.randn(10, 6) * 0.1 for _ in range(5)]

    def run(max_stack: int) -> list[torch.Tensor]:
        monkeypatch.setattr(optim_module, "_MUON_MAX_STACK", max_stack)
        params = [p.detach().clone().requires_grad_(True) for p in shared]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt = MuonWithAuxAdam(
            [{"params": params, "algorithm": "muon", "lr": 0.02, "weight_decay": 0.01}]
        )
        for _ in range(3):
            opt.step()
        return params

    chunked = run(max_stack=2)
    unbounded = run(max_stack=1000)

    for i, (p, q) in enumerate(zip(chunked, unbounded)):
        torch.testing.assert_close(p, q, rtol=2e-2, atol=1e-3, msg=f"param {i} diverged")
