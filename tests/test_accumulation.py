"""The grad-accumulation loop's chunking contract, shared by every training mode.

`split_micro_batch` and `chunk_loss_and_metrics` are the only two things that differ between
pretrain/SFT and DPO inside the step loop; the weighting, the scaled backward and the accumulation
around them are written once. These tests pin the contract that makes sharing them safe: a chunked
micro-batch must produce the same gradient as one un-split backward (which is what the OOM backoff
silently relies on — it changes how many forwards a step takes, never what the step computes), and
each mode must report exactly the metrics its logged series expect.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import Config, DPOConfig, SFTConfig
from radiance.model import DenseTransformer
from radiance.train import (
    _ACCUM_METRICS,
    build_dpo_loss_fn,
    build_loss_fn,
    chunk_loss_and_metrics,
    chunk_reduction_units,
    split_micro_batch,
)
from tests.conftest import TINY_VOCAB


BATCH, SEQ = 4, 16


def _model(cfg: Config) -> DenseTransformer:
    torch.manual_seed(0)
    return DenseTransformer(cfg.model, vocab_size=TINY_VOCAB, eos_id=0)


def _ids(seed: int = 0, batch: int = BATCH) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, TINY_VOCAB, (batch, SEQ), generator=gen)


def _dpo_batch() -> dict:
    gen = torch.Generator().manual_seed(7)
    return {
        "chosen_input_ids": _ids(1),
        "chosen_loss_mask": torch.ones(BATCH, SEQ, dtype=torch.long),
        "rejected_input_ids": _ids(2),
        "rejected_loss_mask": torch.ones(BATCH, SEQ, dtype=torch.long),
        "ref_chosen_logprob": torch.randn(BATCH, generator=gen),
        "ref_rejected_logprob": torch.randn(BATCH, generator=gen),
    }


def _ragged_sft_mask() -> torch.Tensor:
    """An SFT loss_mask whose rows supervise wildly different numbers of positions — a long
    assistant turn, two short ones, and a mid-length one. Real SFT data is ragged like this; a
    uniform mask would let a row-proportional weight pass by accident."""
    mask = torch.zeros(BATCH, SEQ, dtype=torch.long)
    for row, start in enumerate([1, SEQ - 3, SEQ - 2, SEQ // 2]):
        mask[row, start:] = 1
    return mask


def _accumulate(model, cfg, loss_fn, batch, micro_chunk_size) -> dict[str, torch.Tensor]:
    """The step loop's inner body, verbatim minus the GradScaler (a no-op in fp32 on CPU) and the
    autocast block (disabled at fp32). If this drifts from train()'s loop the tests below stop
    testing it — which is exactly the drift the shared helpers exist to prevent."""
    accums = {name: torch.zeros(()) for name in _ACCUM_METRICS}
    units = chunk_reduction_units(batch, cfg, micro_chunk_size)
    total_units = max(sum(units), 1)
    for chunk, chunk_units in zip(split_micro_batch(batch, cfg, "cpu", micro_chunk_size), units):
        chunk_weight = chunk_units / total_units / cfg.train.grad_accum_steps
        chunk_loss, metrics = chunk_loss_and_metrics(model, model, cfg, loss_fn, chunk)
        (chunk_loss * chunk_weight).backward()
        for name, value in metrics.items():
            accums[name] += value.detach() * chunk_weight
    return accums


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


# --- splitting --------------------------------------------------------------------------------


def test_split_micro_batch_takes_only_the_columns_the_mode_uses(tiny_cfg):
    cfg = tiny_cfg()
    batch = {"input_ids": _ids(), "loss_mask": torch.ones(BATCH, SEQ, dtype=torch.long)}
    assert list(split_micro_batch(batch, cfg, "cpu", BATCH)[0]) == ["input_ids"]

    cfg.sft.enabled = True
    assert list(split_micro_batch(batch, cfg, "cpu", BATCH)[0]) == ["input_ids", "loss_mask"]

    cfg.sft.enabled = False
    cfg.dpo.enabled = True
    assert set(split_micro_batch(_dpo_batch(), cfg, "cpu", BATCH)[0]) == set(_dpo_batch())


def test_split_micro_batch_chunks_every_column_in_lockstep(tiny_cfg):
    """Each chunk must hold exactly its own rows of every column — a chunk whose loss_mask or
    cached reference log-prob came from different rows than its input_ids would train on a
    silently mismatched target and never raise."""
    cfg = tiny_cfg()
    cfg.dpo.enabled = True
    batch = _dpo_batch()

    chunks = split_micro_batch(batch, cfg, "cpu", 3)
    assert [c["chosen_input_ids"].size(0) for c in chunks] == [3, 1], "uneven tail chunk kept"

    for column, whole in batch.items():
        torch.testing.assert_close(torch.cat([c[column] for c in chunks]), whole)


def test_split_micro_batch_at_full_width_is_a_single_chunk(tiny_cfg):
    cfg = tiny_cfg()
    batch = {"input_ids": _ids()}
    assert len(split_micro_batch(batch, cfg, "cpu", BATCH)) == 1
    assert len(split_micro_batch(batch, cfg, "cpu", BATCH * 10)) == 1


# --- the reduction units the weighting is built on ---------------------------------------------


def test_reduction_units_are_scored_tokens_for_pretrain(tiny_cfg):
    """Every position but the last has a target, and none is ignored."""
    cfg = tiny_cfg()
    assert chunk_reduction_units({"input_ids": _ids()}, cfg, BATCH) == [BATCH * (SEQ - 1)]
    assert chunk_reduction_units({"input_ids": _ids()}, cfg, 1) == [SEQ - 1] * BATCH


def test_reduction_units_are_supervised_tokens_for_sft(tiny_cfg):
    """The shifted mask compute_sft_loss folds into its labels — loss_mask[:, 1:] — not the row
    count, and not the unshifted mask (whose final column compute_sft_loss always zeroes)."""
    cfg = tiny_cfg()
    cfg.sft = SFTConfig(enabled=True, dataset="unused")
    mask = _ragged_sft_mask()
    batch = {"input_ids": _ids(), "loss_mask": mask}

    per_row = mask[:, 1:].sum(dim=1).tolist()
    assert len(set(per_row)) == BATCH, "the fixture must actually be ragged"
    assert chunk_reduction_units(batch, cfg, 1) == per_row
    assert chunk_reduction_units(batch, cfg, BATCH) == [sum(per_row)]


def test_reduction_units_are_rows_for_dpo(tiny_cfg):
    """compute_dpo_loss reduces over pair rows, not tokens — a pair contributes one term to the
    mean however long it is."""
    cfg = tiny_cfg()
    cfg.dpo = DPOConfig(enabled=True, dataset="unused", reference_checkpoint="unused")
    assert chunk_reduction_units(_dpo_batch(), cfg, 3) == [3, 1]


def test_reduction_units_are_computed_without_touching_the_device(tiny_cfg):
    """They are read off the CPU batch on purpose: a .sum() read back off an accelerator would
    sync per chunk, every step, in the hot loop. Pinned here because it is invisible on CPU."""
    cfg = tiny_cfg()
    cfg.sft = SFTConfig(enabled=True, dataset="unused")

    class _NoDeviceTensor(torch.Tensor):
        def to(self, *args, **kwargs):  # pragma: no cover - must never be reached
            raise AssertionError("chunk_reduction_units moved a tensor to the device")

    mask = _ragged_sft_mask().as_subclass(_NoDeviceTensor)
    units = chunk_reduction_units({"input_ids": _ids(), "loss_mask": mask}, cfg, 2)
    assert sum(units) == int(_ragged_sft_mask()[:, 1:].sum())


# --- the equivalence the OOM backoff rests on --------------------------------------------------


@pytest.mark.parametrize("mode", ["pretrain", "sft", "dpo"])
def test_chunked_accumulation_matches_one_unsplit_backward(tiny_cfg, mode):
    """Splitting a micro-batch must change only how many forward/backward calls a step takes, not
    the gradient it produces. This is the whole premise of tier-1 OOM backoff: it shrinks
    micro_chunk_size mid-run and leaves batch_size/grad_accum_steps/effective_batch_size — and so
    the run's token accounting — untouched, which is only honest if the math agrees.

    The SFT case gives every row a *different* number of supervised positions, which is what real
    SFT data looks like and what makes this more than a restatement of the pretrain case: a weight
    proportional to rows rather than to scored tokens fails here and only here.
    """
    cfg = tiny_cfg()
    cfg.train.batch_size, cfg.train.grad_accum_steps = BATCH, 1
    if mode == "sft":
        cfg.sft = SFTConfig(enabled=True, dataset="unused")
    elif mode == "dpo":
        cfg.dpo = DPOConfig(enabled=True, dataset="unused", reference_checkpoint="unused")

    if mode == "dpo":
        batch, loss_fn = _dpo_batch(), build_dpo_loss_fn(cfg)
    else:
        batch, loss_fn = {"input_ids": _ids()}, build_loss_fn(cfg)
        if mode == "sft":
            batch["loss_mask"] = _ragged_sft_mask()

    unsplit_model = _model(cfg)
    unsplit = _accumulate(unsplit_model, cfg, loss_fn, batch, BATCH)

    chunked_model = _model(cfg)
    chunked = _accumulate(chunked_model, cfg, loss_fn, batch, 1)  # one row per forward

    for name in _ACCUM_METRICS:
        torch.testing.assert_close(chunked[name], unsplit[name], rtol=1e-5, atol=1e-6, msg=name)

    unsplit_grads, chunked_grads = _grads(unsplit_model), _grads(chunked_model)
    assert unsplit_grads, "the backward must actually have produced gradients"
    assert set(unsplit_grads) == set(chunked_grads)
    for name, grad in unsplit_grads.items():
        torch.testing.assert_close(chunked_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_grad_accum_steps_scales_the_chunk_weight(tiny_cfg):
    """The other half of the weighting: N accumulated micro-batches of the same data must give the
    same gradient as one, not N times it."""
    cfg = tiny_cfg()
    cfg.train.batch_size, cfg.train.grad_accum_steps = BATCH, 1
    batch, loss_fn = {"input_ids": _ids()}, build_loss_fn(cfg)

    single = _model(cfg)
    _accumulate(single, cfg, loss_fn, batch, BATCH)

    cfg.train.grad_accum_steps = 3
    accumulated = _model(cfg)
    for _ in range(3):
        _accumulate(accumulated, cfg, loss_fn, batch, BATCH)

    for name, grad in _grads(single).items():
        torch.testing.assert_close(_grads(accumulated)[name], grad, rtol=1e-4, atol=1e-6, msg=name)


# --- what each mode reports --------------------------------------------------------------------


def test_every_reported_metric_has_an_accumulator(tiny_cfg):
    """A metric name the accumulator doesn't know would KeyError in the step loop; one the step
    loop knows but no mode reports silently logs a flat 0. Both are caught here."""
    cfg = tiny_cfg()
    cfg.train.batch_size, cfg.train.grad_accum_steps = BATCH, 1
    reported: set[str] = set()

    _, pretrain_metrics = chunk_loss_and_metrics(
        _model(cfg), _model(cfg), cfg, build_loss_fn(cfg), {"input_ids": _ids()}
    )
    reported |= set(pretrain_metrics)

    cfg.dpo = DPOConfig(enabled=True, dataset="unused", reference_checkpoint="unused")
    model = _model(cfg)
    _, dpo_metrics = chunk_loss_and_metrics(
        model, model, cfg, build_dpo_loss_fn(cfg), split_micro_batch(_dpo_batch(), cfg, "cpu", BATCH)[0]
    )
    reported |= set(dpo_metrics)

    assert reported == set(_ACCUM_METRICS)
    # DPO's total deliberately omits both — mtp_heads > 1 is rejected outright for post-training,
    # and z_loss would need its own reduction over the concatenated logits.
    assert {"z_loss", "mtp_loss"} & set(dpo_metrics) == set()
    assert {"dpo_accuracy", "dpo_margin"} & set(pretrain_metrics) == set()


def test_dpo_chunk_loss_excludes_the_diagnostics_it_reports(tiny_cfg):
    """dpo_accuracy/dpo_margin are logged, never optimised — they must not leak into the total."""
    cfg = tiny_cfg()
    cfg.train.batch_size, cfg.train.grad_accum_steps = BATCH, 1
    cfg.dpo = DPOConfig(enabled=True, dataset="unused", reference_checkpoint="unused")
    model = _model(cfg)

    chunk = split_micro_batch(_dpo_batch(), cfg, "cpu", BATCH)[0]
    chunk_loss, metrics = chunk_loss_and_metrics(model, model, cfg, build_dpo_loss_fn(cfg), chunk)

    # ponder_cost/moe_aux_loss are zero scalars with those features off, so the total is the DPO
    # term alone — and notably not the term plus a margin that happens to be a real number.
    torch.testing.assert_close(chunk_loss, metrics["lm_loss"])
    assert not metrics["dpo_margin"].requires_grad
    assert not metrics["dpo_accuracy"].requires_grad
