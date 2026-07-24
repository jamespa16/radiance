"""Intra-document attention masking.

data.py packs many documents into each seq_len block joined by EOS, and plain causal attention
reads straight across those joins. The masking itself is delegated to flex_attention; what needs
testing here is the part this repo owns — recovering document boundaries from the packed ids, and
the fallbacks that keep every non-CUDA path working.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer, document_ids
from tests.conftest import TINY_VOCAB


EOS = 5

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention is CUDA-only in this codebase"
)


def _build(device="cpu", **overrides) -> DenseTransformer:
    cfg = ModelConfig(
        d_model=64, head_dim=16, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=32, **overrides,
    )
    torch.manual_seed(0)
    return DenseTransformer(cfg, vocab_size=TINY_VOCAB, eos_id=EOS).to(device).eval()


# --- document id recovery (device-independent) -----------------------------------------------


def test_document_ids_assigns_eos_to_the_document_it_ends():
    """An exclusive cumsum, so a document's terminating EOS belongs to that document rather than
    opening the next one — otherwise every document would start with a token it can't attend to."""
    ids = torch.tensor([[1, 2, EOS, 3, 4, 4, EOS, 7, 8, 9]])
    assert document_ids(ids, EOS).tolist() == [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]]


def test_document_ids_handles_no_boundaries_and_all_boundaries():
    assert document_ids(torch.tensor([[1, 2, 3]]), EOS).tolist() == [[0, 0, 0]]
    assert document_ids(torch.tensor([[EOS, EOS, EOS]]), EOS).tolist() == [[0, 1, 2]]


def test_document_ids_is_per_sequence():
    ids = torch.tensor([[1, EOS, 2], [EOS, 3, 4]])
    assert document_ids(ids, EOS).tolist() == [[0, 0, 1], [0, 1, 1]]


# --- fallbacks -------------------------------------------------------------------------------


def test_masking_is_skipped_without_an_eos_id():
    """generate.py rebuilds models from a checkpoint without an eos_id; that must not crash."""
    cfg = ModelConfig(d_model=64, head_dim=16, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                      dropout=0.0, max_seq_len=32)
    torch.manual_seed(0)
    model = DenseTransformer(cfg, vocab_size=TINY_VOCAB, eos_id=None).eval()
    assert model._doc_masks(torch.tensor([[1, EOS, 2]]), 0, None) is None


def test_masking_is_skipped_on_cpu():
    """CLAUDE.md's CPU sanity-check workflow has to keep working; flex_attention there is
    impractically slow, so the model falls back to plain causal SDPA."""
    model = _build()
    assert model._doc_masks(torch.tensor([[1, EOS, 2]]), 0, None) is None


def test_masking_is_skipped_during_generation():
    """A single prompt is one document, so the mask would be all-ones anyway."""
    model = _build()
    cache = model.new_kv_cache()
    assert model._doc_masks(torch.tensor([[1, EOS, 2]]), 0, cache) is None


# --- the mask actually applies (CUDA) ---------------------------------------------------------


@requires_cuda
def test_second_document_cannot_see_the_first():
    """The whole point: tokens after an EOS must not attend back across it.

    Verified behaviourally rather than by inspecting attention weights — the first document's
    logits must be *identical* with and without masking (nothing precedes it to mask away), while
    later documents' must differ (they lose access to everything before their boundary).
    """
    ids = torch.tensor([[1, 2, EOS, 3, 4, 4, EOS, 7, 8, 9, 1, 2]], device="cuda")
    model = _build("cuda", loop_count=2)

    with torch.no_grad():
        masked = model(ids).logits
        model.cfg.doc_attention_mask = False
        plain = model(ids).logits

    per_position = (masked - plain).abs().amax(-1)[0]
    assert per_position[:3].max() < 1e-5, "first document must be unaffected by masking"
    assert per_position[3:].max() > 1e-3, "later documents must lose cross-boundary attention"


@requires_cuda
def test_masking_is_inert_when_the_batch_is_one_document():
    """With no EOS in the block there are no boundaries, so masked and plain must agree."""
    ids = torch.tensor([[1, 2, 3, 4, 6, 7, 8, 9]], device="cuda")
    model = _build("cuda", loop_count=2)

    with torch.no_grad():
        masked = model(ids).logits
        model.cfg.doc_attention_mask = False
        plain = model(ids).logits

    torch.testing.assert_close(masked, plain, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_per_iteration_windows_select_different_masks():
    """cfg.loop_attn_windows gives each loop pass its own receptive field — a knob that only
    exists because the same weights run repeatedly."""
    model = _build("cuda", loop_count=3, loop_attn_windows=[4, 8, 32])
    ids = torch.randint(0, TINY_VOCAB, (1, 16), device="cuda")

    masks = model._doc_masks(ids, 0, None)
    assert len(masks) == 3, "one BlockMask per distinct window size"

    selected = [model._mask_for_iteration(masks, n) for n in (1, 2, 3)]
    assert len({id(m) for m in selected}) == 3, "each iteration must get its own mask"
    # Clamped past the end of the list rather than cycling back to a narrow window.
    assert model._mask_for_iteration(masks, 9) is selected[-1]


@requires_cuda
def test_windowed_attention_changes_output():
    ids = torch.randint(0, TINY_VOCAB, (1, 16), device="cuda")
    windowed = _build("cuda", loop_count=3, loop_attn_windows=[2, 2, 2])
    full = _build("cuda", loop_count=3)

    with torch.no_grad():
        assert (windowed(ids).logits - full(ids).logits).abs().max() > 1e-3
