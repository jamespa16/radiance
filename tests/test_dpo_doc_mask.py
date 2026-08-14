"""Skipping the document-attention BlockMask for DPO runs.

DPO packs one pair side per row — real content, then a tail of EOS padding with loss_mask 0 — so
the document mask cannot change any *scored* logit, while costing a BlockMask build on every
training step and across the whole reference-logprob precompute pass. The skip is therefore an
efficiency change that must be provably inert, and these tests are the proof: the structural claim
about where document boundaries land (device-independent), and the behavioural equivalence of the
scored log-probabilities themselves (CUDA, where the mask actually applies).
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import Config, DPOConfig, ModelConfig
from radiance.data import _tokenize_dpo_row
from radiance.model import DenseTransformer, document_ids, sequence_logprob_sum
from radiance.train import resolve_dpo_doc_mask
from tests.conftest import TINY_VOCAB


EOS = 5

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention is CUDA-only in this codebase"
)


class _FakeTokenizer:
    """Whitespace tokenizer over a tiny vocab — enough for _tokenize_dpo_row, which only needs
    ids and an eos_token_id, not real text.

    Never emits EOS for ordinary text, matching the assumption document_ids already documents and
    the whole packed-data pipeline already rests on: EOS is a special token, so the only EOS in a
    row is the one packing put there.
    """

    eos_token_id = EOS

    def __call__(self, text: str) -> dict:
        ids = [(sum(map(ord, w)) % (TINY_VOCAB - 2)) + 1 for w in text.split()]
        return {"input_ids": [i if i < EOS else i + 1 for i in ids]}


def _dpo_cfg(seq_len: int = 32, **model_kwargs) -> Config:
    return Config(
        model=ModelConfig(max_seq_len=seq_len, **model_kwargs),
        dpo=DPOConfig(
            enabled=True,
            dataset="unused",
            reference_checkpoint="unused",
            seq_len=seq_len,
            prompt_column="prompt",  # the separate-prompt-column shape, so chosen/rejected are plain strings
        ),
    )


# --- the predicate ----------------------------------------------------------------------------


def test_doc_mask_skipped_for_a_plain_dpo_run():
    cfg = _dpo_cfg()
    assert cfg.model.doc_attention_mask is True
    resolve_dpo_doc_mask(cfg)
    assert cfg.model.doc_attention_mask is False


def test_doc_mask_untouched_when_not_a_dpo_run():
    """Pretrain and SFT pack genuinely-multi-document blocks, where the mask is load-bearing."""
    cfg = _dpo_cfg()
    cfg.dpo.enabled = False
    resolve_dpo_doc_mask(cfg)
    assert cfg.model.doc_attention_mask is True


def test_doc_mask_kept_for_dpo_when_loop_attn_windows_are_configured():
    """Windows ride the same BlockMask, and a sliding window restricts attention *within* a
    document — which real DPO content, all of it document 0, genuinely feels. Not inert."""
    cfg = _dpo_cfg(loop_count=3, loop_attn_windows=[4, 8, 32])
    resolve_dpo_doc_mask(cfg)
    assert cfg.model.doc_attention_mask is True


def test_resolving_an_already_off_doc_mask_is_a_no_op():
    cfg = _dpo_cfg(doc_attention_mask=False)
    resolve_dpo_doc_mask(cfg)
    assert cfg.model.doc_attention_mask is False


# --- the structural claim (device-independent) -------------------------------------------------


def test_every_scored_position_of_a_dpo_row_is_in_document_zero():
    """The load-bearing fact. document_ids' exclusive cumsum puts a document's terminating EOS in
    the document it ends, so a DPO row's real content — including the single scored EOS that ends
    it — is all document 0, and only the padding tail is split into further documents. Masking by
    document therefore only ever removes attention *from* padding, which is scored nowhere."""
    seq_len = 32
    row = _tokenize_dpo_row(
        {
            "prompt": "tell me a story",
            "chosen": "once upon a time there was a duck",
            "rejected": "no",
        },
        _FakeTokenizer(),
        _dpo_cfg(seq_len),
        seq_len,
    )
    assert row is not None

    for side in ("chosen", "rejected"):
        ids = torch.tensor([row[f"{side}_input_ids"]])
        mask = torch.tensor([row[f"{side}_loss_mask"]])
        docs = document_ids(ids, EOS)[0]

        scored = mask[0].nonzero().flatten()
        assert len(scored) > 0, "the fixture must actually score something"
        last_scored = int(scored[-1])
        # Everything a scored logit can causally attend to is positions <= its own, so it suffices
        # that the whole prefix through the last scored position is one document.
        assert docs[: last_scored + 1].eq(0).all(), "real content must be a single document"
        # And the padding tail really is separate documents — i.e. the mask is not inert here for
        # the trivial reason of there being no boundaries at all.
        assert docs[last_scored + 1 :].gt(0).all()


def test_a_dpo_row_that_exactly_fills_seq_len_has_no_padding_at_all():
    """The degenerate end of the same claim: with no padding tail there are no boundaries past
    the content either, so the mask is inert for the trivial reason as well as the general one."""
    ids = torch.tensor([[1, 2, 3, EOS]])
    assert document_ids(ids, EOS).tolist() == [[0, 0, 0, 0]]


# --- the behavioural equivalence (CUDA) --------------------------------------------------------


@requires_cuda
def test_doc_mask_does_not_change_dpo_scored_logprobs():
    """The claim the skip actually rests on, checked where the mask is real: the per-row summed
    log-probabilities the DPO loss consumes must be identical with masking on and off, for a
    DPO-shaped row. Contrast tests/test_doc_mask.py's multi-document block, where turning the mask
    off changes the logits substantially — that is the case this one has to be distinguished from.
    """
    torch.manual_seed(0)
    model = (
        DenseTransformer(
            ModelConfig(
                d_model=64, head_dim=16, n_layers=3, ffn_mult=2.0, ffn_depth=1,
                dropout=0.0, max_seq_len=32, loop_count=2,
            ),
            vocab_size=TINY_VOCAB,
            eos_id=EOS,
        )
        .cuda()
        .eval()
    )

    content = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, EOS]
    ids = torch.tensor([content + [EOS] * (16 - len(content))], device="cuda")
    # Scored on the completion half plus the terminating EOS, 0 on the prompt and the padding.
    mask = torch.tensor([[0] * 5 + [1] * (len(content) - 5) + [0] * (16 - len(content))], device="cuda")

    with torch.no_grad():
        masked = sequence_logprob_sum(model(ids).logits, ids, mask)
        model.cfg.doc_attention_mask = False
        plain = sequence_logprob_sum(model(ids).logits, ids, mask)

    torch.testing.assert_close(masked, plain, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_skipping_the_mask_restores_cuda_graphs_for_a_dpo_run():
    """The larger half of what the skip is worth: doc_attention_mask is one of the three things
    that force resolve_compile_mode down to mode=None, so a DPO run that drops it gets
    mode="reduce-overhead" back."""
    from radiance.train import resolve_compile_mode

    cfg = _dpo_cfg()
    cfg.train.compile = True

    def _mode(config: Config) -> str | None:
        torch.manual_seed(0)
        model = DenseTransformer(config.model, vocab_size=TINY_VOCAB, eos_id=EOS).cuda()
        return resolve_compile_mode(model, config, "cuda")

    assert _mode(cfg) is None
    resolve_dpo_doc_mask(cfg)
    assert _mode(cfg) == "reduce-overhead"
