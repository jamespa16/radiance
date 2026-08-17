"""generate(): regression test for the generate_tokens() extraction.

generate() used to build up input_ids and decode once at the end; it's now a thin wrapper over
generate_tokens() (which radiance.serve streams from directly). This pins two things: that the
refactor didn't change generate()'s output, and that the incremental-decode technique the server
uses for streaming deltas (decode-ids-so-far, diff against the previous decode) reconstructs the
same text a single final decode would.
"""

from __future__ import annotations

import torch

from radiance.generate import generate, generate_tokens
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB
from tests._fake_tokenizer import WordTokenizer


def test_generate_matches_manual_id_join_and_decode(tiny_cfg):
    cfg = tiny_cfg(loop_count=2)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)
    prompt = "the quick brown fox jumps"

    torch.manual_seed(0)
    text = generate(model, tokenizer, prompt, max_new_tokens=6, temperature=0, device="cpu")

    prompt_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor([prompt_ids])
    torch.manual_seed(0)
    new_ids = list(generate_tokens(model, tokenizer, input_ids, max_new_tokens=6, temperature=0, device="cpu"))

    expected = tokenizer.decode(prompt_ids + new_ids, skip_special_tokens=True)
    assert text == expected


def test_incremental_decode_of_generate_tokens_matches_final_decode(tiny_cfg):
    cfg = tiny_cfg(loop_count=2)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)
    prompt = "the quick brown fox jumps"

    input_ids = torch.tensor([tokenizer(prompt)["input_ids"]])
    torch.manual_seed(0)
    new_ids = list(generate_tokens(model, tokenizer, input_ids, max_new_tokens=6, temperature=0, device="cpu"))

    ids_so_far: list[int] = []
    text_so_far = ""
    pieces = []
    for token_id in new_ids:
        ids_so_far.append(token_id)
        new_text = tokenizer.decode(ids_so_far, skip_special_tokens=True)
        pieces.append(new_text[len(text_so_far) :])
        text_so_far = new_text

    assert "".join(pieces) == tokenizer.decode(new_ids, skip_special_tokens=True)


def test_generate_tokens_never_yields_eos(tiny_cfg):
    cfg = tiny_cfg(loop_count=2)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)

    input_ids = torch.tensor([tokenizer("the quick")["input_ids"]])
    torch.manual_seed(0)
    new_ids = list(generate_tokens(model, tokenizer, input_ids, max_new_tokens=20, temperature=0, device="cpu"))

    assert tokenizer.eos_token_id not in new_ids


def test_generate_tokens_never_yields_padded_vocab_id(tiny_cfg):
    """The lm_head is padded past the tokenizer's vocab (model.padded_vocab_size); the padded
    rows are never trained toward -inf, so an unmasked sampling step can land on one and decode
    it to nothing. Pinned through the sampling branch rather than greedy: a random-init greedy
    run falls into a real-token fixed point that never reaches the padded rows, while multinomial
    gives them their ~1/9 softmax mass per step — with this seed the unmasked run samples padded
    id 70, so the test fails deterministically if the mask is ever dropped from generate_tokens.
    (The mask logic itself is unit-tested directly in
    test_eval_harness.py::test_mask_vocab_padding_masks_only_padded_rows.)"""
    cfg = tiny_cfg(loop_count=2)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB + 8).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)

    input_ids = torch.tensor([tokenizer("the quick")["input_ids"]])
    torch.manual_seed(2)
    new_ids = list(
        generate_tokens(
            model, tokenizer, input_ids, max_new_tokens=30, temperature=1.0, top_k=0, device="cpu"
        )
    )

    assert new_ids
    assert all(token_id < TINY_VOCAB for token_id in new_ids)
