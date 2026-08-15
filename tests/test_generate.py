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

    torch.manual_seed(0)
    new_ids = list(generate_tokens(model, tokenizer, prompt, max_new_tokens=6, temperature=0, device="cpu"))

    prompt_ids = tokenizer(prompt)["input_ids"]
    expected = tokenizer.decode(prompt_ids + new_ids, skip_special_tokens=True)
    assert text == expected


def test_incremental_decode_of_generate_tokens_matches_final_decode(tiny_cfg):
    cfg = tiny_cfg(loop_count=2)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)
    prompt = "the quick brown fox jumps"

    torch.manual_seed(0)
    new_ids = list(generate_tokens(model, tokenizer, prompt, max_new_tokens=6, temperature=0, device="cpu"))

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

    torch.manual_seed(0)
    new_ids = list(generate_tokens(model, tokenizer, "the quick", max_new_tokens=20, temperature=0, device="cpu"))

    assert tokenizer.eos_token_id not in new_ids
