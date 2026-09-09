"""Train a byte-level BPE tokenizer on TinyStories at a chosen vocabulary size.

Why this exists: under a *total* parameter cap the tied embedding is a tax paid before any
architecture decision — 12.9M parameters at `d_model: 256` with gpt2's 50k vocab, 19.3M at 384,
25.8M at 512. That is 40-50% of a 50M budget spent on a vocabulary built for the whole internet
and then applied to a corpus of three-year-old-level English. Shrinking it is the largest single
source of free budget available, and nothing in this repo has measured it.

The tokenizer is saved in HF format so `data.tokenizer` can point straight at the directory.

**The comparison it enables is not `val/loss`.** Cross-entropy is per *token*, and a smaller vocab
cuts every story into more tokens, so a smaller-vocab model is scored on an easier per-token
question and its `val/loss` is lower for reasons that have nothing to do with being a better
model. `bits_per_char.py` is what makes the arms comparable; read its docstring before using any
number produced with a non-gpt2 tokenizer here.

Usage:
    uv run python experiments/under-50m/train_tokenizer.py --vocab-size 8192 --out .cache/radiance/tok/ts8k
"""

from __future__ import annotations

import argparse
import pathlib

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--samples",
        type=int,
        default=400_000,
        help="stories to train the merges on; TinyStories is repetitive enough that this "
        "saturates well before the full 2.1M and keeps the run to a couple of minutes",
    )
    args = p.parse_args()

    ds = load_dataset("roneneldan/TinyStories", split="train")
    n = min(args.samples, len(ds))
    print(f"[tok] training vocab={args.vocab_size} on {n:,} of {len(ds):,} stories")

    def corpus():
        batch = 1000
        for i in range(0, n, batch):
            yield from ds[i : i + batch]["text"]

    tok = ByteLevelBPETokenizer()
    # <|endoftext|> keeps the same EOS surface form as gpt2, so data.py's document-joining and
    # train.py's eos_id wiring need no special-casing for these tokenizers.
    tok.train_from_iterator(
        corpus(), vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"]
    )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Wrap the trained backend object directly. Going via GPT2TokenizerFast's vocab_file/
    # merges_file constructor instead looks like it works -- it writes a plausible directory --
    # but builds an *empty* backend, and the tokenizer.json it then saves has a 1-token vocab.
    # AutoTokenizer would load that silently and every story would tokenize to unknowns.
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    hf.save_pretrained(str(out))

    reloaded = AutoTokenizer.from_pretrained(str(out))
    if len(reloaded) < args.vocab_size * 0.9:
        raise SystemExit(f"round-trip check failed: reloaded vocab is {len(reloaded):,}")
    print(f"[tok] saved {len(reloaded):,}-token tokenizer to {out} (round-trip verified)")


if __name__ == "__main__":
    main()
