"""Shared word-level fake tokenizer for generate()/serve tests that need real decode/__len__/
return_tensors="pt" support (unlike test_sft_data.py's _FakeTokenizer, which only needs __call__).

Deterministic: each unique whitespace-separated word gets the next free id, so decode(encode(text))
round-trips exactly and tests can reason about exact ids without a real BPE vocabulary.
"""

from __future__ import annotations

import torch


class WordTokenizer:
    eos_token_id = 0

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def _get_id(self, word: str) -> int:
        if word not in self._word_to_id:
            token_id = len(self._word_to_id) + 1  # 0 is reserved for eos_token_id
            self._word_to_id[word] = token_id
            self._id_to_word[token_id] = word
        return self._word_to_id[word]

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = [self._get_id(w) for w in text.split()]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        words = [
            self._id_to_word[i]
            for i in ids
            if i in self._id_to_word and not (skip_special_tokens and i == self.eos_token_id)
        ]
        return " ".join(words)

    def __len__(self) -> int:
        return self.vocab_size
