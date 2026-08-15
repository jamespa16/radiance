"""DPO data pipeline: pair formatting and "packing of one" (one pair per row, each side
independently EOS-tail-padded to seq_len rather than SFT's many-examples-per-block packing).

_format_dpo_pair and _tokenize_dpo_row are the DPO analogues of _format_sft_messages/
_tokenize_sft_example — these pin the two dataset shapes _format_dpo_pair supports, and that
_tokenize_dpo_row pads correctly (with a zeroed loss_mask on the padding) or drops a pair outright
when either side overflows seq_len, rather than truncating it.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from radiance.config import Config, DPOConfig
from radiance.data import _format_dpo_pair, _load_or_build_dpo_packed, _tokenize_dpo_row
import radiance.data as data_mod


class _FakeTokenizer:
    """Deterministic word-level "tokenizer": one id per whitespace-separated word, mirroring
    test_sft_data.py's fixture so token counts are easy to reason about by hand."""

    eos_token_id = 999

    def __call__(self, text: str):
        return {"input_ids": [hash(word) % 900 for word in text.split()]}


def test_format_dpo_pair_reads_message_lists_by_default():
    cfg = Config(dpo=DPOConfig())  # prompt_column unset -> message-list shape
    example = {
        "chosen": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        "rejected": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "go away"},
        ],
    }

    chosen, rejected = _format_dpo_pair(example, cfg)

    assert chosen == example["chosen"]
    assert rejected == example["rejected"]


def test_format_dpo_pair_builds_prompt_from_columns_when_prompt_column_is_set():
    cfg = Config(dpo=DPOConfig(prompt_column="question", system_column="system"))
    example = {
        "system": "be nice",
        "question": "how are you",
        "chosen": "quite well thanks",
        "rejected": "not your business",
    }

    chosen, rejected = _format_dpo_pair(example, cfg)

    assert chosen == [
        {"role": "user", "content": "be nice\nhow are you"},
        {"role": "assistant", "content": "quite well thanks"},
    ]
    assert rejected == [
        {"role": "user", "content": "be nice\nhow are you"},
        {"role": "assistant", "content": "not your business"},
    ]


def test_format_dpo_pair_prompt_column_without_system_column():
    cfg = Config(dpo=DPOConfig(prompt_column="question"))
    example = {"question": "how are you", "chosen": "well", "rejected": "no"}

    chosen, rejected = _format_dpo_pair(example, cfg)

    assert chosen[0] == {"role": "user", "content": "how are you"}
    assert rejected[0] == {"role": "user", "content": "how are you"}


def test_tokenize_dpo_row_pads_both_sides_to_seq_len_with_zero_mask_on_padding():
    cfg = Config(dpo=DPOConfig(user_prefix="U: ", assistant_prefix="A: "))
    tokenizer = _FakeTokenizer()
    example = {
        "chosen": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
        "rejected": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "go away now"},
        ],
    }
    seq_len = 12

    row = _tokenize_dpo_row(example, tokenizer, cfg, seq_len)

    assert row is not None
    for ids_key, mask_key in [
        ("chosen_input_ids", "chosen_loss_mask"),
        ("rejected_input_ids", "rejected_loss_mask"),
    ]:
        ids, mask = row[ids_key], row[mask_key]
        assert len(ids) == seq_len
        assert len(mask) == seq_len
        # Real content ends with the tokenizer's EOS (mask=1); the tail is padding EOS with mask=0.
        eos_positions = [i for i, tok in enumerate(ids) if tok == tokenizer.eos_token_id]
        first_eos = eos_positions[0]
        assert mask[first_eos] == 1  # the real terminating EOS is supervised
        assert all(tok == tokenizer.eos_token_id for tok in ids[first_eos + 1 :])
        assert all(m == 0 for m in mask[first_eos + 1 :])  # padding is never scored

    # The two sides diverge only in their assistant turn, so their tokenized ids differ.
    assert row["chosen_input_ids"] != row["rejected_input_ids"]


def test_tokenize_dpo_row_drops_pair_when_chosen_overflows_seq_len():
    cfg = Config(dpo=DPOConfig())
    tokenizer = _FakeTokenizer()
    example = {
        "chosen": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "this response has quite a few words in it now"},
        ],
        "rejected": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "no"},
        ],
    }

    # seq_len large enough for rejected but not for chosen's longer completion.
    assert _tokenize_dpo_row(example, tokenizer, cfg, seq_len=6) is None


def test_tokenize_dpo_row_drops_pair_when_rejected_overflows_seq_len():
    cfg = Config(dpo=DPOConfig())
    tokenizer = _FakeTokenizer()
    example = {
        "chosen": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "no"},
        ],
        "rejected": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "this response has quite a few words in it now"},
        ],
    }

    assert _tokenize_dpo_row(example, tokenizer, cfg, seq_len=6) is None


def test_tokenize_dpo_row_uses_dpo_prefixes_not_sft_prefixes():
    """Reuses _tokenize_sft_example, but must pass cfg.dpo's own prefixes — a regression guard for
    the _tokenize_sft_example signature fix that made this reuse possible."""
    cfg = Config(dpo=DPOConfig(user_prefix="DPO_USER: ", assistant_prefix="DPO_ASSISTANT: "))
    tokenizer = _FakeTokenizer()
    example = {
        "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "no"}],
    }

    row = _tokenize_dpo_row(example, tokenizer, cfg, seq_len=16)

    expected_user_ids = tokenizer("DPO_USER: hi")["input_ids"]
    assert row["chosen_input_ids"][: len(expected_user_ids)] == expected_user_ids


def _dpo_examples(n: int) -> Dataset:
    return Dataset.from_dict(
        {
            "chosen": [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}]] * n,
            "rejected": [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "b"}]] * n,
        }
    )


def test_load_or_build_dpo_packed_does_not_replace_an_empty_validation_split_with_test(tmp_path, monkeypatch):
    # A present-but-empty "validation" split is a real (if unusual) HF DatasetDict shape - it must
    # not be swapped for "test" just because an empty datasets.Dataset is falsy.
    raw = DatasetDict({"train": _dpo_examples(2), "validation": _dpo_examples(0), "test": _dpo_examples(3)})
    monkeypatch.setattr(data_mod, "load_dataset", lambda name: raw)
    # _add_reference_logprobs would load a real reference checkpoint - irrelevant to the
    # split-selection bug this test targets, so it's replaced with a passthrough.
    monkeypatch.setattr(data_mod, "_add_reference_logprobs", lambda packed, cfg, tok, device: packed)
    monkeypatch.setattr(data_mod, "resolve_device", lambda device: "cpu")

    ref_ckpt = tmp_path / "ref.pt"
    ref_ckpt.write_bytes(b"x")
    cfg = Config(
        dpo=DPOConfig(
            enabled=True,
            dataset="unused",
            reference_checkpoint=str(ref_ckpt),
            cache_dir=str(tmp_path / "cache"),
        )
    )

    packed = _load_or_build_dpo_packed(cfg, _FakeTokenizer())

    assert len(packed["validation"]) == 0, "empty validation split silently replaced by test"
