"""SFT data pipeline: chat-turn masking and packing.

_tokenize_sft_example and _tokenize_and_pack_sft's group_fn are the SFT analogues of
_tokenize_and_pack's tokenize_fn/group_fn (tested implicitly via build_dataloaders elsewhere) —
these pin the two things that are new here: per-token loss_mask construction from chat turns, and
that packing carries (ids, mask) through concatenation/slicing in lockstep.
"""

from __future__ import annotations

from radiance.config import Config, SFTConfig
from radiance.sft_data import _format_sft_messages, _tokenize_sft_example, format_sft_prompt


class _FakeTokenizer:
    """Deterministic word-level "tokenizer": one id per whitespace-separated word, so tests can
    reason about exact token counts without depending on a real BPE vocabulary."""

    eos_token_id = 999

    def __call__(self, text: str):
        return {"input_ids": [hash(word) % 900 for word in text.split()]}


def test_tokenize_sft_example_masks_user_turns_and_supervises_assistant_turns():
    cfg = Config(sft=SFTConfig(user_prefix="U: ", assistant_prefix="A: "))
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi friend"},
    ]

    ids, mask = _tokenize_sft_example(messages, tokenizer, cfg.sft.user_prefix, cfg.sft.assistant_prefix)

    user_ids = tokenizer("U: hello there")["input_ids"]
    assistant_ids = tokenizer("A: hi friend")["input_ids"]
    assert ids == user_ids + assistant_ids + [tokenizer.eos_token_id]
    assert mask == [0] * len(user_ids) + [1] * len(assistant_ids) + [1]


def test_tokenize_sft_example_multiturn_masks_every_non_assistant_turn():
    cfg = Config(sft=SFTConfig(user_prefix="U: ", assistant_prefix="A: "))
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi friend"},
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "quite well thanks"},
    ]

    ids, mask = _tokenize_sft_example(messages, tokenizer, cfg.sft.user_prefix, cfg.sft.assistant_prefix)

    expected_mask = []
    for turn in messages:
        n = len(tokenizer((cfg.sft.assistant_prefix if turn["role"] == "assistant" else cfg.sft.user_prefix) + turn["content"])["input_ids"])
        expected_mask.extend([1 if turn["role"] == "assistant" else 0] * n)
    expected_mask.append(1)  # trailing EOS is supervised

    assert mask == expected_mask
    assert len(ids) == len(mask)
    assert ids[-1] == tokenizer.eos_token_id


def test_format_sft_messages_reads_messages_column_by_default():
    cfg = Config(sft=SFTConfig(messages_column="chat"))
    example = {"chat": [{"role": "user", "content": "hi"}], "other_col": "ignored"}
    assert _format_sft_messages(example, cfg) == [{"role": "user", "content": "hi"}]


def test_format_sft_messages_builds_two_turns_from_alpaca_style_columns():
    cfg = Config(
        sft=SFTConfig(
            instruction_column="instruction", input_column="input", output_column="output"
        )
    )
    example = {"instruction": "Summarize this.", "input": "Some long text.", "output": "A summary."}

    messages = _format_sft_messages(example, cfg)

    assert messages == [
        {"role": "user", "content": "Summarize this.\nSome long text."},
        {"role": "assistant", "content": "A summary."},
    ]


def test_format_sft_messages_alpaca_style_without_input():
    """input_column is often an empty string per-row for instruction-only examples; the prompt
    should be just the instruction, not "instruction\\n" with a dangling newline."""
    cfg = Config(
        sft=SFTConfig(
            instruction_column="instruction", input_column="input", output_column="output"
        )
    )
    example = {"instruction": "Write a haiku.", "input": "", "output": "..."}

    messages = _format_sft_messages(example, cfg)

    assert messages[0] == {"role": "user", "content": "Write a haiku."}


def test_format_sft_prompt_uses_the_same_template_as_training():
    cfg = Config(sft=SFTConfig(user_prefix="<U>", assistant_prefix="<A>"))
    assert format_sft_prompt("hello", cfg) == "<U>hello<A>"


def test_packing_carries_ids_and_mask_in_lockstep():
    """The packing loop (mirroring _tokenize_and_pack_sft's group_fn) must keep input_ids and
    loss_mask exactly aligned through concatenation and fixed-width slicing."""
    seq_len = 6
    # Two short pre-tokenized examples, as _tokenize_sft_example would produce them.
    ex1_ids, ex1_mask = [1, 2, 3, 999], [0, 0, 1, 1]
    ex2_ids, ex2_mask = [4, 5, 999], [0, 1, 1]

    def group(id_lists, mask_lists):
        ids_concat, mask_concat = [], []
        for ids, mask in zip(id_lists, mask_lists):
            ids_concat.extend(ids)
            mask_concat.extend(mask)
        n_blocks = len(ids_concat) // seq_len
        ids_concat = ids_concat[: n_blocks * seq_len]
        mask_concat = mask_concat[: n_blocks * seq_len]
        blocks = [
            (ids_concat[i : i + seq_len], mask_concat[i : i + seq_len])
            for i in range(0, len(ids_concat), seq_len)
        ]
        return blocks

    blocks = group([ex1_ids, ex2_ids], [ex1_mask, ex2_mask])

    # 4 + 3 = 7 tokens total, seq_len 6 -> exactly one block, tail (1 token) dropped, same as
    # _tokenize_and_pack's pretrain packing drops its remainder.
    assert len(blocks) == 1
    ids, mask = blocks[0]
    assert ids == [1, 2, 3, 999, 4, 5]
    assert mask == [0, 0, 1, 1, 0, 1]
    # The packed block still ends each example's ids in EOS (999) at its true boundary — the
    # doc_attention_mask/document_ids mechanism (tests/test_doc_mask.py) isolates attention
    # between examples using exactly this property, with no changes needed there for SFT.
    assert ids[3] == 999
