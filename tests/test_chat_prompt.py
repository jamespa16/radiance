"""format_chat_prompt: generalizes format_sft_prompt to also cover a DPO checkpoint, since
sft.enabled and dpo.enabled are mutually exclusive on any one run's config, so a DPO checkpoint's
own saved cfg.sft.enabled is always False.
"""

from __future__ import annotations

import pytest

from radiance.config import Config, DPOConfig, SFTConfig
from radiance.sft_data import format_chat_messages, format_chat_prompt, format_sft_prompt


def test_format_chat_prompt_uses_sft_prefixes_when_sft_enabled():
    cfg = Config(sft=SFTConfig(enabled=True, user_prefix="U: ", assistant_prefix="A: "))

    assert format_chat_prompt("hi", cfg) == format_sft_prompt("hi", cfg) == "U: hiA: "


def test_format_chat_prompt_uses_dpo_prefixes_when_dpo_enabled():
    cfg = Config(
        dpo=DPOConfig(
            enabled=True, dataset="foo/bar", reference_checkpoint="foo.pt",
            user_prefix="DU: ", assistant_prefix="DA: ",
        )
    )

    assert format_chat_prompt("hi", cfg) == "DU: hiDA: "


def test_format_chat_prompt_raises_when_neither_enabled():
    cfg = Config()

    with pytest.raises(ValueError, match="sft.enabled|dpo.enabled"):
        format_chat_prompt("hi", cfg)


def test_format_chat_prompt_on_pre_pr_checkpoint_hits_guard_not_attributeerror():
    # A checkpoint pickled before sft/dpo existed on Config has no such attributes after
    # unpickle (pickle restores __dict__ without running __init__, so default_factory never
    # fires). Config.__setstate__ backfills them to their (disabled) defaults, so
    # generate --chat on such a checkpoint lands on the intended guard, not an AttributeError.
    state = dict(Config().__dict__)
    del state["sft"], state["dpo"]
    old = object.__new__(Config)
    old.__setstate__(state)

    with pytest.raises(ValueError, match="sft.enabled|dpo.enabled"):
        format_chat_prompt("hi", old)


def test_format_chat_messages_concatenates_multi_turn_conversation():
    cfg = Config(sft=SFTConfig(enabled=True, user_prefix="U: ", assistant_prefix="A: "))
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "more"},
    ]

    assert format_chat_messages(messages, cfg) == "U: hiA: yoU: moreA: "


def test_format_chat_messages_folds_system_role_into_user_prefix():
    cfg = Config(sft=SFTConfig(enabled=True, user_prefix="U: ", assistant_prefix="A: "))
    messages = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]

    assert format_chat_messages(messages, cfg) == "U: be niceU: hiA: "


def test_format_chat_messages_uses_dpo_prefixes_when_dpo_enabled():
    cfg = Config(
        dpo=DPOConfig(
            enabled=True, dataset="foo/bar", reference_checkpoint="foo.pt",
            user_prefix="DU: ", assistant_prefix="DA: ",
        )
    )

    assert format_chat_messages([{"role": "user", "content": "hi"}], cfg) == "DU: hiDA: "


def test_format_chat_messages_raises_when_neither_enabled():
    cfg = Config()

    with pytest.raises(ValueError, match="sft.enabled|dpo.enabled"):
        format_chat_messages([{"role": "user", "content": "hi"}], cfg)


def test_format_chat_messages_raises_on_empty_messages():
    cfg = Config(sft=SFTConfig(enabled=True))

    with pytest.raises(ValueError, match="at least one message"):
        format_chat_messages([], cfg)


def test_chat_prompt_ids_is_the_rlaif_prompt_under_the_checkpoints_template():
    # generate --chat and serve prompt with the same ids radiance-rlaif samples and judges after,
    # under whichever mode's prefixes the checkpoint's own config has on.
    from transformers import AutoTokenizer

    from radiance.sft_data import chat_prompt_ids, tokenize_chat_prompt

    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not in the local HF cache")
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    for cfg, (user, assistant) in (
        (Config(sft=SFTConfig(enabled=True)), (SFTConfig.user_prefix, SFTConfig.assistant_prefix)),
        (Config(dpo=DPOConfig(enabled=True, user_prefix="\n\nQ: ", assistant_prefix="\n\nA: ")), ("\n\nQ: ", "\n\nA: ")),
    ):
        ids = chat_prompt_ids(messages, tok, cfg)
        assert ids == tokenize_chat_prompt(messages, tok, user, assistant)[0]
        assert ids[-1] != 220  # not the bare-space token tokenizing the text whole ends in
        assert tok.decode(ids) == format_chat_messages(messages, cfg).rstrip(" ")
    with pytest.raises(ValueError, match="sft.enabled|dpo.enabled"):
        chat_prompt_ids(messages, tok, Config())
