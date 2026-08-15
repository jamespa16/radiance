"""format_chat_prompt: generalizes format_sft_prompt to also cover a DPO checkpoint, since
sft.enabled and dpo.enabled are mutually exclusive on any one run's config, so a DPO checkpoint's
own saved cfg.sft.enabled is always False.
"""

from __future__ import annotations

import pytest

from radiance.config import Config, DPOConfig, SFTConfig
from radiance.sft_data import format_chat_prompt, format_sft_prompt


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
