"""Loading public post-training datasets: split selection, subsets, local datasets, hh-rlhf
transcripts, the score-margin filter — and that none of it moves an existing cache key.

The split fields exist because the most-used public preference/chat sets do not have a "train"
split at all (HuggingFaceH4/ultrafeedback_binarized: train_prefs/test_prefs; ultrachat_200k:
train_sft/test_sft) and so could not be loaded before; smoltalk needs a config name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

import radiance.dpo_data as dpo_mod
import radiance.sft_data as sft_mod
from radiance.config import Config, DPOConfig, SFTConfig
from radiance.dpo_data import _dpo_cache_path, _format_dpo_pair, _load_or_build_dpo_packed, parse_hh_transcript
from radiance.sft_data import _sft_cache_path, load_chat_dataset, select_splits, split_key_extras
from tests.test_dpo_data import _FakeTokenizer


def _rows(n: int, tag: str = "x") -> Dataset:
    return Dataset.from_dict({"text": [f"{tag}{i}" for i in range(n)]})


# --- split selection --------------------------------------------------------------------------


def test_missing_train_split_names_what_exists():
    raw = DatasetDict({"train_prefs": _rows(3), "test_prefs": _rows(2)})
    with pytest.raises(ValueError, match="has no 'train' split.*test_prefs, train_prefs"):
        select_splits(raw, "uf", "train", None, ("validation", "test"), 0, None)


def test_named_train_and_eval_splits():
    raw = DatasetDict({"train_prefs": _rows(3, "t"), "test_prefs": _rows(2, "e")})
    train, val = select_splits(raw, "uf", "train_prefs", "test_prefs", ("validation", "test"), 0, None)
    assert train["text"] == ["t0", "t1", "t2"] and val["text"] == ["e0", "e1"]
    with pytest.raises(ValueError, match="no 'test' split"):
        select_splits(raw, "uf", "train_prefs", "test", ("validation",), 0, None)


def test_fallbacks_then_carve():
    raw = DatasetDict({"train": _rows(5), "test": _rows(2, "e")})
    _, val = select_splits(raw, "d", "train", None, ("validation", "test"), 0, None)
    assert val["text"] == ["e0", "e1"]
    # SFT's fallbacks are ("validation",) only — unchanged behavior — so this carves instead.
    train, val = select_splits(raw, "d", "train", None, ("validation",), 2, None)
    assert val["text"] == ["x0", "x1"] and len(train) == 3


def test_carving_the_whole_train_split_is_a_clear_error():
    with pytest.raises(ValueError, match="leaving nothing to train on"):
        select_splits(DatasetDict({"train": _rows(2)}), "d", "train", None, (), 2, None)


def test_fallback_never_reuses_the_train_split():
    raw = DatasetDict({"test": _rows(4)})
    train, val = select_splits(raw, "d", "test", None, ("validation", "test"), 1, None)
    assert len(train) == 3 and len(val) == 1


def test_max_examples_samples_after_the_eval_carve_and_is_deterministic():
    raw = DatasetDict({"train": _rows(100)})
    train_a, val_a = select_splits(raw, "d", "train", None, (), 10, 20)
    train_b, val_b = select_splits(raw, "d", "train", None, (), 10, 20)
    assert val_a["text"] == [f"x{i}" for i in range(10)]  # the carve is unchanged by the cap
    assert len(train_a) == 20 and train_a["text"] == train_b["text"]
    assert train_a["text"] != [f"x{i}" for i in range(10, 30)]  # sampled, not the head
    assert not set(train_a["text"]) & set(val_a["text"])


# --- local datasets ---------------------------------------------------------------------------


def test_load_chat_dataset_reads_local_save_to_disk_and_jsonl(tmp_path):
    DatasetDict({"train": _rows(3)}).save_to_disk(str(tmp_path / "dd"))
    assert load_chat_dataset(str(tmp_path / "dd"))["train"]["text"] == ["x0", "x1", "x2"]

    _rows(2).save_to_disk(str(tmp_path / "single"))  # a bare Dataset, not a DatasetDict
    assert list(load_chat_dataset(str(tmp_path / "single")).keys()) == ["train"]

    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(4)))
    assert load_chat_dataset(str(path))["train"]["prompt"] == ["p0", "p1", "p2", "p3"]


def test_a_directory_of_raw_data_files_still_goes_to_load_dataset(tmp_path):
    # Not a save_to_disk directory: a cloned hub repo / a folder of jsonl splits, which
    # load_dataset(dir) has always read.
    data = tmp_path / "raw"
    data.mkdir()
    (data / "train.jsonl").write_text("\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(3)))
    assert load_chat_dataset(str(data))["train"]["prompt"] == ["p0", "p1", "p2"]


def test_hub_subset_is_passed_through(monkeypatch):
    calls = []
    monkeypatch.setattr(sft_mod, "load_dataset", lambda *a: calls.append(a) or DatasetDict({"train": _rows(1)}))
    load_chat_dataset("HuggingFaceTB/smoltalk", "all")
    load_chat_dataset("HuggingFaceH4/no_robots")
    assert calls == [("HuggingFaceTB/smoltalk", "all"), ("HuggingFaceH4/no_robots",)]


# --- cache keys -------------------------------------------------------------------------------


def test_default_fields_leave_existing_cache_keys_unchanged():
    # The pre-existing SFT key formula, verbatim: any cache built before the split fields existed
    # must still be found.
    cfg = Config(sft=SFTConfig(dataset="HuggingFaceH4/no_robots"))
    old_key = "|".join(
        [
            "HuggingFaceH4/no_robots", cfg.data.tokenizer, str(cfg.data.seq_len), "messages",
            "None", "None", "None", cfg.sft.user_prefix, cfg.sft.assistant_prefix, "0",
        ]
    )
    assert _sft_cache_path(cfg).name == hashlib.sha256(old_key.encode()).hexdigest()[:16]
    assert split_key_extras(cfg.sft) == []


def test_split_fields_and_local_contents_change_the_key(tmp_path):
    base = Config(sft=SFTConfig(dataset="d"))
    for field, value in (("subset", "all"), ("train_split", "train_sft"), ("eval_split", "test"), ("max_examples", 5)):
        cfg = Config(sft=SFTConfig(dataset="d", **{field: value}))
        assert _sft_cache_path(cfg) != _sft_cache_path(base), field

    local = tmp_path / "round1"
    DatasetDict({"train": _rows(3)}).save_to_disk(str(local))
    cfg = Config(sft=SFTConfig(dataset=str(local)))
    before = _sft_cache_path(cfg)
    # Regenerated in place (the normal radiance-rlaif round-over-round case): the key must move.
    DatasetDict({"train": _rows(7)}).save_to_disk(str(local))
    assert _sft_cache_path(cfg) != before


def test_dpo_score_filter_is_in_the_key_only_when_set(tmp_path):
    ref = tmp_path / "ref.pt"
    ref.write_bytes(b"x")
    base = Config(dpo=DPOConfig(dataset="d", reference_checkpoint=str(ref)))
    filtered = Config(
        dpo=DPOConfig(
            dataset="d", reference_checkpoint=str(ref),
            score_chosen_column="a", score_rejected_column="b", min_score_margin=1.0,
        )
    )
    unset_columns_only = Config(
        dpo=DPOConfig(dataset="d", reference_checkpoint=str(ref), score_chosen_column="a", score_rejected_column="b")
    )
    assert _dpo_cache_path(base) != _dpo_cache_path(filtered)
    assert _dpo_cache_path(base) == _dpo_cache_path(unset_columns_only)


def test_shared_turn_masking_rekeys_only_the_objectives_it_changes(tmp_path):
    """Unscoring shared turns changes nothing under the summed objectives (the turns cancel), so
    a cache built before it must still be found for them; IPO/sft_weight must not reuse one."""
    ref = tmp_path / "ref.pt"
    ref.write_bytes(b"x")

    def key(**dpo):
        return _dpo_cache_path(Config(dpo=DPOConfig(dataset="d", reference_checkpoint=str(ref), **dpo)))

    old_formula = "|".join(
        ["d", "gpt2", "512", "None", "None", "chosen", "rejected", "\n\nUser: ", "\n\nAssistant: ", "0"]
    )
    assert key().parent.name == hashlib.sha256(old_formula.encode()).hexdigest()[:16]
    assert key(loss_type="hinge") == key(label_smoothing=0.1) == key()
    assert key(loss_type="ipo") != key() and key(sft_weight=1.0) != key()
    assert key(loss_type="ipo", prompt_column="p") == key(prompt_column="p")  # nothing shared is scored


# --- hh-rlhf transcripts ----------------------------------------------------------------------

_HH_CHOSEN = "\n\nHuman: What is 2+2?\n\nAssistant: 4.\n\nHuman: And 3+3?\n\nAssistant: 6."
_HH_REJECTED = "\n\nHuman: What is 2+2?\n\nAssistant: 4.\n\nHuman: And 3+3?\n\nAssistant: Go away."


def test_parse_hh_transcript_turns():
    assert parse_hh_transcript(_HH_CHOSEN) == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6."},
    ]
    # An empty final response (it happens in hh-rlhf) is an empty assistant turn, not an error.
    assert parse_hh_transcript("\n\nHuman: hi\n\nAssistant:")[-1] == {"role": "assistant", "content": ""}
    with pytest.raises(ValueError, match="prompt_column"):
        parse_hh_transcript("just a completion")


def test_hh_pair_shares_every_turn_but_the_last():
    chosen, rejected = _format_dpo_pair({"chosen": _HH_CHOSEN, "rejected": _HH_REJECTED}, Config())
    assert chosen[:-1] == rejected[:-1] and chosen[-1] != rejected[-1]
    # ...so the shared turns tokenize to an identical prefix on both sides.
    tok = _FakeTokenizer()
    c_ids, _ = dpo_mod.tokenize_sft_example(chosen, tok, "\n\nUser: ", "\n\nAssistant: ")
    r_ids, _ = dpo_mod.tokenize_sft_example(rejected, tok, "\n\nUser: ", "\n\nAssistant: ")
    shared, _ = dpo_mod.tokenize_sft_example(chosen[:-1], tok, "\n\nUser: ", "\n\nAssistant: ")
    n = len(shared) - 1  # minus the EOS tokenize_sft_example appends
    assert c_ids[:n] == r_ids[:n]


def test_shared_turns_are_left_out_of_the_loss_mask():
    tok = _FakeTokenizer()
    cfg = Config()
    row = dpo_mod._tokenize_dpo_row({"chosen": _HH_CHOSEN, "rejected": _HH_REJECTED}, tok, cfg, 64)
    chosen, _ = _format_dpo_pair({"chosen": _HH_CHOSEN, "rejected": _HH_REJECTED}, cfg)
    shared, _ = dpo_mod.tokenize_sft_example(chosen[:-1], tok, cfg.dpo.user_prefix, cfg.dpo.assistant_prefix)
    n = len(shared) - 1
    for side in ("chosen", "rejected"):
        mask = row[f"{side}_loss_mask"]
        assert not any(mask[:n])  # the shared "4." assistant turn is not scored...
        assert all(mask[n : mask.index(0, n)]) and mask.index(0, n) > n  # ...the final response is
    # A single-turn pair loses nothing: its one shared turn was a user turn, never scored.
    single = {
        "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a b"}],
        "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "c"}],
    }
    row = dpo_mod._tokenize_dpo_row(single, tok, cfg, 16)
    assert sum(row["chosen_loss_mask"]) == 4 and sum(row["rejected_loss_mask"]) == 3  # Assistant: + words + EOS


def test_ipo_ignores_turns_both_sides_share():
    """IPO divides each side by its own token count, so a shared, scored turn would not cancel.
    With the shared turn unscored, the loss must not move when only the shared turn changes."""
    import torch

    from radiance.losses import compute_dpo_loss_from_logits

    tok = _FakeTokenizer()
    cfg = Config()

    def loss_for(shared_reply: str) -> torch.Tensor:
        chosen = _HH_CHOSEN.replace("4.", shared_reply)
        rejected = _HH_REJECTED.replace("4.", shared_reply)
        row = dpo_mod._tokenize_dpo_row({"chosen": chosen, "rejected": rejected}, tok, cfg, 32)
        ids = torch.tensor([row["chosen_input_ids"], row["rejected_input_ids"]]) % 50
        mask = torch.tensor([row["chosen_loss_mask"], row["rejected_loss_mask"]])
        # A "policy" whose logits are a fixed random function of the current token: the shared
        # turn's per-token log-probs then vary and differ from the reference's (zero), as they do
        # once training moves the policy. (Constant per-token log-probs would leave the per-token
        # mean unchanged by the shared turn, and this test vacuous.)
        logits = torch.randn(50, 50, generator=torch.Generator().manual_seed(0))[ids]
        ref = torch.zeros(1)
        return compute_dpo_loss_from_logits(
            logits[:1], ids[:1], mask[:1], logits[1:], ids[1:], mask[1:], ref, ref, 0.1, loss_type="ipo"
        )[0]

    torch.testing.assert_close(loss_for("4."), loss_for("four, which is a long shared reply"))


def test_rlaif_rows_train_on_the_sampled_ids_not_a_retokenization():
    tok = _FakeTokenizer()
    cfg = Config(dpo=DPOConfig(prompt_column="prompt"))
    # Ids that are deliberately *not* what the text tokenizes to: a BPE can merge the sampled
    # tokens differently once they are text again.
    example = {"prompt": "q", "chosen": "a b", "rejected": "c", "chosen_token_ids": [7, 8, 9], "rejected_token_ids": [5]}
    row = dpo_mod._tokenize_dpo_row(example, tok, cfg, 16)
    prompt_ids, prompt_mask = dpo_mod.tokenize_chat_prompt(
        [{"role": "user", "content": "q"}], tok, cfg.dpo.user_prefix, cfg.dpo.assistant_prefix
    )
    n = len(prompt_ids)
    assert row["chosen_input_ids"][: n + 4] == prompt_ids + [7, 8, 9, tok.eos_token_id]
    assert row["rejected_input_ids"][: n + 2] == prompt_ids + [5, tok.eos_token_id]
    assert row["chosen_loss_mask"][: n + 4] == prompt_mask + [1, 1, 1, 1]
    assert row["chosen_loss_mask"][n + 4 :] == [0] * (16 - n - 4)
    # The text path's prompt and mask layout is the same, so the two agree wherever the ids do.
    text_row = dpo_mod._tokenize_dpo_row({"prompt": "q", "chosen": "a b", "rejected": "c"}, tok, cfg, 16)
    assert text_row["chosen_input_ids"][:n] == prompt_ids


def test_an_empty_train_split_says_so():
    with pytest.raises(ValueError, match="is empty"):
        select_splits(DatasetDict({"train": _rows(0)}), "d", "train", None, (), 0, None)


def test_rlaif_pairs_refuse_another_turn_template(tmp_path):
    (tmp_path / "rlaif_manifest.json").write_text(
        json.dumps({"turn_template": {"user_prefix": "<u>", "assistant_prefix": "<a>"}})
    )
    cfg = Config(dpo=DPOConfig(dataset=str(tmp_path)))
    with pytest.raises(ValueError, match="turn template"):
        dpo_mod.check_rlaif_manifest(cfg)
    cfg.dpo.user_prefix, cfg.dpo.assistant_prefix = "<u>", "<a>"
    dpo_mod.check_rlaif_manifest(cfg)
    dpo_mod.check_rlaif_manifest(Config(dpo=DPOConfig(dataset="hub/id")))  # no manifest: no check

    (tmp_path / "rlaif_manifest.json").write_text(json.dumps({"tokenizer": "other-tok"}))
    with pytest.raises(ValueError, match="other-tok"):
        dpo_mod.check_rlaif_manifest(cfg)


# --- score-margin filter, end to end through the packed DPO dataset ----------------------------


def test_min_score_margin_drops_near_ties(tmp_path, monkeypatch):
    raw = DatasetDict(
        {
            "train_prefs": Dataset.from_dict(
                {
                    "prompt": ["a", "b", "c", "d"],
                    "chosen": ["good one", "good two", "good three", "good four"],
                    "rejected": ["bad one", "bad two", "bad three", "bad four"],
                    "score_chosen": [9.0, 5.0, 8.0, 6.0],
                    "score_rejected": [1.0, 4.5, 2.0, 6.0],
                }
            )
        }
    )
    monkeypatch.setattr(sft_mod, "load_dataset", lambda name: raw)
    monkeypatch.setattr(dpo_mod, "_add_reference_logprobs", lambda packed, cfg, tok, device: packed)
    monkeypatch.setattr(dpo_mod, "resolve_device", lambda device: "cpu")
    ref = tmp_path / "ref.pt"
    ref.write_bytes(b"x")
    cfg = Config(
        dpo=DPOConfig(
            enabled=True, dataset="uf", train_split="train_prefs", prompt_column="prompt",
            score_chosen_column="score_chosen", score_rejected_column="score_rejected",
            min_score_margin=1.0, reference_checkpoint=str(ref), cache_dir=str(tmp_path / "cache"),
        )
    )
    packed = _load_or_build_dpo_packed(cfg, _FakeTokenizer())
    assert len(packed["train"]) == 2  # margins 8, 0.5, 6, 0 -> keep the 8 and the 6
    assert "validation" not in packed
