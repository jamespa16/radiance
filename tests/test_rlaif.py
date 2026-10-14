"""radiance-rlaif: pairing rules, sampling bookkeeping, and the on-disk contract with the DPO
pipeline — what `pairs` writes must load as a DPO dataset with no hand-editing.

The reward model is replaced by a deterministic fake judge and the policy by a tiny random
DenseTransformer over a word-level tokenizer, so this runs on CPU with no downloads; the real
judge path (RewardModelJudge) is a thin wrapper over transformers that these tests do not load.
"""

from __future__ import annotations

import argparse
import json

import pytest
import torch
from datasets import Dataset, DatasetDict

import radiance.dpo_data as dpo_mod
import radiance.rlaif as rlaif
from radiance.config import Config, DataConfig, DPOConfig, ModelConfig, SFTConfig
from radiance.dpo_data import _load_or_build_dpo_packed
from radiance.model import DenseTransformer
from radiance.rlaif import Completion, build_preference_pairs, sample_completions, score_completions, win_rates
from radiance.sft_data import tokenize_chat_prompt, tokenize_sft_example
from tests._fake_tokenizer import WordTokenizer

VOCAB = 64


class _LengthJudge:
    """Scores a response by its word count: deterministic, and easy to reason about."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def score(self, prompts, responses):
        self.calls.append(len(responses))
        return [float(len(r.split())) for r in responses]

    def close(self):
        pass


def _c(text, finished=True):
    return Completion(text, finished)


# --- pairing rules ----------------------------------------------------------------------------


def test_best_finished_vs_worst():
    comps = [[_c("a"), _c("b"), _c("c"), _c("d")]]
    pairs = build_preference_pairs(["p"], comps, [[1.0, 4.0, 0.5, 2.0]])
    assert (pairs[0]["chosen"], pairs[0]["rejected"]) == ("b", "c")
    assert (pairs[0]["chosen_score"], pairs[0]["rejected_score"]) == (4.0, 0.5)
    assert pairs[0]["completions"] == ["a", "b", "c", "d"]


def test_unfinished_is_never_chosen_but_can_be_rejected():
    comps = [[_c("long", finished=False), _c("ok"), _c("meh")]]
    pairs = build_preference_pairs(["p"], comps, [[9.0, 3.0, 5.0]])
    # The truncated sample scored highest, but would teach the policy to stop mid-sentence.
    assert pairs[0]["chosen"] == "meh" and pairs[0]["rejected"] == "ok"
    comps = [[_c("x", False), _c("y", False)]]
    assert build_preference_pairs(["p"], comps, [[1.0, 2.0]]) == []
    comps = [[_c("bad", False), _c("good")]]
    assert build_preference_pairs(["p"], comps, [[0.0, 1.0]])[0]["rejected"] == "bad"


def test_skips_identical_sides_small_margins_and_skipped_prompts():
    assert build_preference_pairs(["p"], [[_c("same"), _c("same")]], [[2.0, 1.0]]) == []
    assert build_preference_pairs(["p"], [[_c("a"), _c("b")]], [[1.0, 0.8]], min_margin=0.5) == []
    assert build_preference_pairs(["p", "q"], [None, [_c("a")]], [None, [1.0]]) == []


def test_sides_differing_only_in_edge_whitespace_or_tied_are_skipped():
    # The judge scores stripped text, so these two are one answer with bf16 noise between them.
    assert build_preference_pairs(["p"], [[_c("Sure."), _c("Sure. ")]], [[1.02, 1.0]]) == []
    # An exact tie carries no preference either, even at the default min_margin=0.0.
    assert build_preference_pairs(["p"], [[_c("a"), _c("b")]], [[1.0, 1.0]]) == []
    assert len(build_preference_pairs(["p"], [[_c("a"), _c("b")]], [[1.0, 0.9]])) == 1


def test_score_completions_regroups_in_order_and_skips_none():
    judge = _LengthJudge()
    comps = [[_c("one"), _c("one two")], None, [_c("  one two three  ")]]
    assert score_completions(judge, ["p", "q", "r"], comps) == [[1.0, 2.0], None, [3.0]]
    assert judge.calls == [3]  # one batched pass


def test_win_rates_count_ties_as_half():
    assert win_rates([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 0.0, 5.0]) == pytest.approx(2.5 / 4)


def test_identical_completions_tie_whatever_their_bf16_scores():
    # The judge's bf16 score for one text moves ~0.03 with its batch; the same answer is a tie.
    assert win_rates([1.00, 1.0], [1.03, 2.0], ["same", "a"], ["same", "b"]) == pytest.approx(1.5 / 2)
    # ...and an edge-whitespace difference is the same answer, since the judge saw it stripped.
    assert win_rates([1.00], [1.03], ["same"], [" same\n"]) == pytest.approx(0.5)


def test_open_prompt_is_a_token_prefix_of_what_sft_trained_on():
    """With gpt2, "\n\nAssistant: " tokenized alone ends in a bare-space token (220) that SFT,
    which tokenizes prefix and reply together, never puts there: the space merges into the reply's
    first word. The prompt rlaif samples after must be a prefix of the SFT sequence instead."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not in the local HF cache")
    user, assistant = "\n\nUser: ", "\n\nAssistant: "
    prompt_ids, prompt_mask = tokenize_chat_prompt([{"role": "user", "content": "Hi"}], tok, user, assistant)
    assert len(prompt_ids) == len(prompt_mask) and prompt_ids[-1] != 220
    for reply in ("Hello there.", "The answer is 4.", "\nA list:", " indented"):
        sft_ids, _ = tokenize_sft_example(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": reply}], tok, user, assistant
        )
        assert sft_ids[: len(prompt_ids)] == prompt_ids, reply
    # The decoded completion is the reply alone: the template's trailing space is not part of it.
    sft_ids, _ = tokenize_sft_example(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}], tok, user, assistant
    )
    gen = sft_ids[len(prompt_ids) : -1]  # what the policy samples after the prompt, minus EOS
    text = rlaif._decode_completion(tok, prompt_ids, gen, user + "Hi" + assistant)
    assert text == "Hello there."


# --- sampling ---------------------------------------------------------------------------------


def _policy(max_seq_len: int = 48):
    torch.manual_seed(0)
    model_cfg = ModelConfig(d_model=32, head_dim=8, n_layers=2, ffn_depth=1, max_seq_len=max_seq_len, vocab_pad_multiple=1)
    cfg = Config(data=DataConfig(tokenizer="unused"), model=model_cfg, sft=SFTConfig(enabled=True))
    return DenseTransformer(model_cfg, vocab_size=VOCAB), cfg


def _tokenizer():
    tok = WordTokenizer(VOCAB)
    # Register every id, so each sampled token decodes to exactly one word: the turn-template
    # words first, then filler words for the rest.
    tok._get_id("User:")
    tok._get_id("Assistant:")
    i = 1
    while len(tok) > len(tok._word_to_id) + 1:
        tok._get_id(f"w{i}")
        i += 1
    return tok


def test_sample_completions_shapes_and_finished_flags():
    model, cfg = _policy()
    tok = _tokenizer()
    long_prompt = " ".join(["w1"] * 60)  # cannot fit max_seq_len - max_new_tokens
    out = sample_completions(model, cfg, tok, ["w1 w2", long_prompt, "w3"], 3, 6, 1.0, 0, 4, "cpu")
    assert out[1] is None
    for row in (out[0], out[2]):
        assert len(row) == 3
        for comp in row:
            n = len(comp.text.split())
            assert n <= 6
            assert comp.finished or n == 6


def test_a_completion_ending_exactly_at_the_cap_is_finished(monkeypatch):
    model, cfg = _policy()
    tok = _tokenizer()
    # Row 0 samples EOS right after its 6th token; row 1 is still going at the 7th.
    scripts = [[5] * 6, [5] * 7]

    def fake_generate(model, tokenizer, items, device, samples_per_item=1):
        # One prompt, prefilled once and expanded to its two samples.
        assert len(items) == 1 and samples_per_item == 2
        assert all(item.max_new_tokens == 7 for item in items)
        for step in range(8):
            yield [(row, s[step] if step < len(s) else None) for row, s in enumerate(scripts) if step <= len(s)]

    monkeypatch.setattr(rlaif, "generate_tokens_batched", fake_generate)
    out = sample_completions(model, cfg, tok, ["w1"], 2, 6, 1.0, 0, 4, "cpu")[0]
    assert [c.finished for c in out] == [True, False]
    assert [len(c.text.split()) for c in out] == [6, 6]  # the overrun is cut back to the cap


def test_load_prompts_handles_strings_and_message_lists(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt": "a", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "m1"}]},
        {"prompt": "b", "messages": [{"role": "user", "content": "m2"}, {"role": "assistant", "content": "x"}]},
        {"prompt": "a", "messages": [{"role": "user", "content": "m1"}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert sorted(rlaif.load_prompts(str(path), "prompt")) == ["a", "b"]  # deduplicated
    assert sorted(rlaif.load_prompts(str(path), "messages")) == ["m1", "m2"]
    assert len(rlaif.load_prompts(str(path), "prompt", max_prompts=1)) == 1


def test_eval_of_a_checkpoint_against_itself_is_an_exact_tie(tmp_path, monkeypatch):
    model, cfg = _policy()
    tok = _tokenizer()
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text("\n".join(json.dumps({"prompt": f"w{i}"}) for i in range(1, 9)))
    monkeypatch.setattr(rlaif, "load_checkpoint", lambda path, device: (model, cfg, tok))
    monkeypatch.setattr(rlaif, "RewardModelJudge", _LengthJudge)
    out = tmp_path / "eval.json"
    args = argparse.Namespace(
        checkpoint=["a.pt", "a.pt"], output=str(out), prompts=str(prompts_path), prompt_column="prompt",
        split="train", subset=None, max_prompts=None, max_new_tokens=10, temperature=1.0, top_k=0,
        batch_size=4, judge="fake", judge_batch_size=8, device="cpu", seed=0,
    )
    rlaif.run_eval(args)
    report = json.loads(out.read_text())
    # Same seed per checkpoint: the same model writes the same completions, so every prompt ties.
    assert report["results"][1]["win_rate_vs_first"] == 0.5
    assert report["results"][0]["mean_reward"] == report["results"][1]["mean_reward"]
    assert len(report["samples"]) == report["prompts"] == 8


def test_eval_with_no_prompt_that_fits_is_a_clear_error(tmp_path, monkeypatch):
    model, cfg = _policy()
    tok = _tokenizer()
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(json.dumps({"prompt": "w1"}))
    monkeypatch.setattr(rlaif, "load_checkpoint", lambda path, device: (model, cfg, tok))
    monkeypatch.setattr(rlaif, "RewardModelJudge", _LengthJudge)
    args = argparse.Namespace(
        checkpoint=["a.pt"], output=None, prompts=str(prompts_path), prompt_column="prompt",
        split="train", subset=None, max_prompts=None, max_new_tokens=48, temperature=1.0, top_k=0,
        batch_size=4, judge="fake", judge_batch_size=8, device="cpu", seed=0,
    )
    with pytest.raises(SystemExit, match="no prompt fits"):
        rlaif.run_eval(args)


def test_pairs_with_no_prompt_that_fits_blames_the_prompts_not_the_policy(tmp_path, monkeypatch):
    model, cfg = _policy()
    tok = _tokenizer()
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(json.dumps({"prompt": "w1"}))
    monkeypatch.setattr(rlaif, "load_checkpoint", lambda path, device: (model, cfg, tok))
    monkeypatch.setattr(rlaif, "RewardModelJudge", _LengthJudge)
    args = argparse.Namespace(
        checkpoint="policy.pt", output=str(tmp_path / "out"), num_samples=2, min_margin=0.0,
        prompts=str(prompts_path), prompt_column="prompt", split="train", subset=None, max_prompts=None,
        max_new_tokens=48, temperature=1.0, top_k=0, batch_size=4, judge="fake", judge_batch_size=8,
        device="cpu", seed=0,
    )
    with pytest.raises(SystemExit, match="no prompt fits"):
        rlaif.run_pairs(args)


# --- pairs -> DPO dataset, end to end ---------------------------------------------------------


def test_pairs_output_loads_as_a_dpo_dataset(tmp_path, monkeypatch):
    model, cfg = _policy()
    tok = _tokenizer()
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text("\n".join(json.dumps({"prompt": f"w{i} w{i + 1}"}) for i in range(1, 13)))

    monkeypatch.setattr(rlaif, "load_checkpoint", lambda path, device: (model, cfg, tok))
    monkeypatch.setattr(rlaif, "RewardModelJudge", _LengthJudge)
    out = tmp_path / "round1"
    args = argparse.Namespace(
        checkpoint="policy.pt", output=str(out), num_samples=4, min_margin=0.0,
        prompts=str(prompts_path), prompt_column="prompt", split="train", subset=None, max_prompts=None,
        max_new_tokens=30, temperature=1.0, top_k=0, batch_size=16, judge="fake", judge_batch_size=8,
        device="cpu", seed=0,
    )
    rlaif.run_pairs(args)
    manifest = json.loads((out / rlaif.MANIFEST_NAME).read_text())
    n_pairs = manifest["stats"]["pairs"]
    assert n_pairs > 0 and manifest["args"]["num_samples"] == 4
    # _policy() is SFT'd under the sft.* defaults, which the DPO config below also uses.
    assert manifest["turn_template"] == {"user_prefix": cfg.sft.user_prefix, "assistant_prefix": cfg.sft.assistant_prefix}

    # The DPO stage configs/post_training/dpo_rlaif.yaml describes, minus the reference forward.
    monkeypatch.setattr(dpo_mod, "_add_reference_logprobs", lambda packed, cfg, tok, device: packed)
    monkeypatch.setattr(dpo_mod, "resolve_device", lambda device: "cpu")
    ref = tmp_path / "ref.pt"
    ref.write_bytes(b"x")
    dpo_cfg = Config(
        data=DataConfig(tokenizer=cfg.data.tokenizer, seq_len=32, num_workers=0),
        dpo=DPOConfig(
            enabled=True, dataset=str(out), prompt_column="prompt", eval_split_size=2,
            score_chosen_column="chosen_score", score_rejected_column="rejected_score", min_score_margin=1.0,
            reference_checkpoint=str(ref), cache_dir=str(tmp_path / "cache"),
        ),
    )
    packed = _load_or_build_dpo_packed(dpo_cfg, tok)
    pairs = DatasetDict.load_from_disk(str(out))["train"]
    kept = sum(c - r >= 1.0 for c, r in zip(pairs["chosen_score"], pairs["rejected_score"]))
    assert len(packed["validation"]) + len(packed["train"]) <= kept
    # Each packed side is the policy's own sampled ids after the prompt it was sampled with, not a
    # re-tokenization of its text.
    expected = set()
    for pair in pairs:
        prompt_ids, _ = tokenize_chat_prompt(
            [{"role": "user", "content": pair["prompt"]}], tok, cfg.sft.user_prefix, cfg.sft.assistant_prefix
        )
        for side in ("chosen", "rejected"):
            expected.add(tuple(prompt_ids + pair[f"{side}_token_ids"] + [tok.eos_token_id]))
    n_rows = 0
    for split in packed.values():
        for row in split:
            for side in ("chosen", "rejected"):
                mask = row[f"{side}_loss_mask"].tolist()
                real_len = len(mask) - mask[::-1].index(1)
                assert tuple(row[f"{side}_input_ids"][:real_len].tolist()) in expected
            n_rows += 1
    assert n_rows > 0
