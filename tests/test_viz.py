"""radiance-viz: the routing trace has to describe the forward that actually ran.

The trace is rebuilt from router hooks rather than read out of the model, so the risk is drift: the
halting replay or the expert selection quietly diverging from what _forward_act / MoEFeedForward
do. These tests pin the replay to the model's own mean_loop_depth across halting regimes, check
the expert pick honours expert_bias the way the forward does, and run the CLI end to end.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pytest
import torch

from radiance import viz
from radiance.config import ModelConfig
from radiance.model import DenseTransformer
from tests._fake_tokenizer import WordTokenizer
from tests.conftest import TINY_VOCAB


def _build(seed: int = 0, **overrides) -> DenseTransformer:
    fields = dict(d_model=32, head_dim=8, n_layers=3, ffn_mult=2.0, ffn_depth=1, dropout=0.0,
                  max_seq_len=32, use_router=True, max_loops=4, use_moe=True, n_experts=4, moe_top_k=2)
    fields.update(overrides)
    torch.manual_seed(seed)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB).eval()


@pytest.mark.parametrize("halt_bias", [-6.0, -1.0, 0.0, 6.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_replayed_depth_matches_the_forward(halt_bias, seed, tiny_ids):
    """trace_routing raises if the replay disagrees with mean_loop_depth; sweep the halting bias
    so the check covers never-halt (forced at max_loops), mid-loop halting and halt-on-first-pass."""
    model = _build(seed)
    with torch.no_grad():
        model.router.proj.bias.fill_(halt_bias)
    trace = viz.trace_routing(model, tiny_ids(batch=2, seq=16, seed=seed))
    assert trace.n_iters == 4
    assert trace.depth.min() >= 1 and trace.depth.max() <= 4
    assert abs(trace.depth.mean() - trace.model_mean_depth) < 1e-4
    if halt_bias == 6.0:
        assert (trace.depth == 1).all()
    if halt_bias == -6.0:
        assert (trace.depth == 4).all()
    # ACT weights are a distribution over the passes a token ran.
    np.testing.assert_allclose(trace.act_weights.sum(axis=1), 1.0, atol=1e-5)
    assert (trace.act_weights[:, :, :][np.arange(4)[None, :, None] >= trace.depth[:, None, :]] == 0).all()


def test_replay_mismatch_is_loud(monkeypatch, tiny_ids):
    model = _build()
    monkeypatch.setattr(viz, "_replay_halting", lambda ps, eps, dt: (
        torch.full(ps[0].shape, 99.0), torch.stack(ps, dim=1)))
    with pytest.raises(RuntimeError, match="halting rule has changed"):
        viz.trace_routing(model, tiny_ids())


def test_loops_override_sets_iteration_count(tiny_ids):
    trace = viz.trace_routing(_build(), tiny_ids(), loop_count=2)
    assert trace.n_iters == 2 and trace.experts.shape[2] == 2


def test_expert_pick_follows_expert_bias(tiny_ids):
    """Selection is probs + expert_bias (bias balancing on by default): a large bias on one expert
    must put it in every token's top-k, on every pass, in that block only."""
    model = _build(n_layers=3)
    moe = [i for i, b in enumerate(model.blocks) if hasattr(b.ffn, "expert_bias")]
    assert moe == [1, 2]  # blocks[0] is never MoE
    model.blocks[2].ffn.expert_bias[3] = 10.0
    trace = viz.trace_routing(model, tiny_ids())
    assert trace.moe_blocks == [1, 2]
    assert (trace.experts[:, 1] == 3).any(axis=-1).all()
    assert not (trace.experts[:, 0] == 3).any(axis=-1).all()
    np.testing.assert_allclose(trace.gates.sum(axis=-1), 1.0, atol=1e-5)


def test_moe_without_act(tiny_ids):
    """A fixed-loop MoE checkpoint still traces: every token runs every pass."""
    model = _build(use_router=False, max_loops=1, loop_count=3)
    trace = viz.trace_routing(model, tiny_ids())
    assert trace.halt_probs is None and (trace.depth == 3).all()
    summary = viz.summarize([trace], {})
    assert summary["depth_hist"] == [0, 0, trace.depth.size]


def test_summary_load_shares_sum_to_one(tiny_ids):
    model = _build()
    traces = [viz.trace_routing(model, tiny_ids(seed=s)) for s in range(3)]
    s = viz.summarize(traces, {i: f"t{i}" for i in range(TINY_VOCAB)}, min_count=1)
    assert s["n_tokens"] == 3 * 2 * 16 and sum(s["depth_hist"]) == s["n_tokens"]
    share = np.array(s["moe"]["load_share"])  # (blocks, iters, experts)
    ran = np.array(s["moe"]["tokens_per_iter"])
    np.testing.assert_allclose(share.sum(-1)[:, ran > 0], 1.0, atol=1e-6)
    assert ran[0] == s["n_tokens"]  # everyone runs pass 1
    kept = np.array(s["moe"]["top1_kept"])
    assert ((kept >= 0) & (kept <= 1)).all()


def test_cli_writes_a_self_contained_page(tmp_path, tiny_cfg, monkeypatch):
    cfg = tiny_cfg(use_router=True, max_loops=3, use_moe=True, n_experts=4, moe_top_k=2)
    torch.manual_seed(0)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB)
    ckpt = tmp_path / "step_7.pt"
    torch.save({"model": model.state_dict(), "step": 7, "config": cfg}, ckpt)
    tok = WordTokenizer(TINY_VOCAB)
    monkeypatch.setattr(viz, "build_tokenizer", lambda cfg: tok)
    out = tmp_path / "page.html"
    monkeypatch.setattr(sys, "argv", ["radiance-viz", "--checkpoint", str(ckpt), "--device", "cpu",
                                      "--prompt", "the cat sat on the </script> mat", "--generate", "4",
                                      "--output", str(out)])
    viz.main()
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>")
    assert "<link" not in page and 'src="' not in page  # self-contained: nothing fetched
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
    data = json.loads(blob.replace("<\\/", "</"))
    assert data["meta"]["step"] == 7 and data["meta"]["n_iters"] == 3
    stream = data["stream"]
    assert stream["n_prompt"] == 7 and len(stream["tokens"]) == len(stream["depth"])
    assert " </script>" in stream["tokens"]  # escaped in the page, intact in the data
    assert "".join(stream["tokens"]).startswith("the cat sat on the </script> mat")


def test_bf16_weights_do_not_trip_the_replay_check(tiny_ids):
    """radiance-viz casts to bf16 on CUDA. The forward's mean depth is then rounded to bf16 while
    the replayed counts are exact; the check must allow for that rounding, not call it drift."""
    from radiance.model.load import cast_params

    for seed in range(4):
        model = _build(seed)
        with torch.no_grad():
            model.router.proj.bias.fill_(0.0)  # mid-loop halting, so the mean is not an integer
        cast_params(model, torch.bfloat16)
        trace = viz.trace_routing(model, tiny_ids(batch=1, seq=27, seed=seed))
        assert 1 <= trace.depth.mean() <= 4


def test_stream_texts_keep_split_characters_whole():
    """A byte-level BPE splits some characters across tokens; the page must not show U+FFFD."""

    class Bytes:  # each token is one UTF-8 byte
        def decode(self, ids):
            return bytes(ids).decode("utf-8", errors="replace")

    ids = list("body’s".encode())
    texts = viz._stream_texts(Bytes(), ids)
    assert len(texts) == len(ids) and "".join(texts) == "body’s"
    assert "�" not in "".join(texts) and texts[4:7] == ["", "", "’"]


@pytest.mark.parametrize("halt_bias", [-1.0, 0.0])
@pytest.mark.parametrize("loops", [None, 2])
def test_traced_generation_matches_a_full_trace(halt_bias, loops):
    """The chat page traces each token on its KV-cached decode step. That has to be the same
    routing a single forward over the finished sequence records, position for position, or the
    live page and the static page would disagree about the same text."""
    model = _build()
    with torch.no_grad():
        model.router.proj.bias.fill_(halt_bias)
    tok = WordTokenizer(TINY_VOCAB)
    tok.eos_token_id = None  # random weights: don't let a sampled 0 end it early
    prompt = [3, 9, 4, 1, 7]
    events = list(viz.generate_traced(model, tok, prompt, max_new_tokens=10, temperature=0.0, loops=loops))
    assert events[0][0] == "prompt" and events[-1] == ("done", "length")
    steps = [events[0][1]] + [e[2] for e in events[1:-1]]
    ids = prompt + [e[1] for e in events[1:-1]]
    assert len(ids) == 15
    full = viz.trace_routing(model, torch.tensor([ids]), loop_count=loops)
    np.testing.assert_array_equal(np.concatenate([t.depth for t in steps], axis=1), full.depth)
    np.testing.assert_allclose(np.concatenate([t.halt_probs for t in steps], axis=2), full.halt_probs, atol=1e-5)
    np.testing.assert_array_equal(np.concatenate([t.experts for t in steps], axis=3), full.experts)


def test_traced_generation_stops_on_eos_and_max_seq_len():
    model = _build()  # max_seq_len 32
    tok = WordTokenizer(TINY_VOCAB)
    tok.eos_token_id = None
    events = list(viz.generate_traced(model, tok, list(range(1, 30)), max_new_tokens=10, temperature=0.0))
    assert sum(e[0] == "token" for e in events) == 3 and events[-1] == ("done", "length")
    # whatever greedy picks first, call it EOS: generation ends before yielding a token
    first = next(e[1] for e in viz.generate_traced(model, tok, [1, 2], 1, temperature=0.0) if e[0] == "token")
    tok.eos_token_id = first
    events = list(viz.generate_traced(model, tok, [1, 2], 5, temperature=0.0))
    assert [e[0] for e in events] == ["prompt", "done"] and events[-1] == ("done", "stop")


def _chat_app(tiny_cfg, **cfg_kw):
    from fastapi.testclient import TestClient

    from radiance import viz_chat

    cfg = tiny_cfg(use_router=True, max_loops=3, use_moe=True, n_experts=4, moe_top_k=2)
    for section, values in cfg_kw.items():
        for k, v in values.items():
            setattr(getattr(cfg, section), k, v)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tok = WordTokenizer(TINY_VOCAB)
    tok.eos_token_id = None
    meta = {"checkpoint": "x.pt", "step": 1, "n_iters": 3, "act": True, "chat": False,
            "max_seq_len": cfg.model.max_seq_len, "moe": {"blocks": [1, 2], "n_experts": 4, "top_k": 2}}
    app = viz_chat.create_app(model, tok, cfg, "cpu", None, meta, "Routing: </script> test")
    return TestClient(app), model, tok


def _events(res):
    assert res.status_code == 200, res.text
    return [json.loads(line) for line in res.text.splitlines() if line]


def test_chat_streams_each_token_with_its_routing(tiny_cfg):
    client, model, tok = _chat_app(tiny_cfg)
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "the cat sat on"}],
                                         "max_tokens": 5, "temperature": 0})
    ev = _events(res)
    assert [e["type"] for e in ev] == ["prompt"] + ["token"] * 5 + ["done"]
    prompt, toks, done = ev[0], ev[1:-1], ev[-1]
    assert prompt["n_prompt"] == 4 and prompt["n_iters"] == 3 and not prompt["truncated"]
    assert len(prompt["depth"]) == len(prompt["tokens"]) == 4 and len(prompt["experts"][0]) == 2
    for t in toks:
        assert len(t["tokens"]) == len(t["depth"]) == 1 and 1 <= t["depth"][0] <= 3
        assert np.array(t["experts"]).shape == (1, 2, 3, 2)  # (seq, blocks, passes, k)
    assert done["finish"] == "length" and done["n_tokens"] == 5
    # the streamed routing is the routing a full forward over the same text gives
    ids = tok("the cat sat on")["input_ids"] + [tok._word_to_id[t["tokens"][0].strip()] for t in toks]
    full = viz.trace_routing(model, torch.tensor([ids]))
    assert prompt["depth"] + [t["depth"][0] for t in toks] == full.depth[0].tolist()


def test_chat_keeps_the_end_of_a_long_conversation(tiny_cfg):
    client, _, _ = _chat_app(tiny_cfg)  # max_seq_len 32
    long = " ".join(f"w{i}" for i in range(40))
    ev = _events(client.post("/api/chat", json={"messages": [{"role": "user", "content": long}], "max_tokens": 8}))
    assert ev[0]["truncated"] and ev[0]["n_prompt"] == 24 and ev[0]["tokens"][-1].strip() == "w39"
    assert ev[-1]["n_tokens"] == 8


def test_chat_uses_the_turn_template_when_the_checkpoint_has_one(tiny_cfg):
    from radiance import viz_chat

    _, _, tok = _chat_app(tiny_cfg)
    cfg = tiny_cfg()
    msgs = [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"}]
    assert viz_chat.build_prompt(msgs, tok, cfg) == tok("hi therehellobye")["input_ids"]
    cfg.sft.enabled = True
    from radiance.sft_data import chat_prompt_ids
    assert viz_chat.build_prompt(msgs, tok, cfg) == chat_prompt_ids(msgs, tok, cfg)


def test_chat_rejects_a_bad_request(tiny_cfg):
    client, _, _ = _chat_app(tiny_cfg)
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}).status_code == 422
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "   "}]})
    assert res.status_code == 400 and "tokenizes to nothing" in res.json()["detail"]


def test_chat_page_is_self_contained(tiny_cfg):
    client, _, _ = _chat_app(tiny_cfg)
    page = client.get("/").text
    assert page.startswith("<!DOCTYPE html>") and "<link" not in page and 'src="' not in page
    assert "Routing: &lt;/script&gt; test" in page and "__" not in page.replace("__proto__", "")
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(blob)["meta"]["n_iters"] == 3
