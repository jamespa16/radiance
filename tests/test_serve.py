"""radiance.serve: /v1/chat/completions and /v1/models against a tiny in-memory model, no
checkpoint file on disk. Response-shape and equivalence-with-generate() checks, in the same
equivalence-invariant style as the rest of the suite (no golden files).
"""

from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from radiance.config import SFTConfig
from radiance.generate import generate
from radiance.model import DenseTransformer
from radiance.sft_data import format_chat_messages
from radiance.serve import _require_chat_enabled, create_app
from tests.conftest import TINY_VOCAB
from tests._fake_tokenizer import WordTokenizer


def _sft_cfg(tiny_cfg, **model_kwargs):
    cfg = tiny_cfg(**model_kwargs)
    cfg.sft = SFTConfig(enabled=True, user_prefix="U: ", assistant_prefix="A: ")
    return cfg


def _client(cfg):
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)
    app = create_app(model, cfg, tokenizer, device="cpu", model_name="tiny-test")
    return TestClient(app), model, tokenizer


def test_models_endpoint_lists_the_loaded_checkpoint(tiny_cfg):
    client, _, _ = _client(_sft_cfg(tiny_cfg))
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["tiny-test"]


def test_non_streaming_response_shape_and_usage(tiny_cfg):
    client, _, _ = _client(_sft_cfg(tiny_cfg, loop_count=2))
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 5, "temperature": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_non_streaming_matches_generate(tiny_cfg):
    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    client, model, tokenizer = _client(cfg)

    torch.manual_seed(0)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 5, "temperature": 0},
    )
    content = resp.json()["choices"][0]["message"]["content"]

    prompt = format_chat_messages([{"role": "user", "content": "hello there"}], cfg)
    torch.manual_seed(0)
    full_text = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=0, device="cpu")

    prompt_ids = tokenizer(prompt)["input_ids"]
    prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    expected = (prompt_decoded + " " + content) if content else prompt_decoded
    assert full_text == expected


def test_streaming_chunks_concatenate_to_non_streaming_content(tiny_cfg):
    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    client, _, _ = _client(cfg)
    req = {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 5, "temperature": 0}

    torch.manual_seed(0)
    non_stream = client.post("/v1/chat/completions", json=req).json()["choices"][0]["message"]["content"]

    torch.manual_seed(0)
    with client.stream("POST", "/v1/chat/completions", json={**req, "stream": True}) as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] is not None
    streamed_text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert streamed_text == non_stream
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_startup_guard_raises_without_sft_or_dpo_enabled(tiny_cfg):
    cfg = tiny_cfg()
    with pytest.raises(ValueError, match="sft.enabled|dpo.enabled"):
        _require_chat_enabled(cfg)


def test_stop_sequence_truncates_output(tiny_cfg):
    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    client, model, tokenizer = _client(cfg)

    torch.manual_seed(0)
    unbounded = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 10, "temperature": 0},
    ).json()["choices"][0]["message"]["content"]
    words = unbounded.split()
    assert len(words) >= 2, "test needs at least two generated words to pick a mid-generation stop sequence"
    stop_word = words[1]

    torch.manual_seed(0)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello there"}],
            "max_tokens": 10,
            "temperature": 0,
            "stop": stop_word,
        },
    )
    body = resp.json()["choices"][0]
    assert stop_word not in body["message"]["content"].split()
    assert body["finish_reason"] == "stop"


def test_max_tokens_truncation_sets_length_finish_reason(tiny_cfg):
    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    client, _, _ = _client(cfg)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 1, "temperature": 0},
    )
    body = resp.json()
    assert body["usage"]["completion_tokens"] <= 1
    if body["usage"]["completion_tokens"] == 1:
        assert body["choices"][0]["finish_reason"] == "length"


def test_top_p_rejected(tiny_cfg):
    client, _, _ = _client(_sft_cfg(tiny_cfg))
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "top_p": 0.5},
    )
    assert resp.status_code == 400
