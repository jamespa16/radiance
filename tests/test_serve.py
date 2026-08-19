"""radiance.serve: /v1/chat/completions and /v1/models against a tiny in-memory model, no
checkpoint file on disk. Response-shape and equivalence-with-generate() checks, in the same
equivalence-invariant style as the rest of the suite (no golden files).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
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


def _app(cfg, **app_kwargs):
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    tokenizer = WordTokenizer(TINY_VOCAB)
    app = create_app(model, cfg, tokenizer, device="cpu", model_name="tiny-test", **app_kwargs)
    return app, model, tokenizer


def _client(cfg, **app_kwargs):
    app, model, tokenizer = _app(cfg, **app_kwargs)
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


def test_empty_messages_rejected(tiny_cfg):
    client, _, _ = _client(_sft_cfg(tiny_cfg))
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400


# --- /v1/completions (#30) -------------------------------------------------------------------


def test_completions_works_on_a_base_checkpoint_without_sft_or_dpo(tiny_cfg):
    client, _, _ = _client(tiny_cfg(loop_count=2))
    resp = client.post("/v1/completions", json={"prompt": "hello there", "max_tokens": 5, "temperature": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    choice = body["choices"][0]
    assert isinstance(choice["text"], str)
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_completions_matches_generate(tiny_cfg):
    cfg = tiny_cfg(loop_count=2)
    client, model, tokenizer = _client(cfg)

    torch.manual_seed(0)
    resp = client.post("/v1/completions", json={"prompt": "hello there", "max_tokens": 5, "temperature": 0})
    text = resp.json()["choices"][0]["text"]

    torch.manual_seed(0)
    full_text = generate(model, tokenizer, "hello there", max_new_tokens=5, temperature=0, device="cpu")
    expected = (("hello there" + " " + text) if text else "hello there")
    assert full_text == expected


def test_completions_streaming_chunks_concatenate_to_non_streaming_text(tiny_cfg):
    client, _, _ = _client(tiny_cfg(loop_count=2))
    req = {"prompt": "hello there", "max_tokens": 5, "temperature": 0}

    torch.manual_seed(0)
    non_stream = client.post("/v1/completions", json=req).json()["choices"][0]["text"]

    torch.manual_seed(0)
    with client.stream("POST", "/v1/completions", json={**req, "stream": True}) as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] is not None
    streamed_text = "".join(c["choices"][0]["text"] for c in chunks)
    assert streamed_text == non_stream
    assert all(c["object"] == "text_completion" for c in chunks)


def test_chat_completions_400s_on_a_base_checkpoint(tiny_cfg):
    client, _, _ = _client(tiny_cfg())
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 400
    assert "sft.enabled" in resp.json()["detail"]


def test_completions_empty_prompt_rejected(tiny_cfg):
    client, _, _ = _client(tiny_cfg())
    resp = client.post("/v1/completions", json={"prompt": ""})
    assert resp.status_code == 400


# --- auth and rate limiting (#35) ------------------------------------------------------------


def test_v1_routes_are_open_without_configured_api_keys(tiny_cfg):
    client, _, _ = _client(tiny_cfg())
    resp = client.get("/v1/models")
    assert resp.status_code == 200


def test_v1_routes_require_a_valid_bearer_token_when_configured(tiny_cfg):
    client, _, _ = _client(tiny_cfg(), api_keys={"secret-key"})

    resp = client.get("/v1/models")
    assert resp.status_code == 401

    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401

    resp = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})
    assert resp.status_code == 200


def test_healthz_metrics_and_readyz_bypass_auth(tiny_cfg):
    client, _, _ = _client(tiny_cfg(), api_keys={"secret-key"})
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_rate_limit_returns_429_once_exceeded(tiny_cfg):
    client, _, _ = _client(tiny_cfg(), rate_limit_per_minute=2)
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/models").status_code == 200
    resp = client.get("/v1/models")
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "60"


def test_rate_limit_is_disabled_by_default(tiny_cfg):
    client, _, _ = _client(tiny_cfg())
    for _ in range(5):
        assert client.get("/v1/models").status_code == 200


def test_rate_limit_rejects_over_quota_wrong_key_requests_without_matching_keys(tiny_cfg):
    """Regression guard: once a client's IP has exhausted its quota with invalid-credential
    requests, further requests from it must 429, without a valid key from a different client
    ever being able to consume that same IP's exhausted quota."""
    client, _, _ = _client(tiny_cfg(), api_keys={"secret-key"}, rate_limit_per_minute=1)
    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401

    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "60"


def test_rate_limit_does_not_cross_contaminate_ip_and_key_buckets(tiny_cfg):
    """Regression guard: a valid API key has its own quota, so it must not be starved by
    invalid-credential requests that exhausted the shared IP bucket from the same client."""
    client, _, _ = _client(tiny_cfg(), api_keys={"secret-key"}, rate_limit_per_minute=1)
    resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401

    resp = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})
    assert resp.status_code == 200


# --- health/readiness/metrics (#36) ----------------------------------------------------------


def test_healthz_and_readyz(tiny_cfg):
    client, _, _ = _client(tiny_cfg())
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


def test_metrics_reports_requests_and_tokens(tiny_cfg):
    client, _, _ = _client(tiny_cfg(loop_count=2))
    client.post("/v1/completions", json={"prompt": "hello there", "max_tokens": 3, "temperature": 0})

    body = client.get("/metrics").json()
    # The /metrics call's own middleware increment lands after this response is built, so only
    # the earlier completion request is reflected here.
    assert body["requests_total"] >= 1
    assert body["prompt_tokens_total"] >= 2
    assert "requests_per_second" in body
    assert "tokens_per_second" in body


def _chat_body(prompt: str, **overrides) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 5, "temperature": 0}
    body.update(overrides)
    return body


async def _post_all_concurrently(app, bodies: list[dict]) -> list[dict]:
    """Fires every request in `bodies` at /v1/chat/completions truly concurrently (a single
    asyncio.gather on one event loop, unlike sequential bare TestClient.post() calls, each of
    which — per create_app's _ensure_dispatcher_running — gets its own throwaway loop and so can
    never actually overlap with another), so they have a real chance to land in the same
    dispatcher batch.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(*(client.post("/v1/chat/completions", json=b) for b in bodies))
    return [r.json()["choices"][0]["message"]["content"] for r in responses]


def test_concurrent_requests_get_correct_independent_completions(tiny_cfg):
    """Three different prompts fired truly concurrently (see _post_all_concurrently) — likely to
    share a dispatcher batch given the default batch_wait_ms — must each get exactly the same
    completion as when run alone, proving the right-padding/key-positions machinery in
    generate_tokens_batched doesn't let one row's content or padding leak into another's.
    """
    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    app, _, _ = _app(cfg, max_batch_size=4, batch_wait_ms=50.0)
    client = TestClient(app)
    prompts = ["hello there", "a slow turtle races", "the quick brown fox jumps high"]
    bodies = [_chat_body(p) for p in prompts]

    sequential = [client.post("/v1/chat/completions", json=b).json()["choices"][0]["message"]["content"] for b in bodies]
    concurrent = asyncio.run(_post_all_concurrently(app, bodies))

    assert concurrent == sequential


def test_requests_with_different_loops_are_never_batched_together(tiny_cfg, monkeypatch):
    """Loop depth is one value per forward call (see generate_tokens_batched's docstring), so two
    concurrent requests asking for different --loops overrides must never end up merged into one
    batched forward pass. Spies on radiance.serve.generate_tokens_batched (rather than comparing
    decoded output, which a tiny randomly-initialized model's greedy argmax turns out to be
    remarkably insensitive to loop depth for) to directly confirm the dispatcher issued two
    separate batch-of-one calls instead of merging them.
    """
    import radiance.serve as serve_module

    cfg = _sft_cfg(tiny_cfg, loop_count=4)
    app, _, _ = _app(cfg, max_batch_size=4, batch_wait_ms=50.0)

    calls: list[tuple[int | None, int]] = []
    real_generate_tokens_batched = serve_module.generate_tokens_batched

    def spy(model, tokenizer, items, device, loops=None):
        calls.append((loops, len(items)))
        return real_generate_tokens_batched(model, tokenizer, items, device, loops)

    monkeypatch.setattr(serve_module, "generate_tokens_batched", spy)

    bodies = [_chat_body("hello there", loops=1), _chat_body("hello there", loops=3)]
    asyncio.run(_post_all_concurrently(app, bodies))

    assert sorted(calls) == [(1, 1), (3, 1)]


def test_lone_request_completes_without_waiting_for_batch_company(tiny_cfg):
    client, _, _ = _client(tiny_cfg(loop_count=2), batch_wait_ms=20.0)
    start = time.monotonic()
    resp = client.post("/v1/completions", json={"prompt": "hello there", "max_tokens": 5, "temperature": 0})
    elapsed = time.monotonic() - start

    assert resp.status_code == 200
    assert elapsed < 2.0, "a lone request must not block waiting for batch company that never arrives"


# --- request batching dispatcher (#46) -------------------------------------------------------


def test_batch_wait_zero_still_batches_requests_already_queued(tiny_cfg, monkeypatch):
    """--batch-wait-ms 0 is documented as "batches opportunistically (whatever's already queued)
    with no added wait", so two requests already in the dispatcher's queue when it forms a batch
    must share one forward pass, not each run solo. Spies on generate_tokens_batched to count the
    actual batch sizes: before the fix, the deadline check ran before the first queue drain, so
    the drain never happened and every request ran as a batch of one."""
    import radiance.serve as serve_module

    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    app, _, _ = _app(cfg, max_batch_size=4, batch_wait_ms=0.0)

    real_generate_tokens_batched = serve_module.generate_tokens_batched
    calls: list[int] = []

    def spy(model, tokenizer, items, device, loops=None):
        calls.append(len(items))
        return real_generate_tokens_batched(model, tokenizer, items, device, loops)

    monkeypatch.setattr(serve_module, "generate_tokens_batched", spy)

    bodies = [_chat_body(p) for p in ["hello there", "a slow turtle races"]]
    asyncio.run(_post_all_concurrently(app, bodies))

    assert calls == [2]


def test_next_batch_starts_without_extra_wait_after_in_flight_batch(tiny_cfg, monkeypatch):
    """A request that arrives while a batch is generating must not pay its --batch-wait-ms on
    top of the in-flight generation: the dispatcher forms the next batch while the current one
    runs (see _dispatcher_loop), so this request's own batch starts the moment the in-flight
    batch finishes. The spy slows only the first (in-flight) batch and records when each batched
    call actually starts generating; the gap between the two calls must be bounded by the
    slowdown alone, not slowdown + batch_wait_ms (what the serialized-formation design produced)."""
    import radiance.serve as serve_module

    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    app, _, _ = _app(cfg, max_batch_size=4, batch_wait_ms=150.0)

    real_generate_tokens_batched = serve_module.generate_tokens_batched
    starts: list[float] = []
    state = {"calls": 0}

    def slow_first(model, tokenizer, items, device, loops=None):
        # A generator function on purpose: the body (including the sleep) runs on the first
        # next(), inside _run_batch's producer thread, so the slowdown simulates a long
        # in-flight generation without blocking the event loop itself.
        starts.append(time.monotonic())
        if state["calls"] == 0:
            time.sleep(0.4)
        state["calls"] += 1
        yield from real_generate_tokens_batched(model, tokenizer, items, device, loops)

    monkeypatch.setattr(serve_module, "generate_tokens_batched", slow_first)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"prompt": "hello there", "max_tokens": 3, "temperature": 0},
                )
            )
            # Arrive after the first batch's 150ms formation window closed (so this request
            # can't join it) but while its slowed generation is still in flight.
            await asyncio.sleep(0.2)
            second = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"prompt": "a slow turtle races", "max_tokens": 3, "temperature": 0},
                )
            )
            return await asyncio.gather(first, second)

    responses = asyncio.run(scenario())

    assert all(r.status_code == 200 for r in responses)
    assert len(starts) == 2, "the two requests must run as separate batches"
    # 0.4s slowdown + a few ms of real generation, not 0.4s + the 150ms formation wait.
    assert starts[1] - starts[0] < 0.5


def test_batch_setup_failure_fails_the_batch_and_server_keeps_serving(tiny_cfg, monkeypatch, caplog):
    """A failure during batch setup — before the producer thread exists, e.g. thread-creation
    exhaustion — must fail the batch's requests with an error (not hang them on their per-request
    queues), and the server must keep serving subsequent requests. Both requests run on one event
    loop (a real uvicorn deployment's shape), so nothing per-call can mask a dispatcher that died."""
    import radiance.serve as serve_module

    cfg = _sft_cfg(tiny_cfg, loop_count=2)
    app, _, _ = _app(cfg, max_batch_size=4, batch_wait_ms=0.0)

    real_generate_tokens_batched = serve_module.generate_tokens_batched
    state = {"calls": 0}

    def fail_first_call(model, tokenizer, items, device, loops=None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("simulated batch setup failure")
        return real_generate_tokens_batched(model, tokenizer, items, device, loops)

    monkeypatch.setattr(serve_module, "generate_tokens_batched", fail_first_call)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # asyncio.wait_for (not just httpx's own timeout, which ASGITransport doesn't
            # enforce) bounds the await so a regression to the old hang-fails-the-request bug
            # fails this test in seconds instead of stalling.
            try:
                first: httpx.Response | RuntimeError = await asyncio.wait_for(
                    client.post(
                        "/v1/completions", json={"prompt": "hello there", "max_tokens": 3, "temperature": 0}
                    ),
                    timeout=10.0,
                )
            except RuntimeError as exc:
                first = exc
            except TimeoutError:
                pytest.fail("first request hung: the setup failure left it waiting on its queue")
            second = await client.post(
                "/v1/completions", json={"prompt": "a slow turtle races", "max_tokens": 3, "temperature": 0}
            )
            return first, second

    with caplog.at_level(logging.ERROR, logger="radiance.serve"):
        first, second = asyncio.run(scenario())

    # Starlette's ServerErrorMiddleware sends the 500 and then re-raises to the ASGI caller, so
    # through ASGITransport the failure surfaces as the raised exception, not a response object.
    # Before the fix, this await instead hung on the request's queue until httpx's read timeout.
    assert isinstance(first, RuntimeError)
    assert "simulated batch setup failure" in str(first)
    assert second.status_code == 200
    assert any("generation batch" in r.getMessage() for r in caplog.records)
