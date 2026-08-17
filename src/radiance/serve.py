from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, resolve_device
from radiance.generate import generate_tokens, load_checkpoint
from radiance.model import DenseTransformer
from radiance.sft_data import format_chat_messages

logger = logging.getLogger("radiance.serve")

_HEALTH_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def _token_matches(token: str, key: str) -> bool:
    """Constant-time comparison that also tolerates non-ASCII input. `secrets.compare_digest`
    raises TypeError when given a `str` containing non-ASCII characters (e.g. a header decoded
    via Starlette's latin-1 header codec, or a non-ASCII --api-key); comparing the UTF-8-encoded
    bytes instead sidesteps that restriction without weakening the constant-time guarantee.
    """
    return secrets.compare_digest(token.encode("utf-8"), key.encode("utf-8"))


def _require_chat_enabled(cfg: Config) -> None:
    if not (cfg.sft.enabled or cfg.dpo.enabled):
        raise ValueError(
            "/v1/chat/completions requires a checkpoint trained with sft.enabled: true or "
            "dpo.enabled: true, since it formats requests with format_chat_messages. Use "
            "/v1/completions for a base checkpoint."
        )


@dataclass
class ServerMetrics:
    """In-memory counters for /metrics. No lock: uvicorn's default single worker runs one asyncio
    event loop, and every increment here happens between `await` points, so updates can't
    interleave — a lock would guard against a race that can't occur in this deployment.
    """

    started_at: float = field(default_factory=time.monotonic)
    requests_total: int = 0
    errors_total: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    generation_seconds_total: float = 0.0

    def record_request(self, is_error: bool) -> None:
        self.requests_total += 1
        if is_error:
            self.errors_total += 1

    def record_generation(self, prompt_tokens: int, completion_tokens: int, elapsed_seconds: float) -> None:
        self.prompt_tokens_total += prompt_tokens
        self.completion_tokens_total += completion_tokens
        self.generation_seconds_total += elapsed_seconds

    def snapshot(self) -> dict:
        uptime = max(time.monotonic() - self.started_at, 1e-9)
        return {
            "uptime_seconds": round(uptime, 3),
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "requests_per_second": round(self.requests_total / uptime, 4),
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "tokens_per_second": (
                round(self.completion_tokens_total / self.generation_seconds_total, 2)
                if self.generation_seconds_total > 0
                else 0.0
            ),
        }


class RateLimiter:
    """Fixed 60-second-window request counter, keyed by API key (or client IP when auth is
    disabled). Good enough for "don't fall over under a traffic spike", not a precise sliding
    window — matching this file's "explicit over abstraction" bar rather than pulling in a
    dependency like slowapi for one counter.
    """

    def __init__(self, limit_per_minute: int):
        self.limit_per_minute = limit_per_minute
        self._windows: dict[str, tuple[int, int]] = {}
        self._last_swept_window: int | None = None

    def allow(self, key: str) -> bool:
        if self.limit_per_minute <= 0:
            return True
        window = int(time.monotonic() // 60)
        self._sweep(window)
        start_window, count = self._windows.get(key, (window, 0))
        if start_window != window:
            start_window, count = window, 0
        count += 1
        self._windows[key] = (start_window, count)
        return count <= self.limit_per_minute

    def _sweep(self, window: int) -> None:
        """Evicts entries from stale windows so `_windows` doesn't grow without bound over the
        life of a long-running server as new client IPs/keys are seen. Runs at most once per
        window (not on every call), so this stays O(1) amortized.
        """
        if self._last_swept_window == window:
            return
        self._last_swept_window = window
        stale_keys = [k for k, (start_window, _) in self._windows.items() if start_window != window]
        for k in stale_keys:
            del self._windows[k]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "radiance"
    messages: list[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 200
    stream: bool = False
    stop: str | list[str] | None = None
    top_k: int = 50
    top_p: float | None = None
    loops: int | None = None


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


class CompletionRequest(BaseModel):
    """Legacy /v1/completions request. `prompt` is a single string, not OpenAI's `str | list[str]`
    — this server handles one request at a time (see docs on request batching), so there is no
    benefit to accepting a prompt list only to run it as a sequential loop.
    """

    model: str = "radiance"
    prompt: str
    temperature: float = 0.8
    max_tokens: int = 200
    stream: bool = False
    stop: str | list[str] | None = None
    top_k: int = 50
    top_p: float | None = None
    loops: int | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: ChatCompletionUsage


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionStreamChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "radiance"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


def create_app(
    model: DenseTransformer,
    cfg: Config,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    model_name: str,
    api_keys: set[str] | None = None,
    rate_limit_per_minute: int = 0,
) -> FastAPI:
    lock = asyncio.Lock()
    server_started = int(time.time())
    metrics = ServerMetrics()
    limiter = RateLimiter(rate_limit_per_minute)

    app = FastAPI()

    async def _auth_and_rate_limit(request: Request) -> None:
        """Dependency on every /v1/* route: bearer-token auth (skipped entirely when no keys are
        configured, matching the quickstart's no-flags-needed default) followed by a rate limit,
        both before the route handler ever touches `lock` — an over-quota, unauthenticated, or
        invalid-key request must never queue behind a real generation. A request presenting a
        valid key is rate-limited per key, so distinct keys get independent quotas as documented
        and can't be starved by unrelated traffic sharing the same IP. Requests with a missing or
        wrong key — which never reach the auth check's success path — are throttled per client IP
        instead, so they don't get unlimited attempts to brute-force a key; when the client IP is
        unavailable (e.g. a Unix domain socket) there is no identifier to key a shared bucket on
        without pooling unrelated callers together, so such requests skip the rate limit and fall
        through to the auth check.
        """
        authorization = request.headers.get("authorization", "")
        token = authorization[len("Bearer ") :] if authorization.startswith("Bearer ") else ""
        matched_key = next((key for key in api_keys if _token_matches(token, key)), None) if api_keys else None

        if matched_key is not None:
            rate_limit_key = f"key:{matched_key}"
        elif request.client is not None:
            rate_limit_key = f"ip:{request.client.host}"
        else:
            rate_limit_key = None

        if rate_limit_key is not None and not limiter.allow(rate_limit_key):
            raise HTTPException(
                status_code=429,
                detail={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                headers={"Retry-After": "60"},
            )

        if api_keys and matched_key is None:
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
            )

    @app.middleware("http")
    async def _log_and_count(request: Request, call_next):
        """Structured per-request logging plus the request/error counters behind /metrics.
        Deliberately does *not* drive tokens_per_second: for a streaming response, `call_next`
        returns once headers are ready, well before the generator finishes, so timing token
        throughput here would undercount every streamed request. Handlers report generation time
        themselves via `metrics.record_generation` once a completion actually finishes.
        """
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            metrics.record_request(is_error=True)
            logger.exception(
                "request failed", extra={"method": request.method, "path": request.url.path}
            )
            raise
        duration_ms = (time.monotonic() - start) * 1000
        metrics.record_request(is_error=response.status_code >= 400)
        # /healthz, /readyz, and /metrics are polled every few seconds by liveness/readiness
        # probes; logging that traffic at INFO would drown out real request logs, so it's demoted
        # to DEBUG unless a probe request actually fails.
        level = (
            logging.DEBUG
            if request.url.path in _HEALTH_PATHS and response.status_code < 400
            else logging.INFO
        )
        logger.log(
            level,
            "%s",
            {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict:
        # The model and tokenizer are loaded synchronously before create_app() is ever called, so
        # by the time this route can be hit the server is always ready to serve.
        return {"status": "ready"}

    @app.get("/metrics")
    async def get_metrics() -> dict:
        return metrics.snapshot()

    @app.get("/v1/models", dependencies=[Depends(_auth_and_rate_limit)])
    async def list_models() -> ModelList:
        return ModelList(data=[ModelInfo(id=model_name, created=server_started)])

    def _stop_sequences(stop: str | list[str] | None) -> list[str]:
        if stop is None:
            return []
        return [stop] if isinstance(stop, str) else stop

    def _matched_stop(text: str, stop_seqs: list[str]) -> str | None:
        matches = [s for s in stop_seqs if s in text]
        if not matches:
            return None
        return min(matches, key=lambda s: text.index(s))

    def _longest_partial_overlap(text: str, stop_seqs: list[str]) -> int:
        """Length of the longest suffix of `text` that is a strict prefix of some stop sequence —
        i.e. text that could still turn into a stop-sequence match once more text arrives, and so
        must not be emitted yet. Used to hold back a streamed delta that ends mid-stop-sequence
        instead of yielding it (and, for a streaming response, sending it to the client) before
        it's known whether the stop sequence will actually complete.
        """
        best = 0
        for s in stop_seqs:
            for n in range(min(len(text), len(s) - 1), 0, -1):
                if text.endswith(s[:n]):
                    best = max(best, n)
                    break
        return best

    def _finish_reason(completion_tokens: int, max_tokens: int, stopped_on_sequence: bool) -> str:
        # generate_tokens only stops early (before max_tokens iterations) via internal EOS
        # detection or a stop-sequence match; either way that's "stop", and exhausting the token
        # budget without either is "length".
        return "length" if completion_tokens >= max_tokens and not stopped_on_sequence else "stop"

    def _tokenize_and_check(prompt: str) -> tuple[torch.Tensor, int]:
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_tokens = prompt_ids.shape[1]
        if prompt_tokens >= model.cfg.max_seq_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"prompt is {prompt_tokens} tokens, which leaves no room to generate within "
                    f"this model's max_seq_len ({model.cfg.max_seq_len}); shorten the prompt"
                ),
            )
        return prompt_ids, prompt_tokens

    def _decode_deltas(ids_iter: Iterator[int], tokenizer: PreTrainedTokenizerBase) -> Iterator[str]:
        """Decodes generated token ids incrementally, yielding only the text new since the last
        yield. Re-decoding resets at each whitespace boundary instead of growing over the whole
        completion, so total decode work stays roughly linear in completion length rather than
        quadratic, while each window still starts from a clean boundary so within-word BPE merges
        (e.g. leading-space joins) decode correctly.
        """
        window_ids: list[int] = []
        window_text = ""
        for token_id in ids_iter:
            window_ids.append(token_id)
            new_text = tokenizer.decode(window_ids, skip_special_tokens=True)
            yield new_text[len(window_text) :]
            window_text = new_text
            if window_text.endswith((" ", "\n")):
                window_ids = []
                window_text = ""

    def _iter_in_thread(sync_iter: Iterator[str]) -> AsyncIterator[str]:
        """Runs a blocking iterator (the generation loop, ultimately) on a background thread and
        re-surfaces its items to the event loop via a queue, so a slow generation doesn't stall
        the event loop for other requests (e.g. a concurrent GET /v1/models) for its whole
        duration — only true generation concurrency is serialized behind `lock`, not the loop
        itself.

        The returned async generator must be closed (via `aclose()`, e.g. through a `finally`
        block, not just abandoned by an early `break`) whenever it isn't drained to completion —
        e.g. a stop-sequence match ending generation before the token budget is exhausted —
        otherwise the background thread keeps calling `call_soon_threadsafe` on this event loop
        after the caller has moved on, which raises once the loop closes (RuntimeError: Event loop
        is closed) instead of exiting cleanly.
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        done = object()

        def producer() -> None:
            try:
                for item in sync_iter:
                    if stop_event.is_set():
                        return
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:  # re-raised on the event-loop side, not swallowed
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        async def consumer() -> AsyncIterator[str]:
            try:
                while True:
                    item = await queue.get()
                    if item is done:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                stop_event.set()
                await asyncio.to_thread(thread.join)

        return consumer()

    async def _generate_deltas(
        input_ids: torch.Tensor,
        req: ChatCompletionRequest | CompletionRequest,
        stop_seqs: list[str],
    ) -> AsyncIterator[tuple[str, bool, int]]:
        """Shared generation core for both /v1/chat/completions and /v1/completions: yields
        `(delta_text, stopped, count)` as tokens arrive, where `stopped` is True on the final
        tuple iff generation ended because a stop sequence matched (as opposed to exhausting
        max_tokens or hitting EOS), and `count` is the number of tokens generated so far.

        Text that could be the start of a stop sequence is held back rather than yielded
        immediately, since a stop sequence can span multiple underlying token deltas — yielding
        eagerly would leak a partial stop sequence into the response (and, for a streaming
        request, onto the wire) before it's known whether it actually completes.
        """
        sync_gen = generate_tokens(
            model, tokenizer, input_ids, req.max_tokens, req.temperature, req.top_k, device, req.loops
        )
        stream = _iter_in_thread(_decode_deltas(sync_gen, tokenizer))
        text_so_far = ""
        yielded_len = 0
        count = 0
        try:
            async for delta in stream:
                count += 1
                text_so_far += delta
                stop_at = _matched_stop(text_so_far, stop_seqs)
                if stop_at is not None:
                    stop_index = text_so_far.index(stop_at)
                    yield text_so_far[yielded_len:stop_index], True, count
                    return
                safe_boundary = len(text_so_far) - _longest_partial_overlap(text_so_far, stop_seqs)
                if safe_boundary > yielded_len:
                    yield text_so_far[yielded_len:safe_boundary], False, count
                    yielded_len = safe_boundary
            if yielded_len < len(text_so_far):
                yield text_so_far[yielded_len:], False, count
        finally:
            await stream.aclose()

    async def _run_generation(
        input_ids: torch.Tensor,
        req: ChatCompletionRequest | CompletionRequest,
        stop_seqs: list[str],
        prompt_tokens: int,
    ) -> AsyncIterator[tuple[str, int, str | None]]:
        """Runs one generation under `lock` and yields `(delta, completion_tokens, finish_reason)`
        for each token delta — `finish_reason` is None for every item except the last, which
        carries an empty delta and the reason computed right after `metrics.record_generation` has
        run. Both chat_completions and completions drive their streaming and non-streaming branches
        off this single generator instead of each duplicating the
        lock/gen_start/metrics/_finish_reason sequence across both branches.
        """
        async with lock:
            gen_start = time.monotonic()
            stopped_on_sequence = False
            completion_tokens = 0
            async for delta, stopped, count in _generate_deltas(input_ids, req, stop_seqs):
                stopped_on_sequence = stopped
                completion_tokens = count
                if delta:
                    yield delta, completion_tokens, None
            metrics.record_generation(prompt_tokens, completion_tokens, time.monotonic() - gen_start)
            finish_reason = _finish_reason(completion_tokens, req.max_tokens, stopped_on_sequence)
            yield "", completion_tokens, finish_reason

    @app.post("/v1/chat/completions", dependencies=[Depends(_auth_and_rate_limit)])
    async def chat_completions(req: ChatCompletionRequest):
        try:
            _require_chat_enabled(cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if req.top_p is not None and req.top_p != 1.0:
            raise HTTPException(status_code=400, detail="top_p is not supported by this server; use top_k")
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        prompt = format_chat_messages([m.model_dump() for m in req.messages], cfg)
        prompt_ids, prompt_tokens = _tokenize_and_check(prompt)
        stop_seqs = _stop_sequences(req.stop)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            text_so_far = ""
            completion_tokens = 0
            finish_reason = "stop"
            async for delta, completion_tokens, reason in _run_generation(
                prompt_ids, req, stop_seqs, prompt_tokens
            ):
                text_so_far += delta
                if reason is not None:
                    finish_reason = reason

            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[
                    ChatCompletionChoice(
                        message=ResponseMessage(content=text_so_far),
                        finish_reason=finish_reason,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        async def event_generator() -> AsyncIterator[str]:
            first_chunk = ChatCompletionStreamChunk(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[ChatCompletionStreamChoice(delta=Delta(role="assistant", content=""))],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            async for delta, _completion_tokens, finish_reason in _run_generation(
                prompt_ids, req, stop_seqs, prompt_tokens
            ):
                if finish_reason is not None:
                    final_chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[ChatCompletionStreamChoice(delta=Delta(), finish_reason=finish_reason)],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                elif delta:
                    chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[ChatCompletionStreamChoice(delta=Delta(content=delta))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/v1/completions", dependencies=[Depends(_auth_and_rate_limit)])
    async def completions(req: CompletionRequest):
        if req.top_p is not None and req.top_p != 1.0:
            raise HTTPException(status_code=400, detail="top_p is not supported by this server; use top_k")
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        prompt_ids, prompt_tokens = _tokenize_and_check(req.prompt)
        stop_seqs = _stop_sequences(req.stop)
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            text_so_far = ""
            completion_tokens = 0
            finish_reason = "stop"
            async for delta, completion_tokens, reason in _run_generation(
                prompt_ids, req, stop_seqs, prompt_tokens
            ):
                text_so_far += delta
                if reason is not None:
                    finish_reason = reason

            return CompletionResponse(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[CompletionChoice(text=text_so_far, finish_reason=finish_reason)],
                usage=ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        async def event_generator() -> AsyncIterator[str]:
            async for delta, _completion_tokens, finish_reason in _run_generation(
                prompt_ids, req, stop_seqs, prompt_tokens
            ):
                if finish_reason is not None:
                    final_chunk = CompletionStreamChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[CompletionStreamChoice(text="", finish_reason=finish_reason)],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                elif delta:
                    chunk = CompletionStreamChunk(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[CompletionStreamChoice(text=delta)],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from radiance.train")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name reported in /v1/models and echoed in responses (default: the checkpoint filename)",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        default=None,
        help="Bearer token required on /v1/* endpoints (Authorization: Bearer <key>). Repeatable "
        "for multiple valid keys. Also read from the RADIANCE_API_KEY env var (comma-separated). "
        "If no key is configured from either source, /v1/* endpoints are unauthenticated — the "
        "default, so the quickstart keeps working with no flags.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Max requests per minute per API key (or per client IP when no --api-key is set). "
        "0 (default) disables rate limiting.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api_keys = {key for key in (args.api_key or []) if key}
    api_keys |= {key.strip() for key in os.environ.get("RADIANCE_API_KEY", "").split(",") if key.strip()}

    model, cfg, tokenizer = load_checkpoint(args.checkpoint, device)

    model_name = args.model_name or args.checkpoint.rsplit("/", 1)[-1]
    app = create_app(model, cfg, tokenizer, device, model_name, api_keys=api_keys, rate_limit_per_minute=args.rate_limit)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
