from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, resolve_device
from radiance.generate import BatchItem, generate_tokens_batched, load_checkpoint
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


_DONE = object()  # sentinel on a _PendingRequest's queue: this request's generation is finished


@dataclass
class _PendingRequest:
    """One request waiting for (or currently inside) a batched generation call.

    Built by _generate_deltas and handed to the dispatcher's queue (see _ensure_dispatcher_running);
    `queue` (this dataclass's own field) is where
    the dispatcher deposits this request's own token ids (int), completion (_DONE), or a
    generation-time exception, as a batch it's part of runs — see _run_batch.
    """

    input_ids: torch.Tensor
    loops: int | None
    max_tokens: int
    temperature: float
    top_k: int
    queue: "asyncio.Queue"


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
    — concurrent requests are already batched together by the dispatcher (see
    _dispatcher_loop/_run_batch), one prompt each, so accepting a prompt list here would just
    mean unpacking it into that same per-request path rather than adding anything.
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
    max_batch_size: int = 8,
    batch_wait_ms: float = 20.0,
) -> FastAPI:
    server_started = int(time.time())
    metrics = ServerMetrics()
    limiter = RateLimiter(rate_limit_per_minute)
    batch_wait_seconds = batch_wait_ms / 1000.0
    # Mutable box (not plain locals) so _ensure_dispatcher_running can rebind it from a closure;
    # see that function for why both the task *and* the queue get recreated together.
    dispatcher_state: dict = {"task": None, "queue": None, "loop": None}

    async def _run_batch(batch: list[_PendingRequest]) -> None:
        """Runs one batched generation call for `batch` on a background thread — the blocking
        (synchronous, ultimately model-forward-calling) generate_tokens_batched iterator can't run
        directly on the event loop — and fans each step's per-row results out to that row's own
        `_PendingRequest.queue`: a real token id, or _DONE the one time generate_tokens_batched
        reports a row finished. A generation-time exception is delivered to every request in the
        batch, since they share one forward pass.

        Guarantees that every request in `batch` receives _DONE or an exception on its own queue
        even when the failure happens outside the producer thread's own try/except — e.g. batch
        setup (building the items, creating/starting the thread) fails before the producer
        exists and has no in-thread path to deliver through. Such a failure is logged and
        swallowed here rather than propagated: one bad batch must not kill the dispatcher task
        and strand the rest of the queue's requests.
        """
        try:
            items = [
                BatchItem(
                    input_ids=p.input_ids, max_new_tokens=p.max_tokens, temperature=p.temperature, top_k=p.top_k
                )
                for p in batch
            ]
            sync_gen = generate_tokens_batched(model, tokenizer, items, device, batch[0].loops)

            running_loop = asyncio.get_running_loop()
            result_queue: asyncio.Queue = asyncio.Queue()
            stop_event = threading.Event()
            batch_done = object()

            def producer() -> None:
                try:
                    for step_results in sync_gen:
                        if stop_event.is_set():
                            return
                        running_loop.call_soon_threadsafe(result_queue.put_nowait, step_results)
                except Exception as exc:  # re-raised on the event-loop side, not swallowed
                    running_loop.call_soon_threadsafe(result_queue.put_nowait, exc)
                finally:
                    running_loop.call_soon_threadsafe(result_queue.put_nowait, batch_done)

            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

            try:
                while True:
                    item = await result_queue.get()
                    if item is batch_done:
                        return
                    if isinstance(item, Exception):
                        for p in batch:
                            p.queue.put_nowait(item)
                        return
                    for row, token_id in item:
                        p = batch[row]
                        p.queue.put_nowait(_DONE if token_id is None else token_id)
            finally:
                stop_event.set()
                await asyncio.to_thread(thread.join)
        except Exception as exc:
            logger.exception(
                "generation batch of %d request(s) failed; failing each of them", len(batch)
            )
            for p in batch:
                p.queue.put_nowait(exc)

    async def _dispatcher_loop(queue: "asyncio.Queue") -> None:
        """Forms and runs batches from `queue` one at a time — one forward pass in flight at a
        time, since the model is a single shared instance, not safe to run concurrently — but
        overlaps the next batch's *formation* with the current batch's *generation*: as soon as
        this batch starts generating, a background task starts forming the next one, so a
        request's `batch_wait_seconds` window is anchored at its arrival, not delayed until the
        in-flight batch happens to finish (its wait is paid alongside the generation it queues
        behind, not on top of it), and a fully-formed batch starts the instant the current one
        completes.

        Formation takes the first request immediately, then drains whatever is already queued
        before ever waiting — so `--batch-wait-ms 0` still batches opportunistically (everything
        already in the queue shares the forward pass, with no added wait) — and then waits up to
        `batch_wait_seconds` for more to arrive (capped at `max_batch_size`) so near-simultaneous
        requests share a forward pass instead of each paying for the shared-weight loop body on
        their own. A request whose `loops` override doesn't match the batch being formed is put
        back for the next round instead of blocking this one, since loop depth is one value per
        forward call and can't vary within a batch.
        """
        loop = asyncio.get_running_loop()

        async def _form_batch() -> list[_PendingRequest]:
            first = await queue.get()
            batch = [first]
            leftover: list[_PendingRequest] = []
            deadline = loop.time() + batch_wait_seconds
            while len(batch) < max_batch_size:
                try:
                    candidate = queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        candidate = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                if candidate.loops == first.loops:
                    batch.append(candidate)
                else:
                    leftover.append(candidate)
            for item in leftover:
                queue.put_nowait(item)
            return batch

        forming: asyncio.Task[list[_PendingRequest]] | None = None
        try:
            batch = await _form_batch()
            while True:
                forming = asyncio.ensure_future(_form_batch())
                await _run_batch(batch)
                batch = await forming
                forming = None
        except BaseException:
            # Don't leave a formation task behind if this one dies — it may already be holding
            # requests pulled off `queue`.
            if forming is not None:
                forming.cancel()
            raise

    async def _ensure_dispatcher_running() -> "asyncio.Queue":
        """Lazily (re)starts the dispatcher, returning the queue requests should enqueue onto.

        Not an ASGI startup hook, and not a one-time lazy-init either: a real uvicorn deployment
        has exactly one persistent event loop for the server's whole life, so this only ever fires
        once there — but FastAPI's TestClient, used bare (not as a `with` context manager,
        throughout this file's test suite) hands every separate call a brand-new throwaway event
        loop. A dispatcher task (and the asyncio.Queue it reads from) created on a now-torn-down
        loop is dead weight at best and a silent hang at worst — its queue's internal
        synchronization stays bound to a loop nothing will ever run again — so this checks the
        *current* running loop against the one the dispatcher was last started on, not just
        whether the old task looks done, and restarts both together whenever they differ.
        """
        current_loop = asyncio.get_running_loop()
        task = dispatcher_state["task"]
        if task is None or task.done() or dispatcher_state["loop"] is not current_loop:
            queue: asyncio.Queue = asyncio.Queue()
            dispatcher_state["queue"] = queue
            dispatcher_state["loop"] = current_loop
            dispatcher_state["task"] = asyncio.create_task(_dispatcher_loop(queue))
        return dispatcher_state["queue"]

    app = FastAPI()

    async def _auth_and_rate_limit(request: Request) -> None:
        """Dependency on every /v1/* route: bearer-token auth (skipped entirely when no keys are
        configured, matching the quickstart's no-flags-needed default) followed by a rate limit,
        both before the route handler ever enqueues onto the batching dispatcher — an over-quota,
        unauthenticated, or invalid-key request must never queue behind a real generation. A
        request presenting a valid key is rate-limited per key, so distinct keys get independent
        quotas as documented and can't be starved by unrelated traffic sharing the same IP.
        Requests with a missing or wrong key — which never reach the auth check's success path —
        are throttled per client IP instead, so they don't get unlimited attempts to brute-force a
        key; when the client IP is unavailable (e.g. a Unix domain socket) there is no identifier
        to key a shared bucket on without pooling unrelated callers together, so such requests skip
        the rate limit and fall through to the auth check.
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

    async def _decode_deltas(ids_iter: AsyncIterator[int], tokenizer: PreTrainedTokenizerBase) -> AsyncIterator[str]:
        """Decodes generated token ids incrementally, yielding only the text new since the last
        yield. Re-decoding resets at each whitespace boundary instead of growing over the whole
        completion, so total decode work stays roughly linear in completion length rather than
        quadratic, while each window still starts from a clean boundary so within-word BPE merges
        (e.g. leading-space joins) decode correctly.
        """
        window_ids: list[int] = []
        window_text = ""
        async for token_id in ids_iter:
            window_ids.append(token_id)
            new_text = tokenizer.decode(window_ids, skip_special_tokens=True)
            yield new_text[len(window_text) :]
            window_text = new_text
            if window_text.endswith((" ", "\n")):
                window_ids = []
                window_text = ""

    async def _request_tokens(queue: "asyncio.Queue") -> AsyncIterator[int]:
        """Drains one request's own slice of the dispatcher's batched output (see _run_batch):
        real token ids until _DONE, or re-raises a generation-time exception if one arrived
        instead. Unlike the old per-request background thread this replaced, there is nothing to
        explicitly stop on early abandonment (e.g. a stop-sequence match) — the shared batch
        generation this request is part of keeps running for its batch-mates regardless, and an
        abandoned queue is simply never read again.
        """
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item

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

        Enqueues this request onto the shared dispatcher (starting it on first use) rather than
        generating directly — see _dispatcher_loop and _run_batch: this request's tokens arrive
        from whichever batch it ends up sharing a forward pass with.
        """
        dispatch_queue = await _ensure_dispatcher_running()
        request_queue: asyncio.Queue = asyncio.Queue()
        await dispatch_queue.put(
            _PendingRequest(
                input_ids=input_ids,
                loops=req.loops,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                queue=request_queue,
            )
        )
        stream = _decode_deltas(_request_tokens(request_queue), tokenizer)
        text_so_far = ""
        yielded_len = 0
        count = 0
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

    async def _run_generation(
        input_ids: torch.Tensor,
        req: ChatCompletionRequest | CompletionRequest,
        stop_seqs: list[str],
        prompt_tokens: int,
    ) -> AsyncIterator[tuple[str, int, str | None]]:
        """Yields `(delta, completion_tokens, finish_reason)` for each token delta —
        `finish_reason` is None for every item except the last, which carries an empty delta and
        the reason computed right after `metrics.record_generation` has run. Both
        chat_completions and completions drive their streaming and non-streaming branches off
        this single generator instead of each duplicating the gen_start/metrics/_finish_reason
        sequence across both branches.

        No lock here: generation concurrency is handled by the dispatcher batching concurrent
        requests into shared forward passes (see _dispatcher_loop/_run_batch), not by serializing
        them — this request's own call to _generate_deltas below enqueues it and waits on its own
        result queue, however many other requests are running alongside it.
        """
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
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Max number of concurrent requests the dispatcher batches into one forward pass "
        "(default: 8). Only requests sharing the same --loops override can share a batch.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=20.0,
        help="How long the dispatcher waits for more requests to join a forming batch, after the "
        "first arrives, before it starts generating (default: 20ms). Raising this trades a bit "
        "of per-request latency for a higher average batch fill rate under concurrent traffic; "
        "0 batches opportunistically (whatever's already queued) with no added wait.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api_keys = {key for key in (args.api_key or []) if key}
    api_keys |= {key.strip() for key in os.environ.get("RADIANCE_API_KEY", "").split(",") if key.strip()}

    model, cfg, tokenizer = load_checkpoint(args.checkpoint, device)

    model_name = args.model_name or args.checkpoint.rsplit("/", 1)[-1]
    app = create_app(
        model, cfg, tokenizer, device, model_name,
        api_keys=api_keys, rate_limit_per_minute=args.rate_limit,
        max_batch_size=args.max_batch_size, batch_wait_ms=args.batch_wait_ms,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
