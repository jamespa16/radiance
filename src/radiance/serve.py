from __future__ import annotations

import argparse
import asyncio
import threading
import time
import uuid
from collections.abc import Iterator
from typing import AsyncIterator, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, resolve_device
from radiance.generate import generate_tokens, load_checkpoint
from radiance.model import DenseTransformer
from radiance.sft_data import format_chat_messages


def _require_chat_enabled(cfg: Config) -> None:
    if not (cfg.sft.enabled or cfg.dpo.enabled):
        raise ValueError(
            "radiance-serve requires a checkpoint trained with sft.enabled: true or dpo.enabled: "
            "true, since /v1/chat/completions formats requests with format_chat_messages."
        )


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


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "radiance"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


def create_app(
    model: DenseTransformer, cfg: Config, tokenizer: PreTrainedTokenizerBase, device: str, model_name: str
) -> FastAPI:
    _require_chat_enabled(cfg)
    lock = asyncio.Lock()
    server_started = int(time.time())

    app = FastAPI()

    @app.get("/v1/models")
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
        input_ids: torch.Tensor, req: ChatCompletionRequest, stop_seqs: list[str]
    ) -> AsyncIterator[tuple[str, bool, int]]:
        """Shared generation core for both handlers below: yields `(delta_text, stopped, count)`
        as tokens arrive, where `stopped` is True on the final tuple iff generation ended because
        a stop sequence matched (as opposed to exhausting max_tokens or hitting EOS), and `count`
        is the number of tokens generated so far.
        """
        sync_gen = generate_tokens(
            model, tokenizer, input_ids, req.max_tokens, req.temperature, req.top_k, device, req.loops
        )
        stream = _iter_in_thread(_decode_deltas(sync_gen, tokenizer))
        text_so_far = ""
        count = 0
        try:
            async for delta in stream:
                count += 1
                text_so_far += delta
                stop_at = _matched_stop(text_so_far, stop_seqs)
                if stop_at is not None:
                    overshoot = len(text_so_far) - len(text_so_far.split(stop_at, 1)[0])
                    yield delta[: len(delta) - overshoot], True, count
                    return
                yield delta, False, count
        finally:
            await stream.aclose()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if req.top_p is not None and req.top_p != 1.0:
            raise HTTPException(status_code=400, detail="top_p is not supported by this server; use top_k")
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        prompt = format_chat_messages([m.model_dump() for m in req.messages], cfg)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_tokens = prompt_ids.shape[1]
        stop_seqs = _stop_sequences(req.stop)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            async with lock:
                text_so_far = ""
                stopped_on_sequence = False
                completion_tokens = 0
                async for delta, stopped, count in _generate_deltas(prompt_ids, req, stop_seqs):
                    text_so_far += delta
                    stopped_on_sequence = stopped
                    completion_tokens = count
                # generate_tokens only stops early (before max_tokens iterations) via internal EOS
                # detection or a stop-sequence match above; either way that's "stop", and
                # exhausting the token budget without either is "length".
                finish_reason = (
                    "length" if completion_tokens >= req.max_tokens and not stopped_on_sequence else "stop"
                )

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
            async with lock:
                first_chunk = ChatCompletionStreamChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[ChatCompletionStreamChoice(delta=Delta(role="assistant", content=""))],
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                stopped_on_sequence = False
                completion_tokens = 0
                async for delta, stopped, count in _generate_deltas(prompt_ids, req, stop_seqs):
                    stopped_on_sequence = stopped
                    completion_tokens = count
                    if delta:
                        chunk = ChatCompletionStreamChunk(
                            id=completion_id,
                            created=created,
                            model=req.model,
                            choices=[ChatCompletionStreamChoice(delta=Delta(content=delta))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                # See the non-streaming handler for why this condition identifies "length".
                finish_reason = (
                    "length" if completion_tokens >= req.max_tokens and not stopped_on_sequence else "stop"
                )

                final_chunk = ChatCompletionStreamChunk(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[ChatCompletionStreamChoice(delta=Delta(), finish_reason=finish_reason)],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
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
    args = parser.parse_args()
    device = resolve_device(args.device)

    model, cfg, tokenizer = load_checkpoint(args.checkpoint, device)
    _require_chat_enabled(cfg)

    model_name = args.model_name or args.checkpoint.rsplit("/", 1)[-1]
    app = create_app(model, cfg, tokenizer, device, model_name)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
