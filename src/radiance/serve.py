from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import AsyncIterator, Literal

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

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if req.top_p is not None and req.top_p != 1.0:
            raise HTTPException(status_code=400, detail="top_p is not supported by this server; use top_k")

        prompt = format_chat_messages([m.model_dump() for m in req.messages], cfg)
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
        stop_seqs = _stop_sequences(req.stop)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not req.stream:
            async with lock:
                ids_so_far: list[int] = []
                text_so_far = ""
                stopped_on_sequence = False
                for token_id in generate_tokens(
                    model, tokenizer, prompt, req.max_tokens, req.temperature, req.top_k, device, req.loops
                ):
                    ids_so_far.append(token_id)
                    text_so_far = tokenizer.decode(ids_so_far, skip_special_tokens=True)
                    stop_at = _matched_stop(text_so_far, stop_seqs)
                    if stop_at is not None:
                        text_so_far = text_so_far.split(stop_at, 1)[0]
                        stopped_on_sequence = True
                        break
                # generate_tokens only stops early (before max_tokens iterations) via internal EOS
                # detection or our own stop-sequence break above; either way that's "stop", and
                # exhausting the token budget without either is "length".
                finish_reason = "length" if len(ids_so_far) >= req.max_tokens and not stopped_on_sequence else "stop"

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
                    completion_tokens=len(ids_so_far),
                    total_tokens=prompt_tokens + len(ids_so_far),
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

                ids_so_far: list[int] = []
                text_so_far = ""
                stopped_on_sequence = False
                for token_id in generate_tokens(
                    model, tokenizer, prompt, req.max_tokens, req.temperature, req.top_k, device, req.loops
                ):
                    ids_so_far.append(token_id)
                    new_text = tokenizer.decode(ids_so_far, skip_special_tokens=True)
                    stop_at = _matched_stop(new_text, stop_seqs)
                    if stop_at is not None:
                        new_text = new_text.split(stop_at, 1)[0]
                        stopped_on_sequence = True

                    delta_text = new_text[len(text_so_far) :]
                    text_so_far = new_text
                    if delta_text:
                        chunk = ChatCompletionStreamChunk(
                            id=completion_id,
                            created=created,
                            model=req.model,
                            choices=[ChatCompletionStreamChoice(delta=Delta(content=delta_text))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    if stopped_on_sequence:
                        break

                # See the non-streaming handler for why this condition identifies "length".
                finish_reason = "length" if len(ids_so_far) >= req.max_tokens and not stopped_on_sequence else "stop"

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
