from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, resolve_device
from radiance.data import build_tokenizer
from radiance.sft_data import format_chat_prompt
from radiance.model import (
    DenseTransformer,
    load_transformer_from_checkpoint,
    mask_vocab_padding,
)


def load_checkpoint(path: str, device: str) -> tuple[DenseTransformer, Config, PreTrainedTokenizerBase]:
    # eos_id intentionally left at its default (None): doc masking stays off during generation,
    # since a single prompt is one document anyway (see model.load_transformer_from_checkpoint).
    model, cfg = load_transformer_from_checkpoint(path, device)
    tokenizer = build_tokenizer(cfg)
    return model, cfg, tokenizer


@torch.no_grad()
def generate_tokens(
    model: DenseTransformer,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
    loops: int | None = None,
) -> Iterator[int]:
    """Core sampling loop shared by generate() and radiance.serve: yields one new token id per
    step. Never yields the EOS token itself (breaks before yielding it), matching generate()'s
    skip_special_tokens=True behavior.

    Takes an already-tokenized `input_ids` (shape `[1, seq_len]`, on `device`) rather than a raw
    prompt string, so callers that also need the prompt token count or the full id sequence (e.g.
    generate()) tokenize the prompt exactly once instead of once here and once more themselves.

    `loops` overrides how many times the weight-shared loop body runs per forward pass.

    A model trained with stochastic loop depth (model.loop_count_min/max) has seen a range of
    depths, so inference can spend *more* compute per token than training did by raising this —
    test-time compute scaling with no change to the weights. The KV cache is sized to match, since
    it needs one slot per (block, iteration) pair actually executed.
    """
    next_input = input_ids[:, -model.cfg.max_seq_len :]
    kv_cache = model.new_kv_cache(loop_count=loops)

    for _ in range(max_new_tokens):
        assert kv_cache.seq_len + next_input.shape[1] <= model.cfg.max_seq_len, (
            f"generation would exceed model.cfg.max_seq_len ({model.cfg.max_seq_len}); "
            "reduce --max-new-tokens or use a checkpoint trained with a larger max_seq_len"
        )
        logits = model(next_input, kv_cache=kv_cache, loop_count=loops).logits[:, -1, :]

        mask_vocab_padding(logits, tokenizer)

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        yield next_token.item()
        next_input = next_token  # every step after the first decodes a single new token


@dataclass
class BatchItem:
    """One request's prompt and sampling settings, for generate_tokens_batched.

    All items in a batch must share the `loops` value passed separately to
    generate_tokens_batched — loop depth is one value per forward call, unlike temperature/top_k/
    max_new_tokens which are genuinely independent per row here.
    """

    input_ids: torch.Tensor  # (1, prompt_len), unpadded, on `device`
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50


def _sample_next_token(logits_row: torch.Tensor, temperature: float, top_k: int) -> int:
    """Samples one token id from a single row's (1, vocab) logits.

    Factored out of generate_tokens' inline version so generate_tokens_batched can apply a
    different temperature/top_k per row; logic is otherwise identical.
    """
    if temperature == 0:
        return logits_row.argmax(dim=-1).item()
    logits_row = logits_row / temperature
    if top_k > 0:
        top_values, _ = torch.topk(logits_row, min(top_k, logits_row.size(-1)))
        logits_row = logits_row.masked_fill(logits_row < top_values[:, [-1]], float("-inf"))
    probs = F.softmax(logits_row, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate_tokens_batched(
    model: DenseTransformer,
    tokenizer: PreTrainedTokenizerBase,
    items: list[BatchItem],
    device: str = "cpu",
    loops: int | None = None,
) -> Iterator[list[tuple[int, int | None]]]:
    """Batched counterpart to generate_tokens: runs `items` (right-padded prompts of possibly
    different lengths, one shared KVCache) through a single forward pass per step instead of one
    call per request, so the shared-weight loop body is paid for once per *batch* rather than once
    per request — see model/masking.py's padded_causal_mask for how padding stays excluded from
    attention.

    Yields, once per generation step, the list of (row_index, token_id) pairs for rows with news
    this step: `token_id` is a real sampled id for a row still producing output, or `None` exactly
    once — the step a row hits its own EOS or its own max_new_tokens — signaling that row is done
    and will never appear again. A done row keeps being fed a placeholder token every later step so
    the batch's tensor shapes stay uniform until every row finishes (this repo's "idle to
    completion" batching design), but since attention is masked per row, it can never influence any
    other row's output, and no further sampling work is spent on it.
    """
    batch = len(items)
    prompts = [item.input_ids[:, -model.cfg.max_seq_len :] for item in items]
    prompt_lens = [p.shape[1] for p in prompts]
    max_prompt_len = max(prompt_lens)

    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    padded = torch.full((batch, max_prompt_len), pad_id, dtype=prompts[0].dtype, device=device)
    key_padding_mask = torch.zeros((batch, max_prompt_len), dtype=torch.bool, device=device)
    for i, (p, plen) in enumerate(zip(prompts, prompt_lens)):
        padded[i, :plen] = p[0]
        key_padding_mask[i, :plen] = True
    # Right-padding keeps every row's real prefill tokens start-aligned with their true absolute
    # position (position == column index here), so the prefill's key_positions is just an arange —
    # the divergence only shows up once decode continues past prefill (see the running real_len
    # tracker below and key_padding_mask's docstring in model/core.py's LoopContext).
    key_positions = torch.arange(max_prompt_len, device=device).unsqueeze(0).expand(batch, -1)
    real_len = list(prompt_lens)

    kv_cache = model.new_kv_cache(loop_count=loops)
    assert kv_cache.seq_len + max_prompt_len <= model.cfg.max_seq_len, (
        f"batched prefill would exceed model.cfg.max_seq_len ({model.cfg.max_seq_len}); "
        "reduce the longest prompt in the batch or use a checkpoint trained with a larger max_seq_len"
    )
    logits = model(
        padded, kv_cache=kv_cache, loop_count=loops,
        key_padding_mask=key_padding_mask, key_positions=key_positions,
    ).logits
    mask_vocab_padding(logits, tokenizer)
    last_idx = torch.tensor([plen - 1 for plen in prompt_lens], device=device)
    next_logits = logits[torch.arange(batch, device=device), last_idx, :]

    done = [False] * batch
    yielded_count = [0] * batch

    while not all(done):
        step_tokens = torch.full((batch, 1), pad_id, dtype=padded.dtype, device=device)
        step_results: list[tuple[int, int | None]] = []
        for i, item in enumerate(items):
            if done[i]:
                continue
            token_id = _sample_next_token(next_logits[i : i + 1], item.temperature, item.top_k)
            step_tokens[i, 0] = token_id
            is_eos = tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id
            if is_eos or yielded_count[i] >= item.max_new_tokens:
                done[i] = True
                step_results.append((i, None))
            else:
                yielded_count[i] += 1
                step_results.append((i, token_id))
        if step_results:
            yield step_results
        if all(done):
            break

        assert kv_cache.seq_len + 1 <= model.cfg.max_seq_len, (
            f"generation would exceed model.cfg.max_seq_len ({model.cfg.max_seq_len}); "
            "reduce --max-new-tokens or use a checkpoint trained with a larger max_seq_len"
        )
        key_padding_mask = torch.cat(
            [key_padding_mask, torch.ones((batch, 1), dtype=torch.bool, device=device)], dim=1
        )
        new_positions = torch.tensor([[real_len[i]] for i in range(batch)], device=device)
        key_positions = torch.cat([key_positions, new_positions], dim=1)
        for i in range(batch):
            real_len[i] += 1
        logits = model(
            step_tokens, kv_cache=kv_cache, loop_count=loops,
            key_padding_mask=key_padding_mask, key_positions=key_positions,
        ).logits[:, -1, :]
        mask_vocab_padding(logits, tokenizer)
        next_logits = logits


def generate(
    model: DenseTransformer,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
    loops: int | None = None,
) -> str:
    """Thin wrapper over generate_tokens(): reassembles the full id sequence (prompt + generated)
    and decodes it in one call, rather than decoding the prompt and completion separately and
    string-concatenating — a BPE tokenizer's decode of a completion is not guaranteed to match its
    decode as a continuation of the prompt (e.g. leading-space merges at the boundary), so the ids
    must be joined before decoding, not the decoded strings.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    input_ids = input_ids[:, -model.cfg.max_seq_len :]
    new_ids = list(
        generate_tokens(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, device, loops)
    )
    if new_ids:
        new_ids_tensor = torch.tensor([new_ids], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, new_ids_tensor], dim=-1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from radiance.train")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=50, help="0 to disable top-k filtering")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Override loop-body iterations per forward pass (default: the checkpoint's own "
        "loop_count_max / max_loops). Raise it to spend more compute per token than training did — "
        "most useful for a model trained with stochastic loop depth.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Wrap --prompt in the checkpoint's chat turn template (cfg.sft.user_prefix/"
        "assistant_prefix or cfg.dpo.user_prefix/assistant_prefix) before generating, matching how "
        "the checkpoint was trained. Requires a checkpoint trained with sft.enabled: true or "
        "dpo.enabled: true.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, cfg, tokenizer = load_checkpoint(args.checkpoint, device)
    prompt = args.prompt
    if args.chat:
        prompt = format_chat_prompt(args.prompt, cfg)

    text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        loops=args.loops,
    )
    print(text)


if __name__ == "__main__":
    main()
