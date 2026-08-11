from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, resolve_device
from radiance.data import build_tokenizer
from radiance.model import DenseTransformer


def load_checkpoint(path: str, device: str) -> tuple[DenseTransformer, Config, PreTrainedTokenizerBase]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]

    tokenizer = build_tokenizer(cfg)
    # Must match training's padded vocab (see model.padded_vocab_size), or load_state_dict fails on
    # the token_emb/lm_head shapes. Read the width off the checkpoint itself rather than recomputing
    # it, so checkpoints saved before vocab padding existed still load.
    vocab_size = ckpt["model"]["token_emb.weight"].shape[0]
    model = DenseTransformer(cfg.model, vocab_size=vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, cfg, tokenizer


@torch.no_grad()
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
    """`loops` overrides how many times the weight-shared loop body runs per forward pass.

    A model trained with stochastic loop depth (model.loop_count_min/max) has seen a range of
    depths, so inference can spend *more* compute per token than training did by raising this —
    test-time compute scaling with no change to the weights. The KV cache is sized to match, since
    it needs one slot per (block, iteration) pair actually executed.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    next_input = input_ids[:, -model.cfg.max_seq_len :]
    kv_cache = model.new_kv_cache(loop_count=loops)

    for _ in range(max_new_tokens):
        assert kv_cache.seq_len + next_input.shape[1] <= model.cfg.max_seq_len, (
            f"generation would exceed model.cfg.max_seq_len ({model.cfg.max_seq_len}); "
            "reduce --max-new-tokens or use a checkpoint trained with a larger max_seq_len"
        )
        logits = model(next_input, kv_cache=kv_cache, loop_count=loops).logits[:, -1, :]

        # Mask the vocab-padding rows (see model.padded_vocab_size): no tokenizer id maps to them,
        # so sampling one would decode to nothing and corrupt the KV cache for every later step.
        # They're only ever trained down implicitly via the softmax denominator, never to -inf.
        if logits.size(-1) > len(tokenizer):
            logits[:, len(tokenizer) :] = float("-inf")

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        next_input = next_token  # every step after the first decodes a single new token

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

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
    args = parser.parse_args()
    device = resolve_device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, _, tokenizer = load_checkpoint(args.checkpoint, device)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        loops=args.loops,
    )
    print(text)


if __name__ == "__main__":
    main()
