from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from radiance.config import Config
from radiance.data import build_tokenizer
from radiance.model import DenseTransformer


def load_checkpoint(path: str, device: str) -> tuple[DenseTransformer, Config, PreTrainedTokenizerBase]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]

    tokenizer = build_tokenizer(cfg)
    model = DenseTransformer(cfg.model, vocab_size=len(tokenizer))
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
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.cfg.max_seq_len :]
        logits = model(context)[:, -1, :]

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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, _, tokenizer = load_checkpoint(args.checkpoint, args.device)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
