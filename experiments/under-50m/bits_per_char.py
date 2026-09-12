"""Score a checkpoint in bits per character, so models with different tokenizers can be compared.

`val/loss` is mean cross-entropy per *token*, which is only a model-quality measure when the
tokenizer is held fixed. It is not, across the vocab study in this directory: a smaller vocabulary
splits the same story into more, individually more predictable tokens, so it lowers per-token loss
for a reason that has nothing to do with the model being better. Taken at face value it would rank
a character-level model above every BPE model on the page.

Bits per character removes the units problem by dividing the *total* negative log-likelihood of a
fixed piece of text by the number of characters in that text rather than the number of tokens:

    bits_per_char = sum(NLL over all predicted tokens) / (ln 2 * total characters)

Both models must predict the same text, so both are answering the same question and the token
count cancels out. This is the standard way the compression literature compares segmentations, and
it is the only number in this directory that is comparable across tokenizers.

Two details that would otherwise bias the comparison:

- **Every token is scored exactly once.** Blocks are cut with stride `seq_len - 1`, not `seq_len`,
  so each block's first position is the previous block's last token re-fed purely as context and
  the scored targets tile the token stream without gap or overlap. Cutting with stride `seq_len`
  instead would silently drop one token per block, and a smaller vocabulary produces more blocks
  over the same text — so the arm being tested would be handed a discount the baseline doesn't get,
  on exactly the high-surprise tokens that sit at a block boundary.
- **The denominator is the text actually scored.** The trailing partial window can't be scored, so
  the character count is taken by decoding the scored token span back to text rather than from the
  input stories. Counting all the stories' characters against a slightly shorter scored span would
  understate bits/char by a different amount in each arm.

Usage:
    uv run python experiments/under-50m/bits_per_char.py --checkpoint path/to/step_6000.pt --stories 5000
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset

from radiance.data import build_tokenizer
from radiance.model import load_transformer_from_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stories", type=int, default=5000, help="validation stories to score")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # eos_id must be passed, and generate.load_checkpoint does not pass it: it defaults to None,
    # which switches document masking *off* (correct for generation, where the prompt is one
    # document). These models were trained with `doc_attention_mask: true`, so scoring them
    # without it is a train/test mismatch -- and not a uniform one. Measured on the ts16k arms, it
    # reordered them against their own val/loss: the looped arm attends across story boundaries it
    # was never trained to see and is penalised more than the plain arm. Reproducing training's
    # attention regime is what makes these numbers rankable.
    tmp_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    tokenizer = build_tokenizer(tmp_cfg)
    model, cfg = load_transformer_from_checkpoint(
        args.checkpoint, args.device, eos_id=tokenizer.eos_token_id
    )
    model.eval()
    seq_len = cfg.data.seq_len

    ds = load_dataset("roneneldan/TinyStories", split="validation")
    texts = ds[: args.stories]["text"]

    ids: list[int] = []
    for chunk in range(0, len(texts), 1000):
        for enc in tokenizer(texts[chunk : chunk + 1000])["input_ids"]:
            ids.extend(enc)
            ids.append(tokenizer.eos_token_id)

    # Stride seq_len - 1: position 0 of each window is context, positions 1.. are targets, so the
    # windows' target spans tile ids[1:] exactly once.
    stride = seq_len - 1
    n_blocks = (len(ids) - 1) // stride
    if n_blocks == 0:
        raise SystemExit(f"only {len(ids)} tokens for seq_len {seq_len}; raise --stories")
    blocks = torch.stack(
        [torch.tensor(ids[i * stride : i * stride + seq_len], dtype=torch.long) for i in range(n_blocks)]
    )

    # The characters actually scored: decode the target span back to text. EOS ids are dropped
    # first -- the model is scored on predicting them (they are real targets) but they are
    # tokenizer bookkeeping and contribute no characters to compress.
    scored_ids = ids[1 : n_blocks * stride + 1]
    total_chars = len(tokenizer.decode([i for i in scored_ids if i != tokenizer.eos_token_id]))

    total_nll = 0.0
    scored = 0
    with torch.no_grad():
        for i in range(0, n_blocks, args.batch_size):
            batch = blocks[i : i + args.batch_size].to(args.device)
            logits = model(batch).logits.float()
            # Position t predicts t+1, so position 0 is context-only -- the overlap the stride
            # above leaves in on purpose, not a dropped token.
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            scored += batch.numel() - batch.size(0)

    bpc = total_nll / (math.log(2) * total_chars)
    print(f"checkpoint      {args.checkpoint}")
    print(f"tokenizer       {cfg.data.tokenizer} (vocab {len(tokenizer):,})")
    print(f"stories         {len(texts):,}   scored chars {total_chars:,}   tokens scored {scored:,}")
    print(f"tokens/char     {scored / total_chars:.4f}")
    print(f"nll/token       {total_nll / scored:.4f}   (NOT comparable across tokenizers)")
    print(f"bits/char       {bpc:.4f}   <- the comparable number")


if __name__ == "__main__":
    main()
