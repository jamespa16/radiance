"""Isolate the training-loop memory growth.

Observed: a 21-executed-block run climbs ~0.9 GB/min monotonically until it exhausts a 32GB card
around step ~1600, with expandable_segments on and no fragmentation warnings. Arm A survives only
because it has headroom, so this silently caps how long any run here can go.

This drives the model directly -- no train.py, no DataLoader -- so a leak that shows up here is in
model.py, and one that doesn't is in the training loop's plumbing. Leading suspect is
doc_attention_mask: _doc_masks calls create_block_mask on every forward (it must run eagerly, it's
@torch._dynamo.disable'd), and every batch has different document boundaries, so every step builds
a fresh BlockMask.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from radiance.config import Config, DataConfig, ModelConfig, TrainConfig
from radiance.model import DenseTransformer, padded_vocab_size
from radiance.optim import build_optimizer

EOS = 50256


def run(doc_mask: bool, compile_model: bool, steps: int, batch: int, seq: int) -> list[tuple[int, float]]:
    vocab = padded_vocab_size(50257, 128)
    model_cfg = ModelConfig(
        d_model=256, head_dim=64, n_layers=6, loop_count=4, ffn_mult=4.0, ffn_depth=3,
        dropout=0.0, max_seq_len=seq, doc_attention_mask=doc_mask,
    )
    cfg = Config(
        data=DataConfig(seq_len=seq),
        model=model_cfg,
        train=TrainConfig(lr=1e-2, optimizer="muon", device="cuda", auto_batch_size=False),
    )

    torch.manual_seed(0)
    model = DenseTransformer(model_cfg, vocab_size=vocab, eos_id=EOS).to("cuda")
    if compile_model:
        model = torch.compile(model)
    opt = build_optimizer(model, cfg, "cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    trace = []

    for step in range(steps):
        ids = torch.randint(0, vocab, (batch, seq), generator=gen)
        # Scatter EOS so document boundaries differ every step -- that is what makes each forward
        # build a *different* BlockMask, which is the condition the leak hypothesis needs.
        n_eos = int(torch.randint(2, 8, (1,), generator=gen).item())
        for _ in range(n_eos):
            r = int(torch.randint(0, batch, (1,), generator=gen).item())
            c = int(torch.randint(0, seq, (1,), generator=gen).item())
            ids[r, c] = EOS
        ids = ids.to("cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=False)
        loss.backward()
        opt.step()

        if step % 25 == 0:
            trace.append((step, torch.cuda.memory_reserved() / 1e9))

    del model, opt
    torch.cuda.empty_cache()
    return trace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    for doc_mask in (True, False):
        trace = run(doc_mask, args.compile, args.steps, args.batch, args.seq)
        first, last = trace[1][1], trace[-1][1]   # skip step 0 (pre-warmup)
        tag = f"doc_attention_mask={str(doc_mask):5s} compile={args.compile}"
        print(f"\n{tag}")
        print("  " + "  ".join(f"s{s}:{m:.2f}" for s, m in trace))
        print(f"  reserved {first:.2f} -> {last:.2f} GB   growth {last - first:+.2f} GB "
              f"over {args.steps} steps")


if __name__ == "__main__":
    main()
