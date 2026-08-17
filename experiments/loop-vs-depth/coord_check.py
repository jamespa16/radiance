"""muP coordinate check for radiance.

The claim muP makes is that activation "coordinate size" (mean |entry|) at every point in the
network is Theta(1) in width, at *every* training step -- not just at init. If that holds, the
optimal LR found at a small proxy width transfers to a large one. If it doesn't, it can't.

This trains the real DenseTransformer through the real build_optimizer at several widths, with
identical data/seeds, and records mean |activation| at instrumented points after each step.

Flat curves across width == muP is working. Curves that grow with width == it isn't.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from radiance.config import Config, DataConfig, ModelConfig, TrainConfig
from radiance.model import DenseTransformer
from radiance.optim import build_optimizer


VOCAB = 512
BASE_WIDTH = 128


def make_cfg(d_model: int, mup: bool, optimizer: str, lr: float, n_layers: int, loop_count: int) -> Config:
    model = ModelConfig(
        d_model=d_model,
        head_dim=32,          # fixed, so width grows by adding heads -- this repo's design
        n_layers=n_layers,
        loop_count=loop_count,
        ffn_mult=4.0,
        ffn_depth=1,          # plain SwiGLU MLP; ffn_depth>1 adds a d_ffn x d_ffn layer that
                              # would dominate the parameter count without changing the check
        dropout=0.0,
        max_seq_len=128,
        doc_attention_mask=False,   # eos_id is None below anyway; keeps every width on one path
        vocab_pad_multiple=1,
        mup_base_d_model=BASE_WIDTH if mup else None,
    )
    train = TrainConfig(
        lr=lr,
        optimizer=optimizer,
        auto_batch_size=False,
        compile=False,
        device="cuda",
        weight_decay=0.0,     # decay is a confound here; the check is about the update's scale
    )
    return Config(data=DataConfig(seq_len=64, num_workers=0), model=model, train=train)


def instrument(model: DenseTransformer) -> tuple[dict, list]:
    """Forward hooks recording mean |output| per instrumented point, averaged over blocks."""
    acts: dict[str, list[float]] = {}
    handles = []

    def record(tag):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            acts.setdefault(tag, []).append(t.detach().float().abs().mean().item())
        return hook

    handles.append(model.token_emb.register_forward_hook(record("embedding")))
    for block in model.blocks:
        handles.append(block.attn.out_proj.register_forward_hook(record("attn_out")))
        handles.append(block.ffn.register_forward_hook(record("ffn_out")))
    handles.append(model.ln_f.register_forward_hook(record("ln_f")))
    return acts, handles


def run(d_model: int, mup: bool, optimizer: str, lr: float, steps: int,
        n_layers: int, loop_count: int, batch: int, seq: int) -> list[dict]:
    cfg = make_cfg(d_model, mup, optimizer, lr, n_layers, loop_count)

    torch.manual_seed(0)
    model = DenseTransformer(cfg.model, vocab_size=VOCAB, eos_id=None).to("cuda")
    opt = build_optimizer(model, cfg, "cuda")
    acts, handles = instrument(model)

    # Identical data at every width.
    gen = torch.Generator(device="cpu").manual_seed(1234)
    batches = [torch.randint(0, VOCAB, (batch, seq), generator=gen).to("cuda") for _ in range(steps + 1)]

    rows = []
    for step in range(steps + 1):
        acts.clear()
        ids = batches[step]
        out = model(ids)
        logits = out.logits
        row = {k: sum(v) / len(v) for k, v in acts.items()}
        row["logits"] = logits.detach().float().abs().mean().item()
        row["step"] = step
        rows.append(row)

        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for h in handles:
        h.remove()
    del model, opt
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--optimizers", nargs="+", default=["muon", "adamw"])
    p.add_argument("--lr", type=float, default=1.0e-2)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--loop-count", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--out", default="coord_check.json")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 throughout: precision is a confound here
    torch.backends.cudnn.allow_tf32 = False

    results = []
    for optimizer in args.optimizers:
        for param in ("mup", "sp"):
            for width in args.widths:
                rows = run(width, param == "mup", optimizer, args.lr, args.steps,
                           args.n_layers, args.loop_count, args.batch, args.seq)
                for r in rows:
                    r.update(width=width, param=param, optimizer=optimizer)
                results.extend(rows)
                last = rows[-1]
                print(f"{optimizer:6s} {param:3s} d={width:5d}  "
                      + "  ".join(f"{k}={last[k]:8.4f}" for k in
                                  ("embedding", "attn_out", "ffn_out", "ln_f", "logits")),
                      flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"\nwrote {args.out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
