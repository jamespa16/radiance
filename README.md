# Radiance

An experimental LLM training framework: a from-scratch PyTorch pipeline that loads a HuggingFace dataset,
tokenizes it with an off-the-shelf HF tokenizer, and trains a configurable transformer, with W&B logging.

It exists to try architecture and training ideas quickly and measure whether they actually help. Everything is
driven by one YAML config, the code is deliberately flat and readable, and there is no `Trainer` abstraction to
fight. It is not a production framework.

## Quick start

No setup step — `uv run` creates and syncs the virtualenv on first use.

```bash
# train
WANDB_MODE=offline uv run radiance-train --config configs/tinystories.yaml

# sample from a checkpoint
uv run radiance-generate --checkpoint checkpoints/tinystories/step_1000.pt --prompt "Once upon a time"

# serve an OpenAI-compatible API
uv run radiance-serve --checkpoint checkpoints/tinystories/step_1000.pt --port 8000

# run standard benchmarks
uv run --group eval radiance-eval --checkpoint checkpoints/tinystories/step_1000.pt --tasks piqa,hellaswag

# tests
uv run --group dev pytest              # everything
uv run --group dev pytest -m "not slow"  # skip the compiled-GPU tests
```

`configs/tinystories.yaml` is the reference config, tuned for a quick first run on a single GPU. Copy it to start
a new one. `configs/v1.yaml` is the shipping model: a 375M dense transformer trained on fineweb for ~9B tokens.

Drop `WANDB_MODE=offline` to log to your W&B account (or set `wandb.mode` in the config).

## What's in it

- **Muon + AdamW** with muP scaling and per-group learning rates (embeddings, norms, gates, routers).
- **A shared-weight loop body.** `blocks[0]` runs once; `blocks[1:]` is re-run a fixed or learned number of times
  per forward. Loop conditioning, input injection, BPTT windowing and ACT all support this.
- **MoE** feed-forward, with shared experts, fine-grained experts and load balancing.
- **Attention variants**: GQA, differential attention, document masking under `flex_attention`, KV cache.
- **NVFP4** 4-bit GEMMs in Triton, behind `model.fp4_linear`.
- **Post-training**: SFT and DPO pipelines with chat formatting and packed pairs.
- **Multi-token prediction**, hyper-connections, ACT capacity sparsity, and a few other things that were tried
  and measured.

## What we've measured

The point of the repo is the measurements, and the negative ones are kept alongside the positive ones. A few
that shaped the defaults:

| result | effect |
|---|---|
| Muon over AdamW | reaches AdamW's final loss in about half the steps |
| Retuning `lr` after Muon took over the hidden weights | worth more than tripling the training budget |
| MoE at matched active parameters | +0.049 val/loss — the largest architecture effect recorded here |
| Differential attention | +0.008–0.012 from step 600 on, at 1.34x step time |
| The weight-shared loop vs. real depth | dominated — depth wins at equal FLOPs, and the gap grows |
| Hyper-connections | neutral-to-negative; the real finding was that they need their own LR |
| NSA | worse than the default it would replace; removed |
| Counterfactual MoE routing | exactly neutral; the signal doesn't reproduce across batches |

Full numbers, methodology and caveats in [docs/results.md](docs/results.md). Three cautions from that page are
worth repeating: a 400-step verdict can invert by step 3000; never compare runs with different `max_steps` at the
same step number; and sweep the LR per arm, because arms' LR curves cross.

## Conventions

1. **Features default on, made inert by construction** — zero-initialised, identity-valued, or range-collapsed, so
   existing configs train exactly as they did.
2. **A feature is kept because of a measurement, not because it's implemented.** `use_nsa` was removed and
   `hyper_conn_streams` stays at `1` on those terms.
3. **The compiled path is the real path.** Several of the worst bugs here were silent under `torch.compile` and
   invisible to eager tests. Run compiled before trusting a change to attention, the loop, or the loss.

## Layout

Everything lives under `src/radiance/`:

| | |
|---|---|
| `config.py` | the dataclass schema and `load_config` — single source of truth for every tunable |
| `data.py`, `sft_data.py`, `dpo_data.py` | tokenizing, packing, streaming, disk cache; SFT and DPO pipelines |
| `model/` | `DenseTransformer`, attention, masking, FFN/MoE, ACT, MTP, norms, hyper-connections |
| `nvfp4/` | NVFP4 4-bit GEMM primitives in Triton |
| `optim.py` | Muon, AdamW, muP, parameter groups |
| `train.py` | the training loop, plus `losses.py`, `batching.py`, `checkpointing.py`, `evaluation.py` |
| `generate.py`, `serve.py`, `eval_harness.py` | sampling, the API server, lm-evaluation-harness backend |

## Docs

[docs/](docs/) has a page per area — config, data, model, nvfp4, optim, train, post-training, eval, results, and
[extending](docs/extending.md) for adding a dataset, model variant, optimizer or numeric format. Most of what's
written there is a measurement or a bug that cost a training run, so read the relevant page before changing that
area. [CLAUDE.md](CLAUDE.md) is the index.

## License

See [LICENSE](LICENSE).
