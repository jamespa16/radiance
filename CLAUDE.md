# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It is the index. Detail lives under `docs/` — **read the relevant page before changing that area**, because most of
what is written down there is a measurement or a bug that cost a training run, not a description of the code.

| page | covers |
|---|---|
| [docs/config.md](docs/config.md) | `config.py`, and **the defaults convention** — read before adding any flag |
| [docs/data.md](docs/data.md) | `data.py` + `sft_data.py`/`dpo_data.py` — loading, tokenizing, packing, streaming, disk cache, SFT/DPO pipelines |
| [docs/model.md](docs/model.md) | `model/` — the loop, attention variants, ACT, MoE, hyper-connections |
| [docs/nvfp4.md](docs/nvfp4.md) | `nvfp4/` — NVFP4 4-bit GEMMs behind `model.fp4_linear` |
| [docs/optim.md](docs/optim.md) | `optim.py` — Muon, muP, parameter groups, the `lr` retune |
| [docs/train.md](docs/train.md) | `train.py` (+ `losses.py`, `batching.py`, `checkpointing.py`, `evaluation.py`) + `generate.py` — loop, loss, compile modes, batch sizing, OOM tiers |
| [docs/post-training.md](docs/post-training.md) | SFT and DPO |
| [docs/eval.md](docs/eval.md) | `eval_harness.py` — standard benchmarks (HellaSwag, PIQA, ARC, ...) via lm-evaluation-harness |
| [docs/results.md](docs/results.md) | **every A/B ever run here**, and the cautions for running a new one |
| [docs/extending.md](docs/extending.md) | how to add a dataset / model variant / optimizer / numeric format |

Comments in `configs/`, `tests/` and `src/` that refer to "CLAUDE.md's `optim.py` section", "Measured results" or
"Extending" predate this split — the table above maps each of those to its file.

## What this is

Radiance is an experimental LLM training framework: a minimal, from-scratch PyTorch training pipeline that loads a
HuggingFace `user/dataset`, tokenizes it with an off-the-shelf HF tokenizer, and trains a configurable transformer on
it, with W&B logging. A hackable base for trying non-standard architectures/training ideas, not a production
framework — **prefer explicit, readable code over abstraction layers.**

## Setup

No manual setup step — `uv run` creates/syncs `.venv` from `pyproject.toml`/`uv.lock` automatically on first use.

## Git

**Create a branch and open a pull request; do not commit straight to `main`.** Commit and push only when asked, as
usual — the instruction here is about *where*, not *when*.

**Never include a Claude Code session link (`claude.ai/code/session_...`) in a PR description or commit message.**
Drop that line/footer entirely rather than including it — the rest of the standard attribution (e.g.
`Co-Authored-By: Claude ...`) is fine.

## Running training

```bash
WANDB_MODE=offline uv run radiance-train --config configs/tinystories.yaml
```

Drop `WANDB_MODE=offline` to log to your W&B account (`wandb.mode` in the config also controls this — `online`,
`offline`, or `disabled`). `configs/tinystories.yaml` is the reference config, tuned for a quick first run against
`roneneldan/TinyStories`. Copy it to start a new config.

**Real training runs should use the GPU** (`train.device: auto`, the default, resolves to `cuda` when one's
available — see `resolve_device` in `config.py`). Don't drop to `train.device: cpu` for an actual run just because the GPU is temporarily busy with another
process; wait for it to free up or ask. CPU is fine only for the tiny pipeline sanity-checks below, which are
explicitly cheap/throwaway, not for anything whose numbers you intend to keep.

## Running inference

```bash
uv run radiance-generate --checkpoint checkpoints/tinystories/step_1000.pt --prompt "Once upon a time"
```

Loads the `Config` embedded in the checkpoint, rebuilds the model and tokenizer, and autoregressively samples
(`--temperature`, `--top-k`; `--temperature 0` for greedy; `--loops N` to override loop depth; `--chat` for an
SFT/DPO checkpoint). See [docs/train.md](docs/train.md) for the KV cache and the loop-depth override.

Entry points: `radiance.train:main` (`--config`) and `radiance.generate:main` — `radiance-train` /
`radiance-generate` console scripts after install.

## Running the OpenAI-compatible server

```bash
uv run radiance-serve --checkpoint checkpoints/tinystories/step_1000.pt --port 8000
```

Loads one checkpoint (same loader as `radiance-generate`) and serves `/v1/completions` (streaming and
non-streaming, works with any checkpoint) plus `/v1/chat/completions` (formats requests through the checkpoint's own
chat turn template, so it 400s unless the checkpoint has `sft.enabled: true` or `dpo.enabled: true`) and
`/v1/models`, on `127.0.0.1` by default. `/healthz`, `/readyz`, and `/metrics` (uptime, request/error counts,
tokens/sec) are unauthenticated and unrate-limited. Concurrent requests are batched: a background dispatcher groups
requests that arrive within `--batch-wait-ms` (default 20ms) of each other, up to `--max-batch-size` (default 8),
into one forward pass through the shared-weight loop body — right-padded prompts of different lengths share a
`KVCache`, and only requests with the same `--loops` override can share a batch, since loop depth is one value per
forward call. Batches run one at a time (the model is a single shared instance, not concurrent-safe), but the
dispatcher forms the *next* batch while the current one is generating, so a request's `--batch-wait-ms` window starts
on its arrival rather than after the in-flight batch finishes, and `--batch-wait-ms 0` still batches whatever is
already queued with no added wait. `radiance.generate.generate_tokens_batched` is the batched sampling-loop generator
the server streams tokens from (`radiance.generate.generate_tokens`, single-sequence, is still what
`radiance-generate` uses).

`--api-key` (repeatable, or the comma-separated `RADIANCE_API_KEY` env var) requires a matching `Authorization:
Bearer <key>` on every `/v1/*` route; omitted, those routes are unauthenticated (the default, so the quickstart
above needs no flags). `--rate-limit` caps requests/minute per key (or per client IP when no key is configured); `0`
(default) disables it. Entry point: `radiance.serve:main` — `radiance-serve` console script after install.

## Running standard benchmarks

```bash
uv run --group eval radiance-eval --checkpoint checkpoints/tinystories/step_1000.pt --tasks piqa,hellaswag,arc_easy
```

Runs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks against a checkpoint
in-process (batched forward passes against the loaded model, not through `radiance-serve` — see
[docs/eval.md](docs/eval.md) for why). `lm-eval` is a separate `eval` dependency group (`uv run --group eval ...`),
not a core dependency. Entry point: `radiance.eval_harness:main` — `radiance-eval` console script after install.

## Running tests

```bash
uv run --group dev pytest                  # everything
uv run --group dev pytest -m "not slow"    # skip the compiled-GPU regression tests
```

The suite is built around **equivalence invariants** rather than golden files, because most features here are
supposed to be mathematically inert until configured: a feature that defaults on must produce bit-identical logits to
the same model with it off (`tests/test_inert_defaults.py`, `tests/test_loop_identity.py`), a cached decode must match
a full forward across every loop mode (`tests/test_kv_cache.py`), and so on. Those tests are self-checking — there is
nothing to regenerate when the model changes.

Four are worth knowing about before touching the areas they cover:

- `test_no_dead_parameters` asserts every allocated parameter is reachable from the forward pass. It caught two
  parameters that were allocated but never differentiated (and so silently never trained).
- `tests/test_act_kv_invariance.py` pins down exactly what is and isn't invariant for a halted ACT position. The
  tempting-but-wrong version of that argument is very easy to re-derive; read the file's docstring before attempting
  to make ACT skip halted tokens.
- `tests/test_optim.py::test_mup_keeps_adamw_activations_flat` is a coordinate check whose 16x width span and 20
  steps are load-bearing — a smaller version isn't a weaker test but a vacuous one.
- `tests/test_nvfp4.py` compares integer nibble codes at `rtol=0, atol=0`. Do not relax it to "close enough"; that
  bar is what caught four separate 1-ULP disagreements.

`tests/test_compile.py` (marked `slow`, CUDA-only) covers failures that eager tests structurally cannot see, each of
which cost a training run to find: document masking under `torch.compile`, stochastic loop depth under
`mode="reduce-overhead"`, and differential attention's two `flex_attention` calls silently diverging from eager.
**If you change anything touching attention masking, the loop count, or compile settings, run the suite on a GPU
without `-m "not slow"`.** The eval path deserves particular attention: `grad_checkpoint` is disabled under `eval()`,
so evaluation traces a *different* graph than training — the doc-masking bug compiled fine for training and blew up
at the first `evaluate()` call. The same file also carries four *fast* tests pinning `resolve_compile_mode`'s
decision, because the third compile failure is silent; see [docs/train.md](docs/train.md).

Still run a tiny config end-to-end (small `seq_len`, `d_model`, `max_steps`) through `radiance.train` on CPU before
trusting a full run — the suite covers the model, not the data pipeline or the training loop's plumbing.

## Architecture map

Everything lives under `src/radiance/`, driven entirely by a single YAML config (`radiance.config.Config`, loaded via
`load_config`). The four pipeline stages (data, model, optim, train) plus the `nvfp4/` numerics kernel package that
only the model package imports. `model/` and `nvfp4/` are packages whose `__init__` re-exports the public API, so
`from radiance.model import DenseTransformer` and `nvfp4.<name>` access work exactly as before.

- **`config.py`** — the dataclass schema and `load_config`. Single source of truth for every tunable; a new
  hyperparameter is added here first, then threaded through. Plain dataclasses, not OmegaConf/Hydra.
- **`data.py`** — tokenizer, dataloaders, causal-LM packing, streaming, disk cache. The SFT pipeline (chat
  formatting/packing, `format_chat_prompt`) is in **`sft_data.py`**; the DPO pipeline (pair packing, reference
  logprobs) is in **`dpo_data.py`**.
- **`model/`** — `DenseTransformer` (`transformer.py`): token + learned positional embeddings, a stack of pre-norm
  `TransformerBlock`s (`block.py`), final LayerNorm, weight-tied LM head. `blocks[0]` runs once; `blocks[1:]` is a
  **shared-weight loop body** re-run a fixed or learned number of times per forward, which is the central
  architectural idea here and what most features exist to support. Shared dataclasses (`ModelOutput`, `LoopContext`)
  in `core.py`, attention + KV cache in `attention.py`, document masking in `masking.py`, dense/MoE feed-forward in
  `ffn.py`, ACT in `act.py`, MTP in `mtp.py`, norms in `norms.py`, hyper-connections in `hyper_connections.py`,
  checkpoint reconstruction in `load.py`, DPO's `sequence_logprob_sum` in `dpo.py`.
- **`nvfp4/`** — NVFP4 4-bit GEMM primitives. Separate from the model package because it is ~600 lines of Triton with
  nothing to do with architecture: `quantize.py` (format constants, pure-torch reference, Triton kernels),
  `linear.py` (`FP4Linear` and the per-step weight refresh).
- **`optim.py`** — Muon + auxiliary AdamW, muP, and parameter-group construction.
- **`train.py`** — plain PyTorch training loop (no HF `Trainer`), plus SFT/DPO mode switches; the losses in
  `losses.py`, batch sizing in `batching.py`, checkpoint save/load in `checkpointing.py`, evaluation in
  `evaluation.py`.
- **`generate.py`** — checkpoint loading and KV-cached autoregressive sampling.
- **`eval_harness.py`** — an lm-evaluation-harness `LM` backend (`RadianceLM`) plus the `radiance-eval` CLI; runs
  standard benchmarks in-process against a loaded checkpoint. `eval` dependency group, not core.

Three cross-cutting conventions that decide most design questions here:

1. **Features default *on*, made inert by construction** — zero-initialised, identity-valued, or range-collapsed, so
   every existing config keeps training exactly as it did. There are six specific reasons a feature stays off
   instead; see [docs/config.md](docs/config.md) before adding a flag.
2. **A feature is kept or defaulted on because of a measurement, not because it is implemented.** `use_nsa` was
   removed and `hyper_conn_streams` stays at `1` on exactly these terms. [docs/results.md](docs/results.md) records
   every A/B, negative ones included, with the reasoning that made them decisive.
3. **The compiled path is the real path.** Several of the worst bugs here were silent under `torch.compile` and
   invisible in eager tests. Run compiled before trusting a change to attention, the loop, or the loss.
