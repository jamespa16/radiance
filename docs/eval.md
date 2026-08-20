# `eval_harness.py` — standard benchmarks via lm-evaluation-harness

```bash
uv run --group eval radiance-eval --checkpoint checkpoints/tinystories/step_1000.pt --tasks piqa,hellaswag,arc_easy
```

`lm-eval` (EleutherAI's `lm-evaluation-harness`) is a separate `eval` dependency group, not a core
dependency — most entry points in this repo never need it. `radiance-eval` loads one checkpoint
(same loader as `radiance-generate`/`radiance-serve`), wraps it as an lm-eval `LM`, and calls
`lm_eval.evaluator.simple_evaluate`. `--tasks` is a comma-separated list of any
[lm-eval task name](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks);
`--num-fewshot` overrides a task's default shot count, `--limit N` caps examples per task (use this
while iterating — a full MMLU run is ~14k questions), and `--output path.json` writes the full
per-task result dict alongside the printed table. `--loops` overrides loop-body iterations per
forward, exactly like `radiance-generate --loops` and `radiance-serve`'s `loops` request field —
sweeping it is the way to benchmark test-time compute scaling on a model trained with stochastic
loop depth, and is why this exists as a flag here rather than something you'd have to bake into a
checkpoint.

## Why in-process, not through `radiance-serve`

The obvious alternative — point lm-eval's built-in `local-completions` model at a running
`radiance-serve` — doesn't work well here for two independent reasons:

1. Even with `radiance-serve`'s request-batching dispatcher (see the server section of the
   top-level `CLAUDE.md`), it only groups requests that happen to *arrive* within a short window
   of each other — it has no notion of a caller-controlled batch, and HTTP round-trip overhead
   per request still applies. lm-eval's loglikelihood tasks issue one HTTP request *per answer
   choice*: 5-shot MMLU is ~14k questions x 4 choices, HellaSwag is ~40k. That's a lot of HTTP
   round-trips for numbers you want to regenerate on every A/B, next to an in-process call that
   controls its own batch directly.
2. `/v1/completions` doesn't return per-token `logprobs`/`echo`, which `local-completions` needs
   for exactly those loglikelihood requests. Adding that is a real, separate feature (OpenAI's
   `text_offset`/token-alignment contract) that wouldn't fix (1) anyway.

`RadianceLM` in `eval_harness.py` is instead a `lm_eval.api.model.TemplateLM` subclass that runs
forward passes directly against the loaded `DenseTransformer`, batching loglikelihood requests
(default `--batch-size 16`) with right-padding and no attention mask — a decoder-only causal
model's logits at position *i* only depend on positions `<= i`, so padding tokens (which live past
every real position) can never influence a real position's logits; they only waste some compute.
This is the same trick `lm_eval`'s own `HFLM._loglikelihood_tokens` uses for causal models. See
`tests/test_eval_harness.py` for the equivalence check (batched must match one-request-at-a-time to
tight tolerance). `generate_until` (GSM8K, BBH, ...) reuses `generate.generate_tokens` one request
at a time rather than `generate.generate_tokens_batched` (the KV-cached, right-padded batched
generator `radiance-serve`'s dispatcher uses) — a scope decision, not a limitation of that
generator, left for whenever `generate_until` throughput on those tasks is actually the bottleneck.

## Which tasks are worth running

Wiring a task doesn't mean its score means anything at this repo's scale. Most of the commonly-cited
suite — MMLU/MMLU-Pro, MATH, HumanEval, MBPP, IFEval, BBH — needs either instruction-tuned models or
enough scale to be above chance; at the sizes trained here (`configs/tinystories.yaml`,
`configs/fineweb_500m.yaml`) they score at or near chance/zero and don't discriminate between
configs. The subset that actually moves and is worth A/Bing is short-context, 0-shot loglikelihood:
**HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, BoolQ, OpenBookQA, SIQA, CommonsenseQA**
(lm-eval task names: `hellaswag`, `piqa`, `winogrande`, `arc_easy`, `arc_challenge`, `boolq`,
`openbookqa`, `social_iqa`, `commonsense_qa`). GSM8K/BBH are wired and work (`generate_until` is
exercised end-to-end either way) but expect near-zero `exact_match` until a model is meaningfully
larger — treat a nonzero score there as a real signal, not a zero score as a bug.

**Check `model.max_seq_len` before picking tasks.** `_loglikelihood_tokens` truncates from the left
when `context + continuation` overflows it (same as lm-eval's own `HFLM`), so a 5-shot MMLU prompt
(1-2k tokens) against `configs/tinystories.yaml`'s `max_seq_len: 512` silently scores a mutilated
prompt rather than erroring. The 0-shot tasks above have short enough contexts to fit comfortably at
512-1024.

## Alternatives considered

- **bigcode-evaluation-harness** is the better-fit tool for HumanEval/MBPP (`lm-eval`'s code tasks
  need `HF_ALLOW_CODE_EVAL=1` and still execute untrusted model-generated code locally) — out of
  scope here; add it separately if code-gen quality becomes a live question.
- **MT-Bench / AlpacaEval** need an LLM judge (FastChat / the `alpaca_eval` package) and are
  meaningless below a model quality bar this repo hasn't hit yet.
- **HELM / OpenCompass** are much heavier than this repo's "explicit over abstraction" bar (see the
  top-level `CLAUDE.md`) for what `lm-evaluation-harness` already covers.
