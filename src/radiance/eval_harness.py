"""lm-evaluation-harness backend for a radiance checkpoint, plus a `radiance-eval` CLI.

Runs entirely in-process against the loaded `DenseTransformer` — it does **not** go through
`radiance.serve`. The server serializes every generation behind a single `asyncio.Lock` (see
`docs/train.md`'s "one request at a time" note on `create_app`); lm-eval's loglikelihood tasks
issue one request per answer choice (5-shot MMLU is ~4 requests x ~14k questions), so a serialized
HTTP round trip per request would take hours where a batched in-process forward takes minutes.
`/v1/completions` also has no `logprobs`/`echo` support, which the OpenAI-compatible
`local-completions` model type would need for the same requests — adding that is a separate,
larger feature with its own OpenAI-compat surface (`text_offset`, token alignment) and wouldn't be
any faster, since the server still serializes requests. See docs/eval.md.

Batches loglikelihood requests with right-padding and no attention mask: a decoder-only causal
model's logits at position i only depend on positions <=i, so right-padded tokens (which live past
every real position) can never influence a real position's logits — they only waste some
compute, exactly like lm-eval's own `HFLM._loglikelihood_tokens` (`pad_and_concat(...,
padding_side="right")`, no `attention_mask` passed for the causal backend).
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils as lm_utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs

from radiance.config import resolve_device
from radiance.generate import generate_tokens, load_checkpoint


def _first_stop_index(text: str, stop_seqs: list[str]) -> int | None:
    indices = [text.index(s) for s in stop_seqs if s and s in text]
    return min(indices) if indices else None


@register_model("radiance")
class RadianceLM(TemplateLM):
    """Wraps a single radiance checkpoint (loaded the same way as `radiance-generate` /
    `radiance-serve`, via `load_transformer_from_checkpoint`) as an lm-eval `TemplateLM`.

    One process, one checkpoint, no data parallelism — matching every other entry point in this
    repo. `batch_size` only affects loglikelihood-request batching; `generate_until` runs one
    request at a time since `generate_tokens` is a single-sequence KV-cached loop.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        batch_size: int = 16,
        loops: int | None = None,
    ) -> None:
        super().__init__()
        self._device = resolve_device(device)
        self.model, self.cfg, self.tokenizer = load_checkpoint(checkpoint, self._device)
        self.model.eval()
        self.batch_size = int(batch_size)
        self.loops = int(loops) if loops is not None else None
        self.max_length = self.model.cfg.max_seq_len

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        return self.tokenizer(string, add_special_tokens=False)["input_ids"]

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def _model_logprobs(self, batched_inps: torch.Tensor) -> torch.Tensor:
        logits = self.model(batched_inps, loop_count=self.loops).logits.float()
        # Mask the vocab-padding rows (see model.padded_vocab_size, and the identical mask in
        # generate.generate_tokens): no tokenizer id maps to them, they're never trained toward
        # -inf (only implicitly, via the softmax denominator), and an untrained row winning the
        # argmax would silently corrupt both is_greedy and every logprob in this batch.
        if logits.size(-1) > len(self.tokenizer):
            logits[..., len(self.tokenizer) :] = float("-inf")
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        res: list[tuple[float, bool] | None] = [None] * len(requests)
        # Longest-first so any OOM surfaces on the first batch, not partway through a long run.
        order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="loglikelihood")

        for start in range(0, len(order), self.batch_size):
            idxs = order[start : start + self.batch_size]
            inps: list[list[int]] = []
            inplens: list[int] = []
            cont_id_lists: list[list[int]] = []

            for i in idxs:
                _, context_enc, continuation_enc = requests[i]
                assert len(continuation_enc) > 0, "continuation must be non-empty"
                # Truncate from the left when context+continuation overflows max_length, keeping
                # the continuation intact (mirrors lm_eval's own HFLM behavior).
                total = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                inp = total[:-1]
                contlen = min(len(continuation_enc), len(inp))
                inps.append(inp)
                inplens.append(len(inp))
                cont_id_lists.append(continuation_enc[-contlen:])

            padded_len = max(inplens)
            batched = torch.full(
                (len(idxs), padded_len), self.eot_token_id, dtype=torch.long, device=self._device
            )
            for row, inp in enumerate(inps):
                batched[row, : len(inp)] = torch.tensor(inp, dtype=torch.long, device=self._device)

            logprobs = self._model_logprobs(batched)

            for row, (inplen, cont_ids) in enumerate(zip(inplens, cont_id_lists)):
                contlen = len(cont_ids)
                seq_logprobs = logprobs[row, inplen - contlen : inplen]
                cont_ids_t = torch.tensor(cont_ids, dtype=torch.long, device=self._device)
                is_greedy = bool((seq_logprobs.argmax(dim=-1) == cont_ids_t).all())
                token_logprobs = seq_logprobs.gather(1, cont_ids_t.unsqueeze(-1)).squeeze(-1)
                res[idxs[row]] = (float(token_logprobs.sum()), is_greedy)
                pbar.update(1)

        pbar.close()
        assert all(r is not None for r in res)
        return res  # type: ignore[return-value]

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        all_windows: list[tuple[None, list[int], list[int]]] = []
        window_counts: list[int] = []
        for (string,) in [req.args for req in requests]:
            windows = [
                lm_utils.make_disjoint_window(w)
                for w in lm_utils.get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )
            ]
            all_windows.extend((None, ctx, cont) for ctx, cont in windows)
            window_counts.append(len(windows))

        nlls = self._loglikelihood_tokens(all_windows, disable_tqdm=disable_tqdm)

        out = []
        idx = 0
        for n in window_counts:
            out.append(sum(nll for nll, _ in nlls[idx : idx + n]))
            idx += n
        return out

    @torch.no_grad()
    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        res = []
        eos_text = self.tok_decode([self.eot_token_id], skip_special_tokens=False)

        for req in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context, gen_kwargs = req.args
            kwargs = normalize_gen_kwargs(gen_kwargs or {}, default_max_gen_toks=256)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos_text)
            max_gen_toks = kwargs.pop("max_gen_toks")

            # Clamp max_gen_toks (a task-config default, e.g. GSM8K's 256) rather than asserting,
            # so a long few-shot context on a small max_seq_len checkpoint degrades to a truncated
            # generation instead of crashing generate_tokens' max_seq_len assertion.
            max_gen_toks = min(max_gen_toks, self.max_length - 1)
            max_ctx_len = self.max_length - max_gen_toks

            input_ids = self.tokenizer(context, return_tensors="pt")["input_ids"].to(self._device)
            input_ids = input_ids[:, -max_ctx_len:]

            generated_ids: list[int] = []
            text = ""
            for token_id in generate_tokens(
                self.model,
                self.tokenizer,
                input_ids,
                max_new_tokens=max_gen_toks,
                temperature=0.0,
                top_k=0,
                device=self._device,
                loops=self.loops,
            ):
                generated_ids.append(token_id)
                text = self.tok_decode(generated_ids)
                stop_at = _first_stop_index(text, until)
                if stop_at is not None:
                    text = text[:stop_at]
                    break
            res.append(text)

        return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a radiance checkpoint with lm-evaluation-harness.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from radiance.train")
    parser.add_argument(
        "--tasks", type=str, required=True, help="Comma-separated lm-eval task names, e.g. hellaswag,piqa,arc_easy"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16, help="Loglikelihood-request batch size")
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Override loop-body iterations per forward pass, matching --loops on radiance-generate "
        "and radiance-serve. Sweeping this is the point of benchmarking a looped architecture.",
    )
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None, help="Cap examples per task (debugging)")
    parser.add_argument("--output", type=str, default=None, help="Write full JSON results to this path")
    args = parser.parse_args()

    # lm_eval itself lazy-loads .evaluator to keep CLI startup fast (see lm_eval/__init__.py); match
    # that here rather than importing it at module level.
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.utils import make_table

    lm = RadianceLM(
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        loops=args.loops,
    )
    results = simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    print(make_table(results))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
