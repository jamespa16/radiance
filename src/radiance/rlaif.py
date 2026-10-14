"""radiance-rlaif: on-policy preference data from AI feedback, and judge-scored evaluation.

RLAIF here is the iterated-DPO form of RLHF (as in Self-Rewarding LMs, Yuan et al. 2024, and
"online"/iterative DPO, Xu et al. 2023 and Dong et al. 2024) rather than a PPO loop:

    1. `radiance-rlaif pairs`: sample K completions per prompt from the current policy, score each
       with a public reward model (the "AI feedback"), keep best-vs-worst as a preference pair,
       and write those pairs to disk as an HF dataset.
    2. `radiance-train` a DPO stage on that dataset (dpo.dataset: <the output directory>), with the
       policy as both init_from and reference.
    3. Repeat from 1 with the new checkpoint.

Each round's pairs are drawn from the policy being trained, which is what separates this from
offline DPO on someone else's completions (ultrafeedback, hh-rlhf): the preference signal lands
on the policy's own distribution, where offline pairs increasingly do not as training moves it.
It is chosen over PPO/GRPO for the reason docs/post-training.md gives for DPO itself: one resident
model at a time, and the train() loop unchanged. Sampling, judging and training run strictly in
sequence, each freeing its model before the next loads, so the peak is one model, not three.

`radiance-rlaif eval` is the same two halves without the pairing: sample one completion per prompt
from each checkpoint, score with the judge, and report mean reward plus each checkpoint's win rate
against the first on identical prompts — the measurement val/loss and the preference accuracies
cannot make, since they only score pairs someone else wrote.

The default judge, Skywork/Skywork-Reward-V2-Qwen3-0.6B, is a 0.6B Bradley-Terry reward model —
small enough to run beside a training job on one GPU. Any `AutoModelForSequenceClassification`
reward model with one output works (`--judge`); one with a chat template is fed the conversation
through it, one without gets a (prompt, response) text pair (e.g.
OpenAssistant/reward-model-deberta-v3-large-v2).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from radiance.config import Config, resolve_device, set_seed
from radiance.evaluation import standard_error
from radiance.generate import BatchItem, generate_tokens_batched, load_checkpoint
from radiance.model import DenseTransformer
from radiance.sft_data import (
    RLAIF_MANIFEST_NAME as MANIFEST_NAME,
    RLAIF_PROMPT_IDS_VERSION,
    load_chat_dataset,
    tokenize_chat_prompt,
)

DEFAULT_JUDGE = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
# Rows read per slice while collecting prompts: large enough that slicing overhead is noise, small
# enough that a capped run reads a few thousand rows of a million-row dataset, not all of them.
_PROMPT_READ_CHUNK = 1024


@dataclass
class Completion:
    text: str
    # False when generation hit max_new_tokens instead of sampling EOS. An unfinished completion
    # can be a pair's *rejected* side but never its chosen one: DPO's tokenizer appends EOS to
    # both, so a truncated "chosen" would teach the policy to stop mid-sentence.
    finished: bool
    # The sampled ids themselves (no EOS). The DPO pipeline trains on these rather than on a
    # re-tokenization of `text`, which can merge differently at the prompt boundary.
    token_ids: list[int] | None = None


def load_prompts(
    dataset: str,
    column: str,
    split: str = "train",
    subset: str | None = None,
    max_prompts: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Single-turn user prompts from a dataset column holding either a string or a message list
    (whose first user turn is taken). Shuffled with `seed` before the `max_prompts` cap, so a cap
    samples the split rather than its head; deduplicated, since a duplicate prompt only re-weights
    the pairs."""
    raw = load_chat_dataset(dataset, subset)
    if split not in raw:
        raise ValueError(f"dataset {dataset!r} has no {split!r} split (it has: {', '.join(raw.keys())})")
    rows = raw[split].select_columns([column]).shuffle(seed=seed)
    prompts: list[str] = []
    seen: set[str] = set()
    # Read in slices and stop at the cap: rows[column] would decode the whole shuffled column
    # (every message list of a ~1M-row dataset) before the cap applied.
    for start in range(0, len(rows), _PROMPT_READ_CHUNK):
        for value in rows[start : start + _PROMPT_READ_CHUNK][column]:
            if isinstance(value, list):
                value = next((m["content"] for m in value if m["role"] == "user"), None)
            if not value or value in seen:
                continue
            seen.add(value)
            prompts.append(value)
            if max_prompts is not None and len(prompts) >= max_prompts:
                return prompts
    return prompts


def turn_template(cfg: Config) -> dict:
    """The (user_prefix, assistant_prefix) a chat-tuned config is prompted with (its sft.* when
    SFT'd, else dpo.*) — recorded in the pair manifest so the DPO pipeline can refuse to train the
    pairs under another."""
    section = cfg.sft if cfg.sft.enabled else cfg.dpo
    return {"user_prefix": section.user_prefix, "assistant_prefix": section.assistant_prefix}


def _chat_cfg(cfg: Config) -> Config:
    """The checkpoint's own config, for turn_template — which needs sft or dpo enabled to know
    the turn template. A plain pretrained checkpoint has neither; it is prompted with the DPO
    defaults, which are also the template the pairs written here will be trained under."""
    if cfg.sft.enabled or cfg.dpo.enabled:
        return cfg
    print("[radiance] rlaif: checkpoint is not chat-tuned; prompting with the default dpo turn template")
    cfg.dpo.enabled = True
    return cfg


@torch.no_grad()
def sample_completions(
    model: DenseTransformer,
    cfg: Config,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    batch_size: int,
    device: str,
) -> list[list[Completion] | None]:
    """`num_samples` completions per prompt, batched through generate_tokens_batched. None for a
    prompt whose formatted length leaves no room for max_new_tokens in the model's context.

    `batch_size` counts sequences, so each batch holds batch_size // num_samples prompts; each
    prompt is prefilled once and its KV cache shared by its num_samples rows (samples_per_item).

    Generates one token past max_new_tokens: generate_tokens_batched ends a row at its budget
    without saying whether the token it sampled there was EOS, so a response that used exactly
    max_new_tokens and then stopped would otherwise count as unfinished. A row that runs into the
    extra token is unfinished, and is cut back to max_new_tokens."""
    template = turn_template(_chat_cfg(cfg))
    budget = model.cfg.max_seq_len - (max_new_tokens + 1)
    prompt_texts = [template["user_prefix"] + p + template["assistant_prefix"] for p in prompts]
    prompt_ids = [
        tokenize_chat_prompt(
            [{"role": "user", "content": p}], tokenizer, template["user_prefix"], template["assistant_prefix"]
        )[0]
        for p in prompts
    ]
    work = [i for i, ids in enumerate(prompt_ids) if len(ids) <= budget]
    results: list[list[Completion] | None] = [None] * len(prompts)
    for i in work:
        results[i] = []
    n_skipped = len(prompts) - len(work)
    if n_skipped:
        print(f"[radiance] rlaif: skipped {n_skipped} prompts longer than max_seq_len - max_new_tokens = {budget}")

    model.eval()
    use_autocast = device.startswith("cuda")
    prompts_per_batch = max(1, batch_size // num_samples)
    for start in range(0, len(work), prompts_per_batch):
        chunk = work[start : start + prompts_per_batch]
        items = [
            BatchItem(
                input_ids=torch.tensor([prompt_ids[i]], device=device),
                max_new_tokens=max_new_tokens + 1,
                temperature=temperature,
                top_k=top_k,
            )
            for i in chunk
        ]
        # Row r is sample r % num_samples of prompt chunk[r // num_samples].
        generated: list[list[int]] = [[] for _ in range(len(chunk) * num_samples)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            for step in generate_tokens_batched(
                model, tokenizer, items, device=device, samples_per_item=num_samples
            ):
                for row, token in step:
                    if token is not None:
                        generated[row].append(token)
        for row, gen in enumerate(generated):
            i = chunk[row // num_samples]
            finished = len(gen) <= max_new_tokens
            gen = gen[:max_new_tokens]
            text = _decode_completion(tokenizer, prompt_ids[i], gen, prompt_texts[i])
            results[i].append(Completion(text, finished, gen))
        done = min(start + prompts_per_batch, len(work)) * num_samples
        print(f"[radiance] rlaif: sampled {done}/{len(work) * num_samples} completions", flush=True)
    return results


def _decode_completion(
    tokenizer: PreTrainedTokenizerBase, prompt_ids: list[int], gen: list[int], prompt_text: str
) -> str:
    """The completion as the text that follows the prompt — decoded jointly with the prompt and
    then split off it, for the reason generate.generate gives (a BPE decode of a continuation is
    not always its decode in isolation). This is what the judge scores and what a reader sees;
    the DPO pipeline trains on the Completion's token_ids instead.

    Split off `prompt_text` (the full formatted prompt) first, not just the decode of prompt_ids:
    tokenize_chat_prompt leaves the assistant prefix's trailing space for the policy to sample as
    part of its first word, and that space belongs to the template, not the reply."""
    full = tokenizer.decode(prompt_ids + gen, skip_special_tokens=True)
    for prefix in (prompt_text, tokenizer.decode(prompt_ids, skip_special_tokens=True)):
        if full.startswith(prefix):
            return full[len(prefix) :]
    return tokenizer.decode(gen, skip_special_tokens=True)


class RewardModelJudge:
    """A public sequence-classification reward model as the AI-feedback source: one scalar per
    (prompt, response), higher is better."""

    def __init__(self, name: str, device: str, batch_size: int = 8, max_length: int = 4096):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "right"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype, num_labels=1)
        self.model.to(device).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Sequence-classification heads pool at the last non-pad token, located via this id.
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.chat = getattr(self.tokenizer, "chat_template", None) is not None

    def _encode(self, prompts: list[str], responses: list[str]):
        if not self.chat:
            return self.tokenizer(
                prompts, responses, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
        texts = []
        for prompt, response in zip(prompts, responses):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False
            )
            # The template already emits BOS where the model expects one; tokenizing its output
            # with add_special_tokens on would add a second.
            texts.append(text)
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            add_special_tokens=False, return_tensors="pt",
        )

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            enc = self._encode(prompts[start : start + self.batch_size], responses[start : start + self.batch_size])
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            scores.extend(logits[:, 0].float().tolist())
        return scores

    def close(self) -> None:
        del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


def score_completions(judge, prompts: list[str], completions: list[list[Completion] | None]) -> list[list[float] | None]:
    """Flatten (prompt, completion) for one batched judge pass, then regroup per prompt."""
    flat_p, flat_c = [], []
    for prompt, comps in zip(prompts, completions):
        for comp in comps or []:
            flat_p.append(prompt)
            # Whitespace at the edges is an artifact of the turn template, not something to judge.
            flat_c.append(comp.text.strip())
    flat_scores = iter(judge.score(flat_p, flat_c)) if flat_p else iter(())
    return [None if comps is None else [next(flat_scores) for _ in comps] for comps in completions]


def build_preference_pairs(
    prompts: list[str],
    completions: list[list[Completion] | None],
    scores: list[list[float] | None],
    min_margin: float = 0.0,
) -> list[dict]:
    """Best-vs-worst pairs: per prompt, the highest-scored *finished* completion is chosen and the
    lowest-scored other completion (finished or not) is rejected.

    Best-vs-worst rather than all pairs or best-vs-random because it takes the pair the judge is
    most confident about — the one least likely to be label noise — and one pair per prompt keeps
    each prompt's weight equal. A prompt is skipped when it has no finished completion, when the
    two sides are the same text once stripped (at low temperature, common: such a pair carries no
    preference and only dilutes the batch mean — and edge whitespace is not something the judge
    sees, see score_completions), when the margin is not positive (a tie, or a finished best
    outscored by every unfinished sample), or when it is below `min_margin`.
    """
    pairs = []
    for prompt, comps, comp_scores in zip(prompts, completions, scores):
        if not comps or len(comps) < 2:
            continue
        finished = [i for i, c in enumerate(comps) if c.finished]
        if not finished:
            continue
        best = max(finished, key=lambda i: comp_scores[i])
        worst = min((i for i in range(len(comps)) if i != best), key=lambda i: comp_scores[i])
        if comps[best].text.strip() == comps[worst].text.strip():
            continue
        margin = comp_scores[best] - comp_scores[worst]
        if margin <= 0 or margin < min_margin:
            continue
        pair = {
            "prompt": prompt,
            "chosen": comps[best].text,
            "rejected": comps[worst].text,
            "chosen_score": comp_scores[best],
            "rejected_score": comp_scores[worst],
            # Every sample and score, so a later round can re-pair (or re-filter) without
            # re-sampling.
            "completions": [c.text for c in comps],
            "completion_scores": list(comp_scores),
            "completion_finished": [c.finished for c in comps],
        }
        if all(c.token_ids is not None for c in comps):
            # What the DPO pipeline trains on (dpo_data._tokenize_dpo_row), so a pair is the
            # policy's own token sequence, not a re-tokenization of its text.
            pair["chosen_token_ids"] = comps[best].token_ids
            pair["rejected_token_ids"] = comps[worst].token_ids
            pair["completion_token_ids"] = [c.token_ids for c in comps]
        pairs.append(pair)
    return pairs


def _mean_stderr(values: list[float]) -> tuple[float, float | None]:
    mean = sum(values) / len(values)
    return mean, standard_error(values, mean)


def _sample_from_checkpoint(path: str, prompts: list[str], args, num_samples: int, device: str):
    """(completions, the manifest fields the DPO pipeline checks the pairs against: the turn
    template they were sampled under, and the tokenizer their token ids belong to)."""
    model, cfg, tokenizer = load_checkpoint(path, device)
    completions = sample_completions(
        model, cfg, tokenizer, prompts, num_samples, args.max_new_tokens,
        args.temperature, args.top_k, args.batch_size, device,
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return completions, {
        "turn_template": turn_template(cfg),
        "tokenizer": cfg.data.tokenizer,
        "prompt_ids_version": RLAIF_PROMPT_IDS_VERSION,
    }


def run_pairs(args) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    prompts = load_prompts(args.prompts, args.prompt_column, args.split, args.subset, args.max_prompts, args.seed)
    print(f"[radiance] rlaif: {len(prompts)} prompts from {args.prompts}:{args.split}")
    completions, sampled_under = _sample_from_checkpoint(args.checkpoint, prompts, args, args.num_samples, device)
    if all(c is None for c in completions):
        raise SystemExit(
            "[radiance] rlaif: no prompt fits the policy's context with room for "
            f"--max-new-tokens={args.max_new_tokens}; see the skipped count above. Lower "
            "--max-new-tokens, or use shorter prompts — the policy's stopping is not the problem."
        )

    judge = RewardModelJudge(args.judge, device, args.judge_batch_size)
    scores = score_completions(judge, prompts, completions)
    judge.close()

    pairs = build_preference_pairs(prompts, completions, scores, args.min_margin)
    all_scores = [s for row in scores if row for s in row]
    all_finished = [c.finished for row in completions if row for c in row]
    finished_fraction = sum(all_finished) / max(len(all_finished), 1)
    if finished_fraction < 0.5:
        # The usual reason a round yields few pairs: an unfinished sample can never be chosen.
        print(
            f"[radiance] rlaif: only {finished_fraction:.0%} of samples ended with EOS within "
            f"--max-new-tokens={args.max_new_tokens}, so most prompts had no candidate for 'chosen'. "
            "Raise --max-new-tokens, or SFT the policy further first (a policy that rarely stops "
            "is not ready for preference tuning)."
        )
    if not pairs:
        raise SystemExit("[radiance] rlaif: no preference pairs survived; see above, or lower --min-margin")
    mean, stderr = _mean_stderr(all_scores)
    stats = {
        "prompts": len(prompts),
        "pairs": len(pairs),
        "mean_score": mean,
        "mean_score_stderr": stderr,
        "mean_pair_margin": sum(p["chosen_score"] - p["rejected_score"] for p in pairs) / len(pairs),
        "finished_fraction": finished_fraction,
    }
    out = Path(args.output)
    DatasetDict({"train": Dataset.from_list(pairs)}).save_to_disk(str(out))
    manifest = {"args": vars(args), "stats": stats, **sampled_under}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"[radiance] rlaif: wrote {len(pairs)} pairs to {out}")
    print(json.dumps(stats, indent=2))
    print(
        "[radiance] rlaif: train on them with a DPO config setting dpo.dataset: "
        f"{out}, dpo.prompt_column: prompt (see configs/post_training/dpo_rlaif.yaml), and "
        f"dpo.user_prefix/assistant_prefix {sampled_under['turn_template']['user_prefix']!r}/"
        f"{sampled_under['turn_template']['assistant_prefix']!r}, the turn template these were sampled under"
    )


def win_rates(
    baseline: list[float],
    other: list[float],
    baseline_texts: list[str] | None = None,
    other_texts: list[str] | None = None,
) -> float:
    """Fraction of prompts where `other` outscores `baseline`, ties counting half.

    Identical completions (compared stripped, as the judge sees them) are ties whatever their
    scores: the judge runs in bf16 and its score for a given text moves by ~0.03 with the rest of
    its padded batch, so without this, two checkpoints that wrote the same answer would be
    separated by numerical noise.
    """
    wins = 0.0
    for i, (b, o) in enumerate(zip(baseline, other)):
        if baseline_texts is not None and baseline_texts[i].strip() == other_texts[i].strip():
            wins += 0.5
        else:
            wins += 1.0 if o > b else 0.5 if o == b else 0.0
    return wins / len(baseline)


def run_eval(args) -> None:
    device = resolve_device(args.device)
    prompts = load_prompts(args.prompts, args.prompt_column, args.split, args.subset, args.max_prompts, args.seed)
    print(f"[radiance] rlaif eval: {len(prompts)} prompts from {args.prompts}:{args.split}")
    per_ckpt: list[list[list[Completion] | None]] = []
    for path in args.checkpoint:
        # Same seed per checkpoint, so sampling noise is matched as closely as the models allow.
        set_seed(args.seed)
        per_ckpt.append(_sample_from_checkpoint(path, prompts, args, 1, device)[0])

    # Only prompts every checkpoint produced a completion for, so the win rates compare like with like.
    keep = [i for i in range(len(prompts)) if all(c[i] is not None for c in per_ckpt)]
    if not keep:
        raise SystemExit(
            "[radiance] rlaif eval: no prompt fits every checkpoint's context with room for "
            f"--max-new-tokens={args.max_new_tokens}; see the skipped counts above, or lower it"
        )
    judge = RewardModelJudge(args.judge, device, args.judge_batch_size)
    rewards = []
    for comps in per_ckpt:
        scored = score_completions(judge, [prompts[i] for i in keep], [comps[i] for i in keep])
        rewards.append([s[0] for s in scored])
    judge.close()

    texts = [[comps[i][0].text for i in keep] for comps in per_ckpt]
    report = []
    for path, comps, r, t in zip(args.checkpoint, per_ckpt, rewards, texts):
        mean, stderr = _mean_stderr(r)
        kept = [comps[i][0] for i in keep]
        row = {
            "checkpoint": path,
            "mean_reward": mean,
            "mean_reward_stderr": stderr,
            "finished_fraction": sum(c.finished for c in kept) / len(kept),
            "mean_chars": sum(len(c.text) for c in kept) / len(kept),
        }
        if len(rewards) > 1:
            row["win_rate_vs_first"] = win_rates(rewards[0], r, texts[0], t)
        report.append(row)
        print(
            f"[radiance] {path}: reward {mean:.3f}"
            + (f" ± {stderr:.3f}" if stderr is not None else "")
            + (f", win rate vs {args.checkpoint[0]} {row['win_rate_vs_first']:.3f}" if len(rewards) > 1 else "")
            + f", finished {row['finished_fraction']:.2f}"
        )
    if args.output:
        # Per-prompt samples too: a reward number is only as trustworthy as a look at what earned it.
        samples = [
            {"prompt": prompts[i], "completions": [t[j] for t in texts], "rewards": [r[j] for r in rewards]}
            for j, i in enumerate(keep)
        ]
        Path(args.output).write_text(
            json.dumps({"args": vars(args), "prompts": len(keep), "results": report, "samples": samples}, indent=2)
        )


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompts", default="HuggingFaceH4/ultrafeedback_binarized", help="prompt dataset (hub id or local)")
    p.add_argument("--prompt-column", default="prompt", help="column holding a prompt string or a message list")
    p.add_argument("--split", default="train_prefs")
    p.add_argument("--subset", default=None)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32, help="sequences generated per batched forward")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help="HF sequence-classification reward model")
    p.add_argument("--judge-batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy RLAIF preference data and judge-scored evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)

    pairs = sub.add_parser("pairs", help="sample from a policy, judge, and write DPO preference pairs")
    pairs.add_argument("--checkpoint", required=True, help="the policy (.pt or radiance-export directory)")
    pairs.add_argument("--output", required=True, help="directory to write the pair dataset to")
    pairs.add_argument("--num-samples", type=int, default=4, help="completions sampled per prompt")
    pairs.add_argument("--min-margin", type=float, default=0.0, help="drop pairs whose judge margin is below this")
    _add_shared_args(pairs)

    ev = sub.add_parser("eval", help="mean judge reward per checkpoint, and win rate against the first")
    ev.add_argument("--checkpoint", required=True, action="append", help="repeat to compare; the first is the baseline")
    ev.add_argument("--output", default=None, help="optional JSON report path")
    _add_shared_args(ev)
    ev.set_defaults(split="test_prefs", max_prompts=500)

    args = parser.parse_args()
    if args.command == "pairs":
        if args.num_samples < 2:
            parser.error("--num-samples must be at least 2 to form a pair")
        run_pairs(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
