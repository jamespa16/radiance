# Post-training: SFT, DPO and RLAIF

Two post-training modes, SFT and DPO, plus `radiance-rlaif` for on-policy preference data from AI feedback, which
feeds DPO. The modes are implemented as mode switches inside `train()` rather than separate trainers — each
swaps `data.py`'s pipeline and `train.py`'s loss function and reuses everything else (optimizer, LR schedule,
`auto_batch_size`, OOM backoff, checkpointing, `train.init_from`, wandb/stdout logging) unchanged. `cfg.sft.enabled`
and `cfg.dpo.enabled` are mutually exclusive — `validate_post_training_config` raises if both are set, called at the
top of `train()`.

## The workflow

`configs/post_training/` is the chain, set up against a V2 base checkpoint, with every stage's input given on the
command line so no file is edited between runs:

```bash
BASE=checkpoints/v2/decay_from_<N>/step_<M>.pt          # an annealed base
radiance-train --config configs/post_training/sft.yaml --set train.init_from=$BASE
SFT=checkpoints/post_training/sft/step_3000.pt

# offline preference tuning — AI feedback (UltraFeedback) or human feedback (hh-rlhf):
radiance-train --config configs/post_training/dpo_ultrafeedback.yaml --set train.init_from=$SFT
radiance-train --config configs/post_training/dpo_hh_rlhf.yaml       --set train.init_from=$SFT

# or on-policy RLAIF, one round (repeat with the new checkpoint as the policy):
radiance-rlaif pairs --checkpoint $SFT --output data/rlaif/round1 --max-prompts 8000
radiance-train --config configs/post_training/dpo_rlaif.yaml \
    --set train.init_from=$SFT dpo.dataset=data/rlaif/round1

# and the judge-scored comparison against the SFT model:
radiance-rlaif eval --checkpoint $SFT --checkpoint checkpoints/post_training/dpo_ultrafeedback/step_1000.pt
```

Three pieces make that work without hand-editing:

- **`radiance-train --set section.field=value`** — the same override syntax `radiance-size` already had (now shared as
  `config.load_config_with_overrides`). Overrides are applied to the raw YAML and go through `load_config` itself, so
  its cross-field checks and `inherit_model_config` see them. Repeatable (`--set a=1 --set b=2` keeps both). Values
  are parsed as YAML, except a string-typed field's, which is taken verbatim: YAML would make `dpo.dataset=2026-10-14`
  a date and cut `a #b` at the comment. A top-level field takes no section — `--set run_name=rlaif_round2` — which is
  how each RLAIF round gets its own W&B run.
- **`train.inherit_model_config`** — the model *architecture* (and `data.tokenizer`) comes from `init_from`'s own
  saved config, with only the keys this YAML's `model:` spells out applied on top. A resuming run reads the resume
  checkpoint's instead, which already holds the inherited config, so a resume no longer needs `init_from` to exist,
  or to be unchanged. Before it, every post-training config
  carried a hand copy of the base's `model:` block, and `load_pretrained_weights` only checks *shape* fields: a copy
  that drops a behavioral one (`loop_count`, a RoPE setting, `qk_norm`) loads cleanly and fine-tunes a different model
  from the one pretrained. Opt-in, because it is not inert — an existing config's *omitted* model keys would change
  from the dataclass defaults to the checkpoint's values. Resolved in `load_config` (only the raw YAML knows which
  keys were written down), and cheaply: `export.read_checkpoint_config` mmaps the `.pt`, or reads an export's
  `config.json` alone. Fields that are how the source run *trained* rather than what its weights compute
  (`config._RUN_TIME_MODEL_FIELDS`: the `fp4_*` numeric format, loss weights such as `z_loss_weight`, `dropout`,
  `grad_checkpoint`, `loop_bptt_window`, `doc_attention_mask`) are not inherited: they come from this stage's YAML or
  their defaults. Inherited, they would carry over silently — `fp4_linear` is resolved from `train.dtype: nvfp4`, so a
  bf16 SFT stage off an NVFP4 base would train on FP4 GEMMs, and a DPO checkpoint from before `dpo_model_config`
  stopped writing its doc-mask skip back holds `doc_attention_mask: false`, which would let an SFT stage's packed
  conversations attend to each other.
- **`dpo.reference_checkpoint` defaults to `train.init_from`** (`Config.resolved_reference_checkpoint`) — a
  range-collapsed default: a config naming the reference is unaffected, and one naming neither still raises.

`tests/test_stage_chaining.py` loads each stage config exactly the way its header says to run it.

**The learning rates in those configs are not measured.** They are about a tenth of V2's pretraining rates for SFT
and a tenth again for DPO — the usual rules of thumb, under Muon, where nothing post-training has been swept.

## Datasets

The split fields — `subset`, `train_split`, `eval_split`, `max_examples`, on both `sft` and `dpo` — exist because the
most-used public sets could not be loaded before: `HuggingFaceH4/ultrafeedback_binarized` has `train_prefs`/
`test_prefs` and `HuggingFaceH4/ultrachat_200k` `train_sft`/`test_sft`, and neither has a `train` split, which the
pipelines hard-coded; `HuggingFaceTB/smoltalk` needs a config name. Selection is `sft_data.select_splits`: an explicit
`eval_split` must exist, otherwise the first fallback present is used (`validation` for SFT, `validation` then
`test` for DPO — each pipeline's old behavior, unchanged), otherwise `eval_split_size` rows are carved off train.
`max_examples` caps train *after* the carve (so the held-out rows don't depend on the cap) and after a fixed-seed
shuffle (hub datasets are often sorted by source, so a head would be one source).

`dataset` may also be local: a `save_to_disk` directory or a `.json`/`.jsonl` file (`sft_data.load_chat_dataset`).
That is how `radiance-rlaif` output reaches DPO. A `radiance-rlaif` directory's cache key includes a fingerprint of
its files (latest mtime + total size, the same identity idea the reference-checkpoint key uses), because the path
alone goes stale the moment a round's directory is regenerated in place. Only rlaif output (recognised by its
`rlaif_manifest.json`) gets one: any other local dataset keeps the path-only key it always had, so its existing
caches — DPO reference log-probs among them, which cannot be rebuilt once the reference checkpoint is gone — are still
found. Rewriting such a dataset in place still needs its cache cleared by hand.

**Every pre-existing cache key is unchanged**: the new fields join the key only when they differ from their defaults
(`split_key_extras`), and `tests/test_post_training_datasets.py` pins the old SFT key formula verbatim.

Shapes — **run** marks those taken through `radiance-train` in the smoke runs under Status; the rest were checked
against the dataset's published schema/splits only:

| dataset | stage | fields | |
|---|---|---|---|
| `HuggingFaceTB/smol-smoltalk` | SFT | `messages_column: messages`, `eval_split: test` | run |
| `HuggingFaceTB/smoltalk` | SFT | `subset: all` | schema |
| `HuggingFaceH4/ultrachat_200k` | SFT | `train_split: train_sft`, `eval_split: test_sft` | schema |
| `HuggingFaceH4/ultrafeedback_binarized` | DPO | `train_split: train_prefs`, `eval_split: test_prefs`, optional score filter on `score_chosen`/`score_rejected` | run |
| `Anthropic/hh-rlhf` | DPO | defaults — transcript strings are parsed | run |
| `argilla/dpo-mix-7k` | DPO | defaults — message lists | earlier worked example |
| a `radiance-rlaif pairs` directory | DPO | `prompt_column: prompt`, score filter on `chosen_score`/`rejected_score` | run |

**hh-rlhf transcripts.** `chosen`/`rejected` there are whole `"\n\nHuman: ...\n\nAssistant: ..."` strings sharing
every turn but the last. With no `prompt_column`, a string-valued side is parsed into turns
(`dpo_data.parse_hh_transcript`); before, it crashed iterating characters as turns, so no working config took that
branch. Parsed, the shared turns tokenize identically on both sides (tokenization is per turn), and attention is
causal, so their log-probabilities are equal and cancel in the summed log-ratio. They are nonetheless left out of
the loss mask (`_tokenize_dpo_row`, for any pair whose sides share leading turns): IPO and `sft_weight` divide each
side by its own token count, and with chosen and rejected of different lengths the shared turns would *not* cancel
there, so the preference would leak onto turns both sides agree on. Only message-list/transcript data has shared
assistant turns, so only its cache key changed (`prompt_column` data's one shared turn was never scored).

**Score-margin filter.** `dpo.min_score_margin` with `score_chosen_column`/`score_rejected_column` drops pairs the
judge barely separated. For AI-judged data the near-ties are where the labels are noisiest and carry the least
signal. It runs as its own pass ahead of tokenization so its drop count prints separately from the `seq_len` one —
on a 300-row UltraFeedback smoke at `seq_len: 512`, margin >= 1.0 kept 215, and then 98 of those were too long. (At
512, ~45% of UltraFeedback overflows; the configs use 1024.)

## SFT

`cfg.sft.enabled`, `SFTConfig`. Chat/instruction fine-tuning on a pretrained checkpoint.

`data._format_sft_messages`/`_tokenize_sft_example` turn each example into `(ids, loss_mask)` — plain-text turn
markers (`sft.user_prefix`/`assistant_prefix`, **not** new special tokens, so no vocab/embedding resize is needed),
`loss_mask` 1 on assistant turns plus the trailing EOS, 0 elsewhere. `_tokenize_and_pack_sft` packs many examples per
fixed-`seq_len` block, EOS-joined exactly like pretraining documents, and `doc_attention_mask` (on by default)
isolates them from each other using the *same* EOS-boundary mechanism (`model.document_ids`) it already uses for
plain text — so SFT needs no `model/` changes at all.

`train.compute_sft_loss` is a strict generalization of `compute_loss`, not a parallel reimplementation: it shifts
`loss_mask` the same way labels are shifted, folds it into the `-100` ignore positions, then calls the same
`_nll_and_logz` helper — bit-identical to `compute_loss` at an all-ones mask
(`tests/test_sft_loss.py::test_all_ones_mask_matches_compute_loss`). See `configs/tinystories_sft.yaml`.

## DPO

`cfg.dpo.enabled`, `DPOConfig`. Direct Preference Optimization (Rafailov et al. 2023) on `(prompt, chosen, rejected)`
triples — deliberately *not* full PPO/reward-model RLHF: no reward model, no rollout loop during training, just a
second (data pipeline, loss function) pair plugged into the same `train()` loop SFT already established.

Two design decisions shape everything, both made to keep training at exactly one resident model's VRAM cost.

### Reference log-probabilities are precomputed once and disk-cached

Never held as a second live model during training. `data._add_reference_logprobs` loads the frozen
`dpo.reference_checkpoint`, forwards it once over every packed pair, caches each side's summed log-probability as a
`ref_chosen_logprob`/`ref_rejected_logprob` column, then frees the model entirely (`del` +
`torch.cuda.empty_cache()`). Not a `datasets.map()` call — a model forward can't run inside a `.map()`
multiprocessing worker the way plain tokenization can — so it's a plain batched `DataLoader` loop under
`torch.no_grad()`.

That pass runs *before* the training loop, so its OOMs never reach the step loop's `micro_chunk_size`/CPU-offload
backoff — and the policy model is already resident, so `dpo.reference_batch_size` has less headroom than the trained
batch size. It carries the same idiom locally: a fetched batch is processed in chunks, and a
`torch.cuda.OutOfMemoryError` halves the (sticky) chunk size, frees the failed call's tensors, and retries the batch;
at chunk size 1 it re-raises with a note. Rows are independent — one pair's summed log-probability never reads
another row's logits — so chunked calls are exact and an unbroken run's numbers are unchanged.

The cache lives under `dpo.cache_dir` as `<base digest>/ref-<identity digest>`, a two-level key: the base digest
covers the dataset/tokenizer/columns/prefixes fields, and the `ref-*` subdirectory digests the reference checkpoint's
path/mtime/size — **not** a content hash, since hashing a multi-GB checkpoint on every run start would defeat the
point of a fast cache-hit path, and nothing else in this repo content-hashes checkpoint files. Changing the reference
model invalidates the cache automatically.

Splitting the levels matters for the archive case: the checkpoint is only needed to *compute* those log-probs, never
afterwards, so if it is later archived or moved away, the single cache built while it was present is found by listing
`ref-*` under the base digest and reused; with zero or several candidates there, which reference model's log-probs a
given cache holds can't be established and it raises instead of guessing.

The pass also validates the checkpoint's saved `data.tokenizer` against this run's before forwarding: if they differ,
the reference model's vocab ids don't correspond to the packed ids and `sequence_logprob_sum` gathers from a
no-smaller ref vocab to return numerically-plausible but semantically meaningless values, silently corrupting the DPO
loss from step 1 with nothing to compare against — so it raises.

### Data is packed "one pair per row"

Not SFT's "many examples per block." Each side of a pair (`prompt+chosen`, `prompt+rejected`) is tokenized
independently via `_tokenize_sft_example` (which now takes explicit `user_prefix`/`assistant_prefix` arguments rather
than reading `cfg.sft.*` directly, so DPO reuses it unchanged with `cfg.dpo`'s own prefixes) and padded to exactly
`seq_len` by appending repeated `eos_token_id` at the tail, with `loss_mask=0` on the padding. A row is **dropped,
not truncated**, if either side's real content overflows `seq_len` (`_tokenize_dpo_row`): truncating the completion
would score a partial response as fully chosen/rejected, truncating the prompt would silently change what it is
conditioned on.

Chosen over SFT-style multi-pair packing because a pair's two halves must survive `DataLoader` shuffling together —
keeping both in the same dataset row makes that automatic, where reconstructing pairing from document ids after
shuffling a multi-pair block would be fragile.

**This needs no `doc_attention_mask` support at all for correctness**, for a stronger reason than "it reuses the
mechanism": attention is strictly causal and the real (scored) content always precedes its own trailing padding
within a row, so real tokens can never attend to padding regardless of whether document-boundary masking is active —
unlike SFT's genuinely-multi-document blocks, where `doc_attention_mask` *is* load-bearing. Confirmed empirically: a
CPU smoke run's step-1 DPO loss came out at exactly `log(2) = 0.6931`, the value the loss must take when the policy
hasn't moved from the reference yet — plain SDPA on CPU, no doc masking involved.

So `losses.dpo_model_config` **turns it off** for a DPO run, rather than leaving a default-on feature to build a
mask nothing reads. Precisely (`config.doc_mask_is_inert_for_dpo`): `document_ids`' exclusive cumsum puts all of a
row's real content — including the single scored EOS terminating it — in document 0, and each padding EOS in a
document of its own, so the only attention the mask removes belongs to a padded position, and no padded position
contributes a scored logit. The model is built from a copy of `cfg.model` with the mask off, which matters because
`doc_attention_mask` is one of the three things that force `resolve_compile_mode` down to `mode=None` — so a DPO run gets CUDA graphs back, which is the larger half of what this is worth (the other
half is one skipped `BlockMask` build per step, plus one per reference-precompute batch — `data._add_reference_logprobs`
applies the same skip to the reference model's own config). A copy, not a mutation of `cfg`: the skip is a property
of this run's packing, not of the weights, so the checkpoint saves the architecture's own value, and a later stage
that packs several examples per block (SFT after DPO) gets its mask back.

**`loop_attn_windows` is the exception**, and the reason this is a predicate and not a bare `cfg.dpo.enabled`: windows
ride the very same `BlockMask`, and a sliding window restricts attention *within* document 0, which real DPO content
does feel. Configured together, the mask stays on. `tests/test_dpo_doc_mask.py` pins both the structural claim
(device-independent) and, on CUDA where the mask is real, the equivalence that matters: the summed log-probabilities
the DPO loss consumes are identical with masking on and off for a DPO-shaped row — the case that has to be
distinguished from `tests/test_doc_mask.py`'s multi-document block, where turning it off changes the logits a lot.

### The loss

`train.compute_dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta)` is the
standard objective, `-E[logsigmoid(beta * (policy_logratio - ref_logratio))]`, plus three `no_grad` diagnostics
(none used in the loss), logged as `train/dpo_margin` (implicit reward margin), `train/dpo_margin_accuracy`
(fraction of the batch where the policy separates chosen from rejected *more than the reference did* — the sign of
the loss's own sigmoid argument, so it can sit near 50% early in training even when the policy already ranks chosen
above rejected in absolute terms), and `train/dpo_reward_accuracy` (the conventional reference-independent
`policy_chosen_logp > policy_rejected_logp`, the number most DPO writeups mean by "accuracy").
`compute_dpo_loss_from_logits` fuses two `sequence_logprob_sum` calls plus `compute_dpo_loss` into one function so
`build_dpo_loss_fn` can `torch.compile` it as a unit, mirroring `build_loss_fn`.

In `train()`'s accumulation loop, a DPO batch concatenates chosen+rejected along the batch dimension into **one**
`model(...)` call rather than two forward passes, then splits the logits back — halving the number of forwards at the
cost of the model seeing 2x the token width per row. Only two things about that are DPO-specific, and each is one
function: `split_micro_batch` (which batch columns to chunk — 6 parallel tensors, vs. `input_ids` plus SFT's
`loss_mask`) and `chunk_loss_and_metrics` (the cat/split forward and the 9-argument loss, plus the named metric terms
to log). Everything that makes accumulation *correct* — `chunk_reduction_units`' `chunk_weight`, the autocast block, the
scaled backward, the weighted accumulation into `_ACCUM_METRICS`, and the enclosing OOM `try` — is written once in the
loop, so a fix to the weighting math or the OOM-chunking contract cannot land in one mode and silently miss the other.
That mattered: the DPO branch started as a parallel copy and had already begun to drift. `tests/test_accumulation.py`
holds the contract the sharing rests on — a chunked micro-batch produces the same gradient as one un-split backward,
in all three modes, which is the premise tier-1 OOM backoff quietly depends on.

The 2x width is exactly what `estimate_batch_size`'s
`train_width_multiplier` (`2` under `cfg.dpo.enabled`, `1` otherwise) and `cfg.resolved_row_tokens` exist to account
for, alongside any active `sft.seq_len`/`dpo.seq_len` override: `auto_batch_size` would otherwise size a DPO batch as
if it were an SFT batch and OOM on the first step, and `tokens_per_param` would derive `max_steps` from the wrong
per-row token width.

`z_loss`/`mtp_loss` are deliberately omitted from the DPO chunk loss (and from the metrics it reports, so those series
log a flat 0 rather than a stale value) — `mtp_heads > 1` is already rejected by
`validate_post_training_config`, and `z_loss` would need its own reduction pass over the concatenated logits that
isn't needed for DPO correctness. `ponder_cost`/`moe_aux_loss` compose in for free, since both are zero scalar
tensors when their feature is off. Because `model.z_loss_weight` defaults nonzero, `train.note_dpo_z_loss_omitted`
prints a one-time startup note under `dpo.enabled` when it's still set, rather than raising the way `mtp_heads > 1`
does — raising would break every existing default DPO config over a term that was never meant to apply here.

### Loss variants

`dpo.loss_type`, `dpo.label_smoothing` and `dpo.sft_weight` select among the standard offline-preference objectives,
all on the same reference-relative margin `(policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)`:

| setting | objective | what it changes |
|---|---|---|
| `loss_type: sigmoid` (default) | DPO, `-logsigmoid(beta * margin)` | — |
| `label_smoothing: e` | cDPO, `(1-e) * DPO(c>r) + e * DPO(r>c)` | optimum is the finite margin `log((1-e)/e) / beta`, not infinity: a pair that may be mislabeled is not pushed apart forever. The case for AI-judged data. |
| `loss_type: ipo` | IPO, `(m - 1/(2*beta))^2` on **per-token-mean** log-ratios | regresses to a fixed margin, so it stops pushing a pair it already separates; per-token means because with sums the target margin would scale with response length (TRL does the same) |
| | | note `beta` means something else here: the target is `1/(2*beta)` nats *per token* — 5 at the default 0.1, so the loss starts near 25 (seen in the hh-rlhf smoke). An IPO run wants its own, larger `beta` |
| `loss_type: hinge` | SLiC, `relu(1 - beta * margin)` | exactly zero past the margin |
| `sft_weight: w` | `+ w * NLL(chosen)`, per-token mean (RPO; Llama 3's DPO+NLL) | DPO constrains only the *gap*; on small models both log-probs commonly fall together, and this anchors the chosen side |

They are bound into the loss once, in `build_dpo_loss_fn` (a `functools.partial` over the possibly-compiled
`compute_dpo_loss_from_logits`), so every call site still passes the same nine positional arguments. `val/loss`
is evaluated with `build_dpo_loss_fn(cfg, plain=True)` instead: always plain DPO, so it compares across variants (an
`sft_weight` run's configured loss carries a chosen-NLL term no other run's does). At the defaults the code path is the
original one operation for operation — `tests/test_preference_losses.py` checks it is bit-identical, and checks each
variant by its defining property (cDPO's zero-gradient point, hinge's zero region, IPO's zero at its target) rather than
against a re-typed formula. Token counts for the two length-normalized terms are the *shifted* mask's,
`loss_mask[:, 1:].sum()` — exactly the positions `sequence_logprob_sum` scores; an off-by-one there would silently
rescale the term. They are only computed when a variant needs them.

### Validation: preference accuracies

`evaluate_detailed` now also returns `dpo_reward_accuracy` and `dpo_margin_accuracy` over the val pairs (row-weighted
across chunks and batches, so a backed-off eval matches an un-split one), logged as `val/dpo_*_accuracy` and printed
beside `val/loss`. `val/loss` cannot say whether the policy *ranks* held-out pairs correctly; these can.

### Dropout biases the DPO loss

The reference log-probabilities are computed once, in eval mode; the policy's are computed in train mode. With
`model.dropout > 0` the policy's log-probs are dropout-noised and the reference's are not, so even at step 1, where the
policy *is* the reference, the margin is noise rather than 0 and the loss does not start at `log(2)`. Measured on a
one-pair CPU smoke run: **0.661 at `dropout: 0.1`, exactly 0.6931 at `0.0`**. TRL disables dropout for DPO by default
for this reason. Here `train()` prints a note (`losses.note_dpo_dropout`) rather than overriding it, since silently
changing `model.dropout` would change what an existing DPO config trains; the `configs/post_training/` DPO stages pin
`model.dropout: 0.0` on top of the inherited model config. (V2 itself trains at 0.0, so the chain above is unaffected
either way; a base with dropout is not.)

### Why `sft.enabled`/`dpo.enabled` stay two booleans, not a mode enum

Considered and rejected. Mutual exclusion is centrally enforced (`validate_post_training_config` raises if both are
set), and every dispatch site that branches on the two booleans — `estimate_batch_size`, `build_loss_fn`,
`chunk_reduction_units`/`split_micro_batch`, `chunk_loss_and_metrics`, `dpo_model_config`'s predicate, `train()`'s
own loader/loss-fn dispatch, `Config.resolved_seq_len`/`train_width_multiplier`, and `data.format_chat_prompt` — was
individually audited and is already correctly guarded; there is no live bug from the two-boolean shape today, only
repeated (not inconsistent) `if dpo.enabled: ... elif sft.enabled: ...` structure. A `PostTrainingMode` enum would
collapse those to one property read, but it means a `Config` dataclass schema change plus pickle-compat handling in
`_backfill_missing_fields` for checkpoints saved with only the two booleans — a real migration, not a local fix, and
not justified without a concrete bug the current shape causes.

## RLAIF: on-policy preference rounds (`radiance-rlaif`)

`radiance-rlaif pairs` turns DPO into iterated, on-policy RLAIF (the form used by Self-Rewarding LMs and
iterative/"online" DPO): sample `--num-samples` completions per prompt from the current policy, score each with a
public reward model, keep the best against the worst as a preference pair, write the pairs as a local HF dataset, DPO
on it, repeat with the new checkpoint. What distinguishes it from offline DPO on UltraFeedback or hh-rlhf is *whose*
completions the preferences are over: offline pairs were sampled from other models, and drift further from the
policy's own distribution the more training moves it; a round's pairs are from the policy being trained.

**Why not PPO/GRPO.** For the reason DPO itself was chosen here: one resident model. A PPO step holds the policy, a
value model, a reference and a reward model, and needs generation inside the training loop. Here sampling, judging
and training run strictly in sequence — `pairs` samples with the policy, frees it, then loads the judge — so the peak
is one model, and `train()` is untouched. The cost is that the data goes stale *within* a round, so rounds should be
short (`dpo_rlaif.yaml` defaults to 250 steps over ~8k prompts). An in-loop GRPO is the natural next step if a round
structure turns out to be the limiter; nothing has measured that.

**The judge** is `--judge`, any one-output `AutoModelForSequenceClassification` reward model; the default,
`Skywork/Skywork-Reward-V2-Qwen3-0.6B`, is a 0.6B Bradley-Terry model, small enough to run beside a training job. A
judge with a chat template gets the conversation through it (tokenized with `add_special_tokens=False`, since the
template already emits any BOS); one without gets a `(prompt, response)` text pair. Checked by hand against the real
default judge: "Paris" for the capital of France scored 2.66, "Berlin" -4.41, "banana banana banana" -8.38. Batched
(right-padded) and single-sequence scores agree to ~0.03 in bf16, against score gaps of several units.

**Pairing rules** (`rlaif.build_preference_pairs`), each for a reason:

- **chosen is the best *finished* sample; rejected the worst other one.** An unfinished sample (cut at
  `--max-new-tokens`; sampling runs one token past the cap to tell "stopped exactly at the cap" from "cut off") can
  be rejected but never chosen: the DPO tokenizer appends EOS to both sides, so a truncated
  "chosen" would teach the policy to stop mid-sentence. Best-vs-worst is the pair the judge is most sure of, and one
  pair per prompt keeps prompts equally weighted.
- **identical sides are skipped** (common at low temperature), compared after stripping edge whitespace, which the
  judge never sees: such a pair carries no preference and only dilutes the batch mean. So is a **non-positive
  margin** — an exact tie, which would otherwise pass the default `--min-margin 0`.
- **`--min-margin`** drops pairs below a judge-score margin at write time; `dpo.min_score_margin` does the same at
  train time, so one sampled round can be re-filtered without re-sampling. Every sample and score is kept in the
  output (`completions`, `completion_scores`, `completion_finished`) for the same reason.

**The pairs are the policy's own token sequences.** Re-tokenizing sampled text is not: BPE can merge the sampled
tokens differently once they are text again, and DPO would then score a sequence the policy never produced. So the
prompt is tokenized turn by turn (`sft_data.tokenize_chat_prompt`, the same construction `tokenize_sft_example`
uses), sampling runs after those ids, and `pairs` writes the sampled ids beside the text (`chosen_token_ids`,
`rejected_token_ids`, `completion_token_ids`). The DPO pipeline, given a row with them and `prompt_column` set,
rebuilds the prompt with the same function and appends the ids plus EOS instead of re-tokenizing. The text (decoded
jointly with the prompt and split off the full formatted prompt, `_decode_completion`) is what the judge scores and
a reader sees.

**The open prompt ends where SFT's tokens do.** With GPT-2, `"\n\nAssistant: "` tokenized alone ends in a bare-space
token (220), but SFT tokenizes prefix and reply together, and there the space merges into the first word
(`":", " Hello"`) — so a prompt ending in 220 is a position SFT never trained after. `tokenize_chat_prompt` drops the
prefix's merged-away tail (found by tokenizing the prefix with a probe word after it), leaving the policy to sample
`" Hello"` itself; a reply that genuinely starts with whitespace can still produce the bare token. Pairs sampled
before this follow the old prompt ids, so the manifest records `prompt_ids_version` and the DPO pipeline refuses
token-id pairs from an older version rather than splice them onto a prompt they were not sampled after.

**Sampling prefills each prompt once.** `--batch-size` counts sequences, so a batch holds `batch_size // num_samples`
prompts; `generate_tokens_batched(samples_per_item=K)` prefills each and expands its KV cache across its K samples,
instead of paying K identical prefills. `load_prompts` reads the shuffled prompt column in slices and stops at
`--max-prompts`, so `eval`'s 500-prompt default does not decode a million-row column first.

That only holds under the turn template and tokenizer the policy was sampled with (its own `sft.*`, or `dpo.*` for a
DPO or plain checkpoint), so `pairs` records both in `rlaif_manifest.json` and the DPO pipeline refuses to train the
pairs under a different `dpo.user_prefix`/`assistant_prefix` or `data.tokenizer`.

A round that yields few pairs almost always means few samples sampled EOS in time; `pairs` prints the finished
fraction and says so below 50%. (When no prompt fits the context at all, it says that instead: raising
`--max-new-tokens` would make it worse.) A policy that rarely stops is not ready for preference tuning — the random-init smoke
policy finished 0.6% of its samples and produced 1 pair from 48 prompts, the intended behavior.

**`radiance-rlaif eval`** samples one completion per prompt from each `--checkpoint` (same seed for each), scores
them with the judge on the prompts every checkpoint could answer, and reports mean reward ± stderr, finished fraction,
mean length and, from the second checkpoint on, the win rate against the first (ties count half). It defaults to 500
prompts from `test_prefs` — disjoint from the `train_prefs` prompts the training rounds draw. This is the
measurement the DPO metrics cannot make: `val/loss` and the preference accuracies score pairs someone else wrote,
not what the policy now generates. Mind the obvious confound when it is the same judge that produced a round's pairs:
the policy was optimized against that judge, so its win rate under it is an upper bound. A second, different judge
(`--judge OpenAssistant/reward-model-deberta-v3-large-v2`, say) is the cheap check. On the smoke models (near-random
text, 29 prompts) the two judges disagreed outright — 0.655 vs 0.500 for the same pair of checkpoints — which is what
judging noise looks like at that n; the default is 500 prompts for a reason.

Identical completions count as exact ties whatever their scores: the judge runs in bf16, where a text's score moves
~0.03 with the rest of its padded batch, so two checkpoints that wrote the same answer would otherwise be separated by
numerical noise. `--output` writes the per-prompt completions and rewards beside the summary.

## Shared plumbing

`model.sequence_logprob_sum(logits, input_ids, loss_mask)` is the primitive both the precompute pass and the training
loss need: a per-row `(batch,)` **sum** of log-probability at `loss_mask==1` positions, using the same
single-`logsumexp`-pass trick `losses._nll_and_logz` uses but keeping the batch dimension — `_nll_and_logz` reduces
to one batch-wide *mean*, the right reduction for a plain LM loss but not for DPO, which needs one log-probability
scalar per sequence. It lives in `model/` (`dpo.sequence_logprob_sum`), not `train.py`, purely because of import
direction: `dpo_data.py`'s precompute needs it too, and `dpo_data.py` must not import `train.py` (which already
imports it) — both already import `model/`. `model.load_transformer_from_checkpoint(path, device, eos_id=None)` moved there for the identical
reason: `generate.py` owns this logic and imports `data.py`, so `data.py` can't import `generate.py` back;
`generate.load_checkpoint` is now a thin wrapper.

`data.format_chat_prompt(user_message, cfg)` generalizes the older `format_sft_prompt` to also cover a DPO
checkpoint: since the two modes are mutually exclusive on any one run's config, a DPO checkpoint's own saved
`cfg.sft.enabled` is always `False` even though it expects the same turn-template prompting it was (transitively)
SFT'd with. It checks `cfg.sft.enabled` then `cfg.dpo.enabled` and picks the matching prefixes, raising if neither is
set. `generate.py --chat` calls this instead of `format_sft_prompt`.

## Status

`configs/post_training/` is the worked chain (see [The workflow](#the-workflow)); the TinyStories-scale
`configs/old_tests/tinystories_{sft,dpo}.yaml` predate it.

**No quality A/B has been run yet** — of DPO, its variants, or RLAIF. What has been established is plumbing, on CPU
with a 2-layer `d_model: 64` random-init base and the real public datasets: SFT on `smol-smoltalk`
(`max_examples: 300`); DPO on `ultrafeedback_binarized` (`train_prefs`/`test_prefs`, score filter, cDPO + RPO terms,
val accuracies); DPO on `hh-rlhf` (transcript parsing, IPO); `radiance-rlaif pairs` with the real Skywork judge on
GPU; DPO on its output; and `radiance-rlaif eval` under both the Skywork (chat-template) and DeBERTa (text-pair)
judges. The step-1 DPO loss is exactly `log(2)` at `dropout: 0` (and is not otherwise — see
[Dropout](#dropout-biases-the-dpo-loss)). The natural first A/B, from an SFT'd V2: SFT alone vs. offline DPO on
UltraFeedback vs. one or two on-policy RLAIF rounds over the same prompt pool, compared by `radiance-rlaif eval` under
two judges, plus `radiance-eval` benchmarks to catch regressions the judge doesn't see.

Unit coverage: `tests/test_dpo_data.py`, `tests/test_dpo_loss.py` and `tests/test_dpo_reference_logprob.py` cover the
data pipeline, the loss (including the policy-equals-reference equivalence check that predicted the `log(2)`
observation), and the precompute pass; `tests/test_sft_train_e2e.py` and `tests/test_dpo_train_e2e.py` run
`train.train()` itself end-to-end (tiny model, faked dataset/tokenizer, CPU) for each mode, checking the loss
actually decreases and the resulting checkpoint reloads and forwards cleanly — the DPO one also pins that
`dpo_model_config` applies inside a full run and that the reference-logprob precompute populates `dpo.cache_dir`.
`tests/test_post_training_datasets.py` covers split selection, local datasets, cache-key stability, hh-rlhf
parsing and the score filter; `tests/test_preference_losses.py` the loss variants and val accuracies;
`tests/test_rlaif.py` the pairing rules, sampling bookkeeping, and that `pairs` output loads as a DPO dataset with no
hand-editing; `tests/test_stage_chaining.py` `inherit_model_config`, `--set`, and every `configs/post_training/` stage.
