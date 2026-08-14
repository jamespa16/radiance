# Post-training: SFT and DPO

Two post-training modes, both implemented as mode switches inside `train()` rather than separate trainers — each
swaps `data.py`'s pipeline and `train.py`'s loss function and reuses everything else (optimizer, LR schedule,
`auto_batch_size`, OOM backoff, checkpointing, `train.init_from`, wandb/stdout logging) unchanged. `cfg.sft.enabled`
and `cfg.dpo.enabled` are mutually exclusive — `validate_post_training_config` raises if both are set, called at the
top of `train()`.

## SFT

`cfg.sft.enabled`, `SFTConfig`. Chat/instruction fine-tuning on a pretrained checkpoint.

`data._format_sft_messages`/`_tokenize_sft_example` turn each example into `(ids, loss_mask)` — plain-text turn
markers (`sft.user_prefix`/`assistant_prefix`, **not** new special tokens, so no vocab/embedding resize is needed),
`loss_mask` 1 on assistant turns plus the trailing EOS, 0 elsewhere. `_tokenize_and_pack_sft` packs many examples per
fixed-`seq_len` block, EOS-joined exactly like pretraining documents, and `doc_attention_mask` (on by default)
isolates them from each other using the *same* EOS-boundary mechanism (`model.document_ids`) it already uses for
plain text — so SFT needs no `model.py` changes at all.

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

So `train.resolve_dpo_doc_mask` **turns it off** for a DPO run, rather than leaving a default-on feature to build a
mask nothing reads. Precisely (`config.doc_mask_is_inert_for_dpo`): `document_ids`' exclusive cumsum puts all of a
row's real content — including the single scored EOS terminating it — in document 0, and each padding EOS in a
document of its own, so the only attention the mask removes belongs to a padded position, and no padded position
contributes a scored logit. It is turned off before the model is built, which matters twice over: `DenseTransformer`
reads `cfg.model` by reference, and `doc_attention_mask` is one of the three things that force `resolve_compile_mode`
down to `mode=None` — so a DPO run gets CUDA graphs back, which is the larger half of what this is worth (the other
half is one skipped `BlockMask` build per step, plus one per reference-precompute batch — `data._add_reference_logprobs`
applies the same skip to the reference model's own config). Because it mutates `cfg`, the value saved into the
checkpoint is the one the run actually trained with.

**`loop_attn_windows` is the exception**, and the reason this is a predicate and not a bare `cfg.dpo.enabled`: windows
ride the very same `BlockMask`, and a sliding window restricts attention *within* document 0, which real DPO content
does feel. Configured together, the mask stays on. `tests/test_dpo_doc_mask.py` pins both the structural claim
(device-independent) and, on CUDA where the mask is real, the equivalence that matters: the summed log-probabilities
the DPO loss consumes are identical with masking on and off for a DPO-shaped row — the case that has to be
distinguished from `tests/test_doc_mask.py`'s multi-document block, where turning it off changes the logits a lot.

### The loss

`train.compute_dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta)` is the
standard objective, `-E[logsigmoid(beta * (policy_logratio - ref_logratio))]`, plus two `no_grad` diagnostics logged
as `train/dpo_accuracy`/`train/dpo_margin` (pairwise accuracy and implicit reward margin, neither used in the loss).
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
tensors when their feature is off.

## Shared plumbing

`model.sequence_logprob_sum(logits, input_ids, loss_mask)` is the primitive both the precompute pass and the training
loss need: a per-row `(batch,)` **sum** of log-probability at `loss_mask==1` positions, using the same
single-`logsumexp`-pass trick `train._nll_and_logz` uses but keeping the batch dimension — `_nll_and_logz` reduces to
one batch-wide *mean*, the right reduction for a plain LM loss but not for DPO, which needs one log-probability
scalar per sequence. It lives in `model.py`, not `train.py`, purely because of import direction: `data.py`'s
precompute needs it too, and `data.py` must not import `train.py` (which already imports `data.py`) — both already
import `model.py`. `model.load_transformer_from_checkpoint(path, device, eos_id=None)` moved there for the identical
reason: `generate.py` owns this logic and imports `data.py`, so `data.py` can't import `generate.py` back;
`generate.load_checkpoint` is now a thin wrapper.

`data.format_chat_prompt(user_message, cfg)` generalizes the older `format_sft_prompt` to also cover a DPO
checkpoint: since the two modes are mutually exclusive on any one run's config, a DPO checkpoint's own saved
`cfg.sft.enabled` is always `False` even though it expects the same turn-template prompting it was (transitively)
SFT'd with. It checks `cfg.sft.enabled` then `cfg.dpo.enabled` and picks the matching prefixes, raising if neither is
set. `generate.py --chat` calls this instead of `format_sft_prompt`.

## Status

See `configs/tinystories_dpo.yaml` for a worked example — a three-stage pretrain -> SFT -> DPO chain, with
`train.init_from` seeding the policy and `dpo.reference_checkpoint` anchoring the loss, usually (but not necessarily
— they're separate fields) the same checkpoint.

**No quality A/B has been run yet.** The CPU smoke run above establishes that the pipeline runs and the loss starts
where the math says it must, not that DPO improves anything at this scale. `tests/test_dpo_data.py`,
`tests/test_dpo_loss.py` and `tests/test_dpo_reference_logprob.py` cover the data pipeline, the loss (including the
policy-equals-reference equivalence check that predicted the smoke run's `log(2)` observation), and the precompute
pass, at the same unit-test depth SFT already had — no end-to-end `train()` integration test exists for either mode.
