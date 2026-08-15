from __future__ import annotations

import torch
import torch.nn.functional as F

from radiance.config import Config, doc_mask_is_inert_for_dpo
from radiance.model import DenseTransformer, ModelOutput, sequence_logprob_sum, target_logit_and_logz
def compute_mtp_loss(
    model: DenseTransformer, mtp_hidden: tuple[torch.Tensor, ...] | None, input_ids: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy over the auxiliary multi-token-prediction heads.

    Head `d` (1-indexed) predicts the token d+1 positions ahead, so its labels are input_ids shifted
    left by d+1 with the tail padded to ignore_index — the same label-shifting trick compute_loss
    uses, for the same reason (never slice the logits).

    Heads are projected and reduced one at a time so only one (batch, seq, vocab_size) tensor is
    live in the loop, rather than materialising all of them and then reducing.
    """
    if not mtp_hidden:
        return input_ids.new_zeros((), dtype=torch.float32)

    total = None
    for depth, hidden in enumerate(mtp_hidden, start=1):
        shift = depth + 1
        labels = torch.cat(
            [input_ids[:, shift:], input_ids.new_full((input_ids.size(0), shift), -100)], dim=1
        )
        logits = model._project_logits(hidden)
        # Same single-pass formulation as compute_loss, for the same reason: F.cross_entropy is on
        # autocast's fp32 list, so it upcast this head's whole (batch, seq, vocab_size) tensor
        # before reducing it. Each head materialises one of those, so the saving is per head.
        head_loss, _ = _nll_and_logz(logits.view(-1, logits.size(-1)), labels.view(-1))
        total = head_loss if total is None else total + head_loss
    return total / len(mtp_hidden)


def _nll_and_logz(flat_logits: torch.Tensor, flat_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean NLL over non-ignored positions, and the mean of logsumexp(logits)^2, from ONE pass.

    The z-loss regulariser squares the same `logsumexp(x)` that cross-entropy's `target_logit`
    comes from; target_logit_and_logz (shared with model.dpo.sequence_logprob_sum, DPO's
    per-row analogue of this NLL) derives both from that single pass, which lets the whole thing
    stay in the compute dtype instead of tripping autocast's fp32 policy on log_softmax.
    """
    keep = flat_labels != -100
    # gather can't take -100, and those rows are masked out of both means below anyway.
    safe_labels = flat_labels.masked_fill(~keep, 0)
    target_logit, z = target_logit_and_logz(flat_logits, safe_labels)

    keep_f = keep.to(torch.float32)
    # clamp: an all-ignored batch is degenerate (it needs seq_len < 2), and 0 beats the nan
    # F.cross_entropy returns there, which would poison the accumulated gradient.
    n_kept = keep_f.sum().clamp(min=1)
    # Identical to F.cross_entropy(ignore_index=-100) with the default 'mean' reduction, which also
    # divides by the kept count rather than the total.
    nll = ((z - target_logit) * keep_f).sum() / n_kept
    return nll, (z.square() * keep_f).sum() / n_kept


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard causal-LM loss: token t's logits predict token t+1. Returns (lm_loss, z_loss).

    The shift is done on the *labels* (cheap, one int64 row) rather than by slicing the logits.
    Slicing logits[:, :-1] yields a non-contiguous view whose .contiguous()/.view() forces a full
    copy of the (batch, seq, vocab_size) tensor — the single largest activation in the model — on
    every forward. Padding the labels with ignore_index instead lets logits.view(-1, vocab_size)
    be a free reshape of the already-contiguous tensor, and cross_entropy skips the padded
    positions, giving a numerically identical mean over exactly the same (seq - 1) targets.

    z_loss is the log-Z regulariser mean(logsumexp(logits)^2) (PaLM/Chinchilla), returned
    separately so callers decide whether to apply it: train() adds cfg.model.z_loss_weight * z_loss
    to the training objective, while evaluate() discards it so val/loss stays a pure LM number and
    remains comparable across configurations — exactly how ponder_cost and moe_aux_loss are already
    handled.

    **Both losses are derived from one logsumexp rather than two passes over the logits**, which is
    what makes this the cheap version. cross_entropy is exactly `logsumexp(x) - x[label]`, and
    z_loss squares that same `logsumexp(x)` — so calling F.cross_entropy *and* torch.logsumexp
    reduced the largest tensor in the model twice over, once inside cross_entropy's fused
    log_softmax and once again for z. Computing `z` once and gathering the target logit off it
    collapses that to a single reduction, forward and backward.

    Wrap this in torch.compile (see `build_loss_fn`) before judging it: inductor fuses the whole
    thing into one pass and keeps the reduction in fp32 without ever materialising an fp32 copy of
    the (batch, seq, vocab_size) logits. Measured at batch 32 x seq 512 x vocab 50304, fwd+bwd:
    22.7 ms and +8.2 GB peak for the two-pass version, 17.0 ms eager here, **5.5 ms and +5.0 GB
    compiled** — and the compiled result is *closer* to an fp32 reference than the original
    (|dlm| 2.5e-5 vs 3.4e-3), because F.cross_entropy under autocast returned a bf16-rounded loss.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    # Reduced over the vocab first, then masked — rather than flat_logits[mask], which would
    # materialise a near-full copy of the largest activation in the model just to drop one row per
    # sequence. logsumexp is internally max-subtracted so it's stable in the autocast dtype.
    return _nll_and_logz(logits.view(-1, logits.size(-1)), labels.view(-1))


def compute_sft_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SFT analogue of compute_loss: the same causal shift, plus loss_mask shifted the same way
    and folded into the ignore positions, so only supervised (assistant-turn) tokens are scored.

    loss_mask is 1 at positions data.py's SFT pipeline marked as assistant-turn tokens (or the
    trailing EOS), 0 at prompt/user-turn tokens. Position i predicts input_ids[i+1], so it should
    be scored iff *that target* is supervised — i.e. iff loss_mask[i+1] == 1 — which is exactly
    what shifting loss_mask the same way labels are shifted gives.

    With an all-ones loss_mask this is bit-identical to compute_loss on the same inputs: it's a
    strict generalization, not a parallel reimplementation, and _nll_and_logz needs no change at
    all — it already treats -100 generically anywhere in the flat label tensor.
    """
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), -100)], dim=1)
    shifted_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    labels = labels.masked_fill(shifted_mask == 0, -100)
    return _nll_and_logz(logits.view(-1, logits.size(-1)), labels.view(-1))


def compute_dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The standard DPO objective (Rafailov et al. 2023): -E[logsigmoid(beta * (policy_logratio -
    ref_logratio))], mean over the batch. Each *_logp is a per-row (batch,) sequence-summed
    log-probability (see model.sequence_logprob_sum), not a mean, and the reference values are read
    from data.py's precomputed cache rather than a second live model.

    Also returns three no_grad diagnostics, logged but not optimized:
    - margin: the mean implicit reward margin (how much more the policy separates chosen from
      rejected than the reference did, in beta-scaled log-odds units).
    - margin_accuracy: fraction of the batch where the policy ranks chosen above rejected by *more
      than the reference did* (logits > 0) — this is what the loss's sigmoid argument's sign
      measures, so it can sit near 50% early in training even when the policy already ranks chosen
      above rejected in absolute terms, since it's relative to the (untrained) reference.
    - reward_accuracy: the conventional reference-independent pairwise accuracy
      (policy_chosen_logp > policy_rejected_logp) — the TRL-style number most DPO writeups mean by
      "accuracy".
    """
    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        margin_accuracy = (logits > 0).float().mean()
        reward_accuracy = (policy_chosen_logp > policy_rejected_logp).float().mean()
        margin = (
            beta * (policy_chosen_logp - ref_chosen_logp) - beta * (policy_rejected_logp - ref_rejected_logp)
        ).mean()
    return loss, margin, margin_accuracy, reward_accuracy


def compute_dpo_loss_from_logits(
    chosen_logits: torch.Tensor,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_logits: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuses sequence_logprob_sum (x2) + compute_dpo_loss into one function so build_dpo_loss_fn
    can torch.compile it as a single unit — the same reasoning build_loss_fn already applies to
    compute_loss/compute_sft_loss."""
    policy_chosen_logp = sequence_logprob_sum(chosen_logits, chosen_ids, chosen_mask)
    policy_rejected_logp = sequence_logprob_sum(rejected_logits, rejected_ids, rejected_mask)
    return compute_dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta)


def build_loss_fn(cfg: Config):
    """compute_loss (or, under cfg.sft.enabled, compute_sft_loss), compiled when the run is compiled.

    Separate from the model's own torch.compile because the loss lives outside DenseTransformer:
    it consumes the (batch, seq, vocab_size) logits the model returns, and that tensor is large
    enough that whether its reductions get fused is worth ~20% of step time on a small-d_model,
    large-vocab config. Tied to cfg.train.compile so the CPU sanity-check path stays eager.

    Not used for cfg.dpo.enabled runs — see build_dpo_loss_fn, which wraps a different-shaped
    function (compute_dpo_loss_from_logits) since DPO's loss needs both chosen and rejected logits
    plus the cached reference log-probs, not just one logits tensor and its own input_ids.
    """
    fn = compute_sft_loss if cfg.sft.enabled else compute_loss
    return torch.compile(fn) if cfg.train.compile else fn


def build_dpo_loss_fn(cfg: Config):
    """compute_dpo_loss_from_logits, compiled when the run is compiled. DPO analogue of
    build_loss_fn — kept separate rather than folded into it because the two functions take
    different-shaped arguments (one logits tensor + input_ids/loss_mask vs. two logits tensors +
    two input_ids/loss_mask pairs + two cached reference log-probs)."""
    return torch.compile(compute_dpo_loss_from_logits) if cfg.train.compile else compute_dpo_loss_from_logits


def forward_dpo_pair(model, chunk: dict) -> tuple[ModelOutput, torch.Tensor, torch.Tensor]:
    """Forward a DPO chunk's chosen+rejected sequences concatenated in one `model()` call, then
    split the output logits back apart.

    One concatenated forward rather than two separate ones keeps the DPO step's per-chunk kernel
    launches and activation shape at chosen+rejected's natural combined width. Shared by
    chunk_loss_and_metrics (training) and evaluate() (eval) since this forward mechanics is
    identical between them — only what each does with the split logits differs; see
    chunk_loss_and_metrics's docstring for why the two loss-*assembly* paths don't also share code.
    """
    c_ids, r_ids = chunk["chosen_input_ids"], chunk["rejected_input_ids"]
    b = c_ids.size(0)
    out = model(torch.cat([c_ids, r_ids], dim=0))
    chosen_logits, rejected_logits = out.logits.split(b, dim=0)
    return out, chosen_logits, rejected_logits


def chunk_loss_and_metrics(
    model, raw_model: DenseTransformer, cfg: Config, loss_fn, chunk: dict
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward one chunk and build its scalar loss plus the terms to log, for either mode.

    Called from inside the step loop's autocast block (and so must not be given a chunk from a
    different device or dtype context). Returns the differentiable total the caller weights and
    backwards, and the individual `_ACCUM_METRICS` terms it produced — the caller weights and
    accumulates those identically regardless of which branch built them, which is the whole point
    of returning them by name instead of as a fixed tuple.

    DPO's branch genuinely differs in shape: it forwards chosen+rejected concatenated in one call
    and splits the logits back, against a 9-argument loss. z_loss/mtp_loss are deliberately absent
    from its total — mtp_heads > 1 is already rejected by validate_post_training_config, and
    z_loss would need its own reduction pass over the concatenated logits that isn't needed for
    DPO correctness. ponder_cost/moe_aux_loss compose in for free, being zero scalars when their
    feature is off.
    """
    if cfg.dpo.enabled:
        c_ids, r_ids = chunk["chosen_input_ids"], chunk["rejected_input_ids"]
        out, chosen_logits, rejected_logits = forward_dpo_pair(model, chunk)
        lm_loss, margin, margin_accuracy, reward_accuracy = loss_fn(
            chosen_logits, c_ids, chunk["chosen_loss_mask"],
            rejected_logits, r_ids, chunk["rejected_loss_mask"],
            chunk["ref_chosen_logprob"], chunk["ref_rejected_logprob"], cfg.dpo.beta,
        )
        chunk_loss = (
            lm_loss
            + cfg.model.ponder_weight * out.ponder_cost
            + cfg.model.moe_aux_loss_weight * out.moe_aux_loss
        )
        extra = {
            "dpo_margin_accuracy": margin_accuracy,
            "dpo_reward_accuracy": reward_accuracy,
            "dpo_margin": margin,
        }
    else:
        input_ids = chunk["input_ids"]
        out = model(input_ids)
        lm_loss, z_loss = (
            loss_fn(out.logits, input_ids, chunk["loss_mask"])
            if cfg.sft.enabled
            else loss_fn(out.logits, input_ids)
        )
        mtp_loss = compute_mtp_loss(raw_model, out.mtp_hidden, input_ids)
        # Summed in this order deliberately: it is the order the two branches were written in
        # before they shared this function, so the refactor is bit-identical rather than merely
        # equivalent-to-tolerance.
        chunk_loss = (
            lm_loss
            + cfg.model.ponder_weight * out.ponder_cost
            + cfg.model.moe_aux_loss_weight * out.moe_aux_loss
            + cfg.model.z_loss_weight * z_loss
            + cfg.model.mtp_weight * mtp_loss
        )
        extra = {"z_loss": z_loss, "mtp_loss": mtp_loss}

    metrics = {
        "loss": chunk_loss,
        "lm_loss": lm_loss,
        "ponder_cost": out.ponder_cost,
        "mean_loop_depth": out.mean_loop_depth,
        "moe_aux_loss": out.moe_aux_loss,
        **extra,
    }
    return chunk_loss, metrics


def validate_post_training_config(cfg: Config) -> None:
    """Raise clear errors for post-training mode combinations train() can't handle, rather than
    letting them fail confusingly deep inside the data pipeline or the accumulation loop."""
    if cfg.sft.enabled and cfg.dpo.enabled:
        raise ValueError("sft.enabled and dpo.enabled are mutually exclusive — pick one post-training mode.")
    if (cfg.sft.enabled or cfg.dpo.enabled) and cfg.model.mtp_heads > 1:
        # compute_mtp_loss would need the same loss_mask treatment compute_sft_loss/compute_dpo_loss
        # got — mechanically similar (shift-by-depth+1 and fold the mask into -100) but not yet
        # built for either. Raise rather than silently score prompt/rejected tokens through the
        # auxiliary heads.
        raise ValueError("sft.enabled/dpo.enabled do not support model.mtp_heads > 1 yet — set mtp_heads: 1.")
    if cfg.dpo.enabled and not cfg.dpo.reference_checkpoint:
        raise ValueError("dpo.enabled requires dpo.reference_checkpoint to be set.")


def resolve_dpo_doc_mask(cfg: Config) -> None:
    """Turn model.doc_attention_mask off for a DPO run whose packing makes it a no-op.

    See config.doc_mask_is_inert_for_dpo for why it is a no-op (and for the loop_attn_windows
    exception that keeps it on). Worth doing for two reasons, the second larger than the first:
    the BlockMask is otherwise rebuilt on every training step and across the entire
    reference-logprob precompute pass for nothing, and doc_attention_mask is one of the three
    things that force resolve_compile_mode down to mode=None — so a DPO run that skips it gets
    CUDA graphs back.

    Mutates cfg rather than branching at each use, the same idiom auto_batch_size and
    tokens_per_param already use here, which also means the value saved into the checkpoint is
    the one the run actually trained with.
    """
    if not (cfg.dpo.enabled and doc_mask_is_inert_for_dpo(cfg.model)):
        return
    cfg.model.doc_attention_mask = False
    print(
        "[radiance] dpo.enabled: turning model.doc_attention_mask off — DPO packs one pair side "
        "per row, so the document mask cannot change any scored logit, and building it every "
        "step is pure overhead (see config.doc_mask_is_inert_for_dpo)."
    )


def note_dpo_z_loss_omitted(cfg: Config) -> None:
    """Surface that model.z_loss_weight has no effect under DPO, since it defaults nonzero.

    z_loss/mtp_loss are deliberately left out of the DPO chunk loss (see docs/post-training.md) —
    z_loss would need its own reduction pass over the concatenated chosen+rejected logits that
    DPO's correctness doesn't need, and mtp_heads > 1 is already rejected outright by
    validate_post_training_config. Unlike that rejection, z_loss_weight's default is nonzero
    (1e-4), so raising here would break every existing default DPO config for a term that was
    never meant to apply to DPO — a startup note keeps the omission visible without doing that.
    """
    if cfg.dpo.enabled and cfg.model.z_loss_weight != 0:
        print(
            f"[radiance] dpo.enabled: model.z_loss_weight={cfg.model.z_loss_weight} has no effect — "
            "z_loss is deliberately omitted from the DPO loss (see docs/post-training.md) and its "
            "metric logs a flat 0. Set model.z_loss_weight: 0 to silence this note."
        )
