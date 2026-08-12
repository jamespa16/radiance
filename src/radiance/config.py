from __future__ import annotations

from dataclasses import dataclass, field

import torch
import yaml


@dataclass
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    text_column: str = "text"
    tokenizer: str = "gpt2"
    seq_len: int = 512
    num_workers: int = 4
    cache_dir: str = ".cache/radiance/tokenized"
    streaming: bool = False
    shuffle_buffer_size: int = 1000
    disk_cache_max_gb: float | None = None
    disk_cache_shard_size: int = 100
    prefetch_factor: int = 2
    eval_split_size: int = 0


@dataclass
class ModelConfig:
    d_model: int = 256
    head_dim: int = 32  # n_heads = d_model // head_dim; d_model must divide evenly
    n_kv_heads: int | None = None  # number of K/V heads for GQA; None = n_heads (standard MHA, the
    # default). Each K/V head is shared by n_heads // n_kv_heads_resolved query heads. Must evenly
    # divide n_heads — see n_kv_heads_resolved.
    qk_norm: bool = True  # RMSNorm applied per-head to q/k (over head_dim) before RoPE, for training
    # stability across blocks[1:]'s weight-shared loop iterations.
    value_residual: bool = True  # mix each block's attention values with blocks[0]'s: v = lam * v +
    # (1 - lam) * v_first, with a learned per-block scalar `lam` initialised to 1.0. At init this is
    # therefore *exactly* the un-mixed model; the mixing only appears if training moves lam. Gives
    # every block (and, since blocks[1:] is a weight-shared loop, every iteration) direct access to
    # the first block's values (ResFormer / nanoGPT-speedrun).
    attn_out_gate: bool = True  # per-head sigmoid gate on the attention output before out_proj.
    # Written as 2 * sigmoid(Linear(x)) with the Linear zero-initialised, so the gate is exactly
    # 1.0 at init (a plain sigmoid cannot reach 1) and the model starts identical to an ungated one.
    mtp_heads: int = 1  # multi-token prediction: how many future tokens each position predicts.
    # 1 (default) is ordinary next-token prediction — exactly the previous behavior. Higher values
    # add auxiliary heads predicting t+2, t+3, ... which densifies the training signal per token
    # and lays the groundwork for speculative decoding. Left off by default despite this file's
    # on-by-default convention because the cost is real rather than nominal: each head materialises
    # its own (batch, seq, vocab_size) logits tensor — the largest activation in the model — so
    # defaulting to 2 would quietly halve the batch size auto_batch_size can fit.
    mtp_weight: float = 0.3  # coefficient on the averaged auxiliary-head loss
    z_loss_weight: float = 1.0e-4  # coefficient on the log-Z regulariser mean(logsumexp(logits)^2),
    # which keeps logit scale from drifting. Mirrors ponder_weight/moe_aux_loss_weight's role and
    # placement. Applied to the *training* loss only — evaluate()'s val/loss stays pure LM loss.
    # 0.0 disables. Matters more here than in a plain transformer because looping multiplies
    # effective depth without adding parameters.
    n_layers: int = 6
    loop_count: int = 1
    loop_iter_conditioning: str = "norm_gains"  # "norm_gains" (default), "lora", or "none". The
    # loop body is weight-shared, so without this iteration 5 is computationally indistinguishable
    # from iteration 1 except through residual-stream norm drift — the block has no idea how deep
    # into the recursion it is. "norm_gains" gives each loop iteration its own RMSNorm gains (the
    # adaLN trick) plus its own router biases, costing ~2 * loop_multiplier * d_model parameters per
    # block; "lora" additionally gives each iteration a rank-loop_lora_rank adapter on the attention
    # and FFN projections. Inert whenever the loop only runs once (loop_count: 1, the default),
    # since there is then exactly one iteration to condition on.
    loop_lora_rank: int = 8  # rank of the per-iteration adapters when loop_iter_conditioning="lora"
    loop_input_injection: bool = True  # re-inject the token embedding at the start of every loop
    # iteration after the first: h = h + W_inj @ x0. Without an anchor a deep recurrence drifts;
    # with one it trains stably to far higher iteration counts. W_inj is zero-initialised, so this
    # is exactly a no-op at init regardless of loop count and can only be learned into.
    loop_count_min: int | None = None  # stochastic loop depth: each training step samples the loop
    loop_count_max: int | None = None  # count uniformly from [loop_count_min, loop_count_max].
    # Both default to None, resolving to loop_count, which collapses the range to a point and makes
    # sampling inert. Set them to train one model across a range of depths, which both regularises
    # the recursion and lets inference dial compute up or down (radiance-generate --loops). Eval and
    # generation always use loop_count_max so their numbers stay deterministic. Note each distinct
    # sampled count compiles its own graph, so raise torch._dynamo.config.cache_size_limit (default
    # 8) if the range is wide — train.py does this automatically.
    loop_bptt_window: int | None = None  # backpropagate through only the last N loop iterations,
    # running earlier ones under no_grad. Activation memory becomes O(N) instead of O(loop_count) —
    # a stronger lever than grad_checkpoint on this axis, and composable with it. Off (full BPTT)
    # by default: truncating the gradient is an approximation you reach for deliberately, not a
    # free win like the zero-init features above.
    hyper_conn_streams: int = 1  # expansion rate n for hyper-connections (Zhu et al., ICLR 2025):
    # the residual stream is replaced by n parallel streams, and each sublayer learns which stream
    # to read, how to mix the streams with each other, and how to distribute its output back across
    # them. 1 (default) collapses this to exactly today's single residual stream and allocates no
    # parameters at all. Aimed squarely at the weight-shared loop: at loop_count 6 a 6-layer model
    # performs 62 residual writes into one accumulator, which is the regime the paper's
    # gradient-vanishing/representation-collapse argument is about — and giving the connection
    # matrices per-iteration variants (see loop_iter_conditioning) lets each pass learn its own
    # routing, so a stream an iteration doesn't write into carries information across the whole
    # recursion untouched. Costs n * the residual stream's activation memory, which is why it
    # stays at 1 by default rather than defaulting on like the free/inert features above — the
    # same reasoning that keeps mtp_heads at 1.
    hyper_conn_dynamic: bool = True  # additionally condition the connection weights on the hidden
    # state: coeff = static + s * tanh(norm(H) @ W), with W zero-initialised. Exactly bit-identical
    # to static hyper-connections at init, and free at n=1 where no hyper-connections exist at all.
    use_router: bool = False  # opt-in: replace fixed loop_count with per-token ACT halting
    max_loops: int = 6  # hard cap on loop iterations when use_router=True; independent of loop_count
    ponder_weight: float = 1.0e-2  # tau: coefficient on the ponder-cost loss term
    halt_epsilon: float = 0.01  # ACT epsilon: a position halts once cumulative halting prob >= 1 - halt_epsilon
    act_capacity_ratio: float = 1.0  # fraction of each sequence's positions that an interior ACT
    # loop iteration actually computes — attention *and* FFN, i.e. the whole block. This is what
    # makes router mode save wall-clock: below 1.0, iterations past the first process only the
    # highest-priority still-running positions, and everything else is carried forward unchanged
    # while its keys/values are served from the previous iteration's retained store.
    #
    # It is an approximation, not an exact speedup, and knowing why matters. Only the *first* block
    # of the loop body has genuinely invariant K/V for a halted position (K and V are per-position
    # projections of frozen_x, which stops changing). Later blocks' K/V do drift, because block 1's
    # output at a halted position mixes in attention over still-running positions that keep
    # evolving — see tests/test_act_kv_invariance.py. Reusing them is the same class of
    # approximation act_ffn_capacity_ratio already made, extended to the whole block.
    #
    # 1.0 (default) disables it entirely, leaving the loop byte-for-byte identical to the dense
    # implementation. Incompatible with grad_checkpoint (see DenseTransformer.__init__).
    act_ffn_capacity_ratio: float = 1.0  # the older, narrower version of the above: sparsifies only
    # the FFN sublayer and leaves attention fully dense, so it saves much less. Superseded by
    # act_capacity_ratio; kept so existing configs keep working. Setting both below 1.0 raises.
    ffn_mult: float = 4.0  # ffn_dim = round(d_model * ffn_mult)
    ffn_depth: int = 2
    use_moe: bool = False  # opt-in: blocks[1:] (the shared loop body) use MoEFeedForward instead of
    # FeedForward; blocks[0] is unaffected and always stays dense — see moe_dense_every for keeping
    # some of blocks[1:] dense too.
    n_experts: int = 8  # experts per MoE FFN layer; only used when use_moe=True
    moe_top_k: int = 2  # experts activated per token (Mixtral-style weighted top-k, not Switch top-1)
    moe_capacity_factor: float = 1.25  # per-expert capacity = round(capacity_factor * n_tokens *
    # moe_top_k / n_experts); tokens routed to an already-full expert are dropped (zero contribution
    # from that expert — see MoEFeedForward).
    moe_aux_loss_weight: float = 1.0e-2  # coefficient on the load-balancing aux loss term; mirrors
    # ponder_weight's role/placement for ACT's ponder cost. Only used when moe_balance includes the
    # aux-loss term.
    moe_balance: str = "both"  # how expert load is balanced: "aux_loss" (the gradient-based term
    # above, the previous behavior), "bias" (a non-learned per-expert bias on the routing logits,
    # nudged after each optimizer step toward whichever experts are under-loaded), or "both"
    # (default, DeepSeek-V3's actual configuration). The bias term balances load without adding a
    # gradient that competes with the LM objective, which is what the aux loss alone does.
    moe_bias_update_rate: float = 1.0e-3  # step size for the "bias"/"both" balancing update
    moe_balance_signal: str = "count"  # what the "bias"/"both" balancing rule drives toward
    # uniformity: "count" (default, DeepSeek-V3's rule — one per token routed to the expert) or
    # "weight" (the token's gate weight, i.e. how much of its output the expert actually supplies).
    # They differ for an expert chosen often but weakly, which "count" calls loaded and "weight"
    # calls idle; gradient reaching an expert scales with the latter. The update is sign-based, so
    # the two are interchangeable without retuning moe_bias_update_rate. Does not touch the aux
    # loss, whose f_i is a count fraction by the Switch formulation's own definition.
    moe_counterfactual_weight: float = 0.0  # coefficient on the counterfactual routing signal:
    # extra router gradient derived from the expert outputs the fixed-capacity dispatch *already*
    # computes for tokens it then discards (see MoEFeedForward._counterfactual_probe_signal). Zero
    # (default) leaves the router trained exactly as before. This is the one MoE knob that is not
    # defaulted on despite being free in the forward — it changes gradients, and nothing has A/B'd
    # it yet; 1.0 is the natural scale to start from (it makes a probe's push exactly as strong per
    # unit of utility as the true gradient on a chosen expert's gate weight).
    moe_n_shared: int = 1  # always-on expert(s) added to every token's FFN output alongside the
    # routed ones (DeepSeekMoE). Absorbs the computation every token needs, so the routed experts
    # are free to specialise instead of each re-learning the common case. 0 disables.
    moe_shared_ffn_mult: float = 1.0  # shared expert width as a fraction of ffn_dim
    moe_expert_ffn_mult: float | None = None  # width of each routed expert as a fraction of
    # ffn_dim. None (default) means ffn_dim, i.e. unchanged. Set below 1.0 to make experts
    # fine-grained: n_experts: 32, moe_top_k: 8, moe_expert_ffn_mult: 0.25 activates the same
    # parameter count per token as 8 experts at top_k 2, but from 4x as many combinations. The
    # batched baddbmm dispatch keeps step time nearly flat in expert count, which is what makes
    # this affordable.
    moe_eval_full_capacity: bool = True  # outside training, size expert capacity to the actual
    # per-expert load instead of the moe_capacity_factor formula, so no token is ever dropped.
    # Capacity limits exist to bound *training* throughput and memory; at eval they only discard
    # computation, and because the limit scales with the token count they made a token's output
    # depend on how many other tokens were passed alongside it — the same prompt scored differently
    # in a batch than alone, and incremental decoding diverged from a full forward.
    moe_dense_every: int | None = None  # opt-in: every Nth block (1-indexed by position within
    # blocks[1:]) uses a plain dense FeedForward instead of MoEFeedForward even when use_moe=True.
    # None (default) means every block in blocks[1:] is MoE.
    mup_base_d_model: int | None = None  # muP: the width this config's hyperparameters were tuned
    # at. None (default) resolves to d_model itself, making the width multiplier exactly 1.0 and
    # every muP correction a no-op — so this is on by default yet changes nothing until you sweep
    # d_model away from the base. Set it once (to the small proxy width you tuned lr at) and the
    # init scale, per-tensor LRs and output logit scale all track d_model automatically, instead of
    # the optimal lr silently moving every time the model gets wider. See mup_width_mult.
    dropout: float = 0.1
    max_seq_len: int = 512
    rope_theta: float = 10000.0  # RoPE base frequency (Su et al. 2021)
    doc_attention_mask: bool = True  # mask attention at document boundaries. data.py packs many
    # documents into each seq_len block joined by EOS, and plain causal attention lets every token
    # attend back across those joins into unrelated text — so the model spends capacity predicting
    # one document's tokens from another's. Document ids are recovered in-model from the EOS
    # positions (no data-pipeline or cache change), and the mask is built once per forward with
    # torch's flex_attention and reused across every block and loop iteration. Automatically
    # disabled without CUDA (flex_attention is impractically slow on CPU) and during generation
    # (a single prompt is one document), both of which fall back to the plain SDPA path.
    loop_attn_windows: list[int] | None = None  # opt-in: per-iteration attention window sizes, e.g.
    # [128, 128, 512, 512] to make the early loop passes local and the later ones global. Indexed by
    # loop iteration and clamped at the end of the list. A knob that only exists because the
    # architecture loops — the same weights get a different receptive field on each pass. Requires
    # doc_attention_mask (it rides the same flex_attention BlockMask). None = every pass is global.

    grad_checkpoint: bool = False  # opt-in: recompute each block's activations during backward instead
    # of storing them. Trades ~20-30% throughput for a large drop in activation memory, and it pays off
    # disproportionately here because blocks[1:] is re-run loop_count/max_loops times per forward with
    # every pass retaining its own activations — see DenseTransformer.forward. Training-only (a no-op
    # under eval/no_grad/kv-cache); raise batch_size or target_effective_batch_size to spend the memory
    # it frees.
    vocab_pad_multiple: int = 128  # round the tokenizer's vocab up to a multiple of this for the
    # token_emb/lm_head matmuls (see model.padded_vocab_size). The padding rows are unreachable by
    # any tokenizer id, so this is behavior-preserving; it just keeps the model's largest matmul on
    # a tensor-core tile boundary. Set to 1 to disable. Defaults on (like qk_norm/auto_batch_size,
    # and unlike this file's usual opt-in-False convention) since it's a pure throughput win.

    @property
    def n_heads(self) -> int:
        if self.d_model % self.head_dim != 0:
            raise ValueError(f"model.d_model ({self.d_model}) must be divisible by model.head_dim ({self.head_dim})")
        return self.d_model // self.head_dim

    @property
    def n_kv_heads_resolved(self) -> int:
        n_heads = self.n_heads  # triggers d_model % head_dim validation
        n_kv_heads = self.n_kv_heads if self.n_kv_heads is not None else n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"model.n_heads ({n_heads}) must be divisible by model.n_kv_heads ({n_kv_heads})")
        return n_kv_heads

    @property
    def ffn_dim(self) -> int:
        return round(self.d_model * self.ffn_mult)

    @property
    def moe_expert_dim(self) -> int:
        """Hidden width of one routed expert. Defaults to ffn_dim (a dense FFN's width)."""
        if self.moe_expert_ffn_mult is None:
            return self.ffn_dim
        return max(1, round(self.ffn_dim * self.moe_expert_ffn_mult))

    @property
    def moe_shared_dim(self) -> int:
        """Hidden width of the always-on shared expert."""
        return max(1, round(self.ffn_dim * self.moe_shared_ffn_mult))

    @property
    def loop_multiplier(self) -> int:
        """The maximum number of times the loop body (blocks[1:]) can run in one forward pass.

        The single source of truth for everything that must be sized to the *worst case* rather
        than the typical one: the KV cache's slot count, the depth-scaled residual init, and the
        activation-memory estimate. Under stochastic depth the actual count varies per step, so
        each of those must use loop_count_max, not loop_count.
        """
        if self.use_router:
            return self.max_loops
        return self.loop_count_max or self.loop_count

    @property
    def mup_width_mult(self) -> float:
        """d_model relative to the width the config's hyperparameters were tuned at.

        Exactly 1.0 when mup_base_d_model is unset, which makes every muP correction an identity —
        that's what lets muP default on without perturbing any existing config.

        Note what is deliberately *not* corrected: the attention logit scale stays 1/sqrt(head_dim)
        rather than becoming 1/head_dim. muP's 1/d attention scale applies when the head dimension
        grows with width; here head_dim is a fixed config constant and width grows by adding heads
        (n_heads = d_model // head_dim), so each q·k dot product is a sum over a fixed number of
        terms and its variance does not grow with d_model.
        """
        base = self.mup_base_d_model or self.d_model
        return self.d_model / base


@dataclass
class TrainConfig:
    batch_size: int = 32  # micro-batch size: what one forward/backward pass consumes
    grad_accum_steps: int = 1  # micro-batches (of batch_size each) accumulated per optimizer.step();
    # effective_batch_size = batch_size * grad_accum_steps. Raise this instead of batch_size to grow the
    # effective batch beyond what fits in VRAM.
    lr: float = 1.0e-2  # AdamW's LR. Under optimizer="muon" this governs only the tensors AdamW
    # still owns (embeddings, norm gains, biases, routers/gates) — see optim.build_muon_param_groups.
    #
    # Was 3.0e-4, which is what it had been tuned to back when AdamW trained *every* tensor. Moving
    # the hidden weights to Muon preserved that value but silently changed its job, leaving the tied
    # embedding — the one large matrix AdamW still owns — training ~30-100x too slowly. A sweep on
    # configs/tinystories.yaml bottoms out flat between 1.0e-2 and 3.0e-2 and is worth ~0.44 val/loss
    # at 1200 steps (2.2341 -> 1.7984), better at every eval point; see "Measured results" in
    # CLAUDE.md. This is one of the few defaults here that changes results deliberately, and unlike
    # the rest of them it is a fix, not a feature. Note it only reaches configs that *omit* lr —
    # most configs in configs/ pin it, and pinning 3.0e-4 restores the old behavior exactly.
    optimizer: str = "muon"  # "muon" (default) or "adamw". Muon orthogonalises the momentum update
    # of every hidden weight matrix via a Newton-Schulz iteration, which lets it take a much larger
    # step than AdamW at equal stability; embeddings/norms/biases stay on an auxiliary AdamW. Set
    # to "adamw" for the previous behavior.
    muon_lr: float = 0.02  # LR for the Muon group only. Deliberately a separate field rather than a
    # multiplier on `lr`: Muon's normalised update wants a ~50x larger LR, so folding the two
    # together would silently invalidate every config's tuned `lr`.
    embed_lr: float | None = None  # LR for the tied token_emb/lm_head matrix only. None (default)
    # resolves to `lr`, collapsing this to exactly the previous single-LR behavior — the
    # range-collapsed inert default this file's convention asks for. Split out because `lr` was
    # tuned when AdamW owned *every* tensor: moving the hidden weights to Muon left the embedding
    # as the one large matrix still on AdamW, and the value that was right for "all of it" is far
    # too small for "just the embedding". The embedding also wants a different LR from the other
    # tensors AdamW still holds (norm gains, biases, routers, gates), whose scale is load-bearing —
    # hence its own field rather than simply raising `lr`. See embed_lr_resolved.
    hyper_conn_lr: float | None = 1.0e-3  # LR for the hyper-connection coefficients only (see
    # model.hyper_conn_streams). None resolves to `lr`; unlike embed_lr this does *not* default to
    # None, because at `lr`'s post-Muon value of 1e-2 hyper-connections are not merely mistuned but
    # actively destructive, and a feature whose default setting breaks it is not a usable default.
    #
    # The reason is that AdamW's update magnitude is ~lr per step almost regardless of gradient
    # scale, so 400 steps at 1e-2 can move a coefficient by O(1) — and these coefficients are
    # *structural*: alpha_m starts one-hot, alpha_r the identity, beta all-ones. A drift of O(1)
    # there does not refine the routing, it erases it (the read becomes a blend of every stream and
    # the depth mix becomes arbitrary), whereas the same drift on an RMSNorm gain is just a
    # rescale. This is the same lesson embed_lr records from the other direction: `lr` reaches a
    # grab-bag of tensors whose ideal step sizes differ by orders of magnitude, and the fix is a
    # separate field rather than a compromise value. Measured: see the sweep in CLAUDE.md.
    muon_momentum: float = 0.95
    weight_decay: float = 0.01
    warmup_ratio: float = 0.04  # warmup_steps = round(max_steps * warmup_ratio)
    min_lr_ratio: float = 0.1  # the schedule decays to min_lr_ratio * lr, not to 0 — the tail of a
    # run at a ~0 LR contributes nothing. 0.0 restores a decay-all-the-way-to-zero schedule.
    lr_schedule: str = "cosine"  # "cosine" (default) or "wsd" (warmup-stable-decay: hold at full lr,
    # then decay only over the last wsd_decay_ratio of the run). Unlike most additions in this file
    # this stays *off* by default: it is not a bug fix, and switching it would silently reshape the
    # LR trajectory of every existing config whose lr was tuned against cosine. WSD's advantage is
    # operational — a run can be branched or extended from a mid-training checkpoint without
    # invalidating the schedule, which cosine cannot do since its shape depends on max_steps.
    wsd_decay_ratio: float = 0.2  # only used when lr_schedule == "wsd": fraction of max_steps spent
    # in the final decay phase.
    max_steps: int = 5000  # ignored (overwritten once the model is built) if tokens_per_param is set
    tokens_per_param: float | None = None  # opt-in: derive max_steps from model size instead of a fixed step
    # count — max_steps = round(tokens_per_param * num_active_parameters / (effective_batch_size *
    # data.seq_len)), computed in train.py once the model is built (num_active_parameters excludes
    # unused MoE expert params when model.use_moe is set). Chinchilla-optimal is ~20 tokens/param.
    auto_batch_size: bool = True  # overwrite batch_size/grad_accum_steps at startup, computed from free
    # VRAM + model size (see train.py's estimate_batch_size) instead of the values configured above.
    # Defaults to True — a deliberate behavior change for every existing config, not the usual
    # default-False opt-in convention (contrast use_router/use_moe) — since it only ever makes the
    # actual micro-batch size *safer* than a hand-picked one, never bigger, and it's what gates the OOM
    # backoff below. Set to False for a manually-chosen/swept batch_size to behave exactly as configured
    # (e.g. a sweep that's already tuning batch_size itself). CUDA-only: on CPU/MPS it's a no-op (prints
    # a note and keeps the configured batch_size/grad_accum_steps) since estimate_batch_size only knows
    # how to read free VRAM. Also enables OOM backoff during training: a CUDA OOM shrinks the internal
    # per-forward-pass chunk size and retries the step instead of ending the run (see train.py's main
    # loop) — this backoff never fires when auto_batch_size is False.
    target_effective_batch_size: int | None = None  # the effective batch size auto_batch_size solves
    # for (grad_accum_steps = ceil(target_effective_batch_size / computed batch_size)). None (default)
    # falls back to whatever effective_batch_size the configured batch_size/grad_accum_steps already
    # imply, so an existing config's effective batch size is preserved even as auto_batch_size splits it
    # differently across batch_size/grad_accum_steps to fit VRAM. Set explicitly to target a different
    # effective batch size than batch_size * grad_accum_steps would otherwise imply.
    vram_safety_margin: float = 0.5  # only used when auto_batch_size is True: fraction of the (already
    # conservative) estimated max token budget to actually use. Lower = more conservative.
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    eval_max_batches: int | None = 50  # cap on batches per evaluate() call. Uncapped, each eval walks
    # the whole validation split (unbounded for a streaming one) every eval_every steps; a fixed count
    # also keeps val/loss comparable across configs with different val split sizes. None = full pass.
    save_every: int = 1000
    output_dir: str = "checkpoints/run"
    resume_from: str | None = None  # opt-in: path to a checkpoint to continue training from, or the
    # literal "auto" to pick the highest-numbered step_*.pt in output_dir (so an interrupted run can be
    # relaunched with the same config unchanged). Restores model + optimizer moments + LR schedule +
    # GradScaler, so the run continues rather than restarting AdamW from zero momentum at warmup LR.
    # The DataLoader position is *not* restored — a resumed run revisits some examples. None = fresh run.
    seed: int = 42
    device: str = "auto"
    compile: bool = True
    dtype: str = "fp32"

    @property
    def embed_lr_resolved(self) -> float:
        """The tied embedding's LR, falling back to `lr` when embed_lr is unset.

        Exactly `lr` by default, which is what makes the separate embedding group inert for every
        config that hasn't set it — the group is still built, but at the same LR it had when it
        shared AdamW's decayed group, so the update is unchanged.
        """
        return self.lr if self.embed_lr is None else self.embed_lr

    @property
    def hyper_conn_lr_resolved(self) -> float:
        """The hyper-connection coefficients' LR, falling back to `lr` when unset.

        Only ever consulted when model.hyper_conn_streams > 1, since no such parameters exist
        otherwise — so this field is inert for every config that hasn't enabled them, despite not
        defaulting to None.
        """
        return self.lr if self.hyper_conn_lr is None else self.hyper_conn_lr

    @property
    def warmup_steps(self) -> int:
        return round(self.max_steps * self.warmup_ratio)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass
class WandbConfig:
    project: str = "radiance"
    entity: str | None = None
    mode: str = "online"


@dataclass
class Config:
    run_name: str = "radiance-run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def resolve_device(device: str) -> str:
    """Resolve "auto" to whatever accelerator is actually available, cuda > mps > cpu."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_dtype(dtype: str) -> torch.dtype:
    """Map a config dtype string ("fp32", "fp16", "bf16") to its torch.dtype."""
    if dtype not in _DTYPES:
        raise ValueError(f"Unknown train.dtype {dtype!r}, expected one of {sorted(_DTYPES)}")
    return _DTYPES[dtype]


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        run_name=raw.get("run_name", Config.run_name),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
    )
