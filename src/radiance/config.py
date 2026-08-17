from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass

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
    use_diff_attn: bool = False  # opt-in: Differential Attention (Ye et al. 2024) — each head
    # computes two softmax attention maps at half head_dim each (Q1K1, Q2K2) and takes a learned
    # difference (A1 - lambda*A2) @ V, which cancels common-mode "attention noise" the two maps
    # share. Reuses qkv_proj's existing width unchanged (two half-width Q heads sum back to
    # d_model, two half-width K heads sum back to kv_dim), so this is not a FLOP/param increase on
    # the projection — only how its output is chunked changes. Unlike qk_norm/value_residual/
    # attn_out_gate there is no zero/identity init under which this reduces to the same
    # *computation* as standard attention (splitting head_dim is structural), so — like use_moe/
    # use_router/n_kv_heads, and like the removed use_nsa — it is opt-in and evaluated by A/B
    # rather than defaulted on. Requires head_dim % 4 == 0 (each branch's head_dim // 2 must itself
    # be even for RoPE's pairwise rotation) and is incompatible with act_capacity_ratio < 1.0 (no
    # sparse/gathered variant of this attention path yet) — both raise in DenseTransformer.__init__.
    fp4_linear: bool = False  # opt-in: run every hidden linear's three GEMMs (forward, dgrad,
    # wgrad) in NVFP4 — 4-bit e2m1 operands with a per-16-element e4m3 block scale, on Blackwell's
    # FP4 tensor cores. Master weights, gradients and optimizer state all stay fp32 and autocast
    # still runs in train.dtype; only the GEMM operands are 4-bit. See nvfp4/.
    #
    # Opt-in rather than default-on, and this file's own convention says why: features default on
    # because their parameters default to an *inert* setting (zero-init, identity-valued, or
    # range-collapsed). FP4 has no such setting — there is no configuration under which quantizing
    # every hidden matmul to 4 bits is bit-identical to bf16. Structurally the same category as
    # use_diff_attn and the removed use_nsa. And "inert is not the same as free" applies twice
    # over: it measures 0.34x at d_model 256, so defaulting it on would make tinystories.yaml
    # three times *slower*. Requires d_model, the qkv width and ffn_dim to be multiples of 128,
    # and is incompatible with use_moe — both raise in DenseTransformer.__init__.
    fp4_grad_gemms: bool = True  # run dgrad and wgrad in FP4 too. False keeps the backward in
    # bf16, which is the lowest-risk arm and the control that separates "FP4 forward is fine, the
    # gradient path is what hurts" from "FP4 hurts". Only consulted when fp4_linear is on.
    fp4_stochastic_rounding: bool = True  # stochastic rounding on gradient tensors only.
    # Round-to-nearest is a biased estimator and the bias accumulates coherently over a run; SR
    # trades it for variance, which averages out over a batch. Weights and activations have no
    # accumulation to protect, so they stay round-to-nearest.
    fp4_hadamard: bool = True  # random Hadamard rotation on the wgrad GEMM's operands, to spread
    # outliers before the per-16 block amax picks a scale. Free: the rotation group and the scale
    # block are the same 16 elements, so it happens in-register on the tile the reduction is about
    # to consume. Both operands must get the same rotation — see nvfp4._FP4LinearFn.
    fp4_save_activations: bool = True  # save each quantized linear's activation for backward in
    # its packed NVFP4 form (1.125 bytes/element) instead of bf16 (2). **Bit-identical**, not an
    # approximation: the backward already quantized exactly this tensor with exactly these
    # parameters, so all that changes is when. That is why it defaults on where nothing else in the
    # fp4_* family does. Skipped automatically when fp4_grad_gemms is off, since a bf16 wgrad needs
    # the real activation back.
    fp4_keep_bf16_blocks: int = 0  # keep the last k *structural* blocks in bf16. blocks[1:] is a
    # weight-shared loop body, so a bf16 block is bf16 in every loop iteration — precision cannot
    # vary per-iteration when the weights are shared, and there is deliberately no per-iteration
    # variant of this knob.
    fp4_keep_bf16_first: bool = False  # keep blocks[0] in bf16. It runs once per forward rather
    # than loop_multiplier times, so it is the cheapest block to keep in high precision, and it
    # produces v_first and the injection anchor that the rest of the recursion depends on.
    fp4_lm_head: bool = True  # quantize the LM head too.
    #
    # These three default to *maximum* FP4 coverage, and the justification is throughput only:
    # full coverage measured 1.20x bf16 end-to-end against 1.16x for the conservative setting
    # (fwd+bwd 1.24x vs 1.18x). **The quality side is unmeasured**, and both of the settings this
    # departs from exist for real reasons — NVIDIA's own recipe keeps the final blocks in high
    # precision, and lm_head is weight-tied to token_emb, the tensor the embed_lr decomposition
    # attributes ~93% of the largest quality win recorded in this repo to. So if an FP4 run
    # regresses on val/loss, turn these back on in this order: fp4_lm_head first (it is the one
    # touching the embedding's gradient), then fp4_keep_bf16_blocks: 1, then fp4_keep_bf16_first.
    # They cost ~4% of throughput between them, which is cheap insurance against a quality loss.
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
    tokens_per_param: float | None = None  # opt-in: derive max_steps from model size instead of a fixed
    # step count — max_steps = round(tokens_per_param * num_active_parameters /
    # (effective_batch_size * resolved_row_tokens)), computed in train.py once the model is built
    # (num_active_parameters excludes unused MoE expert params when model.use_moe is set, and
    # resolved_row_tokens honors any active sft.seq_len / dpo.seq_len override plus DPO's
    # chosen+rejected width). Chinchilla-optimal is ~20 tokens/param.
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
    init_from: str | None = None  # opt-in: path to a checkpoint to seed *model weights only* from,
    # for starting a new run (e.g. SFT) on top of a previously trained model. Unlike resume_from,
    # the optimizer/scheduler/step are NOT restored — this run gets a fresh optimizer/scheduler
    # from its own TrainConfig and starts at step 0. Ignored whenever resume_from finds a
    # checkpoint (resuming an interrupted run of *this* config takes priority). None = fresh init.
    seed: int = 42
    device: str = "auto"
    compile: bool = True
    dtype: str = "fp32"
    native_bf16: bool = False  # opt-in: store parameters, gradients and optimizer moments in bf16
    # instead of the standard fp32-master-weights recipe. `dtype` alone only controls the
    # autocast compute dtype (train.py's forward/loss pass); params/grads/AdamW's exp_avg/
    # exp_avg_sq stay fp32 regardless, so a `dtype: bf16` config saves activation memory only,
    # not the ~12 bytes/param (grad + 2 Adam buffers, all fp32) that dominate a large model's
    # *static* VRAM footprint. This flag halves that: bf16 params/grads/moments cost 2+2+2+2=8
    # bytes/param against fp32's 4+4+4+4=16. Only buffers (RoPE's cos/sin cache, MoE's
    # expert_bias) stay fp32 — they carry no optimizer state, so casting them would buy no
    # memory and would cost precision RMSNorm-style upcasting doesn't protect (a bias nudged by
    # moe_bias_update_rate=1e-3 needs more than bf16's ~3 decimal digits to keep accumulating).
    #
    # No inert setting exists here — like fp4_linear (see model.fp4_linear's docstring, "reason
    # 6"), this is a real numerical-accuracy tradeoff, not a mathematical no-op, so it defaults
    # off rather than on despite being free. Muon's momentum buffer and the built-in
    # torch.optim.AdamW path already allocate state matching each parameter's own dtype, so they
    # need no code change; MuonWithAuxAdam's hand-written auxiliary AdamW groups hardcoded fp32
    # state and are fixed in optim.py to follow suit. Both tiers of CPU-offload OOM recovery
    # deliberately keep upcasting to fp32 on migration regardless — that path only triggers once
    # VRAM is already tight enough to need it, and CPU host memory is the resource under
    # pressure there, not the halved footprint this flag buys. Requires `dtype: "bf16"` (raises
    # otherwise — fp16's narrow exponent range
    # makes native storage without a master copy meaningfully riskier, and "fp32" storage under
    # a non-bf16 compute dtype doesn't match what this flag promises) and is incompatible with
    # model.fp4_linear (FP4's own quality margin assumes fp32 masters — see nvfp4/linear.py).
    # Needs empirical validation (a loss-curve comparison against the fp32-master baseline)
    # before trusting it for a real run — see docs/optim.md.

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
class SFTConfig:
    """Post-training: supervised fine-tuning on chat/instruction data.

    A mode switch, not an inert feature — enabling it swaps the entire data pipeline and loss
    function, so it follows the use_moe/use_router precedent (explicit opt-in bool) rather than
    this file's usual default-on-but-inert convention. See train.TrainConfig.init_from for
    seeding the model from a pretrained checkpoint before this run starts.
    """

    enabled: bool = False
    dataset: str | None = None  # HF `user/dataset`-style instruction dataset. Required when enabled.
    messages_column: str = "messages"  # list[{"role": ..., "content": ...}] per example — the
    # standard shape for chat-formatted HF datasets (e.g. HuggingFaceH4/no_robots). Ignored if
    # instruction_column is set instead.
    instruction_column: str | None = None  # Alpaca-style fallback for datasets with separate
    input_column: str | None = None  # instruction/input/output columns rather than a messages
    output_column: str | None = None  # column. When instruction_column is set, a 2-turn
    # [{"role": "user", ...}, {"role": "assistant", ...}] list is built from these three columns
    # (input_column may be empty-string per row) instead of reading messages_column.
    seq_len: int | None = None  # None (default) resolves to data.seq_len, the range-collapsed
    # inert default this file's convention asks for — packed SFT blocks are the same width as
    # pretraining blocks unless told otherwise.
    cache_dir: str = ".cache/radiance/sft"
    eval_split_size: int = 0  # same semantics as data.eval_split_size, applied to the SFT dataset.
    user_prefix: str = "\n\nUser: "  # plain text turn markers — not new tokens, so they tokenize
    assistant_prefix: str = "\n\nAssistant: "  # through the existing vocab with zero model changes.


@dataclass
class DPOConfig:
    """Post-training: Direct Preference Optimization on (prompt, chosen, rejected) triples.

    A second mode switch alongside SFTConfig, mutually exclusive with sft.enabled (train.py raises
    if both are set) — swaps the data pipeline and loss function the same way sft.enabled does, and
    reuses the rest of train() (optimizer, LR schedule, auto_batch_size, checkpointing, init_from)
    identically. Unlike SFT, DPO's loss needs a frozen reference model's log-probabilities on the
    same sequences; those are precomputed once during data prep and cached on disk alongside the
    tokenized dataset (see data.py's DPO section), so training itself only ever holds the policy
    model in memory — no second live model, no VRAM doubling.
    """

    enabled: bool = False
    dataset: str | None = None  # HF `user/dataset`-style preference dataset. Required when enabled.
    prompt_column: str | None = None  # None (default): chosen_column/rejected_column are full
    # [{"role", "content"}, ...] message lists that already include the prompt turn (e.g.
    # argilla/dpo-mix-7k). Set: chosen_column/rejected_column are plain completion strings, combined
    # with prompt_column (+ optional system_column) into a shared 1-turn user prompt (e.g.
    # Intel/orca_dpo_pairs). Mirrors sft.instruction_column's fallback precedent.
    system_column: str | None = None  # only consulted when prompt_column is set.
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    seq_len: int | None = None  # None (default) resolves to data.seq_len, same range-collapsed
    # inert default convention as sft.seq_len.
    cache_dir: str = ".cache/radiance/dpo"
    eval_split_size: int = 0  # same semantics as data.eval_split_size, applied to the DPO dataset.
    beta: float = 0.1  # DPO temperature / KL-regularization strength (Rafailov et al. 2023).
    reference_checkpoint: str | None = None  # path to a frozen checkpoint whose log-probs anchor
    # the DPO loss. Required when enabled. Often, but not necessarily, the same checkpoint
    # train.init_from seeds the policy from — kept as a separate field since data prep (which needs
    # this) and the train() call (which needs init_from) are different code paths.
    reference_batch_size: int = 32  # batch size for the one-time reference-logprob precompute pass.
    user_prefix: str = "\n\nUser: "  # same plain-text turn-marker convention as sft.user_prefix.
    assistant_prefix: str = "\n\nAssistant: "


def _backfill_missing_fields(obj) -> None:
    """Restore, one field at a time, dataclass fields this pickle predates.

    pickle reconstructs an instance by updating its ``__dict__`` — ``__init__`` never runs —
    so a checkpoint saved before field ``x`` existed has no ``x`` attribute at all: the
    ``default``/``default_factory`` machinery only fires inside ``__init__``. Reading such a
    field then raises ``AttributeError`` instead of falling back to the default (a
    pre-``sft``/``dpo`` checkpoint would AttributeError in ``format_chat_prompt`` and
    ``validate_post_training_config``). A missing field is treated the way a YAML config
    treats an omitted key — with the field's *current* default (see ``load_config``'s
    ``raw.get("sft", {})``) — and the walk recurses into nested dataclasses, which pickle
    restores as separate objects each with their own gaps. A field with no default at all
    predates the schema by more than a default can absorb; there is nothing to restore, so
    fail loudly instead of pretending.
    """
    for f in fields(obj):
        if f.name not in obj.__dict__:
            if f.default is not MISSING:
                setattr(obj, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(obj, f.name, f.default_factory())
            else:
                raise ValueError(
                    f"unpickled {type(obj).__name__} is missing field {f.name!r}, which has no "
                    "default to restore it from: the checkpoint predates this schema. Re-save "
                    "the checkpoint with current code."
                )
        nested = getattr(obj, f.name)
        if is_dataclass(nested) and not isinstance(nested, type):
            _backfill_missing_fields(nested)


@dataclass
class Config:
    run_name: str = "radiance-run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    @property
    def resolved_seq_len(self) -> int:
        """The packed-block width the active data pipeline actually uses.

        ``data.seq_len`` is the pretraining default; SFT and DPO may each override it with their own
        ``seq_len`` (both default to ``None`` and collapse back to ``data.seq_len``). Anything that sizes
        memory or token accounting against the training/eval batches — ``estimate_batch_size``,
        ``tokens_per_step`` — must use this resolved width, not ``data.seq_len`` directly.
        """
        if self.dpo.enabled:
            return self.dpo.seq_len or self.data.seq_len
        if self.sft.enabled:
            return self.sft.seq_len or self.data.seq_len
        return self.data.seq_len

    @property
    def train_width_multiplier(self) -> int:
        """How many packed rows one logical batch row forwards at once.

        DPO concatenates a pair's chosen and rejected sequences into a single forward pass, so one
        DPO row costs twice the token width of one SFT/pretrain row at the same ``seq_len``. This is
        the single source of truth that both ``estimate_batch_size`` (how many rows fit) and
        ``tokens_per_step`` (how many tokens those rows process) consult.
        """
        return 2 if self.dpo.enabled else 1

    @property
    def resolved_row_tokens(self) -> int:
        """Total tokens in one logical training row under the active data pipeline.

        This is the per-row unit that both ``estimate_batch_size`` and ``tokens_per_step`` should
        use: ``resolved_seq_len`` for pretrain/SFT, and two ``resolved_seq_len`` sequences for DPO
        (chosen+rejected), each potentially narrowed by the active mode's ``seq_len`` override.
        """
        return self.resolved_seq_len * self.train_width_multiplier

    def __setstate__(self, state: dict) -> None:
        # The default unpickle is object.__new__ + __dict__.update(state); this adds schema
        # evolution on top, so checkpoints saved before a field was added load with that
        # field at its default — identical to a config file omitting it. Without this,
        # pre-sft/dpo checkpoints AttributeError wherever cfg.sft/cfg.dpo is read, because
        # default_factory only runs inside __init__, which pickle bypasses.
        self.__dict__.update(state)
        _backfill_missing_fields(self)


def doc_mask_is_inert_for_dpo(model_cfg: ModelConfig) -> bool:
    """Whether DPO's packing makes this model config's ``doc_attention_mask`` provably a no-op.

    DPO packs **one pair side per row**: real content, then a tail of repeated ``eos_token_id``
    padding carrying ``loss_mask=0``. ``document_ids``' exclusive cumsum puts every real token —
    including the single EOS that terminates the content, which *is* scored — in document 0, and
    each padding EOS in a document of its own. So the only attention the document mask removes is
    a padded position's, and no padded position contributes a scored logit: every logit the DPO
    loss reads is bit-identical with the mask on or off. Building it is pure wasted wall-clock,
    every training step and across the whole reference-logprob precompute pass.

    ``loop_attn_windows`` is the exception, and the reason this is a predicate rather than a bare
    ``cfg.dpo.enabled``: windows ride the very same BlockMask, and a sliding window restricts
    attention *within* document 0 — real DPO content genuinely feels it, so the mask stays.
    """
    return model_cfg.doc_attention_mask and not model_cfg.loop_attn_windows


def resolve_device(device: str) -> str:
    """Resolve "auto" to whatever accelerator is actually available, cuda > mps > cpu."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    # Deliberately an alias for bf16, not a dtype of its own. NVFP4 is not an autocast dtype — it
    # is a per-GEMM operand format — so a run asking for it still autocasts in bf16 and keeps fp32
    # masters. `_apply_dtype_sugar` turns the string into `model.fp4_linear`; the entry here exists
    # so `resolve_dtype` needs no branching and so the GradScaler predicate (which keys on the
    # resolved torch.dtype, not the string) correctly stays disabled.
    "nvfp4": torch.bfloat16,
}


def resolve_dtype(dtype: str) -> torch.dtype:
    """Map a config dtype string ("fp32", "fp16", "bf16", "nvfp4") to its torch.dtype."""
    if dtype not in _DTYPES:
        raise ValueError(f"Unknown train.dtype {dtype!r}, expected one of {sorted(_DTYPES)}")
    return _DTYPES[dtype]


def _apply_dtype_sugar(cfg: Config) -> Config:
    """Resolve `train.dtype: nvfp4` into `model.fp4_linear`.

    Done here, in `load_config`, rather than in `train()`. Two reasons, both about the resolved
    value being the one that gets recorded: `save_checkpoint` pickles the whole `Config` and
    `generate.py` rebuilds the model from it, so resolving at load time makes the checkpoint
    self-describing; and `train()` logs `vars(cfg.model)` to W&B, so the run's record shows what
    actually ran instead of a flag that got flipped somewhere invisible.

    This is also the only place that sees both `model` and `train`, so the cross-section checks
    live here — `DenseTransformer.__init__`, where this repo's other incompatibility errors live,
    only ever receives a `ModelConfig`.
    """
    if cfg.train.dtype == "nvfp4":
        cfg.model.fp4_linear = True
        print("[radiance] train.dtype: nvfp4 -> model.fp4_linear: true, with bf16 autocast underneath")
    if cfg.model.fp4_linear and cfg.train.dtype == "fp16":
        raise ValueError(
            "model.fp4_linear is incompatible with train.dtype: fp16. Before the global scales are "
            "folded back in, the GEMM output is ~1e5x the true product — inside bf16's exponent "
            "range and outside fp16's — and GradScaler's loss scale would compose with the "
            "per-tensor global scale in a way nothing here has reasoned about. Use bf16 or nvfp4."
        )
    if cfg.train.native_bf16 and cfg.train.dtype != "bf16":
        raise ValueError(
            f"train.native_bf16 requires train.dtype: bf16 (got {cfg.train.dtype!r}). fp16's narrow "
            "exponent range makes storing params/grads/optimizer state natively in it meaningfully "
            "riskier than bf16 (no fp32 master copy to fall back on), and fp32 storage under any "
            "other compute dtype doesn't match what this flag promises."
        )
    if cfg.train.native_bf16 and cfg.model.fp4_linear:
        raise ValueError(
            "train.native_bf16 is incompatible with model.fp4_linear: FP4's quality margin (see "
            "nvfp4/linear.py) assumes fp32 master weights, and this flag replaces them with bf16."
        )
    return cfg


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Caught here rather than in `_apply_dtype_sugar` because only the raw YAML distinguishes
    # "fp4_linear was left at its default" from "fp4_linear: false was written down".
    if raw.get("train", {}).get("dtype") == "nvfp4" and raw.get("model", {}).get("fp4_linear") is False:
        raise ValueError(
            "train.dtype: nvfp4 and model.fp4_linear: false contradict each other. "
            "Set one or the other, rather than leaving it to resolution order."
        )

    return _apply_dtype_sugar(Config(
        run_name=raw.get("run_name", Config.run_name),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
        sft=SFTConfig(**raw.get("sft", {})),
        dpo=DPOConfig(**raw.get("dpo", {})),
    ))
