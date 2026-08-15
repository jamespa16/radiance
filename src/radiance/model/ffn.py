from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance.config import ModelConfig

from .norms import RMSNorm, _make_iter_lora
class FeedForward(nn.Module):
    """SwiGLU-gated MLP with configurable depth: `ffn_depth` hidden layers of width `ffn_dim`.
    The first hidden layer is gated (SiLU(gate_proj(x)) * up_proj(x)); any additional depth
    (`ffn_depth > 1`) stacks plain Linear + SiLU layers at `ffn_dim` width on top, preserving the
    "extra hidden layers deepen the MLP, independent of block count" meaning of `ffn_depth`.
    """

    def __init__(self, cfg: ModelConfig, n_variants: int = 1, ffn_dim: int | None = None):
        super().__init__()
        depth = max(1, cfg.ffn_depth)
        ffn_dim = cfg.ffn_dim if ffn_dim is None else ffn_dim
        self.gate_proj = nn.Linear(cfg.d_model, ffn_dim)
        self.up_proj = nn.Linear(cfg.d_model, ffn_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ffn_dim, ffn_dim) for _ in range(depth - 1)])
        self.down_proj = nn.Linear(ffn_dim, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.down_lora = _make_iter_lora(cfg, ffn_dim, cfg.d_model, n_variants)

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        """`variant` (which loop iteration is running) is accepted and ignored, so this and
        MoEFeedForward share one call signature and TransformerBlock/_sparse_ffn_delta never have
        to ask which kind of FFN they hold. The (*, d_model) -> (*, d_model) shape contract that
        lets the two be interchangeable is unchanged."""
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        for layer in self.hidden_layers:
            h = F.silu(layer(h))
        out = self.down_proj(h)
        if self.down_lora is not None:
            out = out + self.down_lora(h, variant)
        return self.dropout(out)


class MoERouter(nn.Module):
    """Per-token expert-routing head for MoEFeedForward. Mirrors ACTRouter's RMSNorm -> Linear
    shape, but projects to n_experts logits (softmaxed over experts) instead of a single halting
    probability, and is instantiated once per MoE FFN layer (not a model-level singleton like
    DenseTransformer.router, which is ACT's halting router — different concept, different name).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.n_experts)
        # Per-iteration routing bias (cfg.loop_iter_conditioning). Because blocks[1:] is
        # weight-shared but fed an evolving hidden state, a token's expert choice already drifts
        # across iterations for free; this makes that drift something the model can *steer* — a
        # given iteration can learn a standing preference for particular experts. Zero-initialised,
        # so routing is unchanged at init.
        self.iter_bias = None
        if cfg.loop_iter_conditioning != "none" and cfg.loop_multiplier > 1:
            self.iter_bias = nn.Parameter(torch.zeros(cfg.loop_multiplier, cfg.n_experts))

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        logits = self.proj(self.norm(x))
        if self.iter_bias is not None:
            logits = logits + self.iter_bias[min(variant, self.iter_bias.size(0) - 1)]
        # fp32 for a stable softmax under bf16/fp16 autocast, mirroring cross_entropy's own fp32
        # upcast (see activation_bytes_per_token's docstring).
        return F.softmax(logits.float(), dim=-1)  # (n_tokens, n_experts)


# MoE capacity is quantised to this many tokens so torch.compile doesn't retrace on every batch.
_CAPACITY_QUANTUM = 128


def _moe_eval_capacity(assigned: torch.Tensor, n_tokens: int) -> int:
    """True per-expert load, quantised, for cfg.moe_eval_full_capacity. See _moe_capacity."""
    # One sync to read the real load. Acceptable outside training, where the alternative is a
    # silently input-dependent result.
    #
    # Rounded up to a multiple of _CAPACITY_QUANTUM rather than used exactly: capacity is a
    # tensor *shape*, so a value that moves with the data would make torch.compile retrace the
    # MoE dispatch on nearly every eval batch. Quantising leaves a handful of distinct shapes
    # while keeping the no-drop guarantee (rounding only ever makes capacity larger).
    peak = max(1, int(assigned.sum(dim=0).max().item()))
    quantised = -(-peak // _CAPACITY_QUANTUM) * _CAPACITY_QUANTUM
    return min(n_tokens, quantised)


def _moe_capacity(cfg: ModelConfig, n_tokens: int, assigned: torch.Tensor | None = None) -> int:
    """Per-expert token capacity.

    Normally the Switch-Transformer formula, which bounds the gather's size so training throughput
    and memory don't depend on how unbalanced routing happens to be that step.

    When `assigned` is given (eval, via cfg.moe_eval_full_capacity) capacity is instead the true
    maximum per-expert load, so nothing is dropped. Dropping is a *training* throughput tradeoff;
    at inference it only discards computation. Worse, because the formula scales with n_tokens, it
    made a token's output depend on how many other tokens shared its forward pass — the same prompt
    scored differently in a batch than alone, and incremental decoding drifted from a full forward.
    """
    if assigned is not None:
        return _moe_eval_capacity(assigned, n_tokens)
    # Capped at n_tokens: a per-expert capacity above the token count is never meaningful (there
    # aren't enough tokens to fill it) and would make torch.topk's k exceed the dimension size.
    return max(1, min(n_tokens, round(cfg.moe_capacity_factor * n_tokens * cfg.moe_top_k / cfg.n_experts)))


class BatchedExperts(nn.Module):
    """`n_experts` parallel FeedForwards whose weights live in single stacked 3-D tensors, so one
    `baddbmm` per layer replaces a Python loop of `n_experts` separate small matmuls.

    Mathematically identical to `nn.ModuleList([FeedForward(cfg) for _ in range(n_experts)])` run
    one expert at a time; the difference is purely how the work is issued to the GPU. The loop
    version's cost is dominated by launch overhead and low occupancy on narrow per-expert matmuls,
    which is why it measured ~4x a single dense FFN while doing only ~2.5x its FLOPs.

    Weights are stored (in_features, out_features) — the transpose of nn.Linear's (out, in) — so
    the forward is a plain `x @ W` with no transpose per call. See MoEFeedForward's state-dict hook
    for the conversion from the old per-expert Linear layout.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # moe_expert_dim, not ffn_dim: fine-grained MoE uses many narrow experts (see
        # cfg.moe_expert_ffn_mult), keeping active parameters per token constant while multiplying
        # the number of expert combinations available.
        n_experts, d_model, ffn_dim = cfg.n_experts, cfg.d_model, cfg.moe_expert_dim
        depth = max(1, cfg.ffn_depth)
        self.mup_hidden_std = 0.02 / cfg.mup_width_mult**0.5
        self.gate_w = nn.Parameter(torch.empty(n_experts, d_model, ffn_dim))
        self.gate_b = nn.Parameter(torch.zeros(n_experts, 1, ffn_dim))
        self.up_w = nn.Parameter(torch.empty(n_experts, d_model, ffn_dim))
        self.up_b = nn.Parameter(torch.zeros(n_experts, 1, ffn_dim))
        self.hidden_w = nn.ParameterList(
            [nn.Parameter(torch.empty(n_experts, ffn_dim, ffn_dim)) for _ in range(depth - 1)]
        )
        self.hidden_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_experts, 1, ffn_dim)) for _ in range(depth - 1)]
        )
        self.down_w = nn.Parameter(torch.empty(n_experts, ffn_dim, d_model))
        self.down_b = nn.Parameter(torch.zeros(n_experts, 1, d_model))
        self.dropout = nn.Dropout(cfg.dropout)
        # DenseTransformer._init_weights only reaches nn.Linear/nn.Embedding submodules, and these
        # are raw Parameters, so match its std here explicitly — including muP's width scaling.
        for w in (self.gate_w, self.up_w, self.down_w, *self.hidden_w):
            nn.init.normal_(w, mean=0.0, std=self.mup_hidden_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_experts, capacity, d_model) -> same shape."""
        h = F.silu(torch.baddbmm(self.gate_b, x, self.gate_w)) * torch.baddbmm(self.up_b, x, self.up_w)
        for w, b in zip(self.hidden_w, self.hidden_b):
            h = F.silu(torch.baddbmm(b, h, w))
        return self.dropout(torch.baddbmm(self.down_b, h, self.down_w))


class MoEFeedForward(nn.Module):
    """Top-k routed MoE FFN, drop-in replacement for FeedForward: (*, d_model) -> (*, d_model),
    shape-agnostic over leading dims so it works both from TransformerBlock.forward's (batch,
    seq_len, d_model) path and from ACT's _sparse_ffn_delta gather path's flat (capacity, d_model)
    path with no special-casing in either.

    Router weights are Mixtral-style: full softmax over all experts, top-k selected, renormalized
    to sum to 1 over just the selected k. Dispatch mirrors _sparse_ffn_delta's gather/compute/
    scatter idiom, generalized to n_experts writers with per-expert fixed capacity (_moe_capacity)
    and drop-on-overflow (same "assigned + random tiebreak" topk priority trick as
    _sparse_ffn_delta, applied per-expert).

    Uses index_add, not _sparse_ffn_delta's index_copy, to accumulate per-expert writes into a
    shared output buffer: with top_k >= 1, a token can receive nonzero contributions from more than
    one expert (top_k=2 by default), and even with top_k=1, capacity padding for one expert can
    land on a token index another expert legitimately wrote to. index_copy would let a later
    expert's zero padding silently clobber an earlier expert's real output at that index;
    index_add sums correctly because unassigned/padding slots are explicitly zeroed by `valid`
    before being added.

    Those padding slots are not wasted twice. Because capacity is a fixed shape, an under-loaded
    expert runs on `capacity` rows whatever its real load, so the dispatch is already computing
    each expert's output for tokens it did not win and throwing the result away. The tiebreak in
    forward() aims those rows at the tokens the expert ranked just below its top-k, and
    cfg.moe_counterfactual_weight turns them into the counterfactual gradient the router otherwise
    cannot have — see _counterfactual_probe_signal. Neither changes the forward output.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.moe_top_k > cfg.n_experts:
            raise ValueError(f"model.moe_top_k ({cfg.moe_top_k}) must be <= model.n_experts ({cfg.n_experts})")
        if cfg.moe_balance_signal not in ("count", "weight"):
            # Raised rather than ignored: a typo here would silently keep the previous behaviour,
            # which is exactly the kind of "the A/B measured nothing" failure worth failing fast on.
            raise ValueError(
                f"model.moe_balance_signal must be 'count' or 'weight', got {cfg.moe_balance_signal!r}"
            )
        self.cfg = cfg
        self.router = MoERouter(cfg)
        self.experts = BatchedExperts(cfg)
        # Always-on expert (DeepSeekMoE): its output is added to every token unconditionally, so
        # the routed experts don't each have to re-learn the computation every token needs.
        self.shared_expert = None
        if cfg.moe_n_shared > 0:
            self.shared_expert = FeedForward(cfg, ffn_dim=cfg.moe_n_shared * cfg.moe_shared_dim)

        # Loss-free load balancing (cfg.moe_balance): a non-learned per-expert bias added to the
        # routing logits *for top-k selection only*. Buffer, not parameter — it's updated by an
        # explicit rule after each optimizer step (DenseTransformer.update_expert_bias), never by
        # gradient, which is the entire point: it balances load without adding a term that competes
        # with the LM objective the way the aux loss does.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts), persistent=True)
        self.register_buffer("expert_load", torch.zeros(cfg.n_experts), persistent=False)

        self.aux_loss_accum: torch.Tensor | None = None
        self._aux_loss_calls = 0
        self._register_load_state_dict_pre_hook(self._upgrade_legacy_expert_keys, with_module=False)

    def _upgrade_legacy_expert_keys(self, state_dict, prefix, *args) -> None:
        """Convert checkpoints saved when experts were an nn.ModuleList of FeedForwards.

        Old layout: `{prefix}experts.{e}.{gate,up,down}_proj.{weight,bias}` with nn.Linear's
        (out_features, in_features) weights. New layout stacks them into one (n_experts, in, out)
        tensor per projection, so each old weight is transposed and the experts concatenated.
        """
        legacy = [k for k in state_dict if k.startswith(f"{prefix}experts.") and k.split(".")[-2].endswith("_proj")]
        if not legacy:
            return
        n_experts = self.cfg.n_experts
        for old_name, new_w, new_b in (
            ("gate_proj", "gate_w", "gate_b"),
            ("up_proj", "up_w", "up_b"),
            ("down_proj", "down_w", "down_b"),
        ):
            weights = [state_dict.pop(f"{prefix}experts.{e}.{old_name}.weight") for e in range(n_experts)]
            biases = [state_dict.pop(f"{prefix}experts.{e}.{old_name}.bias") for e in range(n_experts)]
            state_dict[f"{prefix}experts.{new_w}"] = torch.stack([w.t() for w in weights])
            state_dict[f"{prefix}experts.{new_b}"] = torch.stack([b.unsqueeze(0) for b in biases])
        for i in range(max(1, self.cfg.ffn_depth) - 1):
            weights = [state_dict.pop(f"{prefix}experts.{e}.hidden_layers.{i}.weight") for e in range(n_experts)]
            biases = [state_dict.pop(f"{prefix}experts.{e}.hidden_layers.{i}.bias") for e in range(n_experts)]
            state_dict[f"{prefix}experts.hidden_w.{i}"] = torch.stack([w.t() for w in weights])
            state_dict[f"{prefix}experts.hidden_b.{i}"] = torch.stack([b.unsqueeze(0) for b in biases])

    def per_expert_parameter_count(self) -> int:
        """Parameters belonging to a single *routed* expert — the stacked tensors divided by
        n_experts. Used by num_active_parameters() to discount the experts a given token never
        activates. Deliberately excludes the shared expert, which every token activates and so
        counts in full."""
        return sum(p.numel() for p in self.experts.parameters()) // self.cfg.n_experts

    def reset_aux_loss(self) -> None:
        self.aux_loss_accum = None
        self._aux_loss_calls = 0

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        orig_shape = x.shape
        d_model = orig_shape[-1]
        flat_x = x.reshape(-1, d_model)
        n_tokens = flat_x.shape[0]

        probs = self.router(flat_x, variant)  # (n_tokens, n_experts) fp32

        # Selection uses probs + expert_bias; the gating *weights* come from the unbiased probs.
        # That separation is what makes the bias loss-free: it can steer which experts a token goes
        # to without distorting how much each contributes, so it never fights the LM objective.
        selection_scores = probs + self.expert_bias if self._bias_balancing else probs
        topk_idx = selection_scores.topk(self.cfg.moe_top_k, dim=-1).indices
        topk_probs = probs.gather(1, topk_idx)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize over selected k
        weight = probs.new_zeros(n_tokens, self.cfg.n_experts).scatter_(1, topk_idx, topk_probs)

        assigned = weight > 0  # (n_tokens, n_experts)
        if self._bias_balancing and self.training:
            # Accumulated here, consumed by update_expert_bias() after the optimizer step — outside
            # the compiled graph, so no graph break in the forward.
            #
            # cfg.moe_balance_signal picks *what* is being balanced. "count" is one per routed
            # token; "weight" is that token's gate weight, i.e. how much of its FFN output this
            # expert actually supplies. The two differ whenever an expert is selected often but
            # weakly, and "weight" is the better proxy for the thing the bias exists to prevent —
            # an expert starved of gradient — because the gradient reaching an expert's weights
            # scales with the gate weight it was applied at, not with how many tokens listed it.
            # Both are pre-drop, matching each other and the previous behaviour.
            signal = assigned if self.cfg.moe_balance_signal == "count" else weight
            self.expert_load += signal.detach().sum(dim=0).float()
        f_i = assigned.float().mean(dim=0).detach()  # fraction of tokens routed to expert i (non-diff)
        p_i = probs.mean(dim=0)  # mean full-softmax prob mass on expert i (differentiable)
        aux_loss = self.cfg.n_experts * (f_i * p_i).sum()
        self.aux_loss_accum = aux_loss if self.aux_loss_accum is None else self.aux_loss_accum + aux_loss
        self._aux_loss_calls += 1

        # Outside training, size capacity to the real load so nothing is dropped — see _moe_capacity.
        full_capacity = self.cfg.moe_eval_full_capacity and not self.training
        capacity = _moe_capacity(self.cfg, n_tokens, assigned if full_capacity else None)

        # Per-expert token selection, computed for all experts at once. Priority within the
        # assigned set is the router's own weight for this expert, not a random draw: when an
        # expert is over capacity, the tokens it drops should be the ones it was least confident
        # about. The `* 2.0` keeps every assigned token (scoring in [2, 3]) strictly above every
        # unassigned one (in [0, 1]), preserving the "assigned always outranks capacity padding"
        # property.
        #
        # Unassigned tokens are ordered by their raw router probability rather than left tied at
        # zero. They are ordered at all because an under-loaded expert still computes a full
        # `capacity` rows and masks the surplus away below, so *some* unrouted tokens are pushed
        # through it either way — at moe_capacity_factor 1.25 that is ~20% of all MoE FLOPs. Tied
        # at zero, topk broke the tie by index and those rows went to whichever tokens sat earliest
        # in the batch; ranked by probability they go to the tokens the expert ranked just below
        # its top-k, which is the informative sample. The forward output is identical either way
        # (the rows are multiplied by `valid` and contribute exactly zero), so this only decides
        # *which* counterfactuals _counterfactual_probe_signal has to work with.
        #
        # This also makes routing deterministic. The previous random tiebreak ran in eval and
        # generation too, so val/loss and even greedy decoding varied run to run for the same
        # weights and inputs — which defeats comparing two configs' eval numbers.
        priority = assigned.float() * 2.0 + torch.where(assigned, weight, probs).detach().float()
        token_idx = priority.topk(capacity, dim=0).indices.t().contiguous()  # (n_experts, capacity)
        valid = assigned.t().gather(1, token_idx)  # (n_experts, capacity) bool
        gathered = flat_x[token_idx]  # (n_experts, capacity, d_model)

        # Kept unmasked: `probe_out` holds every expert's output for every row it computed,
        # including the discarded ones, which is what the counterfactual signal reads.
        probe_out = self.experts(gathered)
        expert_out = probe_out * valid.unsqueeze(-1).to(x.dtype)
        w = (weight.t().gather(1, token_idx) * valid.to(weight.dtype)).to(x.dtype)

        # index_add, not index_copy: a token can receive nonzero contributions from more than one
        # expert (top_k >= 2 by default), and even at top_k=1 one expert's zero capacity-padding can
        # land on an index another expert legitimately wrote. index_copy would let the padding
        # clobber the real output; index_add sums correctly because `valid` zeroed the padding.
        flat_token_idx = token_idx.reshape(-1)
        delta = flat_x.new_zeros(n_tokens, d_model).index_add(
            0, flat_token_idx, (w.unsqueeze(-1) * expert_out).reshape(-1, d_model)
        )
        if self._counterfactual_routing:
            delta = delta + self._counterfactual_probe_signal(
                probs, probe_out, delta, w, valid, token_idx, flat_token_idx
            )
        if self.shared_expert is not None:
            # Unconditional, ungated, no capacity limit: every token gets this.
            delta = delta + self.shared_expert(flat_x)
        return delta.view(orig_shape)

    @property
    def _bias_balancing(self) -> bool:
        return self.cfg.moe_balance in ("bias", "both")

    @property
    def _counterfactual_routing(self) -> bool:
        # Training-only: the term is exactly zero in the forward, so it exists purely to deposit a
        # gradient. Skipping it under eval/no_grad costs nothing and saves the extra index_adds.
        return self.cfg.moe_counterfactual_weight != 0.0 and self.training and torch.is_grad_enabled()

    def _counterfactual_probe_signal(
        self,
        probs: torch.Tensor,
        probe_out: torch.Tensor,
        routed: torch.Tensor,
        w: torch.Tensor,
        valid: torch.Tensor,
        token_idx: torch.Tensor,
        flat_token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Router gradient from the expert outputs this dispatch computes and then throws away.

        Returns a tensor that is **exactly zero** — it is added to the routed output and changes
        nothing about what the model computes — whose only effect is the gradient it deposits on
        `probs`.

        The problem it solves. Backprop already gives the router a perfect utility signal for the
        experts a token *was* sent to: expert e contributes `w[t,e] * out[t,e]` to the layer's
        output, so `dL/dw[t,e] = <g_t, out[t,e]>` where `g_t = dL/d(output_t)`. What it cannot give
        is the counterfactual — how good the experts the token was *not* sent to would have been —
        because `out[t,e]` never enters the graph for those. That is normally where the argument
        ends, since computing it means running experts on tokens they didn't win.

        Here it is already paid for. Capacity is a fixed shape, so an under-loaded expert computes
        `capacity` rows regardless and the surplus is masked away (~20% of MoE FLOPs at the default
        moe_capacity_factor), and the priority tiebreak in forward() steers those rows to the
        tokens that expert ranked just below its top-k. `probe_out` holds the results. This turns
        them into the missing gradient.

        The construction. For a probe row (t, e) the first-order utility of having routed token t
        to expert e is `u[t,e] = -<g_t, out[t,e]>`, and the utility it actually realised is
        `-<g_t, m_t>` where `m_t` is the mixture the routed experts produced per unit of retained
        gate weight — which is just `routed_t / sum_e w[t,e]`, already computed. The advantage is
        `adv[t,e] = -<g_t, out[t,e] - m_t>`, and we want the router to raise `p[t,e]` where that is
        positive.

        `g_t` only exists in backward, so the term cannot be written as a forward loss. Instead of
        a custom autograd.Function (which would break the graph under torch.compile), it uses the
        identity that `p - p.detach()` is exactly 0.0 with unit gradient: adding
        `weight * (p - sg p) * sg(out[t,e] - m_t)` to the output leaves the forward bit-identical
        while making `dL/dp[t,e] = weight * <g_t, out[t,e] - m_t> = -weight * adv[t,e]`, so
        gradient descent moves `p` up exactly where the counterfactual was better than what
        happened. It is scale-consistent with everything else for free, including under fp16's
        GradScaler, because it *is* an ordinary gradient rather than a separately-scaled loss.

        The `sum_e coeff * m_t` half is folded into a per-token scalar rather than materialised per
        row, since `m_t` doesn't depend on e — that keeps this to one transient (n_experts,
        capacity, d_model) tensor instead of two retained ones.
        """
        n_tokens, d_model = routed.shape

        # Per unit of *retained* gate weight: a token whose experts dropped it keeps a mixture
        # that only some of its weight paid for, and the baseline has to be the average of what it
        # got, not the sum. clamp_min guards the token that every one of its experts dropped —
        # `routed` is exactly zero there, so the baseline is zero rather than a division blow-up.
        den = routed.new_zeros(n_tokens).index_add(0, flat_token_idx, w.reshape(-1))
        baseline = routed.detach() / den.clamp_min(1e-6).unsqueeze(-1)

        # Exactly 0.0, unit gradient into probs, and zero on the rows that were really routed —
        # those already get the true gradient through `w`, and pushing on them again would just
        # double-count a signal backprop supplies correctly.
        p = probs.t().gather(1, token_idx).to(routed.dtype)
        coeff = (p - p.detach()) * (~valid).to(routed.dtype) * self.cfg.moe_counterfactual_weight

        signal = routed.new_zeros(n_tokens, d_model).index_add(
            0, flat_token_idx, (coeff.unsqueeze(-1) * probe_out.detach()).reshape(-1, d_model)
        )
        coeff_sum = routed.new_zeros(n_tokens).index_add(0, flat_token_idx, coeff.reshape(-1))
        return signal - coeff_sum.unsqueeze(-1) * baseline

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """Nudge expert_bias toward whichever experts are under-loaded, then reset the counter.

        Called once per optimizer step from train(), never inside forward — it mutates a buffer
        with no gradient, so keeping it out of the graph avoids a torch.compile break.

        The update is sign-based (DeepSeek-V3): a fixed step toward balance regardless of how far
        off an expert is. That makes it insensitive to the scale of the load imbalance and keeps
        the bias from oscillating when one batch happens to be unrepresentative.
        """
        if not self._bias_balancing or self.expert_load.sum() == 0:
            return
        mean_load = self.expert_load.mean()
        self.expert_bias -= self.cfg.moe_bias_update_rate * torch.sign(self.expert_load - mean_load)
        self.expert_load.zero_()
