"""Optimizers and parameter-group construction.

Split out of train.py, which was carrying ~160 lines of optimizer machinery unrelated to the
training loop itself. Everything here is about *how parameters get updated*; train.py keeps the
step-based loop, loss computation, checkpointing and evaluation.

build_param_groups lives here rather than in train.py because migrate_optimizer_to_cpu_offload
needs it, and importing it back from train.py would make the two modules circular.
"""

from __future__ import annotations

import torch

from radiance.config import Config
from radiance.model import RMSNorm


def build_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    embed_lr: float | None = None,
    hyper_conn_lr: float | None = None,
    lr: float | None = None,
    mup_width_mult: float = 1.0,
) -> list[dict]:
    """Split parameters into decayed / non-decayed groups.

    AdamW over model.parameters() applies weight decay uniformly, which decays RMSNorm gains and
    every bias. Those are 1-D scale/shift parameters with no "shrink toward zero is a useful
    prior" interpretation — decaying them just fights the norm layers. Standard practice
    (GPT-2/Llama/nanoGPT) is to decay only the >=2-D weight matrices, which is what this does.

    `embed_lr` (cfg.train.embed_lr) pulls the tied token_emb/lm_head weight into a third group
    carrying its own LR, mirroring build_muon_param_groups. None (the default) leaves it in the
    decayed group exactly as before, so the group list is unchanged for any config that hasn't set
    it — which also keeps migrate_optimizer_to_cpu_offload's positional group zip valid, since both
    call sites pass the same value.

    `hyper_conn_lr` (cfg.train.hyper_conn_lr) does the same for the hyper-connection coefficients,
    for a sharper reason: they are structural (a one-hot read, an identity depth mix) and AdamW's
    step is ~lr regardless of gradient scale, so sharing `lr`'s post-Muon value erases the routing
    within a few hundred steps rather than tuning it. Appended last and only when non-empty, so the
    positional zip stays valid here too.

    `mup_width_mult` (cfg.model.mup_width_mult) applies muP's per-tensor LR correction, which for
    Adam is 1/m on the *hidden* weights and Theta(1) on everything else. It is exactly 1.0 unless
    cfg.model.mup_base_d_model is set, in which case `lr` must be given to scale against. Without
    this correction Adam's update is width-independent where muP requires it to shrink, and hidden
    activations grow linearly in d_model — measured at 16.7x across a 16x width sweep, which is
    precisely the drift muP exists to remove. Note it stays invisible in `val/loss`: ln_f is
    RMSNorm and therefore scale-invariant, so it launders the blown-up residual stream away right
    before the LM head. A coordinate check, not a loss curve, is what catches this.

    build_muon_param_groups needs no equivalent, for a real reason rather than an oversight: Muon's
    update is spectrally normalised and so already approximately width-invariant, and the tensors
    its auxiliary AdamW owns are exactly the ones muP leaves at Theta(1) anyway (the embedding,
    1-D gains/biases, and the tiny routers/gates). Verified by coordinate check across a 16x width
    sweep, looped and unlooped.
    """
    mup = mup_width_mult != 1.0
    if mup and lr is None:
        raise ValueError("build_param_groups needs `lr` to apply muP's 1/width_mult correction")
    # The tied embedding is Theta(1) under muP (a row lookup has fan-in 1, so it does not widen).
    # It otherwise sits in `decay` whenever embed_lr is unset — which is the default — and would
    # ride that group's 1/m correction, so give it its own unscaled group as soon as muP is live.
    if mup and embed_lr is None:
        embed_lr = lr

    decay, no_decay, embed, hyper_static, hyper_dyn = [], [], [], [], []
    norm_gains = norm_gain_param_ids(model)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if hyper_conn_lr is not None and "hyper" in name:
            # Only the dynamic *projection* is decayed. The static coefficients must not be (see
            # _is_hyper_static), and dyn_scale_* are 1-D scale parameters that the ordinary
            # `dim() < 2` rule would have left undecayed anyway.
            decayed = param.dim() >= 2 and not _is_hyper_static(name)
            (hyper_dyn if decayed else hyper_static).append(param)
        elif param.dim() < 2 or _is_hyper_static(name) or id(param) in norm_gains:
            # id(param) in norm_gains: a per-iteration gain bank is 2-D but must not be decayed
            # toward zero any more than a 1-D gain is. See norm_gain_param_ids.
            no_decay.append(param)
        elif embed_lr is not None and _is_embedding(name):
            embed.append(param)
        else:
            decay.append(param)
    groups = [
        # `decay` is the hidden weight matrices plus the routers/gates and MoE experts — all
        # hidden- or readout-like, i.e. all 1/m under muP. `no_decay` is 1-D gains and biases,
        # which muP leaves alone. The key is only set when the correction is non-trivial, so an
        # ordinary run's groups are untouched and inherit the optimizer's own lr as before.
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if mup:
        groups[0]["lr"] = lr / mup_width_mult
    if embed:
        groups.append({"params": embed, "weight_decay": weight_decay, "lr": embed_lr})
    # Appended last and only when non-empty, so the positional group zip in
    # migrate_optimizer_to_cpu_offload stays valid — both call sites pass the same arguments.
    if hyper_static:
        groups.append({"params": hyper_static, "weight_decay": 0.0, "lr": hyper_conn_lr})
    if hyper_dyn:
        groups.append({"params": hyper_dyn, "weight_decay": weight_decay, "lr": hyper_conn_lr})
    return groups


# Parameters excluded from Muon and left to the auxiliary AdamW. Two reasons appear here:
#   - token_emb / lm_head: these are the *tied* embedding matrix. Its rows are per-token, not a
#     linear map between two hidden spaces, so orthogonalising it is not meaningful — and it is the
#     largest tensor in the model, so Newton-Schulz on it would also be the dominant step cost.
#   - router / out_gate / hyper: tiny tensors whose exact scale is load-bearing (a router's softmax
#     calibration, a gate's zero-init, a hyper-connection's one-hot read and identity depth mix).
#     Muon's update has a fixed spectral norm regardless of how small the gradient is, which is the
#     wrong behaviour for these. "hyper" matters because HyperConnection's alpha_r is (variants, n,
#     n) and its dynamic weights are (variants, d_model, n) — both >= 2-D, so without this entry
#     they would fall straight through to Muon.
_MUON_EXCLUDED_SUBSTRINGS = ("token_emb", "lm_head", "router", "out_gate", "hyper")

# Cap on how many individual (in, out) matrices _step_muon stacks into a single orthogonalize()
# call — counted after unbinding any BatchedExperts-style (n_experts, in, out) tensor into its
# n_experts separate matrices, not per *tensor*. Chosen to comfortably exceed any dense model's
# per-shape matrix count in this repo's shipped configs (one tensor per shape per layer, and the
# deepest is fineweb_500m.yaml's 22 layers) so ordinary dense runs keep the full batching win, while
# bounding the transient Newton-Schulz buffer size for a fine-grained MoE model where many experts
# across many MoE layers can share one (in, out) shape. See _step_muon's docstring for the OOM this
# fixes, why counting whole tensors rather than matrices doesn't fix it, and why CPU-offload can't
# stand in for it.
_MUON_MAX_STACK = 32

# The tied embedding matrix, split out of the AdamW-decayed group so it can carry its own LR
# (cfg.train.embed_lr). It is the one *large* tensor AdamW still owns once Muon takes the hidden
# weights, and it wants a much larger step than the routers/gates it would otherwise share a group
# with — those are tiny tensors whose exact scale is load-bearing.
_EMBEDDING_SUBSTRINGS = ("token_emb", "lm_head")


def _is_embedding(name: str) -> bool:
    return any(s in name for s in _EMBEDDING_SUBSTRINGS)


def norm_gain_param_ids(model: torch.nn.Module) -> set[int]:
    """ids of every gain/bias owned by a normalisation layer, **at any rank**.

    Both param-group builders used to identify these by `param.dim() < 2`, which was correct only
    while a norm gain was always `(d_model,)`. cfg.model.loop_iter_conditioning="norm_gains" — the
    *default* — gives RMSNorm a `(n_variants, d_model)` gain bank instead, one row per loop
    iteration. That is still a per-channel scale, but it is 2-D, so the rank rule silently
    reclassified it the moment a config set loop_count > 1:

      - on the Muon path it stopped being an AdamW no-decay tensor and became a **Muon** tensor,
        where Newton-Schulz drives its singular values to 1. A norm gain *is* its scale, so that
        does not refine the conditioning, it erases it;
      - on the AdamW path it started being weight-decayed, pulling gains initialised to exactly
        1.0 toward zero.

    Neither raises, neither shows up as a dead parameter, and neither can happen at loop_count 1 —
    which is why configs/tinystories.yaml never saw it and every looped config did. Keying on the
    owning module instead of the tensor's rank makes the classification independent of how many
    variants a gain bank happens to hold. `recurse=False` so only the norm's own parameters count.
    """
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (RMSNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            ids.update(id(p) for p in module.parameters(recurse=False))
    return ids


def _is_hyper_static(name: str) -> bool:
    """A hyper-connection's *static* connection coefficients (beta / alpha_m / alpha_r).

    They are >= 2-D — alpha_r is (n_variants, n, n) — so the plain `dim() < 2` rule would decay
    them, but they are connection coefficients initialised to exact structural values (a one-hot
    read, an identity depth mix, an all-ones write) rather than weights with a shrink-to-zero
    prior; decaying them would pull the model off that initialisation for no reason. The paper
    makes the same split, decaying only the dynamic projections, which is why this keys on the
    "dyn" prefix HyperConnection gives those.
    """
    return "hyper" in name and "dyn" not in name


def build_muon_param_groups(model: torch.nn.Module, cfg: Config) -> list[dict]:
    """Three-way split for MuonWithAuxAdam: Muon hidden weights, AdamW-decayed, AdamW-undecayed.

    Each group carries its own base `lr`, so the Muon group can run at cfg.train.muon_lr (typically
    ~50x cfg.train.lr) while the AdamW groups keep the LR every existing config already tuned.
    LambdaLR scales each group from its own initial_lr, so one schedule shape drives both.

    `mup_lr_scale` carries muP's per-tensor LR correction (see ModelConfig.mup_width_mult). Every
    group on *this* path is 1.0, and deliberately so rather than by omission: the Muon group needs
    no correction (a spectrally-normalised update is already approximately width-invariant), and
    the tensors left to the auxiliary AdamW are exactly the ones muP leaves at Theta(1) — the tied
    embedding (fan-in 1, so it does not widen), 1-D gains and biases, and the routers/gates. That
    last group is hidden-like and would strictly want 1/m, but it is a rounding error in both
    parameter count and gradient norm, so it stays unscaled rather than earning a correction of
    its own. Confirmed by coordinate check: activation scale is flat to within +-8% across a 16x
    width sweep, looped and unlooped.

    The key is kept per-group anyway, so a future group that *does* need a correction has an
    obvious place to declare it. build_param_groups is where the non-trivial version lives, since
    plain AdamW owns the hidden weights too.
    """
    muon, adam_embed, adam_decay, adam_no_decay = [], [], [], []
    norm_gains = norm_gain_param_ids(model)
    # Hyper-connections get their own pair of groups, at cfg.train.hyper_conn_lr: these are
    # structural coefficients (a one-hot read, an identity depth mix), and AdamW's ~lr-per-step
    # update at `lr`'s post-Muon value erases that structure rather than refining it. See
    # TrainConfig.hyper_conn_lr. Split static/dynamic only so the static half escapes weight decay.
    hyper_static, hyper_dyn = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "hyper" in name:
            # Only the dynamic *projection* is decayed. The static coefficients must not be (see
            # _is_hyper_static), and dyn_scale_* are 1-D scale parameters that the ordinary
            # `dim() < 2` rule would have left undecayed anyway.
            decayed = param.dim() >= 2 and not _is_hyper_static(name)
            (hyper_dyn if decayed else hyper_static).append(param)
        elif param.dim() < 2 or id(param) in norm_gains:
            # A per-iteration norm gain bank is 2-D but is still a gain: undecayed AdamW, never
            # Muon. See norm_gain_param_ids.
            adam_no_decay.append(param)
        elif _is_embedding(name):
            adam_embed.append(param)
        elif any(s in name for s in _MUON_EXCLUDED_SUBSTRINGS):
            adam_decay.append(param)
        else:
            muon.append(param)

    groups = [
        {
            "params": muon,
            "algorithm": "muon",
            "lr": cfg.train.muon_lr,
            "weight_decay": cfg.train.weight_decay,
            "mup_lr_scale": 1.0,
        },
        {
            "params": adam_embed,
            "algorithm": "adamw",
            "lr": cfg.train.embed_lr_resolved,
            "weight_decay": cfg.train.weight_decay,
            # Embeddings are unscaled under muP: a row lookup has fan-in 1, so it does not widen.
            "mup_lr_scale": 1.0,
        },
        {
            "params": adam_decay,
            "algorithm": "adamw",
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            # What's left here is the routers and gates — hidden-like, so muP would want 1/m — but
            # they are a rounding error in both parameter count and gradient norm, so leave the
            # group unscaled rather than give them a correction of their own.
            "mup_lr_scale": 1.0,
        },
        {
            "params": adam_no_decay,
            "algorithm": "adamw",
            "lr": cfg.train.lr,
            "weight_decay": 0.0,
            "mup_lr_scale": 1.0,
        },
        {
            "params": hyper_static,
            "algorithm": "adamw",
            "lr": cfg.train.hyper_conn_lr_resolved,
            # Undecayed: decaying a one-hot read or an identity depth mix drags the model off
            # precisely the initialisation that makes it a residual network.
            "weight_decay": 0.0,
            "mup_lr_scale": 1.0,
        },
        {
            "params": hyper_dyn,
            "algorithm": "adamw",
            "lr": cfg.train.hyper_conn_lr_resolved,
            "weight_decay": cfg.train.weight_decay,
            "mup_lr_scale": 1.0,
        },
    ]
    for group in groups:
        group["lr"] *= group["mup_lr_scale"]
    return [g for g in groups if g["params"]]


# Peak transient bytes inside one orthogonalize() call, per fp32 grad element in the stacked input
# (X, A, A@A, B, B@X collectively — see orthogonalize's body). Measured directly (max_memory_allocated
# delta, fp32 input, isolated orthogonalize() calls) rather than derived, the same convention
# activation_bytes_per_token uses for diff attention's 7*d_model: square (in, out) shapes (the worst
# case — A/B are (rows, rows), which shrinks relative to X as the shape gets more rectangular) landed
# at a flat 12 bytes/elem across batch sizes 8-32 and shapes up to 1024x1024; rectangular shapes
# measured lower (9 bytes/elem), so 12 stays conservative rather than shape-specific.
_MUON_NS_BYTES_PER_ELEM = 12


def muon_orthogonalize_reserve_bytes(model: torch.nn.Module, cfg: Config) -> int:
    """VRAM estimate_batch_size should hold back for _step_muon's per-shape orthogonalize() call,
    which allocates transient Newton-Schulz buffers with no relationship to num_params — see
    _step_muon's docstring and _MUON_MAX_STACK. estimate_batch_size's existing not_yet_allocated_bytes
    covers the *persistent* grad/momentum footprint (which does scale with num_params); this covers
    the *transient* per-optimizer-step spike that footprint estimate has no term for at all.

    Returns 0 when cfg.train.optimizer isn't "muon" (nothing to reserve for) or the model has no
    Muon-owned tensors (e.g. every hidden weight excluded some other way).

    Sizing: _step_muon unbinds every Muon-owned tensor into its individual (in, out) matrices — a
    BatchedExperts-shaped (n_experts, in, out) tensor unbinds to n_experts matrices sharing one
    (in, out) shape with every other layer's same-shaped projection — then stacks up to
    _MUON_MAX_STACK of them per orthogonalize() call. The worst single call is therefore whichever
    (in, out) shape has the most matrices, capped at _MUON_MAX_STACK, not the largest individual
    tensor: a fine-grained MoE model's narrow expert projections can out-number a dense model's wide
    ones by more than the width difference, and it's the matrix *count* at a shape that decides
    how many chunks share one orthogonalize() call.
    """
    if cfg.train.optimizer != "muon":
        return 0
    groups = build_muon_param_groups(model, cfg)
    muon_params = next((g["params"] for g in groups if g["algorithm"] == "muon"), [])
    if not muon_params:
        return 0

    matrix_counts: dict[tuple[int, int], int] = {}
    for p in muon_params:
        if p.dim() == 2:
            shape = (p.shape[0], p.shape[1])
            matrix_counts[shape] = matrix_counts.get(shape, 0) + 1
        else:
            # BatchedExperts-style (n_experts, in, out): unbinds to n_experts matrices of the
            # trailing (in, out) shape, same as _step_muon's by_shape grouping.
            shape = (p.shape[-2], p.shape[-1])
            matrix_counts[shape] = matrix_counts.get(shape, 0) + p.shape[0]

    worst_chunk_elems = max(
        min(count, _MUON_MAX_STACK) * rows * cols for (rows, cols), count in matrix_counts.items()
    )
    return _MUON_NS_BYTES_PER_ELEM * worst_chunk_elems


def orthogonalize(grad: torch.Tensor, steps: int = 5, eps: float = 1.0e-7) -> torch.Tensor:
    """Newton-Schulz quintic iteration: replace a matrix by (approximately) the orthogonal factor
    of its polar decomposition, i.e. the same matrix with every singular value driven toward 1.

    This is the whole idea behind Muon. A raw SGD-momentum update tends to be dominated by a few
    large singular directions; orthogonalising it spreads the step evenly across every direction,
    which is what makes a much larger learning rate stable.

    Shape-generic over leading batch dimensions: `@`/`.mT` are batched, so a (n_experts, in, out)
    tensor — exactly BatchedExperts' stacked layout — is orthogonalised per expert with no
    reshaping, dim 0 acting as a free batch dimension.

    The quintic coefficients are the standard tuned ones: they do not converge to machine-precision
    orthogonality, but they drive the singular values into a band around 1 in very few steps, which
    is all the optimizer needs. bfloat16 is deliberate and also standard on GPU — the iteration is
    self-correcting, so the halved precision costs nothing and halves the bandwidth. On CPU there's
    no bandwidth win to trade for, and without AVX-512-BF16 hardware support PyTorch's CPU bf16
    matmul falls back to a path that's dramatically (not just somewhat) slower than fp32 — enough to
    turn a CPU-only test run into an effective hang — so CPU stays in the input's own dtype instead.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = grad.bfloat16() if grad.is_cuda else grad
    X = X / (torch.linalg.matrix_norm(X, keepdim=True) + eps)

    # The iteration below assumes rows <= cols; transpose the tall case and undo it afterwards.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(grad.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for the hidden weight matrices, AdamW for everything else, in one Optimizer object.

    One object rather than two because everything downstream assumes a single optimizer: the
    GradScaler's unscale_/step bookkeeping is keyed on it, LambdaLR drives its param_groups,
    save_checkpoint serialises its state_dict, and the OOM handler swaps it. Splitting into two
    optimizers would mean touching all of those; a per-group `algorithm` key does not.

    Groups are built by build_muon_param_groups. Each carries `algorithm` ("muon" | "adamw") and,
    for adamw groups, an `offload` flag that moves the moment buffers to pinned CPU memory — the
    tier-2 OOM recovery described in train(). Muon's own state is a single momentum buffer (1x
    params, vs AdamW's 2x) and stays resident: Newton-Schulz is a chain of matmuls per parameter
    per step, and running it against CPU memory would cost far more than the VRAM it reclaims.
    """

    def __init__(
        self,
        param_groups: list[dict],
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        device: str = "cpu",
    ) -> None:
        defaults = dict(
            algorithm="adamw",
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            offload=False,
            mup_lr_scale=1.0,
        )
        super().__init__(param_groups, defaults)
        self._device = device
        self._is_cuda = device.split(":")[0] == "cuda"

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group: dict) -> None:
        """Muon over one group, **batched by parameter shape, in bounded-size chunks**.

        The arithmetic is per-parameter, but issuing it per-parameter is what dominates the step:
        Newton-Schulz is `ns_steps` iterations of three matmuls, so a 42-tensor group costs 630
        matmuls, none of them large enough to saturate the GPU. A transformer has very few distinct
        weight shapes (one per projection role, repeated per layer), so stacking the same-shaped
        updates into one leading batch dimension collapses that to 630 / (tensors per shape)
        launches with no change to the result — `orthogonalize` is already shape-generic over
        leading dims (`@`/`.mT` batch, and matrix_norm(keepdim=True) normalises each matrix
        independently), which is what BatchedExperts' stacked (n_experts, in, out) weights already
        rely on. Measured on configs/tinystories.yaml (42 tensors, 5 shapes): 6.71 -> 1.90 ms.

        Momentum and weight decay move to torch._foreach_* for the same reason.

        Stacking is capped at `_MUON_MAX_STACK` *matrices* per orthogonalize() call, not tensors —
        this distinction is load-bearing and was the bug in this function's first chunking attempt.
        `BatchedExperts` already stores each MoE projection role as one tensor shaped
        `(n_experts, in, out)` (see model/ffn.py), so grouping-and-counting by whole *tensor* (e.g. "16
        tensors of shape (48, 1024, 1024), well under a cap of 32, so don't chunk") completely
        misses that each of those 16 tensors already carries 48 experts stacked in its own leading
        dim. `torch.stack`-ing those 16 tensors produces one `(16, 48, 1024, 1024)` tensor, and
        orthogonalize's batched matmuls see an effective batch of 16*48=768 matrices, not 16 — a
        tensor-count cap of 32 lets that straight through untouched. Confirmed empirically: with the
        first attempt (capping tensor count), n_experts=48 (1060M total, ~321M active, 4 MoE layers,
        d_model=1024, moe_expert_ffn_mult=0.25) still OOMs on the very first optimizer step on an
        RTX 5090, identically to before that fix — because the shape group in question has only 16
        distinct tensors, so `len(idxs) <= _MUON_MAX_STACK` and no chunking ever triggers.

        The fix here operates one level down: every same-shaped Muon-owned tensor — 2-D or
        BatchedExperts' 3-D — is first unbound along any leading batch dim into its individual
        (in, out) matrices (a 2-D tensor unbinds to just itself), and the actual chunking/stacking
        happens over that flat list of matrices, keyed by (in, out) alone. A dense model still gets
        exactly the batching described above (every tensor is already a single matrix). A MoE model
        now gets what the cap is supposed to guarantee: `_MUON_MAX_STACK` matrices per orthogonalize
        call regardless of whether they came from many layers, many experts within one layer, or
        both — e.g. 4 MoE layers x 48 experts = 192 matrices chunks into ceil(192/32)=6 calls of
        <=32 each, the same transient buffer size a plain 32-tensor dense stack would need.

        The transient Newton-Schulz buffers (A = X @ X.mT, B = b*A + c*(A@A), B @ X) scale with this
        matrix-batch size, not with total parameter count, which is why the fix bounds it directly
        rather than trying to predict a safe n_experts from parameter counts. CPU-offload can't
        stand in for this: it only migrates AdamW-owned tensors (the tied embedding, norms, routers
        — a small fraction of a MoE model's parameters), while this pressure is entirely Muon-side.
        """
        lr, weight_decay = group["lr"], group["weight_decay"]
        momentum, nesterov = group["momentum"], group["nesterov"]

        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]
        bufs = []
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            bufs.append(state["momentum_buffer"])

        torch._foreach_lerp_(bufs, grads, 1.0 - momentum)
        # Nesterov: step from where the momentum is about to be, not where it is.
        updates = torch._foreach_lerp(grads, bufs, momentum) if nesterov else bufs

        if weight_decay != 0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

        # Flatten every tensor into its individual (in, out) matrices: a plain 2-D weight unbinds to
        # itself (one entry), a BatchedExperts-style (n_experts, in, out) tensor unbinds to
        # n_experts entries. Each entry remembers where its update belongs — (param index, expert
        # index or None) — so the result can be scattered back after orthogonalizing.
        by_shape: dict[tuple[int, ...], list[tuple[int, int | None, torch.Tensor]]] = {}
        for i, p in enumerate(params):
            u = updates[i]
            if u.dim() == 2:
                by_shape.setdefault(tuple(u.shape), []).append((i, None, u))
            else:
                mat_shape = tuple(u.shape[-2:])
                for e, sub in enumerate(u.unbind(0)):
                    by_shape.setdefault(mat_shape, []).append((i, e, sub))

        for shape, entries in by_shape.items():
            # Newton-Schulz normalises the update's singular values to ~1, so its Frobenius norm is
            # ~sqrt(min(rows, cols)) regardless of the parameter's shape. This factor restores the
            # standard sqrt(fan-out / fan-in) scaling, which is also what makes Muon approximately
            # muP-correct without a separate width correction (see build_muon_param_groups). It
            # depends only on the matrix shape, so it is constant across a batch.
            scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
            for start in range(0, len(entries), _MUON_MAX_STACK):
                chunk = entries[start : start + _MUON_MAX_STACK]
                if len(chunk) == 1:
                    i, e, u = chunk[0]
                    target = params[i] if e is None else params[i][e]
                    target.add_(orthogonalize(u, steps=group["ns_steps"]), alpha=-lr * scale)
                    continue
                stacked = orthogonalize(
                    torch.stack([u for _, _, u in chunk]), steps=group["ns_steps"]
                )
                targets = [params[i] if e is None else params[i][e] for i, e, _ in chunk]
                torch._foreach_add_(targets, list(stacked.unbind(0)), alpha=-lr * scale)

    def _step_adamw(self, group: dict) -> None:
        lr, weight_decay = group["lr"], group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        offload = group["offload"]
        state_device = "cpu" if offload else None

        params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = _new_state_like(p, state_device)
                state["exp_avg_sq"] = _new_state_like(p, state_device)
                if offload:
                    state["grad_cpu"] = torch.empty_like(p, device="cpu").pin_memory()
                state["step"] = 0
            state["step"] += 1
            if offload:
                state["grad_cpu"].copy_(p.grad, non_blocking=True)
                grads.append(state["grad_cpu"])
            else:
                grads.append(p.grad)
            params.append(p)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])

        if not params:
            return
        # Grad copies above were issued non_blocking against pinned buffers; sync once (not
        # per-param) before touching them on the CPU side.
        if offload and self._is_cuda:
            torch.cuda.synchronize(self._device)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        # Every param in a group shares a step count in practice (they enter the group together and
        # step together), so the bias corrections are the same scalar for all of them and the whole
        # update batches through torch._foreach_*. Falling back to the per-parameter path when they
        # ever diverge — a param whose grad was None for some steps — keeps that assumption honest
        # rather than silently applying one param's correction to another's moments.
        if weight_decay != 0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

        if len(set(steps)) == 1:
            bias_correction1 = 1 - beta1 ** steps[0]
            bias_correction2 = 1 - beta2 ** steps[0]
            denom = torch._foreach_div(exp_avg_sqs, bias_correction2)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, eps)
            updates = torch._foreach_div(exp_avgs, denom)
            if offload:
                # CPU-resident moments: one copy up per param, then a device-side foreach add.
                updates = [u.to(p.device, non_blocking=True) for u, p in zip(updates, params)]
            torch._foreach_add_(params, updates, alpha=-lr / bias_correction1)
        else:
            for p, exp_avg, exp_avg_sq, step in zip(params, exp_avgs, exp_avg_sqs, steps):
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                update = (exp_avg / bias_correction1) / denom
                p.add_(update.to(p.device, non_blocking=True), alpha=-lr)

        if offload and self._is_cuda:
            torch.cuda.synchronize(self._device)


def _new_state_like(p: torch.Tensor, device: str | None) -> torch.Tensor:
    """Zeroed optimizer state for `p`, matching its dtype (fp32, or bf16 under train.native_bf16),
    pinned on the CPU when offloading.

    torch.optim.AdamW does this automatically (its exp_avg/exp_avg_sq match the parameter's own
    dtype); MuonWithAuxAdam's hand-written _step_adamw used to hardcode float32 here, which was
    invisible before train.native_bf16 existed (every parameter was fp32 anyway) and would have
    silently doubled the AdamW-owned groups' state memory the moment a bf16-native model tried to
    use it.
    """
    if device is None:
        return torch.zeros_like(p)
    return torch.zeros_like(p, device=device).pin_memory()


def build_optimizer(model: torch.nn.Module, cfg: Config, device: str) -> torch.optim.Optimizer:
    """Construct the optimizer named by cfg.train.optimizer."""
    if cfg.train.optimizer == "adamw":
        from torch.optim import AdamW

        return AdamW(
            build_param_groups(
                model, cfg.train.weight_decay, cfg.train.embed_lr, cfg.train.hyper_conn_lr,
                lr=cfg.train.lr, mup_width_mult=cfg.model.mup_width_mult,
            ),
            lr=cfg.train.lr,
            fused=(device.split(":")[0] == "cuda"),
        )
    if cfg.train.optimizer == "muon":
        return MuonWithAuxAdam(
            build_muon_param_groups(model, cfg),
            momentum=cfg.train.muon_momentum,
            device=device,
        )
    raise ValueError(f"Unknown train.optimizer {cfg.train.optimizer!r}, expected 'muon' or 'adamw'")


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW with exp_avg/exp_avg_sq kept in pinned CPU memory instead of on `device`, freeing the
    ~2x num_params fp32 moment-buffer VRAM cost for the rest of the run. Params/grads stay
    GPU-resident the whole time — only a grad copy (down) and the resulting update (up) cross PCIe,
    once per step() call rather than once per forward/backward — so this only touches optimizer
    bookkeeping, never the forward path or torch.compile's captured graph. Used as auto_batch_size's
    second OOM-recovery tier once chunk-size backoff (micro_chunk_size == 1) is exhausted — see
    migrate_optimizer_to_cpu_offload and the OOM handler in train()."""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        device: str,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))
        self._device = device
        self._is_cuda = device.split(":")[0] == "cuda"

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, weight_decay = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            params, grads_cpu, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, device="cpu", dtype=torch.float32).pin_memory()
                    state["exp_avg_sq"] = torch.zeros_like(p, device="cpu", dtype=torch.float32).pin_memory()
                    state["grad_cpu"] = torch.empty_like(p, device="cpu", dtype=torch.float32).pin_memory()
                    state["step"] = 0
                state["grad_cpu"].copy_(p.grad, non_blocking=True)
                state["step"] += 1
                params.append(p)
                grads_cpu.append(state["grad_cpu"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])

            if not params:
                continue

            # Every grad copy above was issued non_blocking against a pinned buffer; sync once here
            # (rather than per-param) before touching them on the CPU side.
            if self._is_cuda:
                torch.cuda.synchronize(self._device)

            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads_cpu, alpha=1 - beta1)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_cpu, grads_cpu, value=1 - beta2)

            for p, exp_avg, exp_avg_sq, step in zip(params, exp_avgs, exp_avg_sqs, steps):
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                update = (exp_avg / bias_correction1) / denom
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update.to(p.device, non_blocking=True), alpha=-lr)

            if self._is_cuda:
                torch.cuda.synchronize(self._device)

        return loss


def migrate_optimizer_to_cpu_offload(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module, cfg: Config, device: str
) -> torch.optim.Optimizer:
    """Move optimizer state to pinned CPU memory — auto_batch_size's tier-2 OOM recovery.

    Returns the optimizer to use from here on. For MuonWithAuxAdam that is the *same object*,
    mutated in place, so the caller can skip rebuilding the LR scheduler; for a plain AdamW it is a
    new CPUOffloadAdamW and the caller must rebuild. Callers should test identity:

        optimizer = migrate_optimizer_to_cpu_offload(optimizer, model, cfg, device)

    and only rebuild the scheduler when the returned object differs from the one passed in.
    """
    if isinstance(optimizer, MuonWithAuxAdam):
        return _offload_muon_aux_adam(optimizer)
    return _adamw_to_cpu_offload(optimizer, model, cfg, device)


def _offload_muon_aux_adam(optimizer: MuonWithAuxAdam) -> MuonWithAuxAdam:
    """Flip the adamw groups of a live MuonWithAuxAdam to CPU-resident moments, in place.

    Only the adamw groups: Muon's state is a single momentum buffer (half AdamW's footprint), and
    its step is a chain of matmuls per parameter, so keeping that state off-device would cost far
    more time than the VRAM it returns. In a Muon run the AdamW groups hold the embedding matrix,
    which is typically the single largest tensor in the model, so this still reclaims most of what
    the tier is after.

    Deliberately upcasts to fp32 here regardless of the live state's dtype (bf16 under
    train.native_bf16): this only runs once VRAM is already tight enough to trigger a two-tier OOM
    escalation, at which point CPU host memory — not the halved footprint bf16 storage buys — is
    the resource under pressure, and the accumulated moments benefit from the extra precision on
    the (now much cheaper) CPU side.
    """
    for group in optimizer.param_groups:
        if group["algorithm"] != "adamw" or group["offload"]:
            continue
        group["offload"] = True
        for p in group["params"]:
            state = optimizer.state[p]
            if "exp_avg" not in state:
                continue  # never stepped; lazy-init on the next step() will allocate on CPU
            state["exp_avg"] = state["exp_avg"].detach().to("cpu", dtype=torch.float32).pin_memory()
            state["exp_avg_sq"] = state["exp_avg_sq"].detach().to("cpu", dtype=torch.float32).pin_memory()
            state["grad_cpu"] = torch.empty_like(p, device="cpu", dtype=torch.float32).pin_memory()
    return optimizer


def _adamw_to_cpu_offload(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module, cfg: Config, device: str
) -> CPUOffloadAdamW:
    """Swap a live AdamW for CPUOffloadAdamW, migrating any existing exp_avg/exp_avg_sq/step per
    param onto pinned CPU tensors instead of resetting momentum. Params with no prior state (e.g.
    training OOM'd before its first successful step) are left to lazy-init on first step(), matching
    fresh-AdamW behavior."""
    new_optimizer = CPUOffloadAdamW(
        # Argument-identical to build_optimizer's call above, which is what keeps the positional
        # group zip below valid — muP's extra embedding group must appear in both or neither.
        build_param_groups(
            model, cfg.train.weight_decay, cfg.train.embed_lr, cfg.train.hyper_conn_lr,
            lr=cfg.train.lr, mup_width_mult=cfg.model.mup_width_mult,
        ),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        device=device,
    )
    # LambdaLR.__init__ requires 'initial_lr' already present in param_groups whenever it's
    # constructed with last_epoch != -1 (its "resuming a schedule" path) — copy it over from the old
    # optimizer so the caller can rebuild the scheduler against new_optimizer at the current step.
    for old_group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
        if "initial_lr" in old_group:
            new_group["initial_lr"] = old_group["initial_lr"]
    for p, old_state in optimizer.state.items():
        if "exp_avg" not in old_state:
            continue
        step = old_state["step"]
        new_optimizer.state[p] = {
            "exp_avg": old_state["exp_avg"].detach().to("cpu", dtype=torch.float32).pin_memory(),
            "exp_avg_sq": old_state["exp_avg_sq"].detach().to("cpu", dtype=torch.float32).pin_memory(),
            "grad_cpu": torch.empty_like(p, device="cpu", dtype=torch.float32).pin_memory(),
            "step": step.item() if torch.is_tensor(step) else step,
        }
    return new_optimizer
