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
    # stability across blocks[1:]'s weight-shared loop iterations. Defaults to True (a behavior change
    # for every existing config) unlike this file's usual opt-in-False convention — see CLAUDE.md.
    n_layers: int = 6
    loop_count: int = 1
    use_router: bool = False  # opt-in: replace fixed loop_count with per-token ACT halting
    max_loops: int = 6  # hard cap on loop iterations when use_router=True; independent of loop_count
    ponder_weight: float = 1.0e-2  # tau: coefficient on the ponder-cost loss term
    halt_epsilon: float = 0.01  # ACT epsilon: a position halts once cumulative halting prob >= 1 - halt_epsilon
    act_ffn_capacity_ratio: float = 1.0  # fraction of batch*seq_len tokens the FFN sublayer actually
    # processes per interior ACT loop iteration (first/last iteration always run fully dense — see
    # DenseTransformer._forward_act). 1.0 (default) disables the fixed-capacity sparse-FFN path
    # entirely, so the loop is byte-for-byte identical to the fully-dense implementation.
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
    # ponder_weight's role/placement for ACT's ponder cost.
    moe_dense_every: int | None = None  # opt-in: every Nth block (1-indexed by position within
    # blocks[1:]) uses a plain dense FeedForward instead of MoEFeedForward even when use_moe=True.
    # None (default) means every block in blocks[1:] is MoE.
    dropout: float = 0.1
    max_seq_len: int = 512
    rope_theta: float = 10000.0  # RoPE base frequency (Su et al. 2021)
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


@dataclass
class TrainConfig:
    batch_size: int = 32  # micro-batch size: what one forward/backward pass consumes
    grad_accum_steps: int = 1  # micro-batches (of batch_size each) accumulated per optimizer.step();
    # effective_batch_size = batch_size * grad_accum_steps. Raise this instead of batch_size to grow the
    # effective batch beyond what fits in VRAM.
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.04  # warmup_steps = round(max_steps * warmup_ratio)
    min_lr_ratio: float = 0.1  # cosine decays to min_lr_ratio * lr, not to 0 — the tail of a run at
    # a ~0 LR contributes nothing. 0.0 restores a decay-all-the-way-to-zero schedule.
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
