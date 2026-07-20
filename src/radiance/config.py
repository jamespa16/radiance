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
    dropout: float = 0.1
    max_seq_len: int = 512
    rope_theta: float = 10000.0  # RoPE base frequency (Su et al. 2021)

    @property
    def n_heads(self) -> int:
        if self.d_model % self.head_dim != 0:
            raise ValueError(f"model.d_model ({self.d_model}) must be divisible by model.head_dim ({self.head_dim})")
        return self.d_model // self.head_dim

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
    max_steps: int = 5000  # ignored (overwritten once the model is built) if tokens_per_param is set
    tokens_per_param: float | None = None  # opt-in: derive max_steps from model size instead of a fixed step
    # count — max_steps = round(tokens_per_param * num_parameters / (effective_batch_size * data.seq_len)),
    # computed in train.py once the model is built. Chinchilla-optimal is ~20 tokens/param.
    auto_batch_size: bool = False  # opt-in: overwrite batch_size/grad_accum_steps at startup, computed from
    # free VRAM + model size (see train.py's estimate_batch_size) instead of the values configured above.
    # CUDA-only; requires target_effective_batch_size to be set. Also enables OOM backoff during training:
    # a CUDA OOM shrinks the internal per-forward-pass chunk size and retries the step instead of ending the
    # run (see train.py's main loop) — this backoff never fires when auto_batch_size is False, so a manually
    # chosen/swept batch_size always behaves exactly as configured.
    target_effective_batch_size: int | None = None  # required when auto_batch_size is True: grad_accum_steps
    # is derived as ceil(target_effective_batch_size / computed batch_size).
    vram_safety_margin: float = 0.5  # only used when auto_batch_size is True: fraction of the (already
    # conservative) estimated max token budget to actually use. Lower = more conservative.
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = "checkpoints/run"
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
