from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    text_column: str = "text"
    tokenizer: str = "gpt2"
    seq_len: int = 512
    num_workers: int = 4
    cache_dir: str = ".cache/radiance/tokenized"


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    loop_count: int = 1
    ffn_dim: int = 1024
    ffn_depth: int = 2
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = "checkpoints/run"
    seed: int = 42
    device: str = "cuda"
    compile: bool = True


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
