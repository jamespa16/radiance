from __future__ import annotations

import argparse

import wandb

from radiance.config import load_config
from radiance.train import train


def apply_sweep_overrides(cfg, sweep_config: dict) -> None:
    """Overwrite fields on cfg.data/model/train with values wandb.agent injected via wandb.config."""
    for section_name in ("data", "model", "train"):
        overrides = sweep_config.get(section_name)
        if not overrides:
            continue
        section = getattr(cfg, section_name)
        for key, value in dict(overrides).items():
            if not hasattr(section, key):
                raise ValueError(f"Unknown sweep parameter '{section_name}.{key}'")
            setattr(section, key, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Base config.yaml to layer sweep overrides onto")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Joins the sweep this process was launched under (via WANDB_SWEEP_ID etc. in the
    # environment) and populates wandb.config with this run's sampled hyperparameters.
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode)

    apply_sweep_overrides(cfg, wandb.config)
    cfg.run_name = wandb.run.name
    cfg.train.output_dir = f"{cfg.train.output_dir}/{wandb.run.id}"

    train(cfg)


if __name__ == "__main__":
    main()
