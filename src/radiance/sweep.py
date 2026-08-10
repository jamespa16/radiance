from __future__ import annotations

import argparse

import wandb

from radiance.config import load_config
from radiance.train import train


# Sweep configs can't express string values under bayes method, so these
# integer-encoded parameters are mapped back to strings after override.
_SWEEP_INT_TO_STR = {
    "model.loop_iter_conditioning": {0: "none", 1: "norm_gains", 2: "lora"},
}


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
            full_key = f"{section_name}.{key}"
            if full_key in _SWEEP_INT_TO_STR:
                mapping = _SWEEP_INT_TO_STR[full_key]
                if value not in mapping:
                    raise ValueError(f"{full_key}: sweep value {value} not in mapping {mapping}")
                value = mapping[value]
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
