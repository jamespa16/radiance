from __future__ import annotations

from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from radiance.config import Config, ModelConfig
from radiance.model import DenseTransformer, checkpoint_vocab_size
def save_checkpoint(
    path: Path,
    raw_model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    step: int,
    cfg: Config,
) -> None:
    """Write a resumable checkpoint.

    Everything needed to continue the run goes in, not just the weights: without the optimizer's
    moment buffers a resumed run restarts AdamW from zero momentum, which shows up as a visible
    loss spike, and without the scheduler/step the LR trajectory restarts from warmup. `config` is
    the full Config object (pickled), which is what generate.py reads back.

    Not captured: the DataLoader's position, and RNG state. Those two trade off against each other,
    and the resolution here favours data freshness — train() re-seeds off the resumed step, so the
    loader draws a *different* shuffle order rather than replaying batches the run already trained
    on (which is what restoring RNG state verbatim would cause). The cost is that dropout draws a
    different mask stream, so a resumed run is statistically equivalent to an uninterrupted one
    rather than bit-identical. With dropout disabled, resume reproduces an uninterrupted run
    exactly — model weights, AdamW moments, and LR schedule all match to the bit.
    """
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # Which algorithm produced that state. Muon's per-param state is a single momentum
            # buffer where AdamW's is two moments, so loading one into the other silently produces
            # a wrong (or shape-mismatched) resume — see train()'s resume block, which resets
            # optimizer state with a warning rather than crashing when these disagree.
            "optimizer_type": cfg.train.optimizer,
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def find_resume_checkpoint(cfg: Config) -> Path | None:
    """Resolve cfg.train.resume_from to an actual checkpoint path.

    "auto" picks the highest-numbered step_*.pt in output_dir, so a config can be re-launched
    unchanged after an interruption and simply continue. Returns None when there's nothing to
    resume from (including "auto" against an empty/missing output_dir, which is a fresh run, not
    an error) — an explicit path that doesn't exist *is* an error, since silently starting a
    50-hour run from scratch is far worse than failing loudly.
    """
    if not cfg.train.resume_from:
        return None
    if cfg.train.resume_from != "auto":
        path = Path(cfg.train.resume_from)
        if not path.exists():
            raise FileNotFoundError(f"train.resume_from={cfg.train.resume_from!r} does not exist")
        return path
    candidates = sorted(
        Path(cfg.train.output_dir).glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return candidates[-1] if candidates else None


def _loop_conditioning_signature(cfg: ModelConfig) -> tuple[str, int]:
    """The (mode, variant count) that actually allocates per-iteration parameters.

    Raw ``loop_iter_conditioning`` is inert when ``loop_multiplier == 1`` (a single iteration has
    nothing to condition on), so all three modes collapse to the same parameter allocation there.
    """
    if cfg.loop_iter_conditioning == "none" or cfg.loop_multiplier == 1:
        return ("none", 1)
    return (cfg.loop_iter_conditioning, cfg.loop_multiplier)


def load_pretrained_weights(raw_model: DenseTransformer, path: str, cfg: Config, vocab_size: int, device: str) -> None:
    """Load model weights only from a checkpoint (train.init_from) — no optimizer/scheduler/step.

    The counterpart to find_resume_checkpoint's "continue this exact run" behavior: this seeds a
    *new* run (e.g. SFT) from a previously trained model's weights, so the caller builds a fresh
    optimizer/scheduler from its own cfg.train afterward rather than restoring saved ones.

    Checks the checkpoint's saved model shape against this run's cfg.model before touching
    load_state_dict, since a mismatch there would otherwise surface as an opaque tensor-shape
    RuntimeError deep inside torch rather than a clear message naming the field that disagrees.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    source_cfg = ckpt["config"].model
    source_vocab = checkpoint_vocab_size(ckpt)
    mismatches = []
    if source_vocab != vocab_size:
        mismatches.append(f"vocab_size: checkpoint={source_vocab}, this run={vocab_size}")
    for field_name in (
        "d_model",
        "n_layers",
        "head_dim",
        "use_diff_attn",
        "hyper_conn_streams",
    ):
        source_val, this_val = getattr(source_cfg, field_name), getattr(cfg.model, field_name)
        if source_val != this_val:
            mismatches.append(f"model.{field_name}: checkpoint={source_val!r}, this run={this_val!r}")
    # Compare the resolved GQA count rather than the raw field: n_kv_heads=None means n_heads, so a
    # checkpoint written with the default (None) and a run that pins the resolved value must
    # still match, and only the resolved value determines the qkv projection's shape.
    source_gqa = source_cfg.n_kv_heads_resolved
    this_gqa = cfg.model.n_kv_heads_resolved
    if source_gqa != this_gqa:
        mismatches.append(
            f"model.n_kv_heads: checkpoint={source_gqa}, this run={this_gqa} (resolved; None -> n_heads)"
        )
    # Compare the *resolved* loop-conditioning allocation rather than the raw field: at
    # loop_multiplier == 1 the modes are all inert (a single iteration has nothing to condition
    # on), so a raw comparison would reject a checkpoint whose mode happened to differ.
    source_loop = _loop_conditioning_signature(source_cfg)
    this_loop = _loop_conditioning_signature(cfg.model)
    # RMSNorm._broadcast_legacy_gain lets a single 1-D gain seed every variant, so a
    # pre-conditioning checkpoint is a valid source for the norm_gains variant bank. It cannot
    # seed the lora branch (whose keys the source never had) or shrink an existing variant bank
    # back to 1-D.
    if source_loop != this_loop and not (source_loop == ("none", 1) and this_loop[0] == "norm_gains"):
        mismatches.append(
            "model.loop_iter_conditioning: "
            f"checkpoint={source_loop[0]!r} (variants={source_loop[1]}), "
            f"this run={this_loop[0]!r} (variants={this_loop[1]}) "
            "(resolved; inert when loop_multiplier == 1)"
        )
    if (
        source_loop[0] == this_loop[0] == "lora"
        and source_cfg.loop_lora_rank != cfg.model.loop_lora_rank
    ):
        mismatches.append(
            f"model.loop_lora_rank: checkpoint={source_cfg.loop_lora_rank}, "
            f"this run={cfg.model.loop_lora_rank}"
        )
    if mismatches:
        raise ValueError(
            f"train.init_from={path!r} has an incompatible model shape:\n  "
            + "\n  ".join(mismatches)
            + "\ninit_from loads weights into an already-constructed model, so shapes must match exactly."
        )
    raw_model.load_state_dict(ckpt["model"])
