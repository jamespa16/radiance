"""End-to-end SFT integration test.

docs/post-training.md notes SFT is unit-tested (chat masking in test_sft_data.py, the loss in
test_sft_loss.py) but "no end-to-end train() integration test exists" — nothing has ever actually
run radiance.train.train() in SFT mode. This exercises exactly that: a tiny model, a small
in-memory chat dataset (load_dataset and build_tokenizer are both faked so this stays on CPU with
no network), a handful of steps, and checks the loss trends down and the resulting checkpoint
reloads and forwards cleanly — the class of wiring bug the compiled-path tests exist to catch
elsewhere (see CLAUDE.md's "the compiled path is the real path"), just for the SFT branch instead.
"""

from __future__ import annotations

import torch
from datasets import Dataset, DatasetDict

import radiance.sft_data as sft_data_mod
import radiance.train as train_mod
from radiance.checkpointing import save_checkpoint
from radiance.config import Config, DataConfig, ModelConfig, SFTConfig, TrainConfig, WandbConfig
from radiance.model import DenseTransformer, load_transformer_from_checkpoint
from radiance.optim import build_optimizer

from tests._fake_tokenizer import WordTokenizer

TINY_VOCAB = 64


class _FakeWandb:
    """Stands in for the wandb module inside train.py's namespace: wandb.init() in "disabled"
    mode reassigns its own log()/run internals in ways that silently swallow a plain
    monkeypatch.setattr(wandb, "log", ...) applied beforehand, so this replaces the whole module
    reference instead — the test only needs to observe what train() would have logged, not
    exercise the real wandb SDK (that's wandb's own test suite's job, not ours)."""

    def __init__(self):
        self.run = None
        self.logged: list[tuple[int | None, dict]] = []

    def init(self, **kwargs):
        self.run = object()

    def log(self, data, step=None):
        self.logged.append((step, dict(data)))

    def finish(self):
        self.run = None
_WORDS = ["hello", "there", "how", "are", "you", "today", "i", "am", "fine", "thanks", "goodbye", "friend"]


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        head_dim=8,
        n_layers=3,
        ffn_mult=2.0,
        ffn_depth=1,
        dropout=0.0,  # deterministic loss trend, not testing regularization here
        max_seq_len=32,
        vocab_pad_multiple=1,  # keeps the run's vocab_size exactly TINY_VOCAB, matching the base
        # checkpoint below bit-for-bit (see checkpointing.load_pretrained_weights's vocab check).
    )


def _sft_messages(n: int) -> DatasetDict:
    # Small fixed vocabulary, well under TINY_VOCAB, repeated with enough total tokens that
    # packing (multiple examples per seq_len=12 block) yields several full training batches
    # despite the DataLoader's drop_last=True.
    examples = []
    for i in range(n):
        u = " ".join(_WORDS[i % 4 : i % 4 + 3])
        a = " ".join(_WORDS[(i + 4) % 8 : (i + 4) % 8 + 3])
        examples.append([{"role": "user", "content": u}, {"role": "assistant", "content": a}])
    return DatasetDict({"train": Dataset.from_dict({"messages": examples})})


def _save_base_checkpoint(tmp_path, model_cfg: ModelConfig):
    cfg = Config(model=model_cfg, train=TrainConfig(device="cpu", compile=False))
    raw_model = DenseTransformer(model_cfg, vocab_size=TINY_VOCAB)
    optimizer = build_optimizer(raw_model, cfg, "cpu")
    scheduler = train_mod.build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "base.pt"
    save_checkpoint(path, raw_model, optimizer, scheduler, scaler, step=0, cfg=cfg)
    return path


def test_sft_train_loop_runs_end_to_end_and_loss_decreases(tmp_path, monkeypatch):
    model_cfg = _model_cfg()
    base_ckpt = _save_base_checkpoint(tmp_path, model_cfg)

    monkeypatch.setattr(sft_data_mod, "load_dataset", lambda name: _sft_messages(64))
    tokenizer = WordTokenizer(vocab_size=TINY_VOCAB)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg: tokenizer)

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_mod, "wandb", fake_wandb)

    max_steps = 30
    cfg = Config(
        run_name="test-sft-e2e",
        data=DataConfig(tokenizer="unused", seq_len=12, num_workers=0),
        model=model_cfg,
        sft=SFTConfig(enabled=True, dataset="unused", cache_dir=str(tmp_path / "sft_cache")),
        train=TrainConfig(
            init_from=str(base_ckpt),
            batch_size=4,
            auto_batch_size=False,
            max_steps=max_steps,
            log_every=1,
            eval_every=10_000,  # no validation split here; just keep evaluate() from ever firing
            save_every=max_steps,  # exactly one checkpoint, at the final step
            output_dir=str(tmp_path / "out"),
            device="cpu",
            compile=False,
        ),
        wandb=WandbConfig(mode="disabled"),
    )

    train_mod.train(cfg)

    losses = [data["train/loss"] for _, data in fake_wandb.logged if "train/loss" in data]
    assert len(losses) == max_steps, "expected one train/loss log per step at log_every=1"
    early = sum(losses[:3]) / 3
    late = sum(losses[-3:]) / 3
    assert late < early, f"SFT loss did not decrease over the run: early={early:.4f} late={late:.4f}"

    ckpt_path = tmp_path / "out" / f"step_{max_steps}.pt"
    assert ckpt_path.exists(), "expected exactly one checkpoint written at the final step"
    model, _ = load_transformer_from_checkpoint(str(ckpt_path), "cpu")
    logits = model(torch.randint(1, TINY_VOCAB, (2, 8))).logits
    assert logits.shape == (2, 8, TINY_VOCAB)
