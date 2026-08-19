"""End-to-end DPO integration test.

docs/post-training.md notes DPO's data pipeline, loss, and reference-logprob precompute/caching
are each unit-tested but "no end-to-end train() integration test exists" — nothing has ever run
radiance.train.train() in DPO mode and let its pieces (reference-logprob precompute against a real
checkpoint, pair packing, the doc_attention_mask auto-disable) wire together for real. This
exercises exactly that: a tiny reference/policy checkpoint, a small in-memory preference dataset
(load_dataset and build_tokenizer are both faked so this stays on CPU with no network), a handful
of steps, and checks the loss trends down and the resulting checkpoint reloads and forwards
cleanly.
"""

from __future__ import annotations

import torch
from datasets import Dataset, DatasetDict

import radiance.dpo_data as dpo_data_mod
import radiance.train as train_mod
from radiance.checkpointing import save_checkpoint
from radiance.config import Config, DataConfig, DPOConfig, ModelConfig, TrainConfig, WandbConfig
from radiance.model import DenseTransformer, load_transformer_from_checkpoint
from radiance.optim import build_optimizer

from tests._fake_tokenizer import WordTokenizer
from tests.test_sft_train_e2e import _FakeWandb

TINY_VOCAB = 64
_WORDS = ["hello", "there", "how", "are", "you", "today", "i", "am", "fine", "thanks", "goodbye", "friend"]


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        head_dim=8,
        n_layers=3,
        ffn_mult=2.0,
        ffn_depth=1,
        dropout=0.0,
        max_seq_len=32,
        vocab_pad_multiple=1,  # keeps the run's vocab_size exactly TINY_VOCAB, matching both the
        # policy-seed and reference checkpoints below bit-for-bit.
    )


def _dpo_pairs(n: int) -> DatasetDict:
    chosen, rejected = [], []
    for i in range(n):
        prompt = [{"role": "user", "content": " ".join(_WORDS[i % 4 : i % 4 + 2])}]
        chosen.append(prompt + [{"role": "assistant", "content": " ".join(_WORDS[4:7])}])
        rejected.append(prompt + [{"role": "assistant", "content": " ".join(_WORDS[7:10])}])
    return DatasetDict({"train": Dataset.from_dict({"chosen": chosen, "rejected": rejected})})


def _save_base_checkpoint(tmp_path, model_cfg: ModelConfig, name: str):
    # data.tokenizer must match the run's below: _add_reference_logprobs refuses a reference
    # checkpoint whose saved data.tokenizer disagrees, since that would mean scoring this run's
    # token ids against a different vocabulary.
    cfg = Config(data=DataConfig(tokenizer="unused"), model=model_cfg, train=TrainConfig(device="cpu", compile=False))
    raw_model = DenseTransformer(model_cfg, vocab_size=TINY_VOCAB)
    optimizer = build_optimizer(raw_model, cfg, "cpu")
    scheduler = train_mod.build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / name
    save_checkpoint(path, raw_model, optimizer, scheduler, scaler, step=0, cfg=cfg)
    return path


def test_dpo_train_loop_runs_end_to_end_and_loss_decreases(tmp_path, monkeypatch):
    model_cfg = _model_cfg()
    # Same architecture seeds both the policy (init_from) and the frozen reference
    # (dpo.reference_checkpoint) — the common case per configs/tinystories_dpo.yaml's worked
    # example, and it's the reference checkpoint that exercises the precompute pass under test.
    base_ckpt = _save_base_checkpoint(tmp_path, model_cfg, "base.pt")

    monkeypatch.setattr(dpo_data_mod, "load_dataset", lambda name: _dpo_pairs(32))
    tokenizer = WordTokenizer(vocab_size=TINY_VOCAB)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg: tokenizer)

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_mod, "wandb", fake_wandb)

    max_steps = 20
    cache_dir = tmp_path / "dpo_cache"
    cfg = Config(
        run_name="test-dpo-e2e",
        data=DataConfig(tokenizer="unused", seq_len=16, num_workers=0),
        model=model_cfg,
        dpo=DPOConfig(
            enabled=True,
            dataset="unused",
            reference_checkpoint=str(base_ckpt),
            cache_dir=str(cache_dir),
        ),
        train=TrainConfig(
            init_from=str(base_ckpt),
            batch_size=2,
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

    # resolve_dpo_doc_mask must have fired for real inside train(): DPO's "packing of one" makes
    # doc_attention_mask a no-op, and docs/post-training.md's comment on configs/tinystories_dpo.yaml
    # says a DPO run turns it off for itself at startup — this is the first test to actually run
    # train() far enough to see that happen rather than calling resolve_dpo_doc_mask directly.
    assert cfg.model.doc_attention_mask is False

    # The reference-logprob precompute pass actually ran against a real checkpoint and cached its
    # output to disk (not stubbed out, unlike the unit-level test in test_dpo_reference_logprob.py).
    assert any(cache_dir.iterdir()), "expected the reference-logprob precompute to populate dpo.cache_dir"

    losses = [data["train/loss"] for _, data in fake_wandb.logged if "train/loss" in data]
    assert len(losses) == max_steps, "expected one train/loss log per step at log_every=1"
    early = sum(losses[:3]) / 3
    late = sum(losses[-3:]) / 3
    assert late < early, f"DPO loss did not decrease over the run: early={early:.4f} late={late:.4f}"

    ckpt_path = tmp_path / "out" / f"step_{max_steps}.pt"
    assert ckpt_path.exists(), "expected exactly one checkpoint written at the final step"
    model, _ = load_transformer_from_checkpoint(str(ckpt_path), "cpu")
    logits = model(torch.randint(1, TINY_VOCAB, (2, 8))).logits
    assert logits.shape == (2, 8, TINY_VOCAB)
