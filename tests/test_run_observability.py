"""What a run reports about itself: logged metrics, throughput/MFU, and a validation number that
means the same thing every time it is read.

The val-shuffle test is the one with history. V1's validation stream was shuffled and each eval
read only its first eval_max_batches, so every eval scored a different random subset of the
held-out set — val/loss moved by ~0.02 between evals with no change in the model.
"""

from __future__ import annotations

import math

import pytest
import torch
from datasets import DatasetDict, IterableDataset

import radiance.data as d
import radiance.train as train_mod
from radiance.checkpointing import prune_checkpoints
from radiance.config import Config, DataConfig, DPOConfig, ModelConfig, TrainConfig
from radiance.evaluation import evaluate, evaluate_detailed
from radiance.model import DenseTransformer


# --- logged metrics ---------------------------------------------------------------------------


def test_plain_pretraining_logs_no_series_that_are_zero_by_construction():
    assert train_mod.logged_metrics(Config()) == ("loss", "lm_loss", "z_loss")


def test_each_feature_brings_its_own_series():
    moe = Config(model=ModelConfig(use_moe=True))
    assert "moe_aux_loss" in train_mod.logged_metrics(moe)
    router = Config(model=ModelConfig(use_router=True))
    assert {"ponder_cost", "mean_loop_depth"} <= set(train_mod.logged_metrics(router))
    mtp = Config(model=ModelConfig(mtp_heads=2))
    assert "mtp_loss" in train_mod.logged_metrics(mtp)
    dpo = train_mod.logged_metrics(Config(dpo=DPOConfig(enabled=True)))
    assert {"dpo_margin", "dpo_margin_accuracy", "dpo_reward_accuracy"} <= set(dpo)
    assert "z_loss" not in dpo  # DPO's total omits it


def test_logged_metrics_is_a_subset_of_what_is_accumulated():
    for cfg in (Config(), Config(model=ModelConfig(use_moe=True, use_router=True, mtp_heads=2))):
        assert set(train_mod.logged_metrics(cfg)) <= set(train_mod._ACCUM_METRICS)


# --- FLOPs and throughput ---------------------------------------------------------------------


def _model(**kw) -> DenseTransformer:
    cfg = ModelConfig(d_model=64, head_dim=16, n_layers=4, max_seq_len=32, **kw)
    return DenseTransformer(cfg, vocab_size=128)


def test_dense_flops_are_the_palm_formula():
    m = _model()
    assert m.training_flops_per_token(32) == 6 * m.num_active_parameters() + 12 * 4 * 64 * 32


def test_the_loop_is_charged_for_every_pass():
    """num_active_parameters counts the shared loop body once; the FLOPs must count it per pass."""
    once, thrice = _model(), _model(loop_count=3)
    body = sum(p.numel() for p in thrice.blocks[1:].parameters())
    extra_params = thrice.num_active_parameters() - once.num_active_parameters()  # loop-only params
    expected = (
        once.training_flops_per_token(32)
        + 6 * extra_params
        + 6 * 2 * body
        + 12 * 2 * (4 - 1) * 64 * 32
    )
    assert thrice.training_flops_per_token(32) == expected


def test_throughput_meter_skips_warmup_and_excludes_eval(monkeypatch):
    clock = iter([0.0, 100.0, 100.0, 110.0, 110.0])  # start; each read() reads, then restarts
    monkeypatch.setattr(train_mod.time, "perf_counter", lambda: next(clock))
    meter = train_mod.ThroughputMeter(tokens_per_step=1000, flops_per_token=6.0, peak_tflops=1e-9)
    meter.step()
    assert meter.read() == {}  # the compile-warmup interval is never reported
    for _ in range(4):
        meter.step()
    meter.exclude(2.0)
    perf = meter.read()
    assert perf["perf/tokens_per_sec"] == pytest.approx(4000 / 8.0)
    assert perf["perf/step_time_s"] == pytest.approx(2.0)
    assert perf["perf/mfu"] == pytest.approx(500 * 6.0 / 1e3)


# --- evaluation -------------------------------------------------------------------------------


class _UniformModel:
    """Uniform logits: every scored token costs exactly ln(vocab) nats."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def __call__(self, ids):
        class Out:
            pass

        out = Out()
        out.logits = torch.zeros(*ids.shape, self.vocab)
        return out


def test_bits_per_byte_is_nats_over_ln2_over_bytes():
    vocab = 16
    token_bytes = torch.full((vocab,), 2, dtype=torch.long)
    token_bytes[0] = 0  # a special token: scored, but not text
    batches = [{"input_ids": torch.randint(1, vocab, (2, 9))} for _ in range(3)]
    batches[1]["input_ids"][0, 5] = 0
    result = evaluate_detailed(
        _UniformModel(vocab), batches, "cpu", "cpu", torch.float32, token_bytes=token_bytes
    )
    n_scored = 3 * 2 * 8
    n_bytes = 2 * (n_scored - 1)
    assert result.loss == pytest.approx(math.log(vocab))
    assert result.bits_per_byte == pytest.approx(n_scored * math.log(vocab) / (math.log(2) * n_bytes))
    assert result.loss_stderr == pytest.approx(0.0, abs=1e-6)  # identical per-batch losses


def test_evaluate_still_returns_the_plain_mean_and_bpb_is_off_by_default():
    batches = [{"input_ids": torch.randint(0, 16, (2, 9))} for _ in range(2)]
    assert evaluate(_UniformModel(16), batches, "cpu", "cpu", torch.float32) == pytest.approx(math.log(16))
    assert evaluate_detailed(_UniformModel(16), batches, "cpu", "cpu", torch.float32).bits_per_byte is None


def test_gpt2_byte_table_reproduces_utf8_length_exactly():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not in the local HF cache")
    lengths = d.token_byte_lengths(tok, 50304)
    assert lengths is not None and lengths.shape == (50304,)
    text = "Naïve café — 東京 🚀 déjà vu\n\ttabs and  spaces, ¡hola!"
    ids = tok(text)["input_ids"]
    assert int(lengths[ids].sum()) == len(text.encode("utf-8"))
    assert lengths[tok.eos_token_id] == 0
    assert int(lengths[50257:].sum()) == 0  # vocab padding


def test_non_byte_level_tokenizers_get_no_byte_table():
    class Plain:
        pass

    assert d.token_byte_lengths(Plain(), 10) is None


# --- validation is the same set every time ----------------------------------------------------


class _IndexTok:
    eos_token_id = 1

    def __call__(self, texts):
        return {"input_ids": [[int(w[1:]) + 2 for w in t.split()] for t in texts]}


@pytest.fixture
def fake_stream(monkeypatch):
    docs = [" ".join(f"w{i * 15 + j}" for j in range(15)) for i in range(240)]
    shards = [docs[i * 60 : (i + 1) * 60] for i in range(4)]

    def gen(shards):
        for shard in shards:
            for text in shard:
                yield {"text": text}

    def fake(name, streaming=False, split=None, **_kw):
        ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
        return ds if split else DatasetDict({"train": ds})

    monkeypatch.setattr(d, "load_dataset", fake)


def test_streaming_validation_scores_the_same_blocks_every_eval(fake_stream, tmp_path):
    cfg = Config(
        data=DataConfig(
            dataset="test/stream", seq_len=16, streaming=True, cache_dir=str(tmp_path),
            disk_cache_max_gb=1.0, shuffle_buffer_size=16, num_workers=2, eval_split_size=80,
        ),
        train=TrainConfig(batch_size=2, device="cpu", compile=False, auto_batch_size=False),
    )
    _, val_loader = d.build_dataloaders(cfg, _IndexTok())

    def first_blocks(n_batches: int) -> list[int]:
        out = []
        for i, batch in enumerate(val_loader):
            if i == n_batches:
                break
            out.extend(batch["input_ids"][:, 0].tolist())
        return out

    evals = [first_blocks(10) for _ in range(3)]  # persistent workers carry state across these
    assert evals[0] == evals[1] == evals[2]


# --- keep_every -------------------------------------------------------------------------------


def test_keep_every_spares_multiples_and_still_clears_stale_temps(tmp_path):
    for step in (5, 10, 15, 20, 25):
        (tmp_path / f"step_{step}.pt").write_bytes(b"x")
    (tmp_path / "step_30.pt.tmp").write_bytes(b"torn")
    prune_checkpoints(tmp_path, keep=tmp_path / "step_25.pt", keep_every=10)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_10.pt", "step_20.pt", "step_25.pt"]


def test_keep_every_unset_keeps_only_the_newest(tmp_path):
    for step in (10, 20):
        (tmp_path / f"step_{step}.pt").write_bytes(b"x")
    prune_checkpoints(tmp_path, keep=tmp_path / "step_20.pt")
    assert [p.name for p in tmp_path.iterdir()] == ["step_20.pt"]


def test_files_that_only_look_like_checkpoints_are_neither_parsed_nor_deleted(tmp_path):
    # A hand-copied step_final.pt used to crash every save under keep_every (and resume_from:
    # auto always): its "step" does not parse. It is not ours to delete or resume from either.
    from radiance.checkpointing import find_resume_checkpoint
    from radiance.config import Config, TrainConfig

    for name in ("step_10.pt", "step_20.pt", "step_final.pt", "step_5000_backup.pt"):
        (tmp_path / name).write_bytes(b"x")
    cfg = Config(train=TrainConfig(output_dir=str(tmp_path), resume_from="auto"))
    assert find_resume_checkpoint(cfg) == tmp_path / "step_20.pt"
    prune_checkpoints(tmp_path, keep=tmp_path / "step_20.pt", keep_every=5000)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_20.pt", "step_5000_backup.pt", "step_final.pt"]
