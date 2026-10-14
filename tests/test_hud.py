"""The training display: plain mode must keep printing exactly the lines train.py always printed
(log files get grepped), and the live panel must render in every state the loop can leave it in."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

import radiance.hud as hud_mod
from radiance.config import Config, TrainConfig
from radiance.hud import TrainingHUD, hud_wanted, sparkline


def _hud(enabled: bool) -> TrainingHUD:
    cfg = Config(train=TrainConfig(max_steps=100, log_every=10, eval_every=50))
    return TrainingHUD(
        cfg,
        enabled=enabled,
        start_step=0,
        device="cpu",
        compile_mode=None,
        num_parameters=3_500_000,
        num_active_parameters=3_500_000,
        tokens_per_step=256,
    )


def _render(hud: TrainingHUD) -> str:
    out = io.StringIO()
    Console(file=out, width=100, color_system=None).print(hud)
    return out.getvalue()


def test_plain_mode_prints_the_original_lines(capsys):
    hud = _hud(enabled=False)
    with hud:
        hud.on_train_log(
            10,
            {"loss": 2.5, "lm_loss": 2.4, "z_loss": 0.1},
            lr=1e-3,
            grad_norm=0.5,
            mem_gb=1.25,
            perf={"perf/tokens_per_sec": 12_345.0, "perf/step_time_s": 0.02},
            micro_chunk_size=8,
        )
        hud.on_val(50, 2.3, 0.01, 1.1)
        hud.on_checkpoint(50, Path("ckpt/step_50.pt"))
        hud.event("CUDA OOM at step 51, ending run early.")
    assert capsys.readouterr().out.splitlines() == [
        "[radiance] step     10/100 loss 2.5000 lm 2.4000 lr 1.000e-03 mem 1.25GB 12.3k tok/s",
        "[radiance] step     50 val/loss 2.3000 ± 0.0100 bpb 1.1000",
        "[radiance] CUDA OOM at step 51, ending run early.",
    ]


def test_panel_renders_before_any_metrics_and_after_them():
    hud = _hud(enabled=True)
    assert "first log at step 10" in _render(hud)

    for step, loss in ((10, 3.0), (20, 2.5)):
        hud.on_step(step)
        hud.on_train_log(
            step,
            {"loss": loss, "lm_loss": loss},
            lr=1e-3,
            grad_norm=0.5,
            mem_gb=None,
            perf={"perf/tokens_per_sec": 1000.0, "perf/step_time_s": 0.5},
            micro_chunk_size=4,
        )
    text = _render(hud)
    assert "20/100" in text and "2.5000" in text and "▼ 0.5000" in text
    # 80 steps left at the meter's 0.5 s/step.
    assert "ETA 0:00:40" in text


def test_hud_wanted_honours_an_explicit_choice():
    assert hud_wanted(True) is True
    assert hud_wanted(False) is False
    # pytest captures stdout, so auto resolves to off — the case for every log file and pipe.
    assert hud_wanted(None) is False


def test_sparkline_is_a_two_row_braille_plot_scaled_to_its_own_range():
    # 8 dot rows, 2 points per character, filled from the bottom up to each point: the max fills
    # its whole (left) column, the min only the bottom dot of its (right) column.
    top, bottom = sparkline([1.0, 0.0], width=10)
    left_column = 0x01 | 0x02 | 0x04 | 0x40
    assert top == chr(0x2800 | left_column) and len(top) == 1
    assert bottom == chr(0x2800 | left_column | 0x80)
    # Flat: filled to mid-height (level 4 = top row's bottom dot), so the bottom row is solid.
    assert sparkline([5.0, 5.0], width=10) == [chr(0x2800 | 0x40 | 0x80), "⣿"]
    # Keeps only the last 2 * width points.
    assert all(len(row) == 4 for row in sparkline(range(100), width=4))
    assert sparkline([], width=4) == ["", ""]


def _log(hud: TrainingHUD, step: int) -> None:
    hud.on_train_log(
        step, {"loss": 2.0, "lm_loss": 2.0}, lr=1e-3, grad_norm=0.5, mem_gb=17.75, perf={}, micro_chunk_size=4
    )


def test_gpu_temperature_sits_beside_memory(monkeypatch):
    monkeypatch.setattr(hud_mod, "gpu_temperature", lambda device: 81)
    hud = _hud(enabled=True)
    hud._device, hud._read_temperature = "cuda", True
    _log(hud, 10)
    assert "mem 17.75 GB  ·  81°C" in _render(hud)


def test_gpu_temperature_stops_asking_after_nvml_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(hud_mod, "gpu_temperature", lambda device: calls.append(device))
    hud = _hud(enabled=True)
    hud._device, hud._read_temperature = "cuda", True
    _log(hud, 10)
    _log(hud, 20)
    assert len(calls) == 1
    assert "°C" not in _render(hud)
