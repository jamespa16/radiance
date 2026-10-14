"""The training run's terminal display: a live HUD on a TTY, the plain `[radiance]` lines otherwise.

On an interactive terminal the loop's per-step output is a fixed panel redrawn in place — progress,
ETA, the latest train metrics, a loss sparkline, the best validation loss so far — instead of a
new line every `log_every` steps. Things worth keeping in scrollback (validation results,
checkpoint saves, OOM backoffs, anything else that prints) still go above the panel as ordinary
lines, so the terminal history after a run reads as its event log.

Anything that is not a terminal — `nohup ... > run.log`, a pipe, CI, pytest's capsys — gets
exactly the lines train.py always printed, unchanged, so log files stay greppable and nothing that
parses them breaks. `--no-hud` forces that on a terminal too.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import torch
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from radiance.config import Config

# Train losses kept for the sparkline: one per log_every steps, two per braille character.
_HISTORY = 400
_SPARK_ROWS = 2
# Bit for the dot at (row, col) of one braille cell, row 0 at the top. U+2800 + the OR of the set
# dots is the character; the numbering is the historical 6-dot cell plus a 4th row added later.
_BRAILLE_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def sparkline(values, width: int, rows: int = _SPARK_ROWS) -> list[str]:
    """The last `2 * width` values as a braille line plot `rows` characters tall, scaled to their own
    min..max. Each character is a 2x4 dot grid, so the plot has `2 * width` x `4 * rows` dots;
    every column is filled from the bottom up to its point, an area chart rather than a line, which
    reads at a glance where a lone dot per column would look like noise."""
    values = list(values)[-2 * width :]
    if not values:
        return [""] * rows
    levels = 4 * rows
    lo, hi = min(values), max(values)
    span = hi - lo
    # Level 0 is the bottom dot row; a flat series sits mid-height.
    ys = [levels // 2 if span < 1e-12 else round((v - lo) / span * (levels - 1)) for v in values]
    cells = [[0] * ((len(ys) + 1) // 2) for _ in range(rows)]
    for x, y in enumerate(ys):
        for level in range(y + 1):
            dot_row = levels - 1 - level
            cells[dot_row // 4][x // 2] |= _BRAILLE_BITS[dot_row % 4][x % 2]
    return ["".join(chr(0x2800 + c) for c in row) for row in cells]


def gpu_temperature(device: str) -> int | None:
    """Core temperature in °C via NVML, or None off CUDA or when NVML can't answer (no driver,
    or the driver/library version mismatch a kernel update leaves until reboot)."""
    if not device.startswith("cuda"):
        return None
    try:
        return torch.cuda.temperature(torch.device(device))
    except Exception:
        return None


def hud_wanted(requested: bool | None) -> bool:
    """None means auto: on exactly when stdout is an interactive, cursor-capable terminal."""
    if requested is not None:
        return requested
    console = Console()
    return console.is_terminal and not console.is_dumb_terminal


class TrainingHUD:
    """Context manager around the training loop. Every method is safe to call in either mode."""

    def __init__(
        self,
        cfg: Config,
        *,
        enabled: bool,
        start_step: int,
        device: str,
        compile_mode: str | None,
        num_parameters: int,
        num_active_parameters: int,
        tokens_per_step: int,
    ):
        self.cfg = cfg
        self.enabled = enabled
        self.max_steps = cfg.train.max_steps
        self.start_step = start_step
        self.step = start_step
        self._device = device
        # Flips off after the first failed read, so a machine without NVML doesn't retry every log.
        self._read_temperature = device.startswith("cuda")

        compiled = ("compiled" + (f" ({compile_mode})" if compile_mode else "")) if cfg.train.compile else "eager"
        mode = "dpo" if cfg.dpo.enabled else "sft" if cfg.sft.enabled else "pretrain"
        params = (
            f"{num_active_parameters / 1e6:,.1f}M active / {num_parameters / 1e6:,.1f}M"
            if num_active_parameters != num_parameters
            else f"{num_parameters / 1e6:,.1f}M params"
        )
        self._title = f"[b]radiance[/b] · {escape(cfg.run_name)}"
        self._subtitle = f"{mode} · {device} · {cfg.train.dtype} · {compiled}"
        self._setup = (
            f"{params} · batch {cfg.train.batch_size} x {cfg.train.grad_accum_steps} accum "
            f"· {tokens_per_step:,} tok/step"
        )

        # Written by the training thread, read by Live's refresh thread. Each is replaced whole
        # rather than mutated, so a render never sees a half-updated dict.
        self._train: dict[str, float] = {}
        self._perf: dict[str, float] = {}
        self._extra: dict[str, float] = {}
        self._val: dict[str, float] | None = None
        self._best_val: tuple[float, int] | None = None
        self._ckpt: tuple[str, int] | None = None
        self._losses: deque[float] = deque(maxlen=_HISTORY)
        self._prev_loss: float | None = None
        self._status = "warming up"

        self._started = time.monotonic()
        self._console = Console() if enabled else None
        self._live: Live | None = None

    # --- lifecycle -----------------------------------------------------------------------------

    def __enter__(self) -> TrainingHUD:
        if self.enabled:
            self._started = time.monotonic()
            # redirect_stdout/stderr: any print from elsewhere (a dataset shard download, a
            # torch warning, a recompile notice) lands above the panel instead of tearing it.
            self._live = Live(
                self,
                console=self._console,
                refresh_per_second=4,
                redirect_stdout=True,
                redirect_stderr=True,
                transient=False,
            )
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            if exc_type is None:
                self._status = "done"
            elif issubclass(exc_type, KeyboardInterrupt):
                self._status = "interrupted"
            else:
                self._status = "crashed"
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    # --- events from the loop --------------------------------------------------------------------

    def on_step(self, step: int) -> None:
        """Every optimizer step: moves the progress bar. Takes an int the loop already has, no sync."""
        self.step = step

    def on_train_log(
        self,
        step: int,
        metrics: dict[str, float],
        *,
        lr: float,
        grad_norm: float,
        mem_gb: float | None,
        perf: dict[str, float],
        micro_chunk_size: int,
    ) -> None:
        self.step = step
        if not self.enabled:
            # The line train.py always printed: W&B was once the only place a loss appeared, so a
            # wandb.mode=disabled run had no visible signal. `mem` is reserved VRAM, there because
            # a climbing value is the signature of a torch.compile re-record (see compile_mode).
            mem = f" mem {mem_gb:.2f}GB" if mem_gb is not None else ""
            tok_s = f" {perf['perf/tokens_per_sec'] / 1e3:.1f}k tok/s" if perf else ""
            print(
                f"[radiance] step {step:>6}/{self.max_steps} "
                f"loss {metrics['loss']:.4f} lm {metrics['lm_loss']:.4f} "
                f"lr {lr:.3e}{mem}{tok_s}",
                flush=True,
            )
            return
        temp_c = gpu_temperature(self._device) if self._read_temperature else None
        self._read_temperature = temp_c is not None
        self._prev_loss = self._train.get("loss")
        self._train = dict(metrics)
        self._extra = {
            "lr": lr,
            "grad_norm": grad_norm,
            "micro_chunk_size": micro_chunk_size,
            **({"mem_gb": mem_gb} if mem_gb is not None else {}),
            **({"temp_c": temp_c} if temp_c is not None else {}),
        }
        if perf:
            self._perf = dict(perf)
        self._losses.append(metrics["loss"])
        self._status = "training"

    def on_eval_start(self) -> None:
        self._status = "evaluating"

    def on_val(self, step: int, loss: float, stderr: float | None, bits_per_byte: float | None) -> None:
        extra = ""
        if stderr is not None:
            extra += f" ± {stderr:.4f}"
        if bits_per_byte is not None:
            extra += f" bpb {bits_per_byte:.4f}"
        best = self._best_val is None or loss < self._best_val[0]
        if best:
            self._best_val = (loss, step)
        self._val = {
            "step": step,
            "loss": loss,
            **({"stderr": stderr} if stderr is not None else {}),
            **({"bpb": bits_per_byte} if bits_per_byte is not None else {}),
        }
        self._status = "training"
        if not self.enabled:
            print(f"[radiance] step {step:>6} val/loss {loss:.4f}{extra}", flush=True)
            return
        marker = " [green]★ best[/green]" if best else ""
        self._console.print(f"[cyan]\\[radiance][/cyan] step {step:>6} val/loss [b]{loss:.4f}[/b]{extra}{marker}")

    def on_save_start(self) -> None:
        self._status = "saving checkpoint"

    def on_checkpoint(self, step: int, path: Path) -> None:
        self._ckpt = (str(path), step)
        self._status = "training"
        try:
            path = path.resolve().relative_to(Path.cwd())
        except ValueError:
            pass
        # Silent in plain mode, as it always was: the step line already dates the run's progress.
        if self.enabled:
            self._console.print(f"[dim]\\[radiance] step {step:>6} saved {escape(str(path))}[/dim]")

    def event(self, message: str) -> None:
        """Something that happened once and should stay in the scrollback (OOM backoff, ...)."""
        if self.enabled:
            self._console.print(f"[yellow]\\[radiance][/yellow] {escape(message)}")
        else:
            print(f"[radiance] {message}", flush=True)

    # --- rendering --------------------------------------------------------------------------------

    def _eta(self) -> str:
        remaining = self.max_steps - self.step
        if remaining <= 0:
            return "0:00:00"
        # Prefer the meter's step time: it excludes eval and checkpoint pauses and the compile
        # warmup. Before the first interval reports, fall back to wall clock over steps taken.
        step_time = self._perf.get("perf/step_time_s")
        done = self.step - self.start_step
        if step_time is None and done > 0:
            step_time = (time.monotonic() - self._started) / done
        return _duration(remaining * step_time) if step_time else "--:--:--"

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        frac = self.step / self.max_steps if self.max_steps else 0.0
        progress = Table.grid(expand=True, padding=(0, 1))
        progress.add_column(no_wrap=True)
        progress.add_column(ratio=1)
        progress.add_column(no_wrap=True, justify="right")
        progress.add_row(
            Text.from_markup(f"step [b]{self.step:,}[/b]/{self.max_steps:,}"),
            ProgressBar(total=self.max_steps or 1, completed=self.step),
            Text.from_markup(
                f"[b]{frac:6.1%}[/b]  {_duration(time.monotonic() - self._started)} elapsed"
                f"  ETA [b]{self._eta()}[/b]"
            ),
        )

        train = Table.grid(padding=(0, 2))
        train.add_column(style="dim", no_wrap=True)
        train.add_column(justify="right", no_wrap=True)
        train.add_column(no_wrap=True)
        if self._train:
            for name, value in self._train.items():
                if name != "loss":
                    train.add_row(name, f"{value:.4f}")
                    continue
                delta = ""
                if self._prev_loss is not None:
                    d = value - self._prev_loss
                    color = "green" if d < 0 else "red"
                    delta = f"[{color}]{'▼' if d < 0 else '▲'} {abs(d):.4f}[/{color}]"
                train.add_row(name, Text.from_markup(f"[b]{value:.4f}[/b]"), Text.from_markup(delta))
            train.add_row("grad norm", f"{self._extra['grad_norm']:.3f}")
            train.add_row("lr", f"{self._extra['lr']:.3e}")
        else:
            train.add_row("", Text(f"first log at step {self.cfg.train.log_every:,}…", style="dim"))

        val = Table.grid(padding=(0, 2))
        val.add_column(style="dim", no_wrap=True)
        val.add_column(no_wrap=True)
        if self._val is not None:
            v = self._val
            stderr = f" ± {v['stderr']:.4f}" if "stderr" in v else ""
            val.add_row("loss", Text.from_markup(f"[b]{v['loss']:.4f}[/b]{stderr}  [dim]@ {v['step']:,}[/dim]"))
            if "bpb" in v:
                val.add_row("bits/byte", f"{v['bpb']:.4f}")
            best_loss, best_step = self._best_val
            val.add_row("best", Text.from_markup(f"[green]{best_loss:.4f}[/green]  [dim]@ {best_step:,}[/dim]"))
        else:
            val.add_row("", Text(f"first eval at step {self.cfg.train.eval_every:,}…", style="dim"))
        if self._ckpt is not None:
            path, step = self._ckpt
            val.add_row("checkpoint", Text.from_markup(f"{escape(path.rsplit('/', 1)[-1])}  [dim]@ {step:,}[/dim]"))

        columns = Table.grid(expand=True, padding=(0, 4))
        columns.add_column(ratio=1)
        columns.add_column(ratio=1)
        columns.add_row(
            Group(Text("train", style="b cyan"), train),
            Group(Text("validation", style="b cyan"), val),
        )

        perf_bits = []
        if self._perf:
            perf_bits.append(f"[b]{self._perf['perf/tokens_per_sec'] / 1e3:,.1f}k[/b] tok/s")
            perf_bits.append(f"{self._perf['perf/step_time_s'] * 1e3:,.0f} ms/step")
            if "perf/mfu" in self._perf:
                perf_bits.append(f"MFU {self._perf['perf/mfu']:.1%}")
        if "mem_gb" in self._extra:
            perf_bits.append(f"mem {self._extra['mem_gb']:.2f} GB")
        if "temp_c" in self._extra:
            # Consumer cards start clocking down in the mid-80s, which shows up as tok/s sagging
            # with nothing in the run having changed.
            temp = self._extra["temp_c"]
            color = "red" if temp >= 85 else "yellow" if temp >= 78 else ""
            perf_bits.append(f"[{color}]{temp}°C[/{color}]" if color else f"{temp}°C")
        if self._extra and self._extra["micro_chunk_size"] != self.cfg.train.batch_size:
            # Only shown once OOM backoff has split the micro-batch: it means the run is slower
            # than its config says it should be.
            perf_bits.append(f"[yellow]micro_chunk {self._extra['micro_chunk_size']}[/yellow]")
        perf = Text.from_markup("  ·  ".join(perf_bits) if perf_bits else "[dim]measuring throughput…[/dim]")

        rows = [progress, Text(""), columns, Text(""), perf]
        if len(self._losses) >= 2:
            # Panel border + padding (4), the "loss " prefix (5), the range label (~8).
            width = max(10, min(_HISTORY // 2, options.max_width - 17))
            shown = list(self._losses)[-2 * width :]
            lines = sparkline(shown, width)
            # Max beside the top row, min beside the bottom: the plot is scaled to exactly that range.
            labels = [f"{max(shown):.3f}"] + [""] * (len(lines) - 2) + [f"{min(shown):.3f}"]
            for i, (line, label) in enumerate(zip(lines, labels)):
                prefix = "[dim]loss[/dim] " if i == 0 else "     "
                rows.append(Text.from_markup(f"{prefix}[cyan]{line}[/cyan] [dim]{label}[/dim]"))
        rows.append(Text(self._setup, style="dim"))

        status_style = {"training": "green", "done": "green", "crashed": "red"}.get(self._status, "yellow")
        yield Panel(
            Group(*rows),
            title=self._title,
            title_align="left",
            subtitle=f"[{status_style}]● {self._status}[/{status_style}] · {self._subtitle}",
            subtitle_align="right",
            border_style="cyan",
            padding=(0, 1),
        )
