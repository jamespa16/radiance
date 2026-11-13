"""`radiance-viz`: where each token spends its compute, and which experts it uses on each pass.

For a checkpoint trained with ACT (`model.use_router`) and/or MoE (`model.use_moe`), this runs
one ordinary forward pass and records, for every token:

    depth          how many loop-body iterations updated it before its ACT router halted it
    halt prob      the router's per-iteration halting probability p_n
    ACT weight     how much each iteration's hidden state contributes to the output
    experts        the top-k experts every MoE block picked for it, on every iteration

then writes a single self-contained HTML page: the text with each token shaded by depth (hover
or click a token for its blocks x iterations expert grid), plus aggregates over every traced
token: the depth distribution, per-block expert load on each iteration, how often a token keeps
its top-1 expert from one pass to the next, and the tokens that think deepest and shallowest.

Usage:

    radiance-viz --checkpoint checkpoints/showcase_act_moe/step_5000.pt \\
        --prompt "The mitochondria is the powerhouse of the cell because" --generate 80 \\
        --val-sequences 16 --output viz/showcase.html

**Nothing in the model is changed to do this.** The trace comes from forward hooks on the ACT
router (`DenseTransformer.router`) and on each MoE block's `MoERouter`, both of which are called
once per loop iteration with the iteration index as their second positional argument. Halting is
then replayed from the recorded p_n with the same arithmetic as `_forward_act`, and the replay is
checked against the `mean_loop_depth` the forward itself returned — so if that loop's halting
rule ever changes and this file isn't updated, the tool fails loudly instead of drawing the old
rule. Expert choice is recomputed the way `MoEFeedForward.forward` selects (router probs plus
`expert_bias` when bias balancing is on, top-k), with gate weights renormalised over the k picked.

The forward runs in eval mode, where ACT is dense: every position is computed on every
iteration, and a halted position's later updates are discarded. The page shows those discarded
iterations greyed out rather than hiding them, because "the router would have picked expert 7
here, but the token had already stopped" is part of the picture. (`act_capacity_ratio` sparsity
is training-only, so it does not apply here; see `_forward_act`.)

The checkpoint is read onto the CPU and only the model is moved to `--device`, so a run next to a
training job on the same GPU costs it roughly the weights plus one forward, not the optimizer
state a `.pt` also carries.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from radiance.config import resolve_device
from radiance.data import build_tokenizer, read_pretokenized_meta
from radiance.generate import _sample_next_token
from radiance.model import DenseTransformer, MoEFeedForward, load_transformer_from_checkpoint, mask_vocab_padding
from radiance.model.load import cast_params


@dataclass
class RoutingTrace:
    """Per-token routing for one (batch, seq) forward. Arrays are numpy, batch-first."""

    token_ids: np.ndarray  # (batch, seq) int
    n_iters: int  # loop-body iterations run (max_loops, or the --loops override)
    halt_probs: np.ndarray | None  # (batch, iters, seq) float; None without ACT
    act_weights: np.ndarray | None  # (batch, iters, seq) float; None without ACT
    depth: np.ndarray  # (batch, seq) int; n_iters everywhere without ACT
    moe_blocks: list[int] = field(default_factory=list)  # model.blocks indices that are MoE
    experts: np.ndarray | None = None  # (batch, n_moe_blocks, iters, seq, k) int
    gates: np.ndarray | None = None  # (batch, n_moe_blocks, iters, seq, k) float
    n_experts: int = 0
    model_mean_depth: float | None = None  # the forward's own mean_loop_depth, for the check


def _replay_halting(halt_probs: list[torch.Tensor], halt_epsilon: float, dtype: torch.dtype):
    """_forward_act's halting rule, applied to the recorded per-iteration p_n.

    Kept arithmetically identical to transformer.py (same dtype for the running sum, same `>=`), so
    a position that sits exactly on the threshold halts on the same iteration here as there.
    """
    p0 = halt_probs[0]
    cum_prob = torch.zeros(p0.shape, dtype=dtype, device=p0.device)
    still_running = torch.ones(p0.shape, dtype=torch.bool, device=p0.device)
    n_updates = torch.zeros(p0.shape, dtype=dtype, device=p0.device)
    weights = []
    for n, p_n in enumerate(halt_probs, start=1):
        is_last_step = n == len(halt_probs)
        would_exceed = (cum_prob + p_n) >= (1.0 - halt_epsilon)
        halts_now = still_running & (would_exceed | is_last_step)
        remainder = 1.0 - cum_prob
        weight = torch.where(halts_now, remainder, p_n)
        weights.append(torch.where(still_running, weight, torch.zeros_like(weight)))
        n_updates = n_updates + still_running.to(dtype)
        cum_prob = torch.where(still_running & ~halts_now, cum_prob + p_n, cum_prob)
        still_running = still_running & ~halts_now
    return n_updates, torch.stack(weights, dim=1)


class _RouterRecorder:
    """Forward hooks on the ACT router and every MoE router, turned into a RoutingTrace per forward.

    Used as a context manager so the hooks never outlive the trace. `begin` arms it for one
    forward over `input_ids`; `build` turns what that forward recorded into a RoutingTrace. One
    recorder serves many forwards, which is how `generate_traced` traces a KV-cached decode one
    token at a time.
    """

    def __init__(self, model: DenseTransformer):
        self.model = model
        self.moe_blocks = [i for i, b in enumerate(model.blocks) if isinstance(b.ffn, MoEFeedForward)]
        self.handles = []
        self.n_tokens = 0
        self.halt_records: list[torch.Tensor] = []
        self.stream_dtype: list[torch.dtype] = []
        self.moe_records: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __enter__(self) -> "_RouterRecorder":
        model, top_k = self.model, self.model.cfg.moe_top_k
        if model.router is not None:
            def on_halt(module, args, out):
                self.halt_records.append(out.detach())
                self.stream_dtype.append(args[0].dtype)  # cum_prob is x.new_zeros in _forward_act
            self.handles.append(model.router.register_forward_hook(on_halt))

        for i in self.moe_blocks:
            ffn = model.blocks[i].ffn

            def on_route(module, args, probs, i=i, ffn=ffn):
                variant = args[1] if len(args) > 1 else 0
                if probs.shape[0] != self.n_tokens:
                    raise NotImplementedError(
                        f"block {i}'s MoE router saw {probs.shape[0]} tokens, not batch*seq = "
                        f"{self.n_tokens}: a sparse FFN dispatch (model.act_ffn_capacity_ratio < 1) "
                        "is routing a subset, which this trace does not map back to positions."
                    )
                scores = probs + ffn.expert_bias if ffn._bias_balancing else probs
                idx = scores.topk(top_k, dim=-1).indices
                gate = probs.gather(1, idx)
                gate = gate / gate.sum(dim=-1, keepdim=True)
                if (i, variant) in self.moe_records:
                    raise RuntimeError(f"block {i} routed twice on iteration {variant + 1}")
                self.moe_records[(i, variant)] = (idx.detach(), gate.detach().float())
            self.handles.append(ffn.router.register_forward_hook(on_route))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def begin(self, input_ids: torch.Tensor) -> None:
        self.n_tokens = input_ids.numel()
        self.halt_records.clear()
        self.stream_dtype.clear()
        self.moe_records.clear()

    def build(self, out, input_ids: torch.Tensor, loop_count: int | None) -> RoutingTrace:
        model, cfg = self.model, self.model.cfg
        batch, seq_len = input_ids.shape
        if model.router is not None:
            n_iters = len(self.halt_records)
            n_updates, weights = _replay_halting(self.halt_records, cfg.halt_epsilon, self.stream_dtype[0])
            model_mean = float(out.mean_loop_depth)
            # Depth counts are small integers, exact in any dtype; their *mean* is not, so take it
            # in float64 and allow for the rounding of the forward's own mean (bf16 weights put
            # that at ~1/64 near 4).
            replay_mean = float(n_updates.double().mean())
            tolerance = 1e-4 + n_iters * torch.finfo(out.mean_loop_depth.dtype).eps
            if abs(model_mean - replay_mean) > tolerance:
                raise RuntimeError(
                    f"replayed halting gives mean depth {replay_mean:.5f} but the forward reported "
                    f"{model_mean:.5f}: _forward_act's halting rule has changed and viz.py's "
                    "_replay_halting no longer matches it."
                )
            halt_probs = torch.stack(self.halt_records, dim=1).float().cpu().numpy()
            act_weights = weights.float().cpu().numpy()
            depth = n_updates.round().long().cpu().numpy()
        else:
            n_iters = model.resolved_loop_count(loop_count)
            halt_probs = act_weights = None
            depth = np.full((batch, seq_len), n_iters, dtype=np.int64)
            model_mean = None

        experts = gates = None
        if self.moe_blocks:
            k = cfg.moe_top_k
            experts = np.zeros((batch, len(self.moe_blocks), n_iters, seq_len, k), dtype=np.int64)
            gates = np.zeros((batch, len(self.moe_blocks), n_iters, seq_len, k), dtype=np.float32)
            for bi, i in enumerate(self.moe_blocks):
                for it in range(n_iters):
                    # blocks[0] is never MoE, so every MoE block is in the loop body and runs each
                    # iteration once; a missing record means that assumption broke.
                    idx, gate = self.moe_records[(i, it)]
                    experts[:, bi, it] = idx.view(batch, seq_len, k).cpu().numpy()
                    gates[:, bi, it] = gate.view(batch, seq_len, k).cpu().numpy()

        return RoutingTrace(
            token_ids=input_ids.cpu().numpy(), n_iters=n_iters, halt_probs=halt_probs,
            act_weights=act_weights, depth=depth, moe_blocks=self.moe_blocks, experts=experts,
            gates=gates, n_experts=cfg.n_experts if self.moe_blocks else 0, model_mean_depth=model_mean,
        )


def _autocast(device_type: str, dtype: torch.dtype | None):
    return torch.autocast(device_type=device_type, dtype=dtype or torch.float32, enabled=dtype is not None)


@torch.no_grad()
def trace_routing(
    model: DenseTransformer, input_ids: torch.Tensor, loop_count: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> RoutingTrace:
    """One forward over `input_ids` (batch, seq) with the routers hooked. See the module docstring."""
    model.eval()
    with _RouterRecorder(model) as rec:
        rec.begin(input_ids)
        with _autocast(input_ids.device.type, autocast_dtype):
            out = model(input_ids, loop_count=loop_count)
        return rec.build(out, input_ids, loop_count)


@torch.no_grad()
def generate_traced(
    model: DenseTransformer, tokenizer, prompt_ids: list[int], max_new_tokens: int,
    temperature: float = 0.8, top_k: int = 50, loops: int | None = None,
    autocast_dtype: torch.dtype | None = None, device: str = "cpu",
) -> Iterator[tuple]:
    """KV-cached sampling with every forward traced. Yields, in order:

        ("prompt", trace)            the prefill over prompt_ids
        ("token", token_id, trace)   once per sampled token, traced as it is fed back in
        ("done", finish_reason)      "stop" on EOS, "length" on max_new_tokens / max_seq_len

    A token's trace is the compute spent *reading* it (the same convention as the static page:
    position i's depth is the forward at position i), so each sampled token gets one more
    forward before it is yielded. That forward is also the one that samples the next token, so
    the only extra work is a single forward after the last token. The sampling arithmetic is
    generate_tokens' own (`_sample_next_token`, `mask_vocab_padding`); EOS is not yielded.

    Cached decode is exact against a full forward (tests/test_kv_cache.py), so these traces match
    `trace_routing` over the finished sequence; tests/test_viz.py pins that.
    """
    model.eval()
    kv_cache = model.new_kv_cache(loop_count=loops)
    device_type = torch.device(device).type
    with _RouterRecorder(model) as rec:
        def step(inp: torch.Tensor):
            rec.begin(inp)
            with _autocast(device_type, autocast_dtype):
                out = model(inp, kv_cache=kv_cache, loop_count=loops)
            return out.logits[:, -1, :].float(), rec.build(out, inp, loops)

        logits, trace = step(torch.tensor([prompt_ids], device=device))
        yield ("prompt", trace)
        for _ in range(max_new_tokens):
            if kv_cache.seq_len >= model.cfg.max_seq_len:
                yield ("done", "length")
                return
            mask_vocab_padding(logits, tokenizer)
            next_token = _sample_next_token(logits, temperature, top_k)
            token_id = int(next_token.item())
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                yield ("done", "stop")
                return
            logits, trace = step(next_token)
            yield ("token", token_id, trace)
    yield ("done", "length")


# --- aggregates ---------------------------------------------------------------------------------


def summarize(traces: list[RoutingTrace], token_texts: dict[int, str], min_count: int = 4) -> dict:
    """Statistics over every traced token, counting only iterations a token actually ran."""
    n_iters = traces[0].n_iters
    depth = np.concatenate([t.depth.reshape(-1) for t in traces])
    hist = np.bincount(depth, minlength=n_iters + 1)[1 : n_iters + 1]
    summary: dict = {
        "n_tokens": int(depth.size),
        "mean_depth": float(depth.mean()),
        "depth_hist": hist.tolist(),
    }

    by_token: dict[int, list[int]] = defaultdict(list)
    for t in traces:
        for tid, d in zip(t.token_ids.reshape(-1).tolist(), t.depth.reshape(-1).tolist()):
            by_token[tid].append(d)
    rows = [(tid, float(np.mean(ds)), len(ds)) for tid, ds in by_token.items() if len(ds) >= min_count]
    rows.sort(key=lambda r: (r[1], r[2]))
    as_dict = lambda r: {"text": token_texts.get(r[0], ""), "mean_depth": r[1], "count": r[2]}  # noqa: E731
    summary["shallowest"] = [as_dict(r) for r in rows[:15]]
    summary["deepest"] = [as_dict(r) for r in reversed(rows[-15:])] if rows else []
    summary["token_table_min_count"] = min_count

    if traces[0].experts is not None:
        n_blocks, n_experts = len(traces[0].moe_blocks), traces[0].n_experts
        load = np.zeros((n_blocks, n_iters, n_experts))
        ran = np.zeros(n_iters)
        same = np.zeros((n_blocks, max(n_iters - 1, 0)))
        both = np.zeros(max(n_iters - 1, 0))
        for t in traces:
            for it in range(n_iters):
                running = t.depth > it  # (batch, seq): updated on this iteration
                ran[it] += running.sum()
                for bi in range(n_blocks):
                    picks = t.experts[:, bi, it][running]  # (n_running, k)
                    load[bi, it] += np.bincount(picks.reshape(-1), minlength=n_experts)
                if it + 1 < n_iters:
                    cont = t.depth > it + 1
                    both[it] += cont.sum()
                    for bi in range(n_blocks):
                        a = t.experts[:, bi, it, :, 0][cont]
                        b = t.experts[:, bi, it + 1, :, 0][cont]
                        same[bi, it] += (a == b).sum()
        k = traces[0].experts.shape[-1]
        # Share of a pass's routing slots each expert took: rows sum to 1, uniform = 1/n_experts.
        share = load / np.maximum(ran[None, :, None] * k, 1)
        summary["moe"] = {
            "blocks": traces[0].moe_blocks,
            "n_experts": n_experts,
            "top_k": k,
            "load_share": share.tolist(),
            "tokens_per_iter": ran.tolist(),
            "top1_kept": (same / np.maximum(both[None, :], 1)).tolist(),
            "tokens_continuing": both.tolist(),
        }
    return summary


def _stream_payload(trace: RoutingTrace, row: int, texts: list[str], n_prompt: int | None) -> dict:
    payload = {
        "tokens": texts,
        "depth": trace.depth[row].tolist(),
        "n_prompt": n_prompt,
        "halt": None if trace.halt_probs is None else np.round(trace.halt_probs[row].T, 4).tolist(),
        "weight": None if trace.act_weights is None else np.round(trace.act_weights[row].T, 4).tolist(),
    }
    if trace.experts is not None:
        # (seq, blocks, iters, k)
        payload["experts"] = trace.experts[row].transpose(2, 0, 1, 3).tolist()
        payload["gates"] = np.round(trace.gates[row].transpose(2, 0, 1, 3), 3).tolist()
    return payload


# --- page ---------------------------------------------------------------------------------------

# Depth ramps: one hue (blue), light -> dark with depth, checked with the dataviz validator in
# --ordinal mode for both surfaces. Light mode starts at step 250 and dark mode at step 600 so
# even depth 1 clears 2:1 against the surface. Sampled evenly when max_loops != 4.
_RAMP_LIGHT = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
_RAMP_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4"]
# Ink that reads on each ramp step (dark ink on the light end, white past mid-ramp).
_INK_LIGHT = ["#0b0b0b"] * 4 + ["#ffffff"] * 5
_INK_DARK = ["#ffffff"] * 6 + ["#0b0b0b"] * 3


def _ramp(n: int) -> list[int]:
    if n <= 1:
        return [len(_RAMP_LIGHT) - 1]
    return [round(i * (len(_RAMP_LIGHT) - 1) / (n - 1)) for i in range(n)]


def render_html(title: str, meta: dict, stream: dict, summary: dict) -> str:
    steps = _ramp(meta["n_iters"])
    ramp_css = []
    for d, s in enumerate(steps, start=1):
        ramp_css.append(f"--d{d}:{_RAMP_LIGHT[s]};--d{d}-ink:{_INK_LIGHT[s]};")
    ramp_dark = "".join(f"--d{d}:{_RAMP_DARK[s]};--d{d}-ink:{_INK_DARK[s]};" for d, s in enumerate(steps, 1))
    data = json.dumps({"meta": meta, "stream": stream, "summary": summary}, separators=(",", ":"))
    data = data.replace("</", "<\\/")  # never let token text close the <script>
    return (
        _PAGE.replace("__STYLE__", _STYLE)
        .replace("__TITLE__", html.escape(title))
        .replace("__RAMP_LIGHT__", "".join(ramp_css))
        .replace("__RAMP_DARK__", ramp_dark)
        .replace("__DATA__", data)
    )


_STYLE = r""".viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --halted: #f0efec; --halted-ink: #898781;
  --seq-0: #f3f8fe; --seq-1: #104281;
  __RAMP_LIGHT__
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --halted: #262624; --halted-ink: #898781;
    --seq-0: #1e2632; --seq-1: #9ec5f4;
    __RAMP_DARK__
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --halted: #262624; --halted-ink: #898781;
  --seq-0: #1e2632; --seq-1: #9ec5f4;
  __RAMP_DARK__
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root { background: var(--page); color: var(--text-primary); min-height: 100vh;
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 32px 24px 64px; }
.wrap { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; margin: 16px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; }
.tile .note { color: var(--muted); font-size: 12px; }
.legend { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; color: var(--text-secondary);
  font-size: 13px; margin: 8px 0 14px; }
.sw { display: inline-block; width: 14px; height: 14px; border-radius: 4px; vertical-align: -2px; margin-right: 6px; }
.stream { font: 15px/2.1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap;
  word-break: break-word; }
.tok { border-radius: 4px; padding: 2px 0; margin-right: 2px; cursor: pointer; }
.tok:empty { display: inline-block; width: 6px; height: 1.2em; vertical-align: -3px; }
.tok:hover, .tok.pinned { outline: 2px solid var(--text-primary); outline-offset: 0; }
.gen-mark { display: inline-block; font: 12px system-ui, sans-serif; color: var(--muted);
  border: 1px solid var(--axis); border-radius: 4px; padding: 0 6px; margin: 0 6px; line-height: 18px; }
.detail { margin-top: 16px; border-top: 1px solid var(--grid); padding-top: 14px; min-height: 60px; }
.detail .hint { color: var(--muted); }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--grid); font-size: 13px; }
th { color: var(--text-secondary); font-weight: 500; }
td.num, th.num { text-align: right; }
.grid td.cell { text-align: center; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.grid td.halted { background: var(--halted); color: var(--halted-ink); }
.code { font-family: ui-monospace, Menlo, Consolas, monospace; background: var(--halted);
  border-radius: 4px; padding: 0 4px; white-space: pre; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 800px) { .two { grid-template-columns: 1fr; } }
.bars svg, .heat svg { display: block; overflow: visible; }
.heats { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 18px; margin-top: 10px; }
.heat .cap { font-size: 13px; color: var(--text-secondary); margin-bottom: 4px; }
.tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; max-width: 300px; }
.tip b { font-weight: 600; }
.scale { display: inline-block; width: 140px; height: 10px; border-radius: 4px; vertical-align: middle;
  background: linear-gradient(90deg, var(--seq-0), var(--seq-1)); margin: 0 6px; }
.theme { float: right; font: inherit; font-size: 13px; color: var(--text-secondary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 4px 10px; cursor: pointer; }
details summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; margin-top: 10px; }
"""


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__STYLE__</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
  <button class="theme" id="theme" type="button">Toggle theme</button>
  <h1>__TITLE__</h1>
  <p class="sub" id="subtitle"></p>
  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>How long each token thinks</h2>
    <p class="sub" id="stream-sub"></p>
    <div class="legend" id="depth-legend"></div>
    <div class="stream" id="stream"></div>
    <div class="detail" id="detail"><span class="hint">Hover over a token to see its routing. Click to pin it.</span></div>
  </div>

  <div class="two">
    <div class="card bars">
      <h2>Halting depth</h2>
      <p class="sub" id="hist-sub"></p>
      <div id="hist"></div>
      <details><summary>Table view</summary><div id="hist-table"></div></details>
    </div>
    <div class="card">
      <h2>Which tokens think longest</h2>
      <p class="sub" id="tok-sub"></p>
      <div class="two" id="tok-tables"></div>
    </div>
  </div>

  <div class="card" id="moe-card">
    <h2>Expert load on each pass</h2>
    <p class="sub">Share of a pass's routing slots taken by each expert, counting only tokens still running on that pass.
      An even split is <span id="uniform"></span>. <span class="scale"></span> darker = busier.</p>
    <div class="heats" id="heats"></div>
  </div>

  <div class="card" id="keep-card">
    <h2>Does a token keep its expert from one pass to the next?</h2>
    <p class="sub" id="keep-sub"></p>
    <div id="keep"></div>
  </div>
</div></div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const M = D.meta, S = D.stream, Y = D.summary, I = M.n_iters;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (x, d = 2) => Number(x).toFixed(d);
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const show = (s) => s.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
const svgNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const cssVar = (n) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(n).trim();

// theme toggle: stamp data-theme on <html>, which beats the OS setting both ways
$("theme").onclick = () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  drawHeats();
};

// tooltip
const tip = $("tip");
const showTip = (html, ev) => { tip.innerHTML = html; tip.style.display = "block"; moveTip(ev); };
const moveTip = (ev) => { const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad; if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px"; };
const hideTip = () => { tip.style.display = "none"; };

// header
$("subtitle").textContent = `${M.checkpoint} · step ${M.step ?? "?"} · loop body runs up to ${I} times` +
  (M.moe ? ` · ${M.moe.n_experts} experts, top-${M.moe.top_k}, in ${M.moe.blocks.length} looped blocks` : "");
const tiles = [
  ["Mean depth", fmt(Y.mean_depth), `of ${I} passes`],
  ["Tokens traced", Y.n_tokens.toLocaleString(), M.n_val ? `this text + ${M.n_val} held-out sequences` : "this text"],
  ["Stop after pass 1", pct(Y.depth_hist[0] / Y.n_tokens), "of tokens"],
  [`Use all ${I} passes`, pct(Y.depth_hist[I - 1] / Y.n_tokens), "of tokens"],
];
$("tiles").innerHTML = tiles.map(([l, v, n]) =>
  `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${esc(n)}</div></div>`).join("");

// token stream
$("stream-sub").textContent = M.act
  ? "Each token is shaded by how many passes through the shared loop body it took before its ACT router stopped it."
  : "This checkpoint has no ACT router, so every token runs every pass.";
$("depth-legend").innerHTML = "Passes: " + Array.from({length: I}, (_, i) =>
  `<span><span class="sw" style="background:var(--d${i + 1})"></span>${i + 1}</span>`).join("");
const stream = $("stream");
let pinned = null;
S.tokens.forEach((t, i) => {
  if (S.n_prompt !== null && i === S.n_prompt) {
    const m = document.createElement("span"); m.className = "gen-mark"; m.textContent = "model continues ▸"; stream.appendChild(m);
  }
  const s = document.createElement("span");
  const d = S.depth[i];
  s.className = "tok"; s.textContent = t; s.dataset.i = i;
  s.style.background = `var(--d${d})`; s.style.color = `var(--d${d}-ink)`;
  s.onmouseenter = (ev) => { if (pinned === null) detail(i); showTip(`<b>${esc(show(t || "(part of next character)"))}</b> · ${d} pass${d > 1 ? "es" : ""}`, ev); };
  s.onmousemove = moveTip;
  s.onmouseleave = hideTip;
  s.onclick = () => {
    const prev = stream.querySelector(".pinned"); if (prev) prev.classList.remove("pinned");
    if (pinned === i) { pinned = null; return; }
    pinned = i; s.classList.add("pinned"); detail(i);
  };
  stream.appendChild(s);
});

// #pin=N opens the page with token N pinned, so a talk can link straight to an example
const pinMatch = location.hash.match(/pin=(\d+)/);
if (pinMatch) { const t = stream.querySelector(`.tok[data-i="${pinMatch[1]}"]`); if (t) t.click(); }

function detail(i) {
  const d = S.depth[i];
  let h = `<div style="margin-bottom:8px"><span class="code">${esc(show(S.tokens[i]))}</span> · token ${i} · ` +
    `<b>${d}</b> of ${I} passes${pinned === i ? " · pinned (click again to release)" : ""}</div>`;
  h += `<table class="grid"><tr><th></th>` + Array.from({length: I}, (_, it) => `<th>Pass ${it + 1}</th>`).join("") + `</tr>`;
  if (S.halt) {
    h += `<tr><th>Stop probability</th>` + S.halt[i].map((p, it) =>
      `<td class="cell${it >= d ? " halted" : ""}">${it >= d ? "stopped" : fmt(p)}</td>`).join("") + `</tr>`;
    h += `<tr><th>Share of output</th>` + S.weight[i].map((w, it) =>
      `<td class="cell${it >= d ? " halted" : ""}">${it >= d ? "–" : pct(w)}</td>`).join("") + `</tr>`;
  }
  if (S.experts) {
    S.experts[i].forEach((blk, bi) => {
      h += `<tr><th>Block ${M.moe.blocks[bi]} experts</th>` + blk.map((ks, it) => {
        const g = S.gates[i][bi][it];
        const txt = ks.map((e, j) => `E${e}<span style="color:var(--muted)"> ${pct(g[j])}</span>`).join(" ");
        return `<td class="cell${it >= d ? " halted" : ""}" title="${it >= d ? "computed but discarded: the token had already stopped" : ""}">${txt}</td>`;
      }).join("") + `</tr>`;
    });
  }
  h += `</table>`;
  if (S.experts && d < I) h += `<p class="sub" style="margin-top:8px">Grey passes were computed (evaluation runs every pass for every token) but the token had already stopped, so they are thrown away.</p>`;
  $("detail").innerHTML = h;
}

// depth histogram: single series, slot 1
(function drawHist() {
  const W = 420, H = 220, L = 44, B = 28, T = 16, R = 8;
  const svg = el("svg", {width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Share of tokens by number of passes"}, $("hist"));
  const n = Y.n_tokens, vals = Y.depth_hist.map((c) => c / n), max = Math.max(...vals, 1e-9);
  // gridlines on round percentages: every 10% up to 50%, every 20% above
  const stepV = max > 0.5 ? 0.2 : 0.1, top = Math.min(1, Math.ceil(max / stepV - 1e-9) * stepV);
  const ih = H - B - T, bw = (W - L - R) / I;
  for (let g = 0; g <= Math.round(top / stepV); g++) {
    const v = g * stepV, y = T + ih - ih * v / top;
    el("line", {x1: L, x2: W - R, y1: y, y2: y, stroke: g ? "var(--grid)" : "var(--axis)", "stroke-width": 1}, svg);
    el("text", {x: L - 6, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "var(--muted)"}, svg).textContent = pct(v);
  }
  vals.forEach((v, i) => {
    const h = ih * v / top, x = L + i * bw + bw * 0.2, w = bw * 0.6, y = T + ih - h, r = Math.min(4, h / 2, w / 2);
    // rounded data-end, square at the baseline
    el("path", {d: `M${x},${T + ih} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${T + ih} Z`,
      fill: "var(--series-1)"}, svg);
    el("text", {x: x + w / 2, y: H - 8, "text-anchor": "middle", "font-size": 12, fill: "var(--text-secondary)"}, svg).textContent = i + 1;
    el("text", {x: x + w / 2, y: y - 5, "text-anchor": "middle", "font-size": 11, fill: "var(--text-secondary)"}, svg).textContent = pct(v);
    const hit = el("rect", {x: L + i * bw, y: T, width: bw, height: ih, fill: "transparent"}, svg);
    hit.onmouseenter = (ev) => showTip(`<b>${i + 1} pass${i ? "es" : ""}</b><br>${pct(v, 1)} of tokens (${Y.depth_hist[i].toLocaleString()})`, ev);
    hit.onmousemove = moveTip; hit.onmouseleave = hideTip;
  });
  $("hist-sub").textContent = "Share of all traced tokens by the pass they stopped on.";
  $("hist-table").innerHTML = `<table><tr><th>Passes</th><th class="num">Tokens</th><th class="num">Share</th></tr>` +
    Y.depth_hist.map((c, i) => `<tr><td>${i + 1}</td><td class="num">${c.toLocaleString()}</td><td class="num">${pct(c / n, 1)}</td></tr>`).join("") + `</table>`;
})();

// token tables
$("tok-sub").textContent = `Mean passes per token, over tokens seen at least ${Y.token_table_min_count} times.`;
const tokTable = (title, rows) => `<div><table><tr><th>${title}</th><th class="num">Passes</th><th class="num">Seen</th></tr>` +
  rows.map((r) => `<tr><td><span class="code">${esc(show(r.text))}</span></td><td class="num">${fmt(r.mean_depth)}</td><td class="num">${r.count}</td></tr>`).join("") + `</table></div>`;
$("tok-tables").innerHTML = Y.deepest.length
  ? tokTable("Deepest", Y.deepest) + tokTable("Shallowest", Y.shallowest)
  : `<p class="sub">Not enough repeated tokens. Add --val-sequences to trace more text.</p>`;

// expert-load heatmaps: one small multiple per looped MoE block, shared colour scale
function mix(a, b, t) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "rgb(" + x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",") + ")";
}
function drawHeats() {
  if (!Y.moe) return;
  const box = $("heats"); box.innerHTML = "";
  const E = Y.moe.n_experts, lo = cssVar("--seq-0"), hi = cssVar("--seq-1");
  const all = Y.moe.load_share.flat(2), vmax = Math.max(...all);
  Y.moe.load_share.forEach((blk, bi) => {
    const wrap = document.createElement("div"); wrap.className = "heat";
    wrap.innerHTML = `<div class="cap">Block ${Y.moe.blocks[bi]}</div>`; box.appendChild(wrap);
    const L = 52, T = 4, cw = 17, chh = 20, gap = 2, W = L + E * cw + 4, H = T + I * chh + 22;
    const svg = el("svg", {width: W, height: H, role: "img", "aria-label": `Expert load, block ${Y.moe.blocks[bi]}`}, wrap);
    blk.forEach((row, it) => {
      el("text", {x: L - 6, y: T + it * chh + chh / 2 + 4, "text-anchor": "end", "font-size": 11, fill: "var(--text-secondary)"}, svg).textContent = `pass ${it + 1}`;
      row.forEach((v, e) => {
        const r = el("rect", {x: L + e * cw, y: T + it * chh, width: cw - gap, height: chh - gap, rx: 3,
          fill: mix(lo, hi, vmax ? v / vmax : 0)}, svg);
        r.onmouseenter = (ev) => showTip(`<b>Block ${Y.moe.blocks[bi]}, pass ${it + 1}, expert ${e}</b><br>` +
          `${pct(v, 1)} of routing slots (even split ${pct(1 / E, 1)})<br>${Y.moe.tokens_per_iter[it].toLocaleString()} tokens running`, ev);
        r.onmousemove = moveTip; r.onmouseleave = hideTip;
      });
    });
    for (let e = 0; e < E; e += (E > 16 ? 4 : 1)) {
      if (E > 16 || e % 2 === 0) el("text", {x: L + e * cw + (cw - gap) / 2, y: T + I * chh + 14, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)"}, svg).textContent = e;
    }
  });
}
if (Y.moe) {
  $("uniform").textContent = pct(1 / Y.moe.n_experts, 1);
  drawHeats();
  // top-1 persistence table
  const kept = Y.moe.top1_kept;
  $("keep-sub").textContent = `Share of tokens whose top-1 expert on one pass is the same on the next, among tokens that ran both passes. ` +
    `Chance would be about ${pct(1 / Y.moe.n_experts)}.`;
  if (I > 1) {
    let h = `<table><tr><th>Block</th>` + Array.from({length: I - 1}, (_, it) => `<th class="num">Pass ${it + 1} → ${it + 2}</th>`).join("") + `</tr>`;
    kept.forEach((row, bi) => { h += `<tr><td>${Y.moe.blocks[bi]}</td>` + row.map((v) => `<td class="num">${pct(v)}</td>`).join("") + `</tr>`; });
    h += `<tr><td style="color:var(--muted)">Tokens</td>` + Y.moe.tokens_continuing.map((n) => `<td class="num" style="color:var(--muted)">${n.toLocaleString()}</td>`).join("") + `</tr></table>`;
    $("keep").innerHTML = h;
  }
} else {
  $("moe-card").style.display = "none"; $("keep-card").style.display = "none";
}
})();
</script>
</body>
</html>
"""


# --- CLI ----------------------------------------------------------------------------------------


def _val_sequences(cfg, n_seq: int, seq_len: int) -> list[np.ndarray]:
    """The first n_seq * seq_len tokens of the run's own held-out split, cut into rows."""
    if not cfg.data.pretokenized_dir:
        raise SystemExit("--val-sequences needs the checkpoint's data.pretokenized_dir (this config has none)")
    meta = read_pretokenized_meta(cfg.data.pretokenized_dir)
    flat = np.memmap(Path(cfg.data.pretokenized_dir) / meta["val"]["file"], dtype=meta["dtype"], mode="r")
    n_seq = min(n_seq, len(flat) // seq_len)
    return [np.asarray(flat[i * seq_len : (i + 1) * seq_len], dtype=np.int64) for i in range(n_seq)]


def _stream_texts(tokenizer, ids: list[int]) -> list[str]:
    """Per-token text that joins back into the decoded string.

    Decoding token by token breaks characters that a byte-level BPE splits across tokens (a curly
    apostrophe is two or three GPT-2 tokens), and each fragment renders as U+FFFD. Decoding
    prefixes instead gives each character to the token that completes it and an empty string to
    the fragments before it.
    """
    texts, prev = [], ""
    for i in range(len(ids)):
        cur = tokenizer.decode(ids[: i + 1])
        if cur.startswith(prev) and not cur.endswith("\ufffd"):
            texts.append(cur[len(prev):])
            prev = cur
        else:
            texts.append("")  # an incomplete character: wait for the token that finishes it
    if texts and prev != tokenizer.decode(ids):  # never drop text, even on an odd tokenizer
        texts[-1] += tokenizer.decode(ids)[len(prev):]
    return texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help=".pt checkpoint or radiance-export directory")
    p.add_argument("--prompt", default=None, help="text to visualise")
    p.add_argument("--prompt-file", default=None, help="read the text from a file instead")
    p.add_argument("--generate", type=int, default=0, help="sample this many tokens after the prompt first, then trace it all")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-sequences", type=int, default=0,
                   help="also trace this many held-out sequences for the aggregate charts (from data.pretokenized_dir)")
    p.add_argument("--val-seq-len", type=int, default=None, help="length of each held-out sequence (default data.seq_len)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="held-out sequences per forward (1 keeps peak VRAM low beside a training job)")
    p.add_argument("--loops", type=int, default=None, help="override loop depth, as on radiance-generate")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None, help="HTML path (default viz/<checkpoint name>.html)")
    p.add_argument("--serve", action="store_true",
                   help="instead of writing a page, serve one that chats with the model and shows each reply's routing live")
    p.add_argument("--host", default="127.0.0.1", help="--serve: bind address (the page has no auth)")
    p.add_argument("--port", type=int, default=8100, help="--serve: port")
    args = p.parse_args()

    text = args.prompt
    if args.prompt_file:
        text = Path(args.prompt_file).read_text()
    if not text and not args.serve:
        raise SystemExit("give --prompt or --prompt-file (or --serve to chat)")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    # Read onto the CPU and move only the model: map_location=device would also put the optimizer
    # state a training checkpoint carries on the GPU, next to whatever is training there.
    model, cfg = load_transformer_from_checkpoint(args.checkpoint, "cpu")
    tokenizer = build_tokenizer(cfg)
    model.eos_id = tokenizer.eos_token_id  # doc masking for packed held-out rows, as in training
    if device == "cuda":
        # bf16 weights on the GPU: half the footprint, and the forward runs under bf16 autocast
        # anyway. Measured beside the showcase training job (23.5 GB) on the 5090, fp32 weights
        # with 2-row batches peaked the card at 31.9 of 32.6 GB, close enough to OOM the run.
        cast_params(model, torch.bfloat16)
    model.to(device).eval()
    if not (model.router is not None or any(isinstance(b.ffn, MoEFeedForward) for b in model.blocks)):
        raise SystemExit("this checkpoint has neither an ACT router nor MoE blocks: there is no routing to show")
    autocast = torch.bfloat16 if device == "cuda" else None
    ckpt_path = Path(args.checkpoint)
    step = None
    if ckpt_path.suffix == ".pt":  # mmap: reads the header, not the weights a second time
        step = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True).get("step")
    title = f"Routing: {cfg.run_name or ckpt_path.stem}"

    if args.serve:
        import uvicorn

        from radiance import viz_chat

        model.eos_id = None  # one conversation is one document, as in radiance-generate
        moe = [i for i, b in enumerate(model.blocks) if isinstance(b.ffn, MoEFeedForward)]
        meta = {
            "checkpoint": str(ckpt_path), "step": step, "n_iters": model.resolved_loop_count(args.loops),
            "act": model.router is not None, "chat": bool(cfg.sft.enabled or cfg.dpo.enabled),
            "max_seq_len": cfg.model.max_seq_len,
            "moe": {"blocks": moe, "n_experts": cfg.model.n_experts, "top_k": cfg.model.moe_top_k} if moe else None,
        }
        app = viz_chat.create_app(model, tokenizer, cfg, device, autocast, meta, title)
        print(f"[radiance-viz] chat page at http://{args.host}:{args.port}/")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return

    ids = tokenizer(text)["input_ids"]
    n_prompt = None
    if args.generate:
        from radiance.generate import generate_tokens

        n_prompt = len(ids)
        prompt_t = torch.tensor([ids], device=device)
        with torch.autocast(device_type=device, dtype=autocast or torch.float32, enabled=autocast is not None):
            new = list(generate_tokens(model, tokenizer, prompt_t, max_new_tokens=args.generate,
                                       temperature=args.temperature, top_k=args.top_k, device=device,
                                       loops=args.loops))
        ids = ids + new
    ids = ids[: cfg.model.max_seq_len]

    main_trace = trace_routing(model, torch.tensor([ids], device=device), args.loops, autocast)
    traces = [main_trace]
    if args.val_sequences:
        rows = _val_sequences(cfg, args.val_sequences, min(args.val_seq_len or cfg.data.seq_len, cfg.model.max_seq_len))
        for i in range(0, len(rows), args.batch_size):
            batch = torch.tensor(np.stack(rows[i : i + args.batch_size]), device=device)
            traces.append(trace_routing(model, batch, args.loops, autocast))
            print(f"[radiance-viz] traced {min(i + args.batch_size, len(rows))}/{len(rows)} held-out sequences")

    vocab_ids = {int(t) for tr in traces for t in np.unique(tr.token_ids)}
    token_texts = {t: tokenizer.decode([t]) for t in vocab_ids}
    summary = summarize(traces, token_texts)
    stream = _stream_payload(main_trace, 0, _stream_texts(tokenizer, ids), n_prompt)

    meta = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "n_iters": main_trace.n_iters,
        "act": model.router is not None,
        "n_val": sum(t.token_ids.shape[0] for t in traces[1:]),
        "moe": summary.get("moe") and {k: summary["moe"][k] for k in ("blocks", "n_experts", "top_k")},
    }
    out = Path(args.output) if args.output else Path("viz") / f"{ckpt_path.parent.name}_{ckpt_path.stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(title, meta, stream, summary))
    print(f"[radiance-viz] mean depth {summary['mean_depth']:.2f} of {main_trace.n_iters} over {summary['n_tokens']} tokens")
    print(f"[radiance-viz] wrote {out}")


if __name__ == "__main__":
    main()
