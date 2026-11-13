"""`radiance-viz --serve`: chat with a checkpoint and watch each reply token's loop depth and experts.

A local web page (FastAPI, like radiance-serve) with a conversation on the left and, on the right,
the routing of the selected reply: per-token depth as it streams, the stop probability / share
of output / experts grid for a hovered or pinned token, and the depth of every token of the
context the model read before replying.

Each reply is sampled by `viz.generate_traced`: an ordinary KV-cached decode with the router
hooks of `viz.trace_routing` armed on every step, so a token's routing arrives with the token.
tests/test_viz.py pins that a reply traced this way is position-for-position the trace one full
forward over the finished text gives, so the live page and the static `radiance-viz` page agree.

A checkpoint trained with `sft.enabled` or `dpo.enabled` is prompted through its own turn
template (`chat_prompt_ids`, as radiance-serve's /v1/chat/completions). A base checkpoint has no
template; there the conversation is one running document that you and the model take turns
extending, which is the only honest way to "chat" with a pretrained-only model.

The server is single-user and unauthenticated, and binds 127.0.0.1 by default. One model
instance serves one reply at a time; a second request waits for the first.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import threading
import time
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from radiance.config import Config
from radiance.model import DenseTransformer
from radiance.sft_data import chat_prompt_ids
from radiance.viz import _INK_DARK, _INK_LIGHT, _RAMP_DARK, _RAMP_LIGHT, _STYLE, RoutingTrace, \
    _stream_payload, _stream_texts, generate_traced

logger = logging.getLogger("radiance.viz")


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(200, ge=1)
    temperature: float = Field(0.8, ge=0.0)
    top_k: int = Field(50, ge=0)
    loops: int | None = Field(None, ge=1, le=64)


class _IncrementalText:
    """`viz._stream_texts`, one token at a time: each token gets the text it completes, and a
    token that ends partway through a multi-byte character gets "" until the one that finishes it."""

    def __init__(self, tokenizer):
        self.tokenizer, self.ids, self.prev = tokenizer, [], ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        cur = self.tokenizer.decode(self.ids)
        if cur.startswith(self.prev) and not cur.endswith("�"):
            delta, self.prev = cur[len(self.prev):], cur
            return delta
        return ""

    def tail(self) -> str:
        """Anything a final flush would add (an odd tokenizer), so no text is ever dropped."""
        full = self.tokenizer.decode(self.ids) if self.ids else ""
        return full[len(self.prev):] if full.startswith(self.prev) else ""


def build_prompt(messages: list[dict], tokenizer, cfg: Config) -> list[int]:
    if cfg.sft.enabled or cfg.dpo.enabled:
        return chat_prompt_ids(messages, tokenizer, cfg)
    # Base checkpoint: the conversation is one document, so the turns are simply concatenated.
    return tokenizer("".join(m["content"] for m in messages))["input_ids"]


def _payload(trace: RoutingTrace, texts: list[str]) -> dict:
    p = _stream_payload(trace, 0, texts, None)
    del p["n_prompt"]
    return p


def chat_events(model: DenseTransformer, tokenizer, cfg: Config, req: ChatRequest, device: str,
                autocast: torch.dtype | None, should_stop=lambda: False):
    """The NDJSON stream of one reply, as dicts: a "prompt" event (the context, traced), then a
    "token" event per sampled token, then "done". Separate from the HTTP layer so it is testable."""
    max_seq = cfg.model.max_seq_len
    ids = build_prompt([m.model_dump() for m in req.messages], tokenizer, cfg)
    if not ids:
        raise ValueError("the conversation tokenizes to nothing")
    max_tokens = min(req.max_tokens, max_seq - 1)
    room = max_seq - max_tokens
    truncated = len(ids) > room
    ids = ids[-room:]  # keep the end of the conversation, leaving room for the reply

    start = time.perf_counter()
    text = _IncrementalText(tokenizer)
    n_tokens, finish = 0, "length"
    for ev in generate_traced(model, tokenizer, ids, max_tokens, req.temperature, req.top_k,
                              req.loops, autocast, device):
        if should_stop():
            finish = "cancelled"
            break
        if ev[0] == "prompt":
            yield {"type": "prompt", "n_iters": ev[1].n_iters, "n_prompt": len(ids), "truncated": truncated,
                   **_payload(ev[1], _stream_texts(tokenizer, ids))}
        elif ev[0] == "token":
            n_tokens += 1
            yield {"type": "token", **_payload(ev[2], [text.push(ev[1])])}
        else:
            finish = ev[1]
    yield {"type": "done", "finish": finish, "n_tokens": n_tokens, "tail": text.tail(),
           "seconds": round(time.perf_counter() - start, 3)}


def create_app(model: DenseTransformer, tokenizer, cfg: Config, device: str,
               autocast: torch.dtype | None, meta: dict, title: str) -> FastAPI:
    app = FastAPI(title="radiance-viz")
    lock = threading.Lock()  # one model instance, one reply at a time
    page = render_chat_html(title, meta)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return page

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        try:
            if not build_prompt([m.model_dump() for m in req.messages], tokenizer, cfg):
                raise ValueError("the conversation tokenizes to nothing")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def work():
            try:
                with lock:
                    for ev in chat_events(model, tokenizer, cfg, req, device, autocast, stop.is_set):
                        loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception as e:  # surface it on the page rather than as a dead stream
                logger.exception("reply failed")
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=work, daemon=True).start()

        async def body():
            try:
                while (ev := await queue.get()) is not None:
                    yield json.dumps(ev, separators=(",", ":")) + "\n"
            finally:
                stop.set()  # client went away (Stop button, closed tab): end the decode

        return StreamingResponse(body(), media_type="application/x-ndjson")

    return app


def render_chat_html(title: str, meta: dict) -> str:
    ramp_light = "".join(f"--r{i}:{c};--r{i}-ink:{_INK_LIGHT[i]};" for i, c in enumerate(_RAMP_LIGHT))
    ramp_dark = "".join(f"--r{i}:{c};--r{i}-ink:{_INK_DARK[i]};" for i, c in enumerate(_RAMP_DARK))
    data = json.dumps({"meta": meta, "ramp_steps": len(_RAMP_LIGHT)}, separators=(",", ":")).replace("</", "<\\/")
    style = _STYLE.replace("__RAMP_LIGHT__", ramp_light).replace("__RAMP_DARK__", ramp_dark)
    return (_CHAT_PAGE.replace("__STYLE__", style + _CHAT_STYLE)
            .replace("__TITLE__", html.escape(title)).replace("__DATA__", data))


_CHAT_STYLE = r"""
.wrap { max-width: 1400px; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
.side { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto; }
.side .card:first-child { margin-top: 0; }
.convo { display: flex; flex-direction: column; gap: 12px; margin-top: 0; }
.msg { border-radius: 12px; padding: 10px 14px; max-width: 100%; }
.msg .who { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
.msg.user { background: var(--halted); align-self: flex-end; max-width: 85%; white-space: pre-wrap; }
.msg.assistant { border: 1px solid var(--border); cursor: pointer; }
.msg.assistant.selected { border-color: var(--series-1); box-shadow: 0 0 0 1px var(--series-1); }
.msg .stream { font-size: 14px; line-height: 2; }
.msg .foot { font-size: 12px; color: var(--muted); margin-top: 4px; display: flex; gap: 12px; flex-wrap: wrap; }
.linkish { background: none; border: none; color: var(--series-1); font: inherit; font-size: 12px; padding: 0; cursor: pointer; }
.ctx { margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--grid); }
.ctx .stream { font-size: 12px; line-height: 1.9; }
.composer { display: flex; flex-direction: column; gap: 8px; }
textarea { width: 100%; min-height: 72px; resize: vertical; font: 14px/1.5 system-ui, sans-serif; color: var(--text-primary);
  background: var(--page); border: 1px solid var(--axis); border-radius: 8px; padding: 8px 10px; }
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); }
.controls input { width: 64px; font: inherit; color: var(--text-primary); background: var(--page);
  border: 1px solid var(--axis); border-radius: 6px; padding: 2px 6px; }
.btn { font: inherit; font-size: 14px; border-radius: 8px; padding: 6px 14px; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary); }
.btn.primary { background: var(--series-1); color: #fff; border-color: transparent; }
.btn:disabled { opacity: 0.5; cursor: default; }
.spacer { flex: 1; }
.empty { color: var(--muted); }
.err { color: var(--text-primary); border-left: 3px solid var(--series-1); padding-left: 8px; }
.tiles.small .value { font-size: 20px; }
.cursor::after { content: "▍"; color: var(--muted); }
"""


_CHAT_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__STYLE__
</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
  <button class="theme" id="theme" type="button">Toggle theme</button>
  <h1>__TITLE__</h1>
  <p class="sub" id="subtitle"></p>
  <div class="layout">
    <div>
      <div class="card">
        <div class="convo" id="convo"><p class="empty" id="empty"></p></div>
      </div>
      <div class="card composer">
        <textarea id="input" placeholder="Type a message. Enter sends, Shift+Enter adds a new line."></textarea>
        <div class="controls">
          <label>Max tokens <input id="max_tokens" type="number" min="1" value="200"></label>
          <label>Temperature <input id="temperature" type="number" min="0" step="0.1" value="0.8"></label>
          <label>Top-k <input id="top_k" type="number" min="0" value="50"></label>
          <label title="Override how many times the shared loop body may run (blank = the checkpoint's own)">Max passes <input id="loops" type="number" min="1" max="64" placeholder="auto"></label>
          <span class="spacer"></span>
          <button class="btn" id="clear" type="button">Clear</button>
          <button class="btn primary" id="send" type="button">Send</button>
        </div>
      </div>
    </div>
    <div class="side">
      <div class="card">
        <h2>How long each reply token thinks</h2>
        <p class="sub" id="side-sub">Send a message to see the routing of the reply.</p>
        <div class="tiles small" id="tiles"></div>
        <div class="legend" id="legend"></div>
        <div id="strip"></div>
        <div class="detail" id="detail"><span class="hint">Hover over a reply token to see its routing. Click to pin it.</span></div>
      </div>
    </div>
  </div>
</div></div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const M = D.meta, STEPS = D.ramp_steps;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (x, d = 2) => Number(x).toFixed(d);
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const show = (s) => s.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
const svgNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

$("theme").onclick = () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
};

const tip = $("tip");
const showTip = (h, ev) => { tip.innerHTML = h; tip.style.display = "block"; moveTip(ev); };
const moveTip = (ev) => { const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad; if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px"; };
const hideTip = () => { tip.style.display = "none"; };

// depth d of n passes -> one step of the validated blue ramp, sampled evenly as on the static page
const step = (d, n) => n <= 1 ? STEPS - 1 : Math.round((d - 1) * (STEPS - 1) / (n - 1));
const shade = (s, d, n) => { const k = step(d, n); s.style.background = `var(--r${k})`; s.style.color = `var(--r${k}-ink)`; };

$("subtitle").textContent = `${M.checkpoint} · step ${M.step ?? "?"} · loop body runs up to ${M.n_iters} times` +
  (M.moe ? ` · ${M.moe.n_experts} experts, top-${M.moe.top_k}, in ${M.moe.blocks.length} looped blocks` : "") +
  (M.chat ? " · chat template" : " · base model");
$("empty").textContent = M.chat
  ? "Say something. Each reply token is shaded by how many passes through the shared loop body it took."
  : "This is a base checkpoint with no chat template, so the conversation is one running document: write the start of some text and the model continues it. Each token it writes is shaded by how many passes through the shared loop body it took.";
$("max_tokens").max = M.max_seq_len - 1;

const history = [];  // {role, content} as sent to the server
const replies = [];  // per assistant message: routing arrays + DOM handles
let selected = null, pinned = null, busy = null;

function legend(n) {
  $("legend").innerHTML = (M.act ? "Passes: " : "Every token runs all passes: ") + Array.from({length: n}, (_, i) =>
    `<span><span class="sw" style="background:var(--r${step(i + 1, n)})"></span>${i + 1}</span>`).join("");
}

function tokenSpan(r, i, ctx) {
  const s = document.createElement("span");
  const src = ctx ? r.ctx : r;
  s.className = "tok"; s.textContent = src.tokens[i]; shade(s, src.depth[i], r.n_iters);
  s.onmouseenter = (ev) => { if (!pinned) detail(r, i, ctx);
    const d = src.depth[i]; showTip(`<b>${esc(show(src.tokens[i] || "(part of next character)"))}</b> · ${d} pass${d > 1 ? "es" : ""}`, ev); };
  s.onmousemove = moveTip; s.onmouseleave = hideTip;
  s.onclick = (ev) => { ev.stopPropagation(); select(r);
    document.querySelectorAll(".tok.pinned").forEach((p) => p.classList.remove("pinned"));
    if (pinned && pinned.r === r && pinned.i === i && pinned.ctx === ctx) { pinned = null; return; }
    pinned = {r, i, ctx}; s.classList.add("pinned"); detail(r, i, ctx); };
  return s;
}

function addUser(text) {
  $("empty").style.display = "none";
  const m = document.createElement("div"); m.className = "msg user";
  m.innerHTML = `<div class="who">You</div>`; m.appendChild(document.createTextNode(text));
  $("convo").appendChild(m);
}

function addReply() {
  const r = {tokens: [], depth: [], halt: [], weight: [], experts: [], gates: [], ctx: null, n_iters: M.n_iters, done: null};
  const m = document.createElement("div"); m.className = "msg assistant";
  m.innerHTML = `<div class="who">${M.chat ? "Model" : "Model continues"}</div>`;
  r.stream = document.createElement("div"); r.stream.className = "stream cursor"; m.appendChild(r.stream);
  r.foot = document.createElement("div"); r.foot.className = "foot"; m.appendChild(r.foot);
  r.ctxBox = document.createElement("div"); r.ctxBox.className = "ctx"; r.ctxBox.style.display = "none"; m.appendChild(r.ctxBox);
  m.onclick = () => select(r);
  r.el = m; $("convo").appendChild(m); replies.push(r); select(r);
  m.scrollIntoView({block: "end"});
  return r;
}

function append(dst, ev) {
  for (const k of ["tokens", "depth", "halt", "weight", "experts", "gates"]) if (ev[k]) dst[k].push(...ev[k]);
}

function footer(r) {
  const n = r.depth.length, mean = n ? r.depth.reduce((a, b) => a + b, 0) / n : 0;
  const parts = [`${n} tokens`, `mean ${fmt(mean)} of ${r.n_iters} passes`];
  if (r.done) { parts.push(`${fmt(n / Math.max(r.done.seconds, 1e-9), 1)} tok/s`);
    if (r.done.finish === "length") parts.push("stopped at max tokens");
    if (r.done.finish === "cancelled") parts.push("stopped"); }
  if (r.truncated) parts.push("early context cut to fit max_seq_len");
  r.foot.innerHTML = parts.map((p) => `<span>${esc(p)}</span>`).join("");
  if (r.ctx) {
    const b = document.createElement("button"); b.className = "linkish"; b.type = "button";
    b.textContent = r.ctxBox.style.display === "none" ? `Show how it read the ${r.ctx.tokens.length}-token context` : "Hide context";
    b.onclick = (ev) => { ev.stopPropagation(); toggleCtx(r); footer(r); };
    r.foot.appendChild(b);
  }
}

function toggleCtx(r) {
  if (r.ctxBox.style.display === "none") {
    if (!r.ctxBox.childElementCount) {
      const s = document.createElement("div"); s.className = "stream";
      r.ctx.tokens.forEach((_, i) => s.appendChild(tokenSpan(r, i, true)));
      r.ctxBox.innerHTML = `<div class="who">The context, shaded the same way: the compute spent reading each token before the reply began</div>`;
      r.ctxBox.appendChild(s);
    }
    r.ctxBox.style.display = "";
  } else r.ctxBox.style.display = "none";
}

function select(r) {
  if (selected !== r) { replies.forEach((x) => x.el.classList.toggle("selected", x === r)); selected = r;
    if (pinned && pinned.r !== r) { document.querySelectorAll(".tok.pinned").forEach((p) => p.classList.remove("pinned")); pinned = null;
      $("detail").innerHTML = `<span class="hint">Hover over a reply token to see its routing. Click to pin it.</span>`; } }
  side(r);
}

function side(r) {
  if (r !== selected) return;
  const n = r.depth.length, I = r.n_iters;
  legend(I);
  $("side-sub").textContent = M.act
    ? "Passes through the shared loop body each token of this reply took before its ACT router stopped it, in order."
    : "This checkpoint has no ACT router, so every token runs every pass.";
  const mean = n ? r.depth.reduce((a, b) => a + b, 0) / n : 0;
  const share = (d) => n ? pct(r.depth.filter((x) => x === d).length / n) : "–";
  const tiles = [["Mean depth", n ? fmt(mean) : "–", `of ${I} passes`], ["Tokens", n, r.done ? r.done.finish === "stop" ? "ended by the model" : "" : "generating…"],
    ["Stop after pass 1", share(1), "of tokens"], [`Use all ${I} passes`, share(I), "of tokens"]];
  $("tiles").innerHTML = tiles.map(([l, v, t]) =>
    `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${esc(t)}</div></div>`).join("");
  drawStrip(r);
}

// depth strip: one bar per reply token, height = passes; single series, hover for the token
function drawStrip(r) {
  const box = $("strip"); box.innerHTML = "";
  const n = r.depth.length, I = r.n_iters; if (!n) return;
  const W = 560, H = 150, L = 30, B = 18, T = 8, R = 4, ih = H - B - T;
  const svg = el("svg", {width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Passes per reply token, in order"}, box);
  for (let g = 1; g <= I; g++) { const y = T + ih - ih * g / I;
    el("line", {x1: L, x2: W - R, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1}, svg);
    if (I <= 8 || g % 2 === 0 || g === I) el("text", {x: L - 6, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "var(--muted)"}, svg).textContent = g; }
  el("line", {x1: L, x2: W - R, y1: T + ih, y2: T + ih, stroke: "var(--axis)", "stroke-width": 1}, svg);
  const slot = (W - L - R) / Math.max(n, 40), gap = slot > 4 ? 2 : 0, bw = Math.max(slot - gap, 1);
  r.depth.forEach((d, i) => {
    const h = ih * d / I, x = L + i * slot, y = T + ih - h, rr = Math.min(4, bw / 2, h / 2);
    el("path", {d: `M${x},${T + ih} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + bw - rr} Q${x + bw},${y} ${x + bw},${y + rr} V${T + ih} Z`,
      fill: "var(--series-1)", opacity: pinned && pinned.r === r && !pinned.ctx && pinned.i !== i ? 0.45 : 1}, svg);
    const hit = el("rect", {x, y: T, width: slot, height: ih, fill: "transparent", style: "cursor:pointer"}, svg);
    hit.onmouseenter = (ev) => { if (!pinned) detail(r, i, false);
      showTip(`<b>${esc(show(r.tokens[i] || "(part of next character)"))}</b> · token ${i + 1} · ${d} pass${d > 1 ? "es" : ""}`, ev); };
    hit.onmousemove = moveTip; hit.onmouseleave = hideTip;
    hit.onclick = () => r.stream.children[i] && r.stream.children[i].click();
  });
  el("text", {x: L, y: H - 4, "font-size": 11, fill: "var(--muted)"}, svg).textContent = "reply token →";
}

function detail(r, i, ctx) {
  const S = ctx ? r.ctx : r, I = r.n_iters, d = S.depth[i];
  const isPinned = pinned && pinned.r === r && pinned.i === i && pinned.ctx === ctx;
  let h = `<div style="margin-bottom:8px"><span class="code">${esc(show(S.tokens[i]))}</span> · ${ctx ? "context" : "reply"} token ${i + 1} · ` +
    `<b>${d}</b> of ${I} passes${isPinned ? " · pinned (click again to release)" : ""}</div>`;
  h += `<table class="grid"><tr><th></th>` + Array.from({length: I}, (_, it) => `<th>Pass ${it + 1}</th>`).join("") + `</tr>`;
  if (S.halt && S.halt.length) {
    h += `<tr><th>Stop probability</th>` + S.halt[i].map((p, it) =>
      `<td class="cell${it >= d ? " halted" : ""}">${it >= d ? "stopped" : fmt(p)}</td>`).join("") + `</tr>`;
    h += `<tr><th>Share of output</th>` + S.weight[i].map((w, it) =>
      `<td class="cell${it >= d ? " halted" : ""}">${it >= d ? "–" : pct(w)}</td>`).join("") + `</tr>`;
  }
  if (S.experts && S.experts.length) {
    S.experts[i].forEach((blk, bi) => {
      h += `<tr><th>Block ${M.moe.blocks[bi]} experts</th>` + blk.map((ks, it) => {
        const g = S.gates[i][bi][it];
        const txt = ks.map((e, j) => `E${e}<span style="color:var(--muted)"> ${pct(g[j])}</span>`).join(" ");
        return `<td class="cell${it >= d ? " halted" : ""}" title="${it >= d ? "computed but discarded: the token had already stopped" : ""}">${txt}</td>`;
      }).join("") + `</tr>`;
    });
  }
  h += `</table>`;
  if (S.experts && S.experts.length && d < I) h += `<p class="sub" style="margin-top:8px">Grey passes were computed (evaluation runs every pass for every token) but the token had already stopped, so they are thrown away.</p>`;
  $("detail").innerHTML = h;
  if (!ctx) drawStrip(r);
}

let redraw = null;
const scheduleSide = (r) => { if (!redraw) redraw = requestAnimationFrame(() => { redraw = null; side(r); }); };

async function send() {
  const text = $("input").value;
  if (busy || !text.trim()) return;
  $("input").value = "";
  history.push({role: "user", content: text});
  addUser(text);
  const r = addReply();
  const num = (id) => $(id).value === "" ? null : Number($(id).value);
  const body = {messages: history, max_tokens: num("max_tokens") ?? 200, temperature: num("temperature") ?? 0.8,
    top_k: num("top_k") ?? 50, loops: num("loops")};
  busy = new AbortController();
  const t0 = performance.now();
  $("send").textContent = "Stop"; $("clear").disabled = true;
  try {
    const res = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal: busy.signal});
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail || res.statusText)); }
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const ev = JSON.parse(buf.slice(0, nl)); buf = buf.slice(nl + 1);
        if (ev.type === "prompt") {
          r.n_iters = ev.n_iters; r.truncated = ev.truncated;
          r.ctx = {tokens: [], depth: [], halt: [], weight: [], experts: [], gates: []}; append(r.ctx, ev);
        } else if (ev.type === "token") {
          const i = r.tokens.length; append(r, ev); r.stream.appendChild(tokenSpan(r, i, false));
        } else if (ev.type === "done") {
          r.done = ev; if (ev.tail && r.tokens.length) { r.tokens[r.tokens.length - 1] += ev.tail; r.stream.lastChild.textContent += ev.tail; }
        } else if (ev.type === "error") throw new Error(ev.message);
        footer(r); scheduleSide(r);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") { r.done = r.done || {finish: "cancelled", seconds: (performance.now() - t0) / 1000}; }
    else { const p = document.createElement("div"); p.className = "err"; p.textContent = e.message; r.el.appendChild(p); }
  } finally {
    busy = null; $("send").textContent = "Send"; $("clear").disabled = false;
    r.stream.classList.remove("cursor"); footer(r); side(r);
    const reply = r.tokens.join("");
    if (r.tokens.length) history.push({role: "assistant", content: reply});
    else history.pop();  // nothing came back: don't send a dangling turn next time
    $("input").focus();
  }
}

$("send").onclick = () => busy ? busy.abort() : send();
$("input").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(); } });
$("clear").onclick = () => {
  history.length = 0; replies.length = 0; selected = pinned = null;
  $("convo").innerHTML = ""; const p = document.createElement("p"); p.className = "empty"; p.id = "empty";
  p.textContent = M.chat ? "Say something." : "Write the start of some text and the model continues it."; $("convo").appendChild(p);
  $("tiles").innerHTML = ""; $("strip").innerHTML = ""; $("legend").innerHTML = "";
  $("side-sub").textContent = "Send a message to see the routing of the reply.";
  $("detail").innerHTML = `<span class="hint">Hover over a reply token to see its routing. Click to pin it.</span>`;
};
legend(M.n_iters);
$("input").focus();
})();
</script>
</body>
</html>
"""
