#!/usr/bin/env python3
"""Local web chat UI for the Bonsai-2 27B k1om inference card.

Pure stdlib (http.server + threading + json + queue + subprocess). All card
and tokenizer mechanics are imported from chat.py (the proven console
client): CardClient, Tokenizer (+ read_tokenizer_kv), StreamDecoder,
ensure_card_files / _upload_model_scif, and the per-turn protocol constants
USER_BLOCK / TURN_CLOSER.

    http://127.0.0.1:8641

Endpoints:
    GET  /             single-page chat UI (embedded HTML)
    GET  /api/events   SSE stream: status/thinking/token/done/error/reset/load
    POST /api/send     {"text": ...}
    POST /api/reset

The `load` events (card core-busy % + memory %, 1 Hz) come from a second,
independent ssh channel (LoadMonitor) -- the chat protocol channel is fully
occupied while the card streams tokens.

Usage:
    python scripts/chat_server.py              # serve
    python scripts/chat_server.py --selftest   # tokenizer tests only (no ssh)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# chat.py lives next to this file; make sure its directory is importable no
# matter where the interpreter was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chat  # noqa: E402  (scripts/chat.py -- the proven building blocks)
from chat import (  # noqa: E402
    CardClient,
    CardDied,
    CardError,
    StreamDecoder,
    Tokenizer,          # the GGUF tokenizer (pure-stdlib BPE)
    ensure_card_files,
    read_tokenizer_kv,
    GGUF_PATH,
    SSH_HOST,
    MAX_NEW_TOKENS,
    USER_BLOCK,
    TURN_CLOSER,
)

CTX_LEN = 8192


# ---------------------------------------------------------------------------
# Card load monitor: TWO independent ssh channels (the chat protocol channel
# is fully occupied during generation).
#
# Measured cost on the card (BusyBox loop, 2026-10-11):
#   cat /proc/meminfo   ~6 ms/sample   (cheap: 1 Hz, always on)
#   cat /proc/stat     ~420 ms/sample, ~330 ms in-kernel (formats 244 HT
#                       threads' counters) -- at 1 Hz this visibly stalls the
#                       OMP workers: 3.60 -> 2.30 tok/s. So /proc/stat is
#                       SUSPENDED while a turn runs (zero interference) and
#                       sampled at 0.5 Hz only while idle.
# ---------------------------------------------------------------------------
class LoadMonitor:
    # `|| exit` is load-bearing: when the ssh channel dies (local kill,
    # network drop), the remote cat dies on SIGPIPE and the loop EXITS.
    # Without it the shell loop survives forever on the card (observed:
    # 5 leaked loops at 1 Hz -> decode 3.4 -> 0.8 tok/s). BusyBox pkill
    # has no working -f, so cleanup greps /proc/*/cmdline for the tag.
    MEM_LOOP = ("while :; do cat /proc/meminfo || exit; sleep 1; done"
                " # loadmon_mem")
    CPU_LOOP = ("while :; do cat /proc/stat || exit; sleep 2; done"
                " # loadmon_cpu")
    _KILLER = ("for f in /proc/[0-9]*/cmdline; do "
               "grep -q loadmon_ $f 2>/dev/null || continue; "
               "p=${f#/proc/}; kill ${p%/cmdline} 2>/dev/null; "
               "done")

    def __init__(self, host: str):
        self.host = host
        self.mem_proc: subprocess.Popen | None = None
        self.cpu_proc: subprocess.Popen | None = None
        self.stop_flag = False
        self.cpu_suspended = False         # True while a turn generates
        self.last = {"cpu": None, "mem": None}     # replayed to late joiners

    def start(self):
        threading.Thread(target=self._mem_run, daemon=True).start()
        threading.Thread(target=self._cpu_run, daemon=True).start()

    def close(self):
        self.stop_flag = True
        for p in (self.mem_proc, self.cpu_proc):
            if p is not None:
                try:
                    p.kill()
                except OSError:
                    pass
        self._remote_cleanup()           # belt: sweep any leaked loops

    def _remote_cleanup(self):
        try:
            subprocess.run(["ssh", *chat._ssh_base(self.host), self.host,
                            self._KILLER], capture_output=True, timeout=15)
        except Exception:
            pass

    # -- cpu sampling suspension (called by the turn worker) -----------------
    def suspend_cpu(self):
        self.cpu_suspended = True
        p = self.cpu_proc
        self.cpu_proc = None
        if p is not None:
            try:
                p.kill()               # channel death -> remote `|| exit`
            except OSError:
                pass
        self.last["cpu"] = None
        publish("load", {"cpu": None, "mem": self.last["mem"],
                         "paused": True})

    def resume_cpu(self):
        self.cpu_suspended = False     # _cpu_run respawns on its own

    # -- ssh plumbing ---------------------------------------------------------
    def _spawn(self, loop: str):
        cmd = ["ssh", *chat._ssh_base(self.host), self.host, loop]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True,
                                errors="replace", bufsize=1)

    def _wait_or_stop(self, seconds: float):
        for _ in range(int(seconds * 10)):
            if self.stop_flag:
                return False
            time.sleep(0.1)
        return True

    def _mem_run(self):
        mem_total = None
        while not self.stop_flag:
            try:
                if self.mem_proc is None or self.mem_proc.poll() is not None:
                    self.mem_proc = self._spawn(self.MEM_LOOP)
                line = self.mem_proc.stdout.readline()
                if not line:
                    raise ConnectionError("mem monitor channel closed")
                parts = line.split()
                if parts and parts[0] == "MemTotal:":
                    mem_total = int(parts[1])
                elif parts and parts[0] == "MemFree:" and mem_total:
                    self.last["mem"] = 100.0 * (mem_total - int(parts[1])) / mem_total
                    publish("load", {"cpu": self.last["cpu"],
                                     "mem": self.last["mem"],
                                     "paused": self.cpu_suspended})
            except Exception:
                if self.mem_proc is not None:
                    try:
                        self.mem_proc.kill()
                    except OSError:
                        pass
                self.mem_proc = None
                self.last["mem"] = None
                publish("load", {"cpu": self.last["cpu"], "mem": None,
                                 "paused": self.cpu_suspended})
                if not self._wait_or_stop(5):
                    return

    def _cpu_run(self):
        prev = None                                  # (busy, total) jiffies
        while not self.stop_flag:
            try:
                if self.cpu_suspended:
                    prev = None                      # jiffies jump: rebase
                    if not self._wait_or_stop(0.5):
                        return
                    continue
                if self.cpu_proc is None or self.cpu_proc.poll() is not None:
                    self.cpu_proc = self._spawn(self.CPU_LOOP)
                line = self.cpu_proc.stdout.readline()
                if not line:
                    raise ConnectionError("cpu monitor channel closed")
                parts = line.split()
                if parts and parts[0] == "cpu":
                    vals = [int(x) for x in parts[1:9]]
                    busy = vals[0] + vals[1] + vals[2] + vals[5] + vals[6]
                    total = sum(vals)
                    if prev is not None and total > prev[1]:
                        self.last["cpu"] = 100.0 * (busy - prev[0]) / (total - prev[1])
                    prev = (busy, total)
            except Exception:
                if self.cpu_proc is not None:
                    try:
                        self.cpu_proc.kill()
                    except OSError:
                        pass
                self.cpu_proc = None
                self.last["cpu"] = None
                if not self._wait_or_stop(5):
                    return


# ---------------------------------------------------------------------------
# HTML page (dark theme, Chinese labels)
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bonsai-2 27B · k1om</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --fg: #e6edf3; --dim: #8b949e; --accent: #58a6ff;
    --user: #1f6feb; --think: #9e8a5a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.6 "Segoe UI", "Microsoft YaHei", sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 10px 16px; background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .sub { color: var(--dim); font-size: 12px; }
  #statusbar {
    margin-left: auto; display: flex; gap: 14px; font-size: 12px;
    color: var(--dim); font-variant-numeric: tabular-nums;
  }
  #statusbar b { color: var(--fg); font-weight: 600; }
  #conn-dot { display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: #888; margin-right: 4px; }
  #conn-dot.ok { background: #3fb950; }
  #conn-dot.bad { background: #f85149; }
  #conn-dot.wait { background: #d29922; }
  #chat { flex: 1; overflow-y: auto; padding: 18px 12px; }
  .msg { max-width: 820px; margin: 0 auto 14px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble {
    padding: 10px 14px; border-radius: 12px; max-width: 85%;
    white-space: pre-wrap; word-wrap: break-word;
  }
  .msg.user .bubble { background: var(--user); color: #fff;
    border-bottom-right-radius: 4px; }
  .msg.assistant .bubble { background: var(--panel);
    border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .bubble details {
    background: #1c2128; border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; margin: 4px 0 8px;
    color: var(--think); font-size: 13px;
  }
  .bubble details summary { cursor: pointer; color: var(--dim);
    user-select: none; }
  .bubble details > div { white-space: pre-wrap; margin-top: 4px; }
  .shimmer {
    display: inline-block; min-width: 90px; color: transparent;
    background: linear-gradient(90deg, #444 25%, #999 50%, #444 75%);
    background-size: 200% 100%; -webkit-background-clip: text;
    background-clip: text; animation: sh 1.2s linear infinite;
  }
  @keyframes sh { to { background-position: -200% 0; } }
  #inputbar {
    padding: 12px; background: var(--panel);
    border-top: 1px solid var(--border);
  }
  #inputbar .row { max-width: 820px; margin: 0 auto; display: flex; gap: 8px; }
  #input {
    flex: 1; resize: none; background: #0d1117; color: var(--fg);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; font: inherit; min-height: 46px; max-height: 160px;
  }
  #input:focus { outline: none; border-color: var(--accent); }
  button {
    background: var(--accent); color: #06101f; border: 0;
    border-radius: 8px; padding: 0 18px; font: inherit; font-weight: 600;
    cursor: pointer;
  }
  button:disabled { opacity: .45; cursor: not-allowed; }
  #reset { background: transparent; color: var(--dim);
    border: 1px solid var(--border); }
  #reset:hover { color: var(--fg); }
  .meter { display: inline-block; width: 64px; height: 6px;
    background: #21262d; border-radius: 3px; overflow: hidden;
    vertical-align: middle; margin-left: 4px; }
  .meter i { display: block; height: 100%; width: 0%;
    background: #3fb950; transition: width .5s; }
  .meter i.warn { background: #d29922; }
  .meter i.hot  { background: #f85149; }
</style>
</head>
<body>
<header>
  <h1>Bonsai-2 27B <span class="sub">· Xeon Phi k1om 推理卡 · ~3.4 tok/s</span></h1>
  <div id="statusbar">
    <span><span id="conn-dot" class="wait"></span><b id="st-conn">连接中…</b></span>
    <span>速度 <b id="st-speed">–</b> tok/s</span>
    <span>已生成 <b id="st-total">0</b> tok</span>
    <span>上下文 <b id="st-ctx">0/8192</b></span>
    <span title="61 核 244 线程合计占用。生成期间暂停采样(读 /proc/stat 会拖慢推理 36%),显示 ⏸">内核 <b id="st-cpu">–</b>%<span class="meter"><i id="bar-cpu"></i></span></span>
    <span title="卡上 15.5 GB 内存(tmpfs,含模型文件驻留)">内存 <b id="st-mem">–</b>%<span class="meter"><i id="bar-mem"></i></span></span>
  </div>
</header>
<div id="chat"></div>
<div id="inputbar">
  <div class="row">
    <textarea id="input" rows="1" placeholder="输入消息… (Enter 发送, Shift+Enter 换行)"></textarea>
    <button id="send" disabled>发送</button>
    <button id="reset" title="清空对话并重置卡片上下文">重置</button>
  </div>
</div>
<script>
"use strict";
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const resetBtn = document.getElementById("reset");
const stConn = document.getElementById("st-conn");
const stSpeed = document.getElementById("st-speed");
const stTotal = document.getElementById("st-total");
const stCtx = document.getElementById("st-ctx");
const stCpu = document.getElementById("st-cpu");
const stMem = document.getElementById("st-mem");
const barCpu = document.getElementById("bar-cpu");
const barMem = document.getElementById("bar-mem");
const connDot = document.getElementById("conn-dot");

function setMeter(bar, label, pct) {
  if (pct === null || pct === undefined) { label.textContent = "–"; bar.style.width = "0%"; return; }
  const v = Math.max(0, Math.min(100, pct));
  label.textContent = v.toFixed(0);
  bar.style.width = v + "%";
  bar.className = v >= 95 ? "hot" : (v >= 80 ? "warn" : "");
}

let curBubble = null;   // bubble element of the in-flight assistant turn
let curRaw = "";        // raw accumulated text of that turn
let busy = false;

function setConn(state, label) {
  connDot.className = state;           // ok | bad | wait
  stConn.textContent = label;
}

function addBubble(role) {
  const msg = document.createElement("div");
  msg.className = "msg " + role;
  const b = document.createElement("div");
  b.className = "bubble";
  msg.appendChild(b);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return b;
}

// Split raw model text into think / answer segments.
function parseSegments(raw) {
  const segs = [];
  let rest = raw;
  while (true) {
    const i = rest.indexOf("<think>");
    if (i === -1) { if (rest) segs.push({t: "text", c: rest, closed: true}); break; }
    if (i > 0) segs.push({t: "text", c: rest.slice(0, i), closed: true});
    rest = rest.slice(i + 7);
    const j = rest.indexOf("</think>");
    if (j === -1) { segs.push({t: "think", c: rest, closed: false}); break; }
    segs.push({t: "think", c: rest.slice(0, j), closed: true});
    rest = rest.slice(j + 8);
  }
  return segs;
}

function renderAssistant(bubble, raw, thinking) {
  bubble.textContent = "";
  const segs = parseSegments(raw);
  if (thinking && !segs.some(s => s.t === "think" && !s.closed)) {
    const s = document.createElement("span");
    s.className = "shimmer";
    s.textContent = "思考中…";
    bubble.appendChild(s);
  }
  for (const s of segs) {
    if (s.t === "think") {
      const d = document.createElement("details");
      if (!s.closed) d.open = true;
      const sum = document.createElement("summary");
      sum.textContent = s.closed ? "思考过程（点击展开）" : "正在思考…";
      d.appendChild(sum);
      const div = document.createElement("div");
      div.textContent = s.c;
      d.appendChild(div);
      bubble.appendChild(d);
    } else if (s.c.trim() || bubble.childNodes.length === 0) {
      const div = document.createElement("div");
      div.textContent = s.c;
      bubble.appendChild(div);
    }
  }
  chat.scrollTop = chat.scrollHeight;
}

function setBusy(b) {
  busy = b;
  sendBtn.disabled = b;
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  setBusy(true);
  input.value = "";
  addBubble("user").textContent = text;
  curBubble = addBubble("assistant");
  curRaw = "";
  renderAssistant(curBubble, "", true);      // thinking shimmer
  try {
    const r = await fetch("/api/send", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      renderAssistant(curBubble, "⚠ " + (e.error || ("HTTP " + r.status)), false);
      setBusy(false);
    }
  } catch (e) {
    renderAssistant(curBubble, "⚠ 网络错误: " + e, false);
    setBusy(false);
  }
}

sendBtn.onclick = send;
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

resetBtn.onclick = async () => {
  const r = await fetch("/api/reset", {method: "POST"});
  if (r.ok) {
    chat.innerHTML = "";
    curBubble = null; curRaw = "";
    stTotal.textContent = "0";
    stCtx.textContent = "0/8192";
  }
};

// ---- SSE ------------------------------------------------------------------
const es = new EventSource("/api/events");
es.addEventListener("status", e => {
  const d = JSON.parse(e.data);
  if (d.state === "ready") { setConn("ok", "已连接"); }
  else if (d.state === "dead") { setConn("bad", "连接断开"); }
  else { setConn("wait", d.message || "连接中…"); }
});
es.addEventListener("thinking", () => {
  if (curBubble) renderAssistant(curBubble, curRaw, true);
});
es.addEventListener("token", e => {
  const d = JSON.parse(e.data);
  if (!curBubble) { curBubble = addBubble("assistant"); curRaw = ""; }
  curRaw += d.text;
  renderAssistant(curBubble, curRaw, false);
  stSpeed.textContent = d.tok_s.toFixed(2);
  stTotal.textContent = d.total;
  stCtx.textContent = d.pos + "/8192";
});
es.addEventListener("done", e => {
  const d = JSON.parse(e.data);
  if (curBubble) renderAssistant(curBubble, curRaw, false);
  stSpeed.textContent = d.tok_s.toFixed(2);
  stTotal.textContent = d.total;
  stCtx.textContent = d.pos + "/8192";
  setBusy(false);
});
es.addEventListener("error", e => {
  let msg = "未知错误";
  try { msg = JSON.parse(e.data).message; } catch (_) {}
  if (curBubble) renderAssistant(curBubble, curRaw + "\n⚠ " + msg, false);
  else addBubble("assistant").textContent = "⚠ " + msg;
  setConn("bad", "出错");
  setBusy(false);
});
es.addEventListener("reset", () => {
  chat.innerHTML = "";
  curBubble = null; curRaw = "";
  setBusy(false);
});
es.addEventListener("load", e => {
  const d = JSON.parse(e.data);
  if (d.paused) { stCpu.textContent = "⏸"; barCpu.style.width = "0%"; }
  else setMeter(barCpu, stCpu, d.cpu);
  setMeter(barMem, stMem, d.mem);
});
es.onerror = () => setConn("bad", "SSE 断开,重连中…");
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Shared server state
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.tok: Tokenizer | None = None
        self.client: CardClient | None = None
        self.alive = False               # card connection usable
        self.first_turn = True           # mirror chat.py: reset before turn 1
        self.pos = 0                     # tokens currently in card state
        self.total_gen = 0               # tokens generated this session
        self.turn_lock = threading.Lock()   # serializes card turns
        self.subs: list[queue.Queue] = []   # SSE subscriber queues
        self.subs_lock = threading.Lock()
        self.last_status = {"state": "connecting",
                            "message": "正在连接推理卡…"}
        self.monitor: LoadMonitor | None = None   # card load sampler


STATE = State()


def publish(etype: str, data: dict):
    """Broadcast one SSE event to every connected page."""
    payload = (etype, json.dumps(data, ensure_ascii=False))
    with STATE.subs_lock:
        subs = list(STATE.subs)
    for q in subs:
        q.put(payload)


def set_status(state: str, message: str = ""):
    STATE.last_status = {"state": state, "message": message}
    publish("status", STATE.last_status)


# ---------------------------------------------------------------------------
# Turn worker (mirrors chat.py's REPL turn exactly)
# ---------------------------------------------------------------------------
class WebCardClient(CardClient):
    """CardClient with an event-publishing generate loop (same wire
    protocol as chat.py's generate(), but tokens go to the SSE queue)."""

    def generate_events(self, tok: Tokenizer, max_new: int,
                        pos0: int, total0: int):
        self._send(f"G {max_new}")
        dec = StreamDecoder(tok)
        ids = []
        t0 = time.time()
        while True:
            line = self._read_line()
            if line.isdigit():
                tid = int(line)
                ids.append(tid)
                s = dec.feed(tid)
                dt = time.time() - t0
                # pos: ingested base + emitted non-EOS tokens so far
                emitted = len(ids) - (1 if ids[-1] == tok.eos_id else 0)
                publish("token", {
                    "text": s,
                    "tok_s": len(ids) / dt if dt > 0 else 0.0,
                    "total": total0 + len(ids),
                    "pos": pos0 + emitted,
                })
            elif line.startswith("END"):
                break
            elif line.startswith("ERR"):
                raise CardError(line)
            # anything else is card-side diagnostics; ignore for the UI
        tail = dec.flush()
        dt = time.time() - t0
        return ids, tail, (len(ids) / dt if dt > 0 else 0.0)


def run_turn(text: str):
    """One full user turn. Called in a worker thread with turn_lock held."""
    tok, client = STATE.tok, STATE.client
    total0 = STATE.total_gen
    # /proc/stat sampling stalls the OMP workers (measured: -36% tok/s);
    # suspend it for the whole OMP-hot turn (prefill + generation).
    if STATE.monitor is not None:
        STATE.monitor.suspend_cpu()
    try:
        ids = tok.encode(USER_BLOCK.format(text))
        if STATE.first_turn:
            client.reset()                 # fresh context for turn 1
            STATE.first_turn = False
            STATE.pos = 0
        publish("thinking", {"stage": "prefill", "tokens": len(ids)})
        client.ingest(ids, verbose=True)   # progress goes to the console
        STATE.pos += len(ids)
        pos0 = STATE.pos

        publish("thinking", {"stage": "generate"})
        gen_ids, tail, rate = client.generate_events(
            tok, MAX_NEW_TOKENS, pos0, total0)
        if tail:
            publish("token", {"text": tail, "tok_s": rate,
                              "total": total0 + len(gen_ids),
                              "pos": pos0 + len(gen_ids)})

        # KV bookkeeping, identical to chat.py: the emitted non-EOS tokens
        # are already in card state; ingest the <|im_end|>\n turn closer so
        # the next user block lines up.
        emitted = (len(gen_ids) - 1
                   if gen_ids and gen_ids[-1] == tok.eos_id else len(gen_ids))
        closer = tok.encode(TURN_CLOSER)
        client.ingest(closer, verbose=False)
        STATE.pos = pos0 + emitted + len(closer)
        STATE.total_gen = total0 + len(gen_ids)
        publish("done", {"tok_s": rate, "tokens": len(gen_ids),
                         "total": STATE.total_gen, "pos": STATE.pos,
                         "eos": bool(gen_ids and gen_ids[-1] == tok.eos_id)})
    except CardError as e:
        publish("error", {"message": f"卡片错误: {e} — 请点“重置”"})
    except CardDied as e:
        STATE.alive = False
        set_status("dead", "连接断开")
        publish("error", {"message": f"卡片连接中断: {e}"})
    except Exception as e:                 # never kill the worker silently
        publish("error", {"message": f"内部错误: {e!r}"})
    finally:
        if STATE.monitor is not None:
            STATE.monitor.resume_cpu()
        STATE.turn_lock.release()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):     # keep the console clean-ish
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    # -- helpers ------------------------------------------------------------
    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/events":
            self._sse()
        else:
            self._send_json(404, {"error": "not found"})

    def _sse(self):
        q: queue.Queue = queue.Queue()
        with STATE.subs_lock:
            STATE.subs.append(q)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            # replay the current status so a late-joining page catches up
            self._sse_send("status", STATE.last_status)
            if STATE.monitor is not None:
                self._sse_send("load", STATE.monitor.last)
            while True:
                try:
                    etype, data = q.get(timeout=20)
                    self._sse_send(etype, json.loads(data))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                              # browser went away
        finally:
            self.close_connection = True      # nothing follows SSE here
            with STATE.subs_lock:
                if q in STATE.subs:
                    STATE.subs.remove(q)

    def _sse_send(self, etype: str, obj):
        data = json.dumps(obj, ensure_ascii=False)
        self.wfile.write(f"event: {etype}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- POST ---------------------------------------------------------------
    def do_POST(self):
        if self.path == "/api/send":
            self._api_send()
        elif self.path == "/api/reset":
            self._api_reset()
        else:
            self._send_json(404, {"error": "not found"})

    def _api_send(self):
        try:
            text = str(self._read_json().get("text", "")).strip()
        except (ValueError, UnicodeDecodeError):
            return self._send_json(400, {"error": "bad JSON body"})
        if not text:
            return self._send_json(400, {"error": "empty text"})
        if not STATE.alive:
            return self._send_json(503, {"error": "卡片未连接"})
        if not STATE.turn_lock.acquire(blocking=False):
            return self._send_json(409, {"error": "上一轮还在生成中,请稍候"})
        # lock is released by the worker (run_turn's finally)
        threading.Thread(target=run_turn, args=(text,), daemon=True).start()
        self._send_json(200, {"ok": True})

    def _api_reset(self):
        if not STATE.alive:
            return self._send_json(503, {"error": "卡片未连接"})
        if not STATE.turn_lock.acquire(timeout=240):
            return self._send_json(409, {"error": "上一轮还在生成中,请稍候"})
        try:
            STATE.client.reset()
            STATE.first_turn = True
            STATE.pos = 0
        except (CardDied, CardError) as e:
            STATE.alive = isinstance(e, CardError)
            return self._send_json(500, {"error": str(e)})
        finally:
            STATE.turn_lock.release()
        publish("reset", {})
        self._send_json(200, {"ok": True})


# ---------------------------------------------------------------------------
# Startup: files on the card -> CardClient -> READY -> E <eos>
# ---------------------------------------------------------------------------
def startup(host: str, gguf: str) -> bool:
    """Runs in a background thread; reports progress as SSE status events."""
    try:
        # load monitor comes up first: its ssh channel is independent of the
        # chat protocol, so it keeps sampling during upload and generation
        STATE.monitor = LoadMonitor(host)
        STATE.monitor.start()

        set_status("connecting", "检查卡片文件(如需会自动上传 7.2 GB 模型)…")
        if not ensure_card_files(host, gguf):
            set_status("dead", "卡片文件准备失败")
            return False

        set_status("connecting", "加载分词器…")
        meta = read_tokenizer_kv(gguf)
        STATE.tok = Tokenizer(meta)
        print(f"tokenizer ready: {len(STATE.tok.id2tok):,} tokens, "
              f"{len(STATE.tok.ranks):,} merges", flush=True)

        set_status("connecting", "卡片加载模型中(约 15 s)…")
        client = WebCardClient(host=host)
        STATE.client = client
        client.wait_ready()

        eos_id = STATE.tok.eos_id           # 248046 (<|im_end|>)
        client.set_eos(eos_id)
        print(f"EOS set to {eos_id} ({STATE.tok.id2tok[eos_id]})", flush=True)

        STATE.alive = True
        set_status("ready", "已连接")
        print("card READY -- the web UI is fully operational.", flush=True)
        return True
    except (CardDied, CardError) as e:
        set_status("dead", "连接断开")
        print(f"[fatal] {e}", file=sys.stderr)
        return False
    except (OSError, KeyError, ValueError) as e:
        set_status("dead", f"启动失败: {e}")
        print(f"[fatal] startup failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Self test: reuse chat.py's known vectors verbatim (no ssh, no server)
# ---------------------------------------------------------------------------
def selftest(gguf: str) -> int:
    return chat.selftest(argparse.Namespace(gguf=gguf))


def main(argv=None) -> int:
    # the Windows console may be GBK; utf-8 keeps Chinese output safe
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run tokenizer tests only (no ssh, no server)")
    ap.add_argument("--gguf", default=GGUF_PATH, help="local GGUF path")
    ap.add_argument("--host", default=SSH_HOST, help="ssh target")
    ap.add_argument("--port", type=int, default=8641, help="listen port")
    ap.add_argument("--bind", default="127.0.0.1", help="listen address")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args.gguf)

    url = f"http://{args.bind}:{args.port}"
    print(f"Bonsai-2 27B web chat -> {url}", flush=True)

    # Bring the HTTP server up first so the page can show startup progress
    # (incl. the ~3-4 min SCIF model upload) through the status bar.
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    boot = threading.Thread(target=startup, args=(args.host, args.gguf),
                            daemon=True)
    boot.start()
    print("serving; the card connection comes up in the background.",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ...", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        if STATE.monitor is not None:
            STATE.monitor.close()      # kill monitor ssh (no orphan loop)
        if STATE.client is not None:
            STATE.client.close()          # kill ssh so no orphan card process
            print("card client closed (no orphan bonsai_k1om_chat).",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
