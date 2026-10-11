"""E2E probe for chat_server.py's LoadMonitor: verifies /api/events emits
'load' events, and that core-busy% visibly rises during generation."""
import json, subprocess, sys, threading, time, urllib.request

PY = sys.executable
URL = "http://127.0.0.1:8641"
# bypass the system proxy (127.0.0.1:7888): it can reset long-lived SSE streams
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

srv = subprocess.Popen([PY, "scripts/chat_server.py"],
                       stdout=open("e2e_server.log", "w", encoding="utf-8"),
                       stderr=subprocess.STDOUT)

try:
    # wait for the page
    for _ in range(60):
        try:
            body = NO_PROXY.open(URL, timeout=2).read().decode()
            assert "Bonsai-2" in body and "st-cpu" in body and "bar-mem" in body
            print("[e2e] GET / OK, load widget markers present", flush=True)
            break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit("server page never came up")

    # SSE: collect load + status events
    events, stop = [], threading.Event()
    def sse():
        try:
            req = NO_PROXY.open(URL + "/api/events", timeout=1800)
            etype = None
            for raw in req:
                if stop.is_set():
                    break
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event: "):
                    etype = line[7:]
                elif line.startswith("data: ") and etype:
                    try:
                        events.append((etype, json.loads(line[6:])))
                    except ValueError:
                        pass
                    etype = None
        except Exception as e:
            print(f"[e2e] SSE channel died: {e!r} (events so far: {len(events)})",
                  flush=True)
            events.append(("__sse_died__", {}))
    t = threading.Thread(target=sse, daemon=True)
    t.start()

    # wait for card READY (model re-upload can take ~13 min via scp fallback)
    deadline = time.time() + 1500
    while time.time() < deadline:
        if any(e == "status" and d.get("state") == "ready" for e, d in events):
            break
        if any(e == "__sse_died__" for e, _ in events):
            raise SystemExit("SSE channel died before READY")
        time.sleep(2)
    else:
        raise SystemExit("card never became READY")
    print("[e2e] card READY", flush=True)

    # load events flowing while idle?
    idle = [(e, d) for e, d in events if e == "load" and d.get("cpu") is not None]
    assert len(idle) >= 2, f"no load events yet: {events[:8]}"
    print(f"[e2e] idle load events OK: cpu={idle[-1][1]['cpu']:.1f}% "
          f"mem={idle[-1][1]['mem']:.1f}%", flush=True)

    # one generation turn; load monitor must keep publishing through it
    NO_PROXY.open(urllib.request.Request(
        URL + "/api/send", data=json.dumps({"text": "用一句话介绍北京"}).encode(),
        headers={"Content-Type": "application/json"}), timeout=30)
    n0 = len(events)
    deadline = time.time() + 300
    while time.time() < deadline:
        if any(e == "done" for e, d in events[n0:]):
            break
        time.sleep(2)
    gen_loads = [d for e, d in events[n0:] if e == "load" and d.get("cpu") is not None]
    done = [d for e, d in events[n0:] if e == "done"]
    assert done, "turn never finished"
    assert gen_loads, "load monitor stalled during generation"
    peak = max(d["cpu"] for d in gen_loads)
    print(f"[e2e] turn done: {done[-1]}", flush=True)
    print(f"[e2e] load events during generation: {len(gen_loads)}, "
          f"peak cpu={peak:.0f}% (idle was {idle[-1][1]['cpu']:.0f}%)", flush=True)
    assert peak > idle[-1][1]["cpu"] + 20, "cpu% did not rise during generation?"
    print("LOAD E2E PASS", flush=True)
finally:
    srv.kill()
    time.sleep(2)
    # surface any server-side handler tracebacks
    try:
        log = open("e2e_server.log", encoding="utf-8", errors="replace").read()
        tb = [l for l in log.splitlines() if "Traceback" in l or "Error" in l]
        if tb:
            print("[e2e] server log anomalies:", *tb[:10], sep="\n  ", flush=True)
    except OSError:
        pass
