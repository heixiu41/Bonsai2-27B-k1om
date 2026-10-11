"""Watch the running chat_server's SSE stream and record per-token arrival
times during one turn. Even spacing ~860ms => card genuinely slow.
Bursts => transport/ssh. Usage: run, then send a message in the browser."""
import json, time, urllib.request

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
req = NO_PROXY.open("http://127.0.0.1:8641/api/events", timeout=600)
etype = None
t0 = None
arrivals = []
print("watching; send a message in the browser now...", flush=True)
try:
    for raw in req:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if line.startswith("event: "):
            etype = line[7:]
        elif line.startswith("data: ") and etype:
            now = time.time()
            if etype == "token":
                if t0 is None:
                    t0 = now
                arrivals.append(now)
                if len(arrivals) % 10 == 0:
                    gaps = [b - a for a, b in zip(arrivals[-11:-1], arrivals[-10:])]
                    print(f"tok {len(arrivals)}: last-10 gaps "
                          f"avg={sum(gaps)/len(gaps)*1000:.0f}ms "
                          f"min={min(gaps)*1000:.0f} max={max(gaps)*1000:.0f}",
                          flush=True)
            elif etype == "done" and arrivals:
                d = json.loads(line[6:])
                gaps = [b - a for a, b in zip(arrivals[:-1], arrivals[1:])]
                gaps_sorted = sorted(gaps)
                n = len(gaps)
                print(f"DONE: server tok_s={d.get('tok_s'):.2f}, "
                      f"arrivals={len(arrivals)}")
                print(f"gap ms: avg={sum(gaps)/n*1000:.0f} "
                      f"p50={gaps_sorted[n//2]*1000:.0f} "
                      f"min={gaps_sorted[0]*1000:.0f} "
                      f"max={gaps_sorted[-1]*1000:.0f}")
                even = gaps_sorted[-1] < 3 * gaps_sorted[n//2]
                print("VERDICT:", "EVEN spacing -> card-side slow"
                      if even else "BURSTY -> transport/ssh stall")
                break
            etype = None
except Exception as e:
    print("watcher ended:", repr(e))
