#!/usr/bin/env python3
"""Tune the RMA ring geometry (SCIF_RMA_SLOTS x SCIF_RMA_SLOT_MB).

The card sink's window is capped at 256 MiB by the brief, so slot size and ring
depth trade off against each other at a fixed window.  Ring depth is what
absorbs fence/latency jitter: if the card ever has to spin waiting for
ctrl[HOST_SEQ] ("card-wait-host"), the ring drained and the host is late.

Reports, per configuration: end-to-end MB/s, the card's flush rate, how long
the card spent waiting for the host, and how long the host spent waiting for a
free slot.  All of it comes from the two programs' own timers.

Usage: python tools/tune_rma_slots.py [--configs 8x32,16x16,...] [--runs 1]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
GGUF = r"D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf"
HOST = "root@192.168.1.100"
KEY = os.path.join(os.environ.get("USERPROFILE", ""), ".ssh", "id_rsa_mic")
KNOWN = os.path.join(os.environ.get("USERPROFILE", ""), ".ssh", "known_hosts")
SSH_BASE = ["-i", KEY, "-o", f"UserKnownHostsFile={KNOWN}",
            "-o", "StrictHostKeyChecking=no"]

DEFAULT_CONFIGS = ["8x32", "16x16", "4x64", "32x8"]


def ssh(cmd, timeout=120):
    return subprocess.run(["ssh", *SSH_BASE, HOST, cmd], capture_output=True,
                          text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def deploy_sink():
    local = os.path.join(TOOLS, "scif_rma_sink")
    if not os.path.exists(local):
        sys.exit("build tools/scif_rma_sink first (tools/build_rma.ps1)")
    r = ssh("cat /proc/self/status >/dev/null; killall scif_rma_sink 2>/dev/null; true")
    subprocess.run(["scp", "-O", "-c", "chacha20-poly1305@openssh.com",
                    *SSH_BASE, local, f"{HOST}:/tmp/"],
                   stdin=subprocess.DEVNULL, check=False)
    ssh("chmod +x /tmp/scif_rma_sink")


def one(cfg, runs):
    nslot, mb = (int(x) for x in cfg.split("x"))
    print(f"\n=== {cfg}  ({nslot} x {mb} MiB = {nslot * mb} MiB window) " + "=" * 20)
    out = []
    for run in range(1, runs + 1):
        ssh("killall scif_rma_sink 2>/dev/null; rm -f /tmp/bonsai.gguf; true")
        sink = subprocess.Popen(
            ["ssh", *SSH_BASE, HOST,
             f"export LD_LIBRARY_PATH=/tmp; SCIF_RMA_SLOT_MB={mb} "
             f"SCIF_RMA_SLOTS={nslot} /tmp/scif_rma_sink"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        time.sleep(2)
        t0 = time.time()
        r = subprocess.run([os.path.join(TOOLS, "scif_rma_send.exe"), GGUF,
                            "bonsai.gguf"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=1800)
        dt = time.time() - t0
        sink_out = sink.communicate(timeout=300)[0]
        # card line: "... = NNN MB/s (flush NNN MB/s, ck wait X s, host wait Y s)"
        m = re.search(r"= (\d+) MB/s \(flush (\d+) MB/s, ck wait ([\d.]+) s, "
                      r"host wait ([\d.]+) s\)", sink_out)
        s = re.search(r"sent \S+ bytes in ([\d.]+) s = ([\d.]+) MB/s", r.stderr)
        w = re.search(r"ring-wait ([\d.]+) s", r.stderr)
        ok = r.returncode == 0 and "checksum OK" in r.stderr
        rec = {
            "ok": ok,
            "wall": dt,
            "mbps": float(s.group(2)) if s else 0.0,
            "card_mbps": int(m.group(1)) if m else 0,
            "flush_mbps": int(m.group(2)) if m else 0,
            "ck_wait": float(m.group(3)) if m else -1,
            "host_wait": float(m.group(4)) if m else -1,
            "host_ring_wait": float(w.group(1)) if w else -1,
        }
        out.append(rec)
        print(f"  run {run}: {'OK ' if ok else 'FAIL'} "
              f"{rec['mbps']:6.1f} MB/s | card {rec['card_mbps']} MB/s "
              f"flush {rec['flush_mbps']} MB/s | card-wait-host "
              f"{rec['host_wait']:.3f} s | host ring-wait "
              f"{rec['host_ring_wait']:.1f} s | ck-wait {rec['ck_wait']:.3f} s")
    ssh("killall scif_rma_sink 2>/dev/null; true")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args(argv)

    deploy_sink()
    results = {}
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        results[cfg] = one(cfg, args.runs)

    print("\n=== summary " + "=" * 52)
    print(f"{'config':>8} {'window':>9} {'MB/s':>8} {'flush':>7} "
          f"{'card-wait-host':>15} {'host-ring-wait':>15} {'ok':>4}")
    for cfg, recs in results.items():
        good = [r for r in recs if r["ok"]]
        if not good:
            print(f"{cfg:>8} {'':>9} {'':>8} {'':>7} {'':>15} {'':>15} "
                  f"{'FAIL':>4}")
            continue
        n = len(good)
        win = cfg.split("x")
        print(f"{cfg:>8} {int(win[0]) * int(win[1]):>6} MiB "
              f"{sum(r['mbps'] for r in good) / n:8.1f} "
              f"{sum(r['flush_mbps'] for r in good) / n:7.0f} "
              f"{sum(r['host_wait'] for r in good) / n:14.3f}s "
              f"{sum(r['host_ring_wait'] for r in good) / n:14.1f}s "
              f"{n:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
