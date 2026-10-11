#!/usr/bin/env python3
"""Prove all three upload layers still work, without three 7.2 GB transfers.

ensure_card_files() falls back RMA SCIF -> message SCIF -> scp.  A fallback
chain is only as good as its weakest link, and the two slow layers would take
~15 minutes on the real model, so this drives each layer on its own with a
small slice of the model (BONSAI_TRANSFER_LAYER=<layer>) and then checks the
bytes that actually landed on the card with BusyBox md5sum.

Usage:
    python scripts/verify_transfer_layers.py                 # all three, 64 MiB
    python scripts/verify_transfer_layers.py --mb 256
    python scripts/verify_transfer_layers.py --layers rma
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat  # noqa: E402  (path juggling must come first)


def ssh_run(host, base, cmd, timeout=120):
    return subprocess.run(["ssh", *base, host, cmd],
                          capture_output=True, text=True, timeout=timeout)


def md5_local(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def md5_card(host, base, path: str, timeout=900):
    r = ssh_run(host, base, f"md5sum {path}", timeout=timeout)
    tok = r.stdout.split()
    return tok[0] if tok else None


def card_size(host, base, path: str):
    # BusyBox has no `stat` and its `wc -c` is 32-bit; chat._card_file_size
    # parses `ls -l` instead.
    return chat._card_file_size(host, base, path)


def make_slice(src: str, mb: int, dst: str):
    """First `mb` MiB of the model -- real, non-uniform bytes."""
    want = mb << 20
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while want > 0:
            blk = fi.read(min(1 << 22, want))
            if not blk:
                break
            fo.write(blk)
            want -= len(blk)
    return os.path.getsize(dst)


def check_md5(args) -> int:
    """Positive and negative test of the opt-in whole-file md5 comparison."""
    base = chat._ssh_base(args.host)
    tmp = os.path.join(tempfile.gettempdir(), "bonsai_layer_probe.bin")
    size = make_slice(args.gguf, args.mb, tmp)
    print(f"slice: {size:,} bytes (md5={md5_local(tmp)})\n")

    ssh_run(args.host, base, "killall scif_rma_sink scif_sink 2>/dev/null; "
                             "rm -f /tmp/bonsai.gguf; true")
    os.environ.pop("BONSAI_TRANSFER_LAYER", None)
    os.environ["BONSAI_VERIFY_MD5"] = "1"
    ok1 = caught = False
    try:
        print("=== positive: upload + md5 verify ===", flush=True)
        ok1 = chat.ensure_card_files(args.host, tmp)
        print(f"[md5 positive] {'PASS' if ok1 else 'FAIL'}\n", flush=True)

        print("=== negative: overwrite the card's copy with random data "
              "(same size) ===", flush=True)
        # BusyBox dd has no conv=notrunc, so an in-place poke is out; replacing
        # the whole file keeps the LENGTH identical, which proves the check
        # catches content corruption and not just a size change.
        ssh_run(args.host, base,
                f"dd if=/dev/urandom of=/tmp/bonsai.gguf bs=1M "
                f"count={args.mb} 2>/dev/null; true")
        # the file is present, so ensure_card_files skips the upload and only
        # runs the verification -> it must now refuse
        ok2 = chat.ensure_card_files(args.host, tmp)
        caught = not ok2
        print(f"[md5 negative] {'PASS (mismatch detected)' if caught else 'FAIL (corruption NOT detected)'}\n",
              flush=True)
    finally:
        os.environ.pop("BONSAI_VERIFY_MD5", None)
        ssh_run(args.host, base, "rm -f /tmp/bonsai.gguf")
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if ok1 and caught:
        print("md5 switch: PASS")
        return 0
    print(f"[fatal] md5 switch: FAIL (positive={ok1} negative_caught={caught})",
          file=sys.stderr)
    return 1


def check_fallback(args) -> int:
    """Drive the real fallback: hide the host senders so the chosen layer
    cannot run at all, then require auto mode to deliver the file anyway."""
    base = chat._ssh_base(args.host)
    tools = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    names = ["scif_rma_send.exe"]
    if args.fallback == "scif":
        names.append("scif_send.exe")

    paths = [os.path.join(tools, n) for n in names]
    for p in paths:
        if not os.path.exists(p):
            print(f"[fatal] {p} does not exist, nothing to hide", file=sys.stderr)
            return 2

    tmp = os.path.join(tempfile.gettempdir(), "bonsai_layer_probe.bin")
    size = make_slice(args.gguf, args.mb, tmp)
    want_md5 = md5_local(tmp)
    expect = "scp" if args.fallback == "scif" else "message SCIF"
    print(f"slice: {size:,} bytes  md5={want_md5}")
    print(f"hiding {', '.join(paths)} so the earlier layer(s) must fail; "
          f"expect the {expect} path to deliver the file\n")

    os.environ.pop("BONSAI_TRANSFER_LAYER", None)
    ssh_run(args.host, base, "killall scif_rma_sink scif_sink 2>/dev/null; "
                             "rm -f /tmp/bonsai.gguf; true")
    t0 = time.time()
    try:
        for p in paths:
            os.rename(p, p + ".hidden")
        ok = chat.ensure_card_files(args.host, tmp)
    finally:
        for p in paths:
            if os.path.exists(p + ".hidden"):
                os.rename(p + ".hidden", p)
    dt = time.time() - t0

    got_size = card_size(args.host, base, "/tmp/bonsai.gguf")
    got_md5 = md5_card(args.host, base, "/tmp/bonsai.gguf")
    good = bool(ok and got_size == size and got_md5 == want_md5)
    print(f"\n[fallback -> {expect}] {'PASS' if good else 'FAIL'}  {dt:.1f} s  "
          f"size={got_size} md5={got_md5}")
    if not args.keep:
        ssh_run(args.host, base, "rm -f /tmp/bonsai.gguf")
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return 0 if good else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=64, help="slice size in MiB")
    ap.add_argument("--layers", default="rma,scif,scp",
                    help="comma separated subset of rma,scif,scp")
    ap.add_argument("--gguf", default=chat.GGUF_PATH)
    ap.add_argument("--host", default=chat.SSH_HOST)
    ap.add_argument("--keep", action="store_true",
                    help="leave the slice on the card")
    ap.add_argument("--fallback", nargs="?", const="rma", default=None,
                    choices=["rma", "scif"],
                    help="instead of forcing layers, hide a layer's host sender "
                         "and check that auto mode falls through: 'rma' proves "
                         "RMA -> message SCIF, 'scif' proves both SCIF layers "
                         "fail over to scp")
    ap.add_argument("--md5", action="store_true",
                    help="exercise the BONSAI_VERIFY_MD5 switch: a clean upload "
                         "must verify OK, and a one-megabyte hole punched into "
                         "the card's copy must be detected")
    args = ap.parse_args(argv)

    if args.md5:
        return check_md5(args)
    if args.fallback:
        return check_fallback(args)

    layers = [s.strip().lower() for s in args.layers.split(",") if s.strip()]
    bad = [l for l in layers if l not in ("rma", "scif", "scp")]
    if bad:
        print(f"[fatal] unknown layer(s): {', '.join(bad)}", file=sys.stderr)
        return 2

    base = chat._ssh_base(args.host)
    tmp = os.path.join(tempfile.gettempdir(), "bonsai_layer_probe.bin")
    size = make_slice(args.gguf, args.mb, tmp)
    want_md5 = md5_local(tmp)
    print(f"slice: {size:,} bytes  md5={want_md5}\n{tmp}\n")

    failures = []
    results = []
    for layer in layers:
        print(f"=== layer {layer} " + "=" * 40, flush=True)
        ssh_run(args.host, base, "killall scif_rma_sink scif_sink 2>/dev/null; "
                                 "rm -f /tmp/bonsai.gguf; true")
        os.environ["BONSAI_TRANSFER_LAYER"] = layer
        t0 = time.time()
        try:
            ok = chat.ensure_card_files(args.host, tmp)
        except Exception as e:                       # noqa: BLE001
            print(f"[{layer}] raised {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
        dt = time.time() - t0

        got_size = card_size(args.host, base, "/tmp/bonsai.gguf")
        got_md5 = md5_card(args.host, base, "/tmp/bonsai.gguf")
        good = bool(ok and got_size == size and got_md5 == want_md5)
        results.append((layer, good, dt, got_size, got_md5))
        print(f"[{layer}] {'PASS' if good else 'FAIL'}  {dt:.1f} s  "
              f"{size / 1e6 / dt:.1f} MB/s  size={got_size} md5={got_md5}",
              flush=True)
        if not good:
            failures.append(layer)

    os.environ.pop("BONSAI_TRANSFER_LAYER", None)
    if not args.keep:
        ssh_run(args.host, base, "rm -f /tmp/bonsai.gguf")
    try:
        os.unlink(tmp)
    except OSError:
        pass

    print("\n=== summary " + "=" * 44)
    for layer, good, dt, got_size, got_md5 in results:
        print(f"  {layer:5s} {'PASS' if good else 'FAIL'}  {dt:6.1f} s  "
              f"{size / 1e6 / dt:7.1f} MB/s  size={got_size} md5={got_md5}")
    if failures:
        print(f"\n[fatal] failed layer(s): {', '.join(failures)}",
              file=sys.stderr)
        return 1
    print("\nall layers PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
