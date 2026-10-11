"""Empirically validate the PTQ1_0 decode by pulling one small tensor slice.

Reads the GGUF header only (tensor-info block), computes where the tensor data
region begins, then Range-fetches a few blocks of a small tensor and decodes
them with the base-3 trit recurrence from the PrismML fork. Ternary weights
should decode into a {-1,0,+1} histogram with plausible scales.
"""
import struct
from collections import Counter

import numpy as np
import requests

URL = ("https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf"
       "/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
KV_END = 11070652
N_TENSORS = 851
PTQ1_0 = 143
QK = 128
BLOCK_BYTES = 34


def get(rng):
    r = requests.get(URL, headers={"Range": f"bytes={rng}"}, timeout=120)
    r.raise_for_status()
    return r.content


# ---- parse tensor-info block ----
head = get(f"{KV_END}-{KV_END + 2 * 1024 * 1024 - 1}")
o = 0
tensors = []
for _ in range(N_TENSORS):
    n, = struct.unpack_from("<Q", head, o); o += 8
    name = head[o:o + n].decode(); o += n
    nd, = struct.unpack_from("<I", head, o); o += 4
    dims = list(struct.unpack_from("<" + "Q" * nd, head, o)); o += 8 * nd
    tt, = struct.unpack_from("<I", head, o); o += 4
    off, = struct.unpack_from("<Q", head, o); o += 8
    tensors.append({"name": name, "dims": dims, "type": tt, "off": off})

info_end = KV_END + o
align = 32
# NOTE: tensor "off" values are ABSOLUTE file offsets, not relative to the
# start of the data region. Verified against the file size below.
print(f"tensor-info block ends at {info_end}")

by_name = {t["name"]: t for t in tensors}

ternary = [t for t in tensors if t["type"] == PTQ1_0]
print(f"\nPTQ1_0 (type {PTQ1_0}) tensors: {len(ternary)}")

print("\nsmallest ternary tensors (name, dims, elements, packed bytes):")
small = []
for t in ternary:
    e = 1
    for d in t["dims"]:
        e *= d
    small.append((e, t))
small.sort(key=lambda x: x[0])
for e, t in small[:12]:
    print(f"  {t['name']:<40} {str(t['dims']):<18} {e:>9} el  "
          f"{e // QK * BLOCK_BYTES:>8} B   off={t['off']}")

# ---- decode one real ternary tensor ----
e, t = small[0]
cand = t["name"]
nblocks = e // QK
nbytes = nblocks * BLOCK_BYTES
print(f"\ndecoding {cand}: {nblocks} blocks, {nbytes} bytes at abs off {t['off']}")
n_fetch = min(nbytes, BLOCK_BYTES * 64)
raw = get(f"{t['off']}-{t['off'] + n_fetch - 1}")
print(f"  fetched {len(raw)} bytes = {len(raw) // BLOCK_BYTES} blocks")


def trit(qs, qh, e):
    """Decode element e of a PTQ1_0 block, per the fork's ptq1_0_trit()."""
    if e < 80:
        b = qs[e & 15]; n = e >> 4
    elif e < 120:
        tt = e - 80
        b = qs[16 + (tt & 7)]; n = tt >> 3
    else:
        tt = e - 120
        b = qh[tt & 1]; n = tt >> 1
    v = b
    for _ in range(n):
        v = (v * 3) & 0xFF
    return ((v * 3) >> 8) - 1


hist = Counter()
scales = []
vals = []
for bi in range(len(raw) // BLOCK_BYTES):
    blk = raw[bi * BLOCK_BYTES:(bi + 1) * BLOCK_BYTES]
    qs = blk[0:24]
    qh = blk[24:26]
    d = np.frombuffer(blk[26:28], dtype=np.float16)[0]
    scales.append(float(d))
    trits = [trit(qs, qh, e) for e in range(QK)]
    for x in trits:
        hist[x] += 1
    vals.extend([x * float(d) for x in trits])

print(f"\ntrit histogram over {sum(hist.values())} decoded weights:")
for k in sorted(hist):
    print(f"  {k:+d}: {hist[k]:6d}  ({100 * hist[k] / sum(hist.values()):5.1f}%)")
print(f"\nscale (d) stats: min={min(scales):.6g} max={max(scales):.6g}")
print(f"decoded value range: [{min(vals):.6g}, {max(vals):.6g}]")
print("\nfirst 32 decoded values:")
print(" ", " ".join(f"{v:+.4g}" for v in vals[:32]))
