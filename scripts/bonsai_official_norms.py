"""Recover the official per-layer residual L2 from dump_single.txt.

`l_out-N` (the block-N output residual) is followed by `norm-{N+1} =
RMS_NORM(l_out-N)`, an unweighted RMS norm.  The layer norm is linear, so for
any component j:  norm_j = l_out_j / sqrt(mean(l_out^2) + eps)  ->
    rms(l_out-N) = 1 / (norm_j / l_out_j)
Taking the median ratio over the 6 printed components recovers the official
residual RMS without any extrapolation guesswork (eps=1e-6 is negligible once
rms > 1e-2, and exact where rms >> 1e-3).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

DUMP = Path(r"D:\bonsai\dump_single.txt")
NUM = re.compile(r"[-+]?\d+\.\d+")


def block_values(lines, i):
    """Values of the first [ ... ] row following the header at lines[i]."""
    for j in range(i + 1, min(i + 12, len(lines))):
        if "[" in lines[j] and NUM.search(lines[j]):
            return [float(x) for x in NUM.findall(lines[j])]
    return []


def main():
    lines = DUMP.read_text(errors="replace").splitlines()
    hdr_l = {}
    hdr_n = {}
    for i, ln in enumerate(lines):
        m = re.search(r"\s(l_out-(\d+)) = \(f32\)", ln)
        if m:
            hdr_l[int(m.group(2))] = i
        m = re.search(r"\s(norm(?:-(\d+))?) = \(f32\)\s+RMS_NORM\(l_out-(\d+)", ln)
        if m:
            hdr_n[int(m.group(3))] = i
        m = re.search(r"\snorm-(\d+) = \(f32\)\s+RMS_NORM\(l_out-(\d+)", ln)
        if m:
            hdr_n[int(m.group(2))] = i
    print(f"{'layer':>5} {'rms(l_out)':>11} {'l2(l_out)':>11} {'factor':>8} "
          f"{'l_out first3':>28} {'sum':>12}")
    rows = []
    for n in sorted(hdr_l):
        lv = block_values(lines, hdr_l[n])
        ni = hdr_n.get(n)
        if ni is None or not lv:
            continue
        nv = block_values(lines, ni)
        k = min(len(lv), len(nv))
        if k < 2:
            continue
        ratios = [nv[j] / lv[j] for j in range(k) if abs(lv[j]) > 1e-4]
        if not ratios:
            continue
        fac = float(np.median(ratios))
        rms = 1.0 / fac
        # the plain sum of l_out-N is printed right after its values
        sm = None
        for j in range(hdr_l[n], min(hdr_l[n] + 14, len(lines))):
            if "sum =" in lines[j]:
                sm = float(lines[j].split("=")[1])
                break
        rows.append((n, rms, rms * np.sqrt(5120), fac, lv[:3], sm))
    for n, rms, l2, fac, f3, sm in rows:
        print(f"{n:5d} {rms:11.6f} {l2:11.4f} {fac:8.5f} "
              f"{str([round(x,4) for x in f3]):>28} {sm:12.5f}")
    print("\nfirst/last layer l2:", rows[0][2], rows[-1][2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
