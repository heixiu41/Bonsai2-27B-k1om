#!/usr/bin/env python3
"""asmcount.py -- per-function instruction histograms from an icc -mmic -S listing.

Usage: python asmcount.py file.s [name ...]

Handles icc's internal labels (..B*, ..L*) without treating them as new
functions.  For each function it prints the whole-function mnemonic histogram
and the histogram of the largest basic-block loop body (the region between the
target label of a backward branch and that branch).
"""
import re
import sys
from collections import Counter

LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):")
MNEM = re.compile(r"^\s+([a-z][a-z0-9._]*)\b")
BRANCH = re.compile(r"^\s+(j\w+)\s+([.\w$]+)")


def strip_comment(line):
    i = line.find("#")
    return line[:i] if i >= 0 else line


def parse(path):
    """-> (funcs: list of [name, [ (lineno, mnem, text) ]], label_lineno: dict)"""
    funcs = []
    labels = {}
    cur = None
    for ln, raw in enumerate(open(path, errors="replace"), 1):
        line = strip_comment(raw.rstrip("\n"))
        if not line.strip():
            continue
        if line[0] in " \t":
            if cur is not None:
                mm = MNEM.match(line)
                if mm:
                    cur[1].append((ln, mm.group(1), line.strip()))
            continue
        m = LABEL.match(line)
        if m:
            name = m.group(1)
            labels[name] = ln
            if name.startswith("..") or name.startswith(".L"):
                continue                      # icc internal label
            cur = [name, []]
            funcs.append(cur)
    return funcs, labels


def main():
    path = sys.argv[1]
    want = sys.argv[2:]
    funcs, labels = parse(path)
    out = []
    for name, instrs in funcs:
        if not instrs:
            continue
        if want and name not in want:
            continue
        best = None
        for i, (ln, mn, txt) in enumerate(instrs):
            bm = BRANCH.match(txt)
            if not bm:
                continue
            tl = labels.get(bm.group(2))
            if tl is None or tl > ln:
                continue
            body = [x for x in instrs if tl <= x[0] <= ln]
            if best is None or len(body) > len(best[1]):
                best = (bm.group(2), body)
        out.append("=" * 76)
        out.append(f"FUNCTION {name}: {len(instrs)} instructions")
        h = Counter(m for _, m, _ in instrs)
        out.append("  whole: " + ", ".join(f"{m}x{c}" for m, c in h.most_common(16)))
        if best:
            hb = Counter(m for _, m, _ in best[1])
            out.append(f"  largest loop @{best[0]}: {len(best[1])} instructions")
            out.append("   " + ", ".join(f"{m}x{c}" for m, c in hb.most_common(24)))
    text = "\n".join(out) + "\n"
    sys.stderr.write(text)
    with open(path + ".counts.txt", "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
