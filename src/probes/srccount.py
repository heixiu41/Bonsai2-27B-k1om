#!/usr/bin/env python3
"""srccount.py -- instructions emitted per SOURCE LINE, from an icc -mmic -S file.

icc annotates each instruction with '#<srcline>.<col>'.  Grouping by that is
much more robust than guessing basic blocks, and it is exactly the evidence
needed for TRAP 2 ("count the instructions actually emitted per 128 weights").

Usage:
  python srccount.py listing.s [srcline ...]
    listing.s   an icc -S output file
    srcline...  optional source line numbers to limit the report to

Output: one row per source line that produced instructions, with the total and
the mnemonic histogram.  Lines with no source annotation are reported as '?'.
"""
import re
import sys
from collections import defaultdict, Counter

INSTR = re.compile(r"^[ \t]+([a-z][a-z0-9._]*)\b[^#]*#\s*(?:\[spill\])?\s*(\d+)\.(\d+)")
INSTR_NOSRC = re.compile(r"^[ \t]+([a-z][a-z0-9._]*)\b")
SRCLINE = re.compile(r"^\s*#\s*---\s*(.*)$")


def main():
    path = sys.argv[1]
    want = set(int(x) for x in sys.argv[2:])
    per = defaultdict(Counter)
    for raw in open(path, errors="replace"):
        line = raw.rstrip("\n")
        m = INSTR.match(line)
        if m:
            per[int(m.group(2))][m.group(1)] += 1
            continue
        m2 = INSTR_NOSRC.match(line)
        if m2 and not line.lstrip().startswith((".", "#")):
            per[-1][m2.group(1)] += 1
    print(f"# {path}")
    print(f"{'srcline':>8} {'total':>6}  mnemonics")
    for ln in sorted(per):
        if want and ln not in want:
            continue
        h = per[ln]
        if not h:
            continue
        print(f"{ln:>8} {sum(h.values()):>6}  " +
              ", ".join(f"{k}x{v}" for k, v in h.most_common(30)))


if __name__ == "__main__":
    main()
