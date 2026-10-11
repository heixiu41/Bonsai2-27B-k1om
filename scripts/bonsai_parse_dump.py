"""Parse a llama-eval-callback dump into per-tensor numeric arrays.

`common_debug_print_tensor` (common/debug.cpp) prints each tensor as nested
brackets with `%12.4f` values, keeping only the first/last `n=3` entries of
every dimension. So a row is: f0, f1, f2, ..., l2, l1, l0 -- SIX numbers when
the row is longer than 6 elements, not four. Treating the tail as a contiguous
continuation of the head is the easy mistake this module exists to avoid.

Line format:
    common_debug_cb_eval: <name padded to 24> = (<type>) <OP>(<src>{dims}, ...) = {dims}
    <indented value matrix follows>
"""
from __future__ import annotations

import re
import sys

import numpy as np

HEADER = re.compile(
    r"cb_eval:\s*(?P<name>.+?)\s*=\s*\((?P<type>\w+)\)\s*(?P<op>\S+?)\("
    r"(?P<srcs>.*)\)\s*=\s*\{(?P<dims>[^}]*)\}\s*$")
NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+")


def parse_dump(path: str):
    """Yield {name, op, type, dims, rows} where rows[k] = (head, tail).

    head = up to 3 leading values, tail = up to 3 trailing values (empty when
    the row is short enough that head already covers it).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i, n = 0, len(lines)
    while i < n:
        m = HEADER.search(lines[i])
        if not m:
            i += 1
            continue
        name = m.group("name").strip()
        dims = [int(x) for x in m.group("dims").split(",") if x.strip()]
        rows = []
        j = i + 1
        while j < n and "cb_eval:" not in lines[j]:
            t = lines[j]
            if "[" in t and "]" in t:
                body = t.strip().strip(",").strip()
                if body.startswith("["):
                    body = body[1:]
                if body.endswith("]"):
                    body = body[:-1]
                parts = [p.strip() for p in body.split(",")]
                parts = [p for p in parts if p and p != "..."]
                nums = []
                for p in parts:
                    mm = NUM.fullmatch(p)
                    if mm:
                        nums.append(float(p))
                if nums:
                    head, tail = nums[:3], (nums[-3:] if len(nums) > 3 else [])
                    if len(nums) <= 3:
                        tail = []
                    rows.append((head, tail))
            j += 1
        yield {"name": name, "op": m.group("op"), "type": m.group("type"),
               "dims": dims, "rows": rows}
        i = j


def first_last(rec):
    """Flatten to (first3, last3) across the whole tensor, as printed."""
    if not rec["rows"]:
        return [], []
    head = rec["rows"][0][0]
    tail = rec["rows"][-1][1] or rec["rows"][-1][0][-3:]
    return head, tail


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    path = argv[1]
    want = argv[2:] if len(argv) > 2 else None
    seen = set()
    for rec in parse_dump(path):
        if want and not any(w in rec["name"] for w in want):
            continue
        key = (rec["name"], tuple(rec["dims"]))
        if key in seen:
            continue
        seen.add(key)
        h, t = first_last(rec)
        print(f"{rec['name']:<32} {rec['op']:<12} dims={rec['dims']} "
              f"rows={len(rec['rows'])}")
        print(f"    first3 = {np.array2string(np.array(h), precision=5)}")
        print(f"    last3  = {np.array2string(np.array(t), precision=5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

