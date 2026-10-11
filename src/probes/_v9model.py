#!/usr/bin/env python3
"""Host-side semantic model of the V9 spec, using ONLY the five measured card
premises (pq2sem.c): permutevar_epi32(idx,data)->data[idx&15]; set_epi32 last
arg is lane 0; srlv/sllv is per-lane; LUT lane j = (j&3)-1, so the LUT output
for an index dword x is (x & 3) - 1; mullo(k,0x01010101) replicates the low byte.

Emulates the V9 instruction sequence exactly as specified:
    pre = slli_epi32(word, sh_l)        (word = lane l's own dword)
    t   = mullo_epi32(pre, 0x01010101)
    g   = permutevar_epi32(t, lut)      -> (t & 3) - 1
and compares the 2-bit code that reaches the LUT against the reference codec.
"""
import random

M32 = 0xFFFFFFFF
MULC = 0x01010101


def ref_code(qs, w):
    return (qs[w >> 2] >> (2 * (w & 3))) & 3


def v9_code(qs, m, lane, half, sh):
    """The V9 spec as the hardware actually executes it (lane-local)."""
    base = 0 if half == 0 else 16
    word = qs[base + lane]                     # zero-extended byte per dword
    pre = (word << sh) & M32
    t = (pre * MULC) & M32
    return t & 3                               # LUT index low bits


# ---- shift-vector interpretations -----------------------------------------
def sh_doc_vectors(m, lane):
    """The four vectors as printed in the doc/task, read lane-ordered:
       m=0 -> 0,8,16,24  m=1 -> 24,0,8,16 ... == 8*((l&3)-m mod 4)"""
    return 8 * (((lane % 4) - m) % 4)


def sh_doc_prose(m, lane):
    """The doc's derivation sentence: 8*((m - (l&3)) mod 4)."""
    return 8 * ((m - (lane % 4)) % 4)


def sh_task_formula(m, lane):
    """The task's restated formula: 8*((4m + l/4 - (l&3)) mod 4)."""
    return 8 * ((4 * m + lane // 4 - (lane % 4)) % 4)


def sh_uniform(m, lane):
    """A uniform per-group shift 8*m (one scalar shift for the whole vector)."""
    return 8 * m


INTERPS = [("doc vectors 8*((l&3)-m)", sh_doc_vectors),
           ("doc prose   8*((m-(l&3)))", sh_doc_prose),
           ("task formula 8*((4m+l/4-l&3))", sh_task_formula),
           ("uniform     8*m", sh_uniform)]


def main():
    random.seed(7)
    print("=== V9 (spec) vs reference, per output lane, 4 random blocks ===")
    for name, fn in INTERPS:
        bad = 0
        first = None
        for _ in range(4):
            qs = [random.randrange(256) for _ in range(32)]
            for half in (0, 1):
                for m in range(4):
                    for lane in range(16):
                        want = ref_code(qs, 64 * half + 16 * m + lane)
                        got = v9_code(qs, m, lane, half, fn(m, lane))
                        if got != want:
                            bad += 1
                            if first is None:
                                first = (half, m, lane, want, got)
        print("%-32s mismatches = %4d / 1024   first=%s" % (name, bad, first))

    print()
    print("=== mechanism: pre = byte<<8s, t = pre*0x01010101 (mod 2^32) ===")
    for s in range(4):
        row = []
        for w in (0x00, 0x01, 0xE4, 0xFF):
            pre = (w << (8 * s)) & M32
            t = (pre * MULC) & M32
            row.append("b=%02X pre=%08X t=%08X idx=%d" % (w, pre, t, t & 3))
        print("  s=%d  " % s + " | ".join(row))
    print("  -> for any nonzero byte shift, byte 0 of the product is 0,")
    print("     so the LUT index is forced to 0 and the weight to -1.")

    print()
    print("=== what a lane-local shift+multiply CAN produce ===")
    qs = [random.randrange(256) for _ in range(32)]
    print("  lane  qs[lane]  field f  code  = code(4*lane+f)   V9 lane coverage")
    for lane in range(4):
        for f in range(4):
            c = (qs[lane] >> (2 * f)) & 3
            print("   %2d    0x%02X       %d      %d      weight %2d        "
                  "(g[%d] lane %d wants weight %d)"
                  % (lane, qs[lane], f, c, 4 * lane + f, lane // 4, lane,
                     16 * (lane // 4) + lane))

    print()
    print("=== exhaustive per-lane shift search: does ANY shift work? ===")
    # for each (half,m,lane) test all 4 byte shifts against 200 random blocks
    fails = []
    for half in (0, 1):
        for m in range(4):
            for lane in range(16):
                ok_any = False
                for s in range(4):
                    good = True
                    for _ in range(200):
                        blk = [random.randrange(256) for _ in range(32)]
                        if v9_code(blk, m, lane, half, 8 * s) != \
                           ref_code(blk, 64 * half + 16 * m + lane):
                            good = False
                            break
                    if good:
                        ok_any = True
                if not ok_any:
                    fails.append((half, m, lane))
    print("  lanes where NO shift works: %d / 128" % len(fails))
    own = [(h, m, l) for (h, m, l) in fails if (0 if h == 0 else 16) + l == 4 * m + l // 4]
    print("  of those, lanes whose own byte IS the wanted byte: %d (sanity)" % len(own))
    print("  example failures:", fails[:6])


if __name__ == "__main__":
    main()
