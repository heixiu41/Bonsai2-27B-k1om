#!/usr/bin/env python3
"""Emit deliberately malformed / truncated GGUF v3 files used to probe the
loader's error handling.  Only the header + KV + tensor table are written; the
data region is optional."""
import struct, os, sys

OUT = os.path.dirname(os.path.abspath(__file__))
ALIGN = 32

def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

def kv_u32(key, v):  return s(key) + struct.pack("<II", 4, v)
def kv_str(key, v):  return s(key) + struct.pack("<I", 8) + s(v)
def kv_arr_u32(key, vals):
    return (s(key) + struct.pack("<IIQ", 9, 4, len(vals)) +
            b"".join(struct.pack("<I", v) for v in vals))

def tensor(name, dims, typ, off):
    return (s(name) + struct.pack("<I", len(dims)) +
            b"".join(struct.pack("<Q", d) for d in dims) +
            struct.pack("<IQ", typ, off))

def header(n_tensors, n_kv):
    return b"GGUF" + struct.pack("<IQQ", 3, n_tensors, n_kv)

def finish(kvs, tensors, data=b"", align=ALIGN):
    body = header(len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)
    info_end = len(body)
    data_off = (info_end + align - 1) // align * align
    return body + b"\0" * (data_off - info_end) + data, data_off

def w(name, blob):
    p = os.path.join(OUT, name)
    with open(p, "wb") as f:
        f.write(blob)
    print("%-34s %8d bytes" % (name, len(blob)))

# ---- 1. a minimal valid file: one small F32 tensor, data present ----------
kvs = [kv_u32("general.alignment", ALIGN)]
ts = [tensor("t0", [4], 0, 0)]
blob, doff = finish(kvs, ts, b"\0" * 16)
w("min_valid.gguf", blob)

# ---- 2. truncated in the middle of a scalar KV VALUE ---------------------
# key + type are intact, the 4-byte value is cut off: gguf.c's scalar cases
# never call bg_need, they just dereference c.p and advance it.
body = header(1, 1) + s("some.u32") + struct.pack("<I", 4)
w("trunc_kv_scalar.gguf", body)

# ---- 3. truncated inside an ARRAY: values are read before bg_need --------
arr = s("arr.u32") + struct.pack("<IIQ", 9, 4, 8) + struct.pack("<IIII", 1, 2, 3, 4)
body = header(1, 1) + arr          # header claims 8 elements, only 4 present
w("trunc_array.gguf", body)

# ---- 4. prism.hadamard.sign_values truncated ----------------------------
big = (s("prism.hadamard.sign_values") + struct.pack("<IIQ", 9, 4, 4096) +
       b"\x01\x00\x00\x00" * 8)     # header claims 4096, 8 present
body = header(1, 1) + big
w("trunc_signvalues.gguf", body)

# ---- 5. tensor table ends at EOF: data_offset points past the buffer -----
kvs = [kv_u32("general.alignment", ALIGN)]
ts = [tensor("t0", [4], 0, 0)]     # claims 16 bytes at offset 0 of the data area
body, doff = finish(kvs, ts, b"")  # ... but there is no data area at all
w("no_data.gguf", body)

# ---- 6. tensor offset points past EOF -----------------------------------
kvs = [kv_u32("general.alignment", ALIGN)]
ts = [tensor("t0", [4], 0, 1 << 20)]
blob, doff = finish(kvs, ts, b"\0" * 16)
w("bad_offset.gguf", blob)

# ---- 7. tensor table itself truncated ----------------------------------
body = header(3, 1) + kv_u32("general.alignment", ALIGN) + tensor("t0", [4], 0, 0)
w("trunc_tensors.gguf", body)
