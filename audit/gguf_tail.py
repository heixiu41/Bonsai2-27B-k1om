#!/usr/bin/env python3
"""Parse the Bonsai GGUF header/tensor table only and report the on-disk tail
layout, so we can tell which tensor owns the final bytes of the file."""
import struct, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf"
f = open(path, "rb")
file_size = f.seek(0, 2)
f.seek(0)

def u32():
    return struct.unpack("<I", f.read(4))[0]
def u64():
    return struct.unpack("<Q", f.read(8))[0]
def s():
    n = u64()
    b = f.read(n)
    return b.decode("utf-8", "replace")
def val(t):
    if t in (0, 1, 7): return struct.unpack("<B", f.read(1))[0]
    if t in (2, 3):    return struct.unpack("<H", f.read(2))[0]
    if t in (4, 5, 6): return struct.unpack("<I", f.read(4))[0]
    if t in (10, 11, 12): return struct.unpack("<Q", f.read(8))[0]
    if t == 8:  return s()
    if t == 9:
        et = u32(); n = u64()
        return ("ARRAY", et, n)
    raise ValueError(t)

magic = f.read(4)
assert magic == b"GGUF", magic
version = u32(); n_tensors = u64(); n_kv = u64()
print("version", version, "n_tensors", n_tensors, "n_kv", n_kv)

kv = {}
for i in range(n_kv):
    k = s(); t = u32(); v = val(t)
    if t == 9:
        et, n = v[1], v[2]
        esz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(et)
        if esz:
            data = f.read(esz*n)
            if k == "prism.hadamard.sign_values":
                v = struct.unpack("<%di" % n, data)
        else:
            for j in range(n): s()
            v = ("ARRAY_STR", et, n)
    kv[k] = v

TSIZE = {0:4, 1:2, 30:2}
tensors = []
for i in range(n_tensors):
    name = s()
    nd = u32()
    dims = [u64() for _ in range(nd)]
    typ = u32()
    off = u64()
    ne = 1
    for d in dims: ne *= d
    if typ in (142, 143):  # PQ2_0 / PTQ1_0
        nb = ne // 128 * 34 if ne % 128 == 0 else 0
    elif typ in TSIZE:
        nb = ne * TSIZE[typ]
    else:
        nb = 0
    tensors.append(dict(name=name, dims=dims, type=typ, off=off, nb=nb, ne=ne))

info_end = f.tell()
align = kv.get("general.alignment", 32)
if isinstance(align, tuple): align = align[0]
data_offset = (info_end + align - 1)//align*align
print("info_end", info_end, "alignment", align, "data_offset", data_offset)
print("file_size", file_size, "mod4096", file_size % 4096)

highest = max(data_offset + t["off"] + t["nb"] for t in tensors)
print("highest tensor end", highest, "slack vs EOF", file_size - highest)

tensors.sort(key=lambda t: t["off"])
print("\n--- last 8 tensors in file order ---")
for t in tensors[-8:]:
    print("  %-42s type=%-3d dims=%-22s off=%-12d nb=%-12d end=%d" %
          (t["name"], t["type"], t["dims"], t["off"], t["nb"],
           data_offset + t["off"] + t["nb"]))

# which tensors are consumed by the IMCI (over-reading) path?
def overread(t):
    # mm_folded on a PQ2_0 matrix: per row the last block's qs=blk+2 needs
    # readable bytes up to qs+16+127 -> blk+144, block is 34 -> 110 past the block
    return t["type"] == 142 and t["nb"] > 0

tail = [t for t in tensors if overread(t)]
last = max(tail, key=lambda t: t["off"])
print("\nlast IMCI-decoded PQ2_0 tensor by offset: %s off=%d nb=%d end=%d" %
      (last["name"], last["off"], last["nb"], data_offset + last["off"] + last["nb"]))
print("  bytes of unowned slack needed past its end: 110")
print("  slack actually in the malloc'd buffer (page-aligned end): %d" %
      (4096 - (file_size % 4096)))
f.close()
