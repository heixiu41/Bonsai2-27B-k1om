"""Fetch only the GGUF header of Ternary-Bonsai-2-27B and dump its architecture.

Reads just the first few MB with an HTTP Range request so we never pull the
whole 5.5 GB file, then parses the GGUF metadata key/value block (with the
tensor-info block when it fits) to learn the exact architecture and tensor
inventory needed to write a k1om runtime for it.
"""
import json
import struct
import sys

import requests

URL = ("https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf"
       "/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
HEAD_BYTES = 140 * 1024 * 1024

# GGUF metadata value type ids
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12
# 13-15 are unused; 16+ are the newer float formats used by some exporters.
T_FLOAT16, T_FLOAT8E4M3FN, T_FLOAT8E4M3FNUZ = 16, 17, 18
T_FLOAT8E5M2, T_FLOAT8E5M2FNUZ = 19, 20
TYPE_NAMES = {
    T_UINT8: "u8", T_INT8: "i8", T_UINT16: "u16", T_INT16: "i16",
    T_UINT32: "u32", T_INT32: "i32", T_FLOAT32: "f32", T_BOOL: "bool",
    T_STRING: "str", T_ARRAY: "arr", T_UINT64: "u64", T_INT64: "i64",
    T_FLOAT64: "f64", T_FLOAT16: "f16", T_FLOAT8E4M3FN: "f8e4m3",
    T_FLOAT8E4M3FNUZ: "f8e4m3uz", T_FLOAT8E5M2: "f8e5m2",
    T_FLOAT8E5M2FNUZ: "f8e5m2uz",
}

FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
    T_FLOAT16: "<e",
}
SIZE = {T_UINT8: 1, T_INT8: 1, T_UINT16: 2, T_INT16: 2, T_UINT32: 4,
        T_INT32: 4, T_FLOAT32: 4, T_BOOL: 1, T_UINT64: 8, T_INT64: 8,
        T_FLOAT64: 8, T_FLOAT16: 2, T_FLOAT8E4M3FN: 1, T_FLOAT8E4M3FNUZ: 1,
        T_FLOAT8E5M2: 1, T_FLOAT8E5M2FNUZ: 1}

# GGML tensor types we care to name
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16",
}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def raw(self, n):
        if self.o + n > len(self.b):
            raise EOFError(f"need {n} bytes at {self.o}, have {len(self.b)}")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def scalar(self, t):
        (v,) = struct.unpack(FMT[t], self.raw(SIZE[t]))
        return v

    def string(self):
        (n,) = struct.unpack("<Q", self.raw(8))
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            (et,) = struct.unpack("<I", self.raw(4))
            (n,) = struct.unpack("<Q", self.raw(8))
            # Only materialise short arrays; skip long ones cheaply.
            if et == T_STRING and n > 64:
                return f"<{n} strings>"
            if n > 4096:
                return f"<array len={n} type={et}>"
            return [self.value(et) for _ in range(n)]
        if t in FMT:
            return self.scalar(t)
        raise ValueError(f"unknown gguf type {t}")


def main():
    r = requests.get(URL, headers={"Range": f"bytes=0-{HEAD_BYTES - 1}"},
                     timeout=120)
    r.raise_for_status()
    buf = r.content
    print(f"fetched {len(buf)} bytes (status {r.status_code})")

    rd = Reader(buf)
    magic = rd.raw(4)
    print("magic   :", magic)
    (version,) = struct.unpack("<I", rd.raw(4))
    (n_tensors,) = struct.unpack("<Q", rd.raw(8))
    (n_kv,) = struct.unpack("<Q", rd.raw(8))
    print(f"version : {version}")
    print(f"tensors : {n_tensors}")
    print(f"kv pairs: {n_kv}")

    meta = {}
    for i in range(n_kv):
        key = rd.string()
        (t,) = struct.unpack("<I", rd.raw(4))
        try:
            meta[key] = rd.value(t)
        except EOFError:
            print(f"(metadata truncated at kv #{i} key={key!r} type={t})")
            break
        except ValueError as exc:
            print(f"(parse error at kv #{i} key={key!r} type={t}: {exc});"
                  f" offset now {rd.o}")
            break

    print("\n===== METADATA =====")
    for k, v in meta.items():
        s = str(v)
        if len(s) > 160:
            s = s[:160] + "..."
        print(f"  {k} = {s}")

    print("\n===== TENSORS =====")
    tensors = []
    try:
        for i in range(n_tensors):
            name = rd.string()
            (n_dims,) = struct.unpack("<I", rd.raw(4))
            dims = [struct.unpack("<Q", rd.raw(8))[0] for _ in range(n_dims)]
            (tt,) = struct.unpack("<I", rd.raw(4))
            (off,) = struct.unpack("<Q", rd.raw(8))
            tensors.append((name, dims, tt, off))
    except EOFError:
        print(f"(tensor block truncated after {len(tensors)}/{n_tensors})")

    from collections import Counter
    print(f"parsed {len(tensors)} tensor infos")
    print("type histogram:",
          {GGML_TYPES.get(t, f"type{t}"): c
           for t, c in Counter(x[2] for x in tensors).most_common()})
    for name, dims, tt, off in tensors[:80]:
        print(f"  {name:<58} {str(dims):<22} {GGML_TYPES.get(tt, tt)}")
    if len(tensors) > 80:
        print(f"  ... and {len(tensors) - 80} more")

    with open("_bonsai_gguf_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "tensors": [
            {"name": n, "dims": d, "type": t, "offset": o}
            for n, d, t, o in tensors]}, fh, indent=1)
    print("\nwrote _bonsai_gguf_meta.json")


if __name__ == "__main__":
    sys.exit(main())


