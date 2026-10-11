"""Ternary Bonsai 2 27B — GGUF loader + PQ2_0 dequantizer (reference implementation).

Stage-1 groundwork for the k1om port. This module is correctness-first and
deliberately independent of llama.cpp: everything it does is checked against the
GGUF header alone, so it can run without the 7.2 GB of weights present.

Verified facts it encodes (see docs/BONSAI2_FORWARD_SPEC.md):
  * PQ2_0 = block_pq2_0 { fp16 d; uint8_t qs[32]; } -> 34 B per 128 weights
  * code -> value : 0 -> -1, 1 -> 0, 2 -> +1, 3 -> +2  (i.e. (q - 1) * d)
  * tensor `offset` in the info block is RELATIVE TO THE START OF THE TENSOR
    DATA REGION, which begins at align32(end of the tensor-info block).
    Reading them as absolute file offsets lands inside the wrong tensor and
    silently yields garbage (e.g. F32 norms decoding as 1e+25).
  * Hadamard: 401 folded weights, sign widths [5120, 6144, 17408],
    transform is  x' = H . (s * x)  with H a 1024-block normalized WHT

Usage:
    python scripts/bonsai_ref.py verify          # header/size self-consistency
    python scripts/bonsai_ref.py hadamard        # dump the Hadamard plan
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

import numpy as np
import requests

REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MIRROR = "https://hf-mirror.com"          # huggingface.co is unreachable here
PTQ1_FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
PQ2_FILE = "Ternary-Bonsai-2-27B-PQ2_0.gguf"

# ---- ggml tensor types we handle -------------------------------------------
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30
GGML_PQ2_0, GGML_PTQ1_0 = 142, 143
TYPE_NAMES = {GGML_F32: "F32", GGML_F16: "F16", GGML_BF16: "BF16",
              GGML_PQ2_0: "PQ2_0", GGML_PTQ1_0: "PTQ1_0"}

# bytes per element (0 => block-quantised, use block size)
TYPE_BYTES = {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2}
BLOCK_ELEMS = {GGML_PQ2_0: 128, GGML_PTQ1_0: 128}
BLOCK_BYTES = {GGML_PQ2_0: 34, GGML_PTQ1_0: 34}

# ---- GGUF metadata value types ---------------------------------------------
_KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
            10: 8, 11: 8, 12: 8, 16: 2, 17: 1, 18: 1, 19: 1, 20: 1}
_KV_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d", 16: "<e"}
_T_UINT32, _T_FLOAT32, _T_BOOL = 4, 6, 7
_T_STRING, _T_ARRAY, _T_INT32 = 8, 9, 5


class _Reader:
    def __init__(self, buf: bytes, off: int = 0):
        self.b, self.o = buf, off

    def raw(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise EOFError(f"need {n} at {self.o}, have {len(self.b)}")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def string(self) -> str:
        (n,) = struct.unpack("<Q", self.raw(8))
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t: int):
        if t == _T_STRING:
            return self.string()
        if t == _T_ARRAY:
            (et,) = struct.unpack("<I", self.raw(4))
            (n,) = struct.unpack("<Q", self.raw(8))
            if et == _T_STRING:
                return [self.string() for _ in range(n)]
            fmt = _KV_FMT.get(et)
            if fmt is None or et in (17, 18, 19, 20):
                # float8 arrays: skip, we never consume them
                self.o += _KV_SIZE[et] * n
                return None
            return list(struct.unpack("<" + fmt[1] * n, self.raw(_KV_SIZE[et] * n)))
        if t in _KV_FMT:
            return struct.unpack(_KV_FMT[t], self.raw(_KV_SIZE[t]))[0]
        raise ValueError(f"unknown GGUF kv type {t}")


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    type: int
    offset: int          # relative to the tensor-data region (see Gguf.data_base)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        if self.type in TYPE_BYTES:
            return self.n_elements * TYPE_BYTES[self.type]
        blk = BLOCK_ELEMS[self.type]
        assert self.n_elements % blk == 0, self.name
        return self.n_elements // blk * BLOCK_BYTES[self.type]

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, f"type{self.type}")


@dataclass
class Gguf:
    path: str
    url: str
    version: int
    meta: dict
    tensors: "OrderedDict[str, TensorInfo]"
    info_end: int
    file_size: int
    _http: requests.Session = field(default_factory=requests.Session, repr=False)

    @property
    def data_base(self) -> int:
        """Absolute file offset where the tensor-data region starts.

        Tensor `offset` values are relative to this. llama.cpp aligns the region
        to the context's alignment (32 B in practice); the alignment is confirmed
        by the byte sums closing exactly against the file size.
        """
        return (self.info_end + 31) // 32 * 32

    # -- construction --------------------------------------------------------
    @classmethod
    def open(cls, filename: str = PQ2_FILE, head_span: int = 16 << 20,
             cache: bool = True, local: str | None = None):
        """Parse the header.

        If `local` names an existing .gguf it is used directly (no network);
        otherwise the header is fetched over HTTP and cached on disk, because
        a model file's metadata never changes and the tokenizer block alone is
        ~11 MB, which is wasteful to refetch on every run.
        """
        url = f"{MIRROR}/{REPO}/resolve/main/{filename}"
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"_bonsai_head_{filename.replace('.gguf', '')}.bin")
        file_size = -1
        head = None

        if local:
            src = LocalSource(local)
            file_size = src.size
            if file_size < head_span:
                head = src._read(0, file_size)
            else:
                head = src._read(0, head_span)
        else:
            if cache and os.path.exists(cache_path) and \
                    os.path.getsize(cache_path) >= head_span // 2:
                with open(cache_path, "rb") as fh:
                    head = fh.read()
                sidecar = cache_path + ".size"
                if os.path.exists(sidecar):
                    file_size = int(open(sidecar).read().strip())

            src = requests.Session()
            if head is None:
                r = src.get(url, headers={"Range": f"bytes=0-{head_span - 1}"},
                            timeout=300)
                r.raise_for_status()
                head = r.content
                total = r.headers.get("Content-Range", "").split("/")[-1]
                file_size = int(total) if total.isdigit() else -1
                if cache:
                    with open(cache_path, "wb") as fh:
                        fh.write(head)
                    with open(cache_path + ".size", "w") as fh:
                        fh.write(str(file_size))

        if head[:4] != b"GGUF":
            raise ValueError("not a GGUF file")

        rd = _Reader(head, 4)
        (version,) = struct.unpack("<I", rd.raw(4))
        (n_tensors,) = struct.unpack("<Q", rd.raw(8))
        (n_kv,) = struct.unpack("<Q", rd.raw(8))

        meta = {}
        for _ in range(n_kv):
            key = rd.string()
            (t,) = struct.unpack("<I", rd.raw(4))
            meta[key] = rd.value(t)

        tensors: "OrderedDict[str, TensorInfo]" = OrderedDict()
        for _ in range(n_tensors):
            name = rd.string()
            (nd,) = struct.unpack("<I", rd.raw(4))
            dims = list(struct.unpack("<" + "Q" * nd, rd.raw(8 * nd)))
            (tt,) = struct.unpack("<I", rd.raw(4))
            (off,) = struct.unpack("<Q", rd.raw(8))
            tensors[name] = TensorInfo(name, dims, tt, off)

        return cls(filename, url, version, meta, tensors, rd.o, file_size, src)

    # -- tensor data ---------------------------------------------------------
    def read(self, name: str) -> bytes:
        t = self.tensors[name]
        start = self.data_base + t.offset
        r = self._http.get(self.url, headers={
            "Range": f"bytes={start}-{start + t.nbytes - 1}"}, timeout=600)
        r.raise_for_status()
        if len(r.content) != t.nbytes:
            raise IOError(f"{name}: got {len(r.content)} want {t.nbytes}")
        return r.content

    def dequant(self, name: str) -> np.ndarray:
        """Return the tensor as fp32, shape = reversed(dims) (ggml is col-major)."""
        t = self.tensors[name]
        raw = self.read(name)
        shape = tuple(reversed(t.dims))
        if t.type == GGML_F32:
            return np.frombuffer(raw, dtype="<f4").reshape(shape)
        if t.type == GGML_F16:
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
        if t.type == GGML_BF16:
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if t.type == GGML_PQ2_0:
            return _dequant_pq2_0(raw, shape)
        if t.type == GGML_PTQ1_0:
            return _dequant_ptq1_0(raw, shape)
        raise NotImplementedError(t.type_name)

    # -- derived plans -------------------------------------------------------
    def hadamard_plan(self) -> dict:
        """Build the activation-transform table described in the forward spec."""
        if "prism.hadamard.weight_names" not in self.meta:
            return {"enabled": False}
        widths = list(self.meta["prism.hadamard.sign_widths"])
        vals = list(self.meta["prism.hadamard.sign_values"])
        signs, off = {}, 0
        for w in widths:
            signs[w] = np.array(vals[off:off + w], dtype=np.float32)
            off += w
        assert off == len(vals), "sign_values length mismatch"

        fwd = {}
        for name in self.meta["prism.hadamard.weight_names"]:
            t = self.tensors[name]
            in_dim = t.dims[0]                      # ggml ne[0] = input dim
            assert in_dim in signs, f"no sign vector of width {in_dim} for {name}"
            fwd[name] = in_dim
        inv = list(self.meta.get("prism.hadamard.inverse_weight_names", []))
        return {"enabled": True,
                "block_size": int(self.meta["prism.hadamard.block_size"]),
                "signs": signs,
                "forward": fwd,
                "inverse": inv}


# ---------------------------------------------------------------------------
# PQ2_0 / PTQ1_0 codecs
# ---------------------------------------------------------------------------
def _dequant_pq2_0(raw: bytes, shape) -> np.ndarray:
    """34 B blocks: fp16 scale, then 32 B of 2-bit codes.

    code -> value is {0:-1, 1:0, 2:+1, 3:+2}, i.e. (q - 1) * d.
    """
    n = int(np.prod(shape))
    assert len(raw) == n // 128 * 34, (len(raw), n)
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    codes = blk[:, 2:34]                                    # (nb, 32)
    # unpack 4 codes per byte, little-end first (matches j%4*2 shift)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    q = (codes[:, :, None] >> shifts[None, None, :]) & 0x3   # (nb, 32, 4)
    q = q.reshape(-1, 128).astype(np.float32)
    return ((q - 1.0) * d).reshape(-1).astype(np.float32).reshape(shape)


def _trit_from(v: int, n: int) -> int:
    for _ in range(n):
        v = (v * 3) & 0xFF
    return ((v * 3) >> 8) - 1


def _dequant_ptq1_0(raw: bytes, shape) -> np.ndarray:
    """34 B blocks: 24 B (5 trits/byte -> 120) + 2 B (4 trits/byte -> 8) + fp16 d.

    Element order is NOT positional (see the fork's ptq1_0.glsl):
      e < 80    : qs[e & 15],  n = e >> 4
      80..119   : qs[16 + (e-80 & 7)], n = (e-80) >> 3
      120..127  : qh[(e-120) & 1],     n = (e-120) >> 1
    """
    n = int(np.prod(shape))
    assert len(raw) == n // 128 * 34
    out = np.empty(n, dtype=np.float32)
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    for bi, b in enumerate(blk):
        d = float(np.frombuffer(b[26:28].tobytes(), dtype="<f2")[0])
        qs, qh = b[0:24], b[24:26]
        base = bi * 128
        for e in range(128):
            if e < 80:
                v, k = int(qs[e & 15]), e >> 4
            elif e < 120:
                t = e - 80
                v, k = int(qs[16 + (t & 7)]), t >> 3
            else:
                t = e - 120
                v, k = int(qh[t & 1]), t >> 1
            out[base + e] = _trit_from(v, k) * d
    return out.reshape(shape)


class _Resp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, content: bytes, status: int = 206, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise IOError(f"HTTP {self.status_code}")


class LocalSource:
    """Reads byte ranges from a local GGUF file.

    Deliberately mimics the requests.Session interface used by the remote path
    (`.get(url, headers={"Range": ...}, timeout=...)`), so every consumer --
    BonsaiForward included -- works against a local file unchanged. This
    removes the per-tensor HTTP round trip that dominates iteration time.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        self.size = os.path.getsize(path)

    def get(self, url=None, headers=None, timeout=None):
        hdrs = headers or {}
        rng = hdrs.get("Range")
        if rng is None:
            data = self._read(0, self.size)
            return _Resp(data, 200, {"Content-Length": str(self.size)})
        spec = rng.split("=", 1)[1]
        a, _, b = spec.partition("-")
        start = int(a)
        end = int(b) if b else self.size - 1
        end = min(end, self.size - 1)
        if start > end:
            return _Resp(b"", 416)
        data = self._read(start, end - start + 1)
        return _Resp(data, 206, {
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes {start}-{end}/{self.size}"})

    def _read(self, off: int, n: int) -> bytes:
        self._fh.seek(off)
        return self._fh.read(n)

    def close(self):
        try:
            self._fh.close()
        except Exception:                                 # noqa: BLE001
            pass


def wht_matrix(block: int) -> np.ndarray:
    """Normalized Sylvester-Walsh-Hadamard matrix, matching llama-model.cpp:2054."""
    idx = np.arange(block, dtype=np.uint32)
    parity = idx[:, None] & idx[None, :]
    # popcount parity via the same shift-xor cascade the fork uses
    for sh in (16, 8, 4, 2, 1):
        parity ^= parity >> sh
    return np.where(parity & 1, -1.0, 1.0).astype(np.float32) / np.sqrt(block)


def apply_hadamard(x: np.ndarray, signs, block: int) -> np.ndarray:
    """x' = H . (s * x), applied on the last (input) dimension, block-wise.

    `signs` may be None to apply the bare H transform (the embedding inverse
    applies its sign vector separately, on the flattened row).
    """
    if signs is not None:
        x = x * signs
    n = x.shape[-1]
    assert n % block == 0
    head = x.shape[:-1]
    xb = x.reshape(-1, n // block, block)
    return (xb @ wht_matrix(block).T).reshape(*head, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def fwht_inplace(a: np.ndarray, block: int) -> np.ndarray:
    """In-place-style fast Walsh-Hadamard transform, mirroring src/bonsai/pq2_0.h.

    The C port replaces llama.cpp's explicit 1024x1024 dense Hadamard matmul
    with this butterfly, so the two must agree exactly. This is the reference
    used to prove that equivalence.
    """
    a = np.array(a, dtype=np.float32, copy=True)
    a = a.reshape(-1, block)          # operate on the last axis, batched
    h = 1
    while h < block:
        for i in range(0, block, h << 1):
            u = a[:, i:i + h].copy()
            v = a[:, i + h:i + 2 * h].copy()
            a[:, i:i + h] = u + v
            a[:, i + h:i + 2 * h] = u - v
        h <<= 1
    return a.reshape(-1) if a.shape[0] == 1 else a


def apply_hadamard_fast(x: np.ndarray, signs, block: int) -> np.ndarray:
    """x' = H . (s * x) computed with the butterfly, i.e. O(n log n).

    Matches apply_hadamard (the dense form) to within fp32 rounding.
    """
    x = x.copy()
    n = x.shape[-1]
    assert n % block == 0
    nb = n // block
    head = x.shape[:-1]
    # reshape to (-1, nb, block) explicitly: with a 1-D input and nb == 1 the
    # inferred -1 would collapse the block axis away
    xb = x.reshape(-1, nb, block) if x.ndim > 1 else x.reshape(1, nb, block)
    if signs is not None:
        xb = xb * signs.reshape(1, nb, block)
    norm = np.float32(1.0 / np.sqrt(block))
    out = np.empty_like(xb)
    for i in range(xb.shape[0]):
        out[i] = fwht_inplace(xb[i], block) * norm
    return out.reshape(*head, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_selftest(args):
    """Checks that need no network and no C compiler."""
    rng = np.random.default_rng(1234)
    ok = True

    # 1. butterfly Hadamard == dense Hadamard matrix
    for block in (8, 64, 1024):
        H = wht_matrix(block)
        signs = rng.choice([-1.0, 1.0], size=block).astype(np.float32)
        x = rng.standard_normal(block).astype(np.float32)
        dense = apply_hadamard(x, signs, block)
        fast = apply_hadamard_fast(x, signs, block)
        err = float(np.abs(dense - fast).max())
        good = err < 1e-5
        ok &= good
        print(f"  block {block:>4}: dense vs butterfly max|diff| = {err:.3e} "
              f"{'OK' if good else 'FAIL'}")

    # 2. multi-block stitching (17408 = 17 blocks of 1024)
    block = 1024
    n = 17408
    signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    err = float(np.abs(apply_hadamard(x, signs, block)
                       - apply_hadamard_fast(x, signs, block)).max())
    ok &= err < 1e-4
    print(f"  n={n} (input dim of ffn_down): max|diff| = {err:.3e} "
          f"{'OK' if err < 1e-4 else 'FAIL'}")

    # 3. PQ2_0 codec against an independent scalar reference
    from bonsai_ref import _dequant_pq2_0
    H = 6
    raw = bytes(rng.integers(0, 256, size=H * 34, dtype=np.uint8))
    got = _dequant_pq2_0(raw, (H * 128,))
    ref = np.empty(H * 128, dtype=np.float32)
    for b in range(H):
        blk = raw[b * 34:(b + 1) * 34]
        d = float(np.frombuffer(blk[0:2], dtype="<f2")[0])
        for j in range(128):
            q = (blk[2 + (j >> 2)] >> ((j & 3) * 2)) & 0x3
            ref[b * 128 + j] = (q - 1) * d
    err = float(np.abs(got - ref).max())
    ok &= err == 0.0
    print(f"  PQ2_0 codec: vectorised vs scalar max|diff| = {err:.3e} "
          f"{'OK' if err == 0 else 'FAIL'}")

    # 4. fp16 scale decode matches numpy for a spread of values
    import struct
    worst = 0.0
    for v in (0.0, 1e-3, 0.0115, 1.0, 556.5, 65504.0, -0.047):
        h = struct.unpack("<H", np.float16(v).tobytes())[0]
        # pure-python version of bonsai_fp16_to_fp32
        sign = (h >> 15) << 31
        exp = (h >> 10) & 0x1F
        mant = h & 0x3FF
        if exp == 0:
            bits = sign
        elif exp == 0x1F:
            bits = sign | 0x7F800000 | (mant << 13)
        else:
            bits = sign | ((exp + 112) << 23) | (mant << 13)
        f = struct.unpack("<f", struct.pack("<I", bits))[0]
        worst = max(worst, abs(f - float(np.float16(v))))
    ok &= worst < 1e-3
    print(f"  fp16 decode : max|diff| vs numpy = {worst:.3e} "
          f"{'OK' if worst < 1e-3 else 'FAIL'}")

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(args):
    g = Gguf.open(args.file)
    print(f"file      : {args.file}")
    print(f"gguf ver  : {g.version}   file size: {g.file_size:,} B "
          f"({g.file_size / 1e9:.2f} GB)")
    print(f"tensors   : {len(g.tensors)}   kv pairs: {len(g.meta)}")
    print(f"arch      : {g.meta.get('general.architecture')}")

    print("\n-- tensor type histogram --")
    for tn, c in Counter(t.type_name for t in g.tensors.values()).most_common():
        print(f"   {tn:<8} x{c}")

    # self-consistency: tensor offsets are relative to data_base, and the
    # regions must tile the data area without overlapping or running past EOF
    base = g.data_base
    spans = sorted(((t.offset, t.offset + t.nbytes, n)
                    for n, t in g.tensors.items()))
    bad = [n for (a, b, n) in spans if base + b > g.file_size]
    if bad:
        print(f"\n!! {len(bad)} tensors extend past EOF, e.g. {bad[:3]}")
    overlap = []
    for (a1, b1, n1), (a2, b2, n2) in zip(spans, spans[1:]):
        if a2 < b1:
            overlap.append((n1, n2))
    print(f"\ndata region starts at : {base:,} B  (align32 of info end)")
    print(f"tensor regions        : {len(spans)}, "
          f"{len(overlap)} overlaps, {len(bad)} past-EOF")
    last_end = max(b for _, b, _ in spans)
    print(f"highest tensor end    : {base + last_end:,} B")
    slack = g.file_size - (base + last_end)
    print(f"file size - data end  : {slack:,} B  (trailing padding)")
    print(f"regions tile exactly  : {spans[0][0] == 0 and last_end == g.file_size - base}")

    tot = sum(t.nbytes for t in g.tensors.values())
    print(f"sum of tensor bytes   : {tot:,} B ({tot / 1e9:.2f} GB)")
    assert tot == last_end, "gaps between tensors"
    assert not overlap and not bad

    print("\n-- first 5 tensors --")
    for i, (n, t) in enumerate(g.tensors.items()):
        if i >= 5:
            break
        print(f"   {n:<28} {str(t.dims):<18} {t.type_name:<7} "
              f"{t.nbytes:>12,} B  @{t.offset:,}")

    plan = g.hadamard_plan()
    print("\n-- hadamard plan --")
    if not plan["enabled"]:
        print("   not present")
    else:
        print(f"   block_size : {plan['block_size']}")
        print(f"   sign widths: {sorted(plan['signs'])}")
        print(f"   forward    : {len(plan['forward'])} tensors")
        print(f"   inverse    : {plan['inverse']}")
        by_in = Counter(plan["forward"].values())
        print(f"   forward by input dim: {dict(sorted(by_in.items()))}")
    return 0


def cmd_hadamard(args):
    g = Gguf.open(args.file)
    plan = g.hadamard_plan()
    if not plan["enabled"]:
        print("no hadamard metadata")
        return 1
    bs = plan["block_size"]
    print(f"block_size = {bs}, sqrt = {np.sqrt(bs)}")
    H = wht_matrix(bs)
    print(f"H shape {H.shape}, orthogonality max|H@H.T - I| = "
          f"{np.abs(H @ H.T - np.eye(bs)).max():.3e}")
    for w, s in sorted(plan["signs"].items()):
        print(f"  sign width {w:<6} +1={int((s > 0).sum()):<5} -1={int((s < 0).sum()):<5} "
              f"first8={s[:8].astype(int)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=PQ2_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("hadamard").set_defaults(func=cmd_hadamard)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
