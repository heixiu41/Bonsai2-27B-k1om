#!/usr/bin/env python3
"""Console chat client for the ternary Bonsai-2 27B card over ssh.

Pure stdlib (no tokenizers/transformers/numpy). The tokenizer is read straight
out of the local GGUF header and implements GPT-2 byte-level BPE with the
Qwen pre-split regex.

Card-side protocol (line based, stdin/stdout of the ssh process):
    card prints "READY" when the model is loaded
    E <id>          set EOS token id            -> OK
    R               reset recurrent state       -> OK
    P <id> ...      ingest prompt tokens        -> OK / ERR prompt (code N)
    G <max>         greedy generate; one line per token id, then "END <n>".
                    Stops early at EOS; a trailing EOS is NOT fed back into
                    the model state, everything before it is.

Bookkeeping (verified): the card state always holds exactly the tokens we
have P-ingested plus the non-EOS tokens it emitted. So per turn we ingest
    <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
and after the reply we ingest the closer
    <|im_end|>\n
so the next user turn continues at the right position.

Usage:
    python scripts/chat.py               # chat
    python scripts/chat.py --selftest    # tokenizer tests, no ssh
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import os
import re
import struct
import subprocess
import sys
import time

GGUF_PATH = r"D:\bonsai\Ternary-Bonsai-2-27B-PQ2_0.gguf"
SSH_HOST = "root@192.168.1.100"
REMOTE_CMD = ("export LD_LIBRARY_PATH=/tmp && "
              "/tmp/bonsai_k1om_chat /tmp/bonsai.gguf chat")
MAX_NEW_TOKENS = 512
PREFILL_TOK_S = 3.4          # measured card prefill speed, for the ETA display

# ---------------------------------------------------------------------------
# GGUF metadata (kv section only; tensor directory is never touched)
# ---------------------------------------------------------------------------
_T_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_T_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
          10: "Q", 11: "q", 12: "d"}
_T_STRING, _T_ARRAY = 8, 9

WANTED_KV = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
}


def _read_exact(fh, n: int) -> bytes:
    b = fh.read(n)
    if len(b) != n:
        raise EOFError(f"GGUF truncated: wanted {n} bytes, got {len(b)}")
    return b


def _read_u32(fh) -> int:
    return struct.unpack("<I", _read_exact(fh, 4))[0]


def _read_u64(fh) -> int:
    return struct.unpack("<Q", _read_exact(fh, 8))[0]


def _read_string(fh) -> str:
    n = _read_u64(fh)
    return _read_exact(fh, n).decode("utf-8", "replace")


def read_tokenizer_kv(path: str) -> dict:
    """Stream through the GGUF kv section, keeping only the tokenizer keys."""
    meta = {}
    with open(path, "rb") as fh:
        if _read_exact(fh, 4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = _read_u32(fh)
        _n_tensors = _read_u64(fh)
        n_kv = _read_u64(fh)
        if version != 3:
            print(f"[warn] GGUF version {version}, expected 3; trying anyway",
                  file=sys.stderr)
        for _ in range(n_kv):
            key = _read_string(fh)
            t = _read_u32(fh)
            if t == _T_STRING:
                v = _read_string(fh)
                if key in WANTED_KV:
                    meta[key] = v
            elif t == _T_ARRAY:
                et = _read_u32(fh)
                n = _read_u64(fh)
                if key in WANTED_KV:
                    if et == _T_STRING:
                        meta[key] = [_read_string(fh) for _ in range(n)]
                    else:
                        raw = _read_exact(fh, _T_SIZE[et] * n)
                        meta[key] = list(struct.unpack(
                            "<" + _T_FMT[et] * n, raw))
                else:
                    if et == _T_STRING:
                        for _ in range(n):
                            _read_string(fh)
                    else:
                        fh.seek(_T_SIZE[et] * n, os.SEEK_CUR)
            else:
                v = struct.unpack("<" + _T_FMT[t],
                                  _read_exact(fh, _T_SIZE[t]))[0]
                if key in WANTED_KV:
                    meta[key] = v
        # stop here: tensor directory and data are never read
    return meta


# ---------------------------------------------------------------------------
# GPT-2 byte-level BPE tokenizer
# ---------------------------------------------------------------------------
def _bytes_to_unicode() -> dict:
    """The GPT-2 byte<->unicode alphabet (byte -> printable char)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(0xA1, 0xAC + 1))
          + list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENC = _bytes_to_unicode()                    # int byte -> str char
BYTE_DEC = {c: b for b, c in BYTE_ENC.items()}    # str char -> int byte

# Qwen pre-split, stdlib-re translation of
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
#    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
# with \p{L} ~= [^\W\d_], \p{N} ~= \d (re.UNICODE is default for str).
#   [^\r\n\p{L}\p{N}]  = (?:[^\w\r\n]|_)
#   [^\s\p{L}\p{N}]    = (?:[^\s\w]|_)
QWEN_PAT = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\w\r\n]|_)?[^\W\d_]+"
    r"|\d{1,3}"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# llama.cpp token types (tokenizer.ggml.token_type)
TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6


class Tokenizer:
    def __init__(self, meta: dict):
        for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                  "tokenizer.ggml.model", "tokenizer.ggml.eos_token_id"):
            if k not in meta:
                raise KeyError(f"GGUF is missing required key {k!r}")
        self.id2tok = list(meta["tokenizer.ggml.tokens"])
        self.vocab = {t: i for i, t in enumerate(self.id2tok)}
        self.ranks = {}
        for i, m in enumerate(meta["tokenizer.ggml.merges"]):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = i
        n = len(self.id2tok)
        ttypes = meta.get("tokenizer.ggml.token_type")
        if ttypes is None or len(ttypes) != n:
            # fallback: guess control tokens by the <|...|> naming scheme
            ttypes = [TT_CONTROL if (t.startswith("<|") and t.endswith("|>"))
                      else TT_NORMAL for t in self.id2tok]
        self.ttypes = list(ttypes)
        self.eos_id = int(meta["tokenizer.ggml.eos_token_id"])
        self.bos_id = (int(meta["tokenizer.ggml.bos_token_id"])
                       if "tokenizer.ggml.bos_token_id" in meta else None)
        self.model = meta.get("tokenizer.ggml.model")
        self.pre = meta.get("tokenizer.ggml.pre")
        # special tokens we may inject while encoding (control + user-defined)
        self.specials = {t: i for i, t in enumerate(self.id2tok)
                         if self.ttypes[i] in (TT_CONTROL, TT_USER)}
        self._cache: dict[str, tuple[int, ...]] = {}

    # -- encoding -----------------------------------------------------------
    def _bpe(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        word = [BYTE_ENC[b] for b in chunk.encode("utf-8")]
        while len(word) > 1:
            best_rank, best_i = None, -1
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            word[best_i:best_i + 2] = [word[best_i] + word[best_i + 1]]
        ids = tuple(self.vocab[t] for t in word)
        self._cache[chunk] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids = []
        for m in QWEN_PAT.finditer(text):
            ids.extend(self._bpe(m.group(0)))
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode text; control tokens like <|im_start|> become their ids."""
        if not allow_special or not self.specials:
            return self._encode_plain(text)
        ids = []
        while text:
            pos, hit = -1, None
            for s in self.specials:
                p = text.find(s)
                if p != -1 and (pos == -1 or p < pos):
                    pos, hit = p, s
            if hit is None:
                ids.extend(self._encode_plain(text))
                break
            if pos:
                ids.extend(self._encode_plain(text[:pos]))
            ids.append(self.specials[hit])
            text = text[pos + len(hit):]
        return ids

    # -- decoding -----------------------------------------------------------
    def token_bytes(self, tid: int) -> bytes:
        t = self.id2tok[tid]
        if self.ttypes[tid] in (TT_CONTROL, TT_USER, TT_UNUSED):
            # not byte-level; render literally so it round-trips readably
            return t.encode("utf-8")
        try:
            return bytes(BYTE_DEC[c] for c in t)
        except KeyError:
            return t.encode("utf-8", "replace")

    def decode(self, ids) -> str:
        out = bytearray()
        for tid in ids:
            out += self.token_bytes(tid)
        return out.decode("utf-8", "replace")

    def is_displayable(self, tid: int) -> bool:
        return self.ttypes[tid] not in (TT_CONTROL, TT_UNUSED)


class StreamDecoder:
    """Incremental utf-8 decode of a token-id stream (holds back partial
    multi-byte tails until the rest of the character arrives)."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self._inc = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, tid: int) -> str:
        if not self.tok.is_displayable(tid):
            return ""
        return self._inc.decode(self.tok.token_bytes(tid), final=False)

    def flush(self) -> str:
        return self._inc.decode(b"", final=True)


# ---------------------------------------------------------------------------
# Card client (ssh subprocess, line protocol)
# ---------------------------------------------------------------------------
class CardDied(Exception):
    pass


class CardError(Exception):
    pass


def _ssh_base(host: str) -> list:
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return ["-i", os.path.join(home, ".ssh", "id_rsa_mic"),
            "-o", f"UserKnownHostsFile={os.path.join(home, '.ssh', 'known_hosts')}",
            "-o", "StrictHostKeyChecking=no"]


# The k1om OpenMP runtime the card-side chat binary links against.  PSXE 2017
# ships it under the mic compiler lib dir; the MPSS sysroot does not have it,
# and the card's tmpfs loses it on every boot.
_MIC_OMP_LIB = (r"C:\Program Files (x86)\IntelSWTools"
                r"\compilers_and_libraries_2017.0.109\windows\compiler"
                r"\lib\intel64_win_mic\libiomp5.so")


def _run_detached(argv, **kw):
    """subprocess.run with the child detached from our stdin.

    chat.py's stdin belongs to the conversation (a terminal, or a pipe when it
    is driven by a script).  ssh forwards its stdin to the remote command, so
    without this the upload helpers eat whatever the user typed next -- the
    SCIF sink used to swallow the whole piped input, making a cold start look
    like an immediate EOF."""
    kw.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(argv, **kw)


def _mic_runtime_libs() -> list:
    """Local copies of the k1om shared libraries the card binary needs."""
    cands = [os.environ.get("BONSAI_MIC_OMP_LIB"), _MIC_OMP_LIB]
    return [p for p in cands if p and os.path.exists(p)]


def _ensure_card_binary(host: str, base: list, local: str, remote: str) -> bool:
    """Put a host binary on the card unless that exact build is already there.

    The card is BusyBox with no sftp subsystem, so scp needs -O; a stale binary
    left over from an earlier build is a far worse failure than a 1 s copy."""
    if not os.path.exists(local):
        return False
    if _card_file_size(host, base, f"/tmp/{remote}") == os.path.getsize(local):
        return True
    # scp to the *remote* name explicitly: with a bare "host:/tmp/" target scp
    # keeps the local basename, so `remote` would be a lie whenever the two
    # differ (and only the size check would ever notice).
    r = _run_detached(["scp", "-O", "-c", "chacha20-poly1305@openssh.com",
                       *base, local, f"{host}:/tmp/{remote}"])
    if r.returncode != 0:
        return False
    _run_detached(["ssh", *base, host, f"chmod +x /tmp/{remote}"],
                  capture_output=True)
    return True


def _card_file_size(host: str, base: list, path: str):
    """Size of a file on the card, or None.

    Both obvious probes are broken on this BusyBox: there is no `stat` at all,
    and `wc -c` is 32-bit, so it reports the 7,206,168,928-byte model as
    2,911,201,632 (= size mod 2^32).  `ls -l` prints the real 64-bit size."""
    try:
        r = _run_detached(["ssh", *base, host, f"ls -l {path} 2>/dev/null"],
                          capture_output=True, text=True, timeout=60)
        parts = r.stdout.split()
        return int(parts[4]) if len(parts) >= 5 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _kill_card_sinks(host: str, base: list):
    """A sink left behind by a failed attempt still holds port 20500, which
    would make the next fallback layer fail for the wrong reason."""
    try:
        _run_detached(["ssh", *base, host,
                        "killall scif_rma_sink scif_sink 2>/dev/null; true"],
                       capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _md5_file(path: str, label: str) -> str:
    """Streaming md5 with progress (~30 s for the 7.2 GB model on the host)."""
    h = hashlib.md5()
    total = os.path.getsize(path)
    done = 0
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            blk = f.read(1 << 22)
            if not blk:
                break
            h.update(blk)
            done += len(blk)
            pct = 100.0 * done / total if total else 100.0
            print(f"\r  {label} md5 {pct:5.1f}% ({done / 1e9:.2f}/"
                  f"{total / 1e9:.2f} GB)", end="", flush=True)
    print(f"\r  {label} md5 done in {time.time() - t0:.0f} s" + " " * 24,
          flush=True)
    return h.hexdigest()


def _card_md5(host: str, base: list, path: str) -> str:
    """Whole-file md5 on the card.  BusyBox here has no sha256sum (only md5sum
    and sha3sum), and it is scalar on 1.2 GHz cores, so this takes minutes."""
    print("  card md5sum (no sha256sum in BusyBox; ~1-3 min for 7.2 GB) ...",
          flush=True)
    r = _run_detached(["ssh", *base, host, f"md5sum {path}"],
                      capture_output=True, text=True, timeout=3600)
    tok = r.stdout.split()
    return tok[0] if tok else ""


def _want_md5() -> bool:
    return os.environ.get("BONSAI_VERIFY_MD5", "0").lower() not in \
        ("", "0", "no", "false", "off")


def _verify_card_model(host: str, base: list, gguf: str) -> bool:
    """Opt-in whole-file md5 comparison (BONSAI_VERIFY_MD5=1, or --verify-md5).

    Every transfer already checks a full-length rolling checksum card-side
    against host-side, so this is belt-and-braces with a real cryptographic
    hash: worth it when the card has been misbehaving, too slow for every
    start."""
    want = _md5_file(gguf, "host")
    got = _card_md5(host, base, "/tmp/bonsai.gguf")
    if want and want == got:
        print(f"  md5 OK: {want}", flush=True)
        return True
    print(f"[fatal] md5 MISMATCH -- host={want} card={got}", file=sys.stderr)
    return False


def _model_ready(host: str, base: list, gguf: str) -> bool:
    """Tail of ensure_card_files: the model is on the card."""
    if _want_md5():
        return _verify_card_model(host, base, gguf)
    return True


def ensure_card_files(host: str, gguf: str) -> bool:
    """Upload the chat binary and the model to the card if missing.

    The model is 7.2 GB and the card deletes it after every chat session
    (see note in main()), so this is the normal per-session path.  The model
    comes over SCIF, with three independent layers that must all keep working:
    RMA (measured 357 MB/s) -> message SCIF (9-39 MB/s) -> scp (9-15 MB/s).

    Set BONSAI_VERIFY_MD5=1 (or pass --verify-md5) to md5 the file on the card
    and compare it with the local one after the upload.  Off by default: the
    transfer's own full-length checksum already covers every byte, and the
    card's scalar md5sum costs minutes."""
    base = _ssh_base(host)
    # The card reboots from time to time (tmpfs /tmp is wiped); its sshd
    # comes up well after the network stack, so a single 30 s probe can land
    # in the boot window. Retry for up to ~3 minutes before giving up.
    r = None
    for attempt in range(6):
        try:
            r = _run_detached(["ssh", *base, host,
                                "ls /tmp/bonsai_k1om_chat /tmp/libiomp5.so "
                                "/tmp/bonsai.gguf 2>/dev/null"],
                               capture_output=True, text=True, timeout=35)
            break
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"[card] reach attempt {attempt + 1}/6 failed: {e}; "
                  f"retrying in 25 s (card may still be booting)",
                  file=sys.stderr, flush=True)
            time.sleep(25)
    if r is None:
        print("[fatal] cannot reach the card after 6 attempts",
              file=sys.stderr)
        return False
    have = r.stdout
    # Size-checked, not just "exists": the chat binary is rebuilt far more
    # often than the model changes, and a stale copy on the card silently
    # invalidates every measurement taken through it (it made an A/B compare
    # two identical builds and look like a clean null result).
    local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "bonsai_k1om_chat")
    if "bonsai_k1om_chat" not in have or \
            _card_file_size(host, base, "/tmp/bonsai_k1om_chat") != \
            os.path.getsize(local_bin):
        print("uploading chat binary ...", flush=True)
        if not _ensure_card_binary(host, base, local_bin, "bonsai_k1om_chat"):
            print("[fatal] binary upload failed", file=sys.stderr)
            return False
    if "libiomp5.so" not in have:
        # The chat binary is an OpenMP build and the card has no libiomp5.so of
        # its own, so a cold start must push it too -- previously this was a
        # manual step and chat.py died with "error while loading shared
        # libraries: libiomp5.so".
        libs = _mic_runtime_libs()
        if not libs:
            print("[fatal] /tmp/libiomp5.so is missing on the card and no "
                  "local copy was found (set BONSAI_MIC_OMP_LIB to the k1om "
                  "libiomp5.so)", file=sys.stderr)
            return False
        for lib in libs:
            if not _ensure_card_binary(host, base, lib, os.path.basename(lib)):
                print(f"[fatal] could not upload {os.path.basename(lib)}",
                      file=sys.stderr)
                return False
    if "bonsai.gguf" not in have:
        layer = os.environ.get("BONSAI_TRANSFER_LAYER", "auto").lower()
        if layer not in ("auto", "rma", "scif", "scp"):
            layer = "auto"
        _kill_card_sinks(host, base)

        # Layer 1: SCIF RMA (~357 MB/s).  Layer 2: the message SCIF sink
        # (9-39 MB/s).  Layer 3: scp (9-15 MB/s).  A forced layer disables the
        # fallbacks so the verification harness can prove each one alone.
        ok = False
        if layer in ("auto", "rma"):
            ok = _upload_model_rma(host, base, gguf)
            if not ok and layer == "auto":
                print("SCIF RMA upload failed -- falling back to the message "
                      "SCIF sink", flush=True)
                _kill_card_sinks(host, base)
        if not ok and layer in ("auto", "scif"):
            ok = _upload_model_scif(host, base, gguf)
            if not ok and layer == "auto":
                print("message SCIF upload failed -- falling back to scp",
                      flush=True)
        if ok:
            return _model_ready(host, base, gguf)
        if layer in ("rma", "scif"):
            print(f"[fatal] BONSAI_TRANSFER_LAYER={layer}: that data path "
                  f"failed and fallback is disabled", file=sys.stderr)
            return False

        # scp is both the last layer and the target of BONSAI_TRANSFER_LAYER=scp
        if layer == "auto" and \
                _card_file_size(host, base, "/tmp/bonsai.gguf") == \
                os.path.getsize(gguf):
            print("SCIF handshake failed but the file is complete "
                  "on the card -- skipping the scp fallback", flush=True)
            return _model_ready(host, base, gguf)
        print("uploading over scp (~7-8 min for 7.2 GB) ...", flush=True)
        t0 = time.time()
        r = _run_detached(["scp", "-O", "-c", "chacha20-poly1305@openssh.com",
                            *base, gguf, f"{host}:/tmp/bonsai.gguf"])
        if r.returncode != 0:
            print("[fatal] model upload failed (card tmpfs full? "
                  "it holds only 12.9 GB total)", file=sys.stderr)
            return False
        print(f"model uploaded in {time.time() - t0:.0f} s (scp)", flush=True)
    return _model_ready(host, base, gguf)


def _scif_upload(host: str, base: list, sink_local: str, sink_remote: str,
                 send_exe: str, gguf: str, label: str) -> bool:
    """Shared orchestration for both SCIF data planes.

    Deploy the card-side sink, start it over a background ssh, run the host
    sender, then require BOTH sides to exit 0.  Any failure returns False so
    the caller can fall through to the next layer."""
    if not os.path.exists(send_exe):
        print(f"[{label}] host sender {send_exe} is missing", flush=True)
        return False
    if not _ensure_card_binary(host, base, sink_local, sink_remote):
        print(f"[{label}] could not deploy {sink_remote} to the card", flush=True)
        return False

    for attempt in (1, 2):
        # the sink's own ssh channel has proven flaky: verify the launch and
        # retry once before giving up to the next layer
        sink = subprocess.Popen(
            ["ssh", *base, host,
             f"export LD_LIBRARY_PATH=/tmp && /tmp/{sink_remote}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            # stderr inherited: sink errors must be visible
        try:
            time.sleep(2)
            if sink.poll() is not None:
                print(f"[{label}] attempt {attempt}: sink ssh died at "
                      f"launch (rc={sink.returncode})", flush=True)
                continue
            r = _run_detached([send_exe, gguf, "bonsai.gguf"], timeout=1800)
            sink_rc = sink.wait(timeout=300)
            if r.returncode == 0 and sink_rc == 0:
                return True
            print(f"[{label}] attempt {attempt} failed "
                  f"(send rc={r.returncode}, sink rc={sink_rc})", flush=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"[{label}] attempt {attempt} error: {e}", flush=True)
        finally:
            if sink.poll() is None:
                sink.kill()
        time.sleep(2)
    return False


def _upload_model_rma(host: str, base: list, gguf: str) -> bool:
    """Model upload over SCIF registered-memory RMA (~357 MB/s measured).

    The card registers a 256 MB ring window and the host DMAs whole slots into
    it with scif_writeto, publishing progress with scif_fence_signal -- the
    file never travels through the SCIF message path.  The sender also checks
    the byte count and a full-file checksum the card computed over exactly what
    it wrote, so a zero exit means the file is intact."""
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
    print("model not on the card -- uploading 7.2 GB over SCIF RMA "
          "(~20 s, measured 357 MB/s) ...", flush=True)
    t0 = time.time()
    ok = _scif_upload(host, base,
                      os.path.join(tools, "scif_rma_sink"), "scif_rma_sink",
                      os.path.join(tools, "scif_rma_send.exe"), gguf, "scif-rma")
    if ok:
        print(f"model uploaded in {time.time() - t0:.0f} s (SCIF RMA)",
              flush=True)
    return ok


def _upload_model_scif(host: str, base: list, gguf: str) -> bool:
    """Model upload over the original message SCIF (9-39 MB/s measured).

    Kept as the second layer: the host sends 16 MB scif_send messages that the
    card-side sink turns into scif_recv + fwrite."""
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
    print("uploading 7.2 GB over the message SCIF sink "
          "(measured 9-39 MB/s) ...", flush=True)
    t0 = time.time()
    ok = _scif_upload(host, base,
                      os.path.join(tools, "scif_sink"), "scif_sink",
                      os.path.join(tools, "scif_send.exe"), gguf, "scif-msg")
    if ok:
        print(f"model uploaded in {time.time() - t0:.0f} s (message SCIF)",
              flush=True)
    return ok


class CardClient:
    def __init__(self, host: str = SSH_HOST, remote_cmd: str = REMOTE_CMD):
        home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
        key = os.path.join(home, ".ssh", "id_rsa_mic")
        known = os.path.join(home, ".ssh", "known_hosts")
        cmd = ["ssh", "-i", key,
               "-o", f"UserKnownHostsFile={known}",
               "-o", "StrictHostKeyChecking=no",
               host, remote_cmd]
        try:
            self.p = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,          # diagnostics merged; the
                text=True, bufsize=1,              # [card] filter sorts them
                encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise CardDied("ssh executable not found on PATH")

    # -- low level ----------------------------------------------------------
    def _send(self, line: str):
        try:
            self.p.stdin.write(line + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise CardDied(f"cannot write to ssh: {e}")

    def _read_line(self) -> str:
        line = self.p.stdout.readline()
        if line == "":
            rc = self.p.poll()
            raise CardDied(f"ssh/card closed its output (exit {rc})")
        return line.rstrip("\n").rstrip("\r")

    def _expect_ok(self):
        while True:
            line = self._read_line()
            if line == "OK":
                return
            if line.startswith("ERR"):
                raise CardError(line)
            print(f"[card] {line}", file=sys.stderr)

    # -- protocol -----------------------------------------------------------
    def wait_ready(self):
        print("waiting for the card to load the model (~15 s)...", flush=True)
        t0 = time.time()
        while True:
            line = self._read_line()
            if line == "READY":
                print(f"card READY after {time.time() - t0:.1f} s", flush=True)
                return
            print(f"[card] {line}", file=sys.stderr)

    def set_eos(self, eos_id: int):
        self._send(f"E {eos_id}")
        self._expect_ok()

    def reset(self):
        self._send("R")
        self._expect_ok()

    def ingest(self, ids, verbose: bool = True):
        if not ids:
            return
        if verbose:
            est = len(ids) / PREFILL_TOK_S
            print(f"[prefill] ingesting {len(ids)} tokens "
                  f"(~{est:.0f} s at ~{PREFILL_TOK_S} tok/s)...", flush=True)
        t0 = time.time()
        self._send("P " + " ".join(map(str, ids)))
        self._expect_ok()
        if verbose:
            print(f"[prefill] prompt ingested in {time.time() - t0:.1f} s",
                  flush=True)

    def generate(self, tok: Tokenizer, max_new: int,
                 temperature: float = 0.0) -> list[int]:
        """Stream a reply. Returns the emitted token ids (incl. a trailing
        EOS if the model stopped on its own). temperature > 0 switches the
        card from greedy argmax to softmax(logits/T) sampling."""
        if temperature and temperature > 0:
            self._send(f"G {max_new} {temperature:g}")
        else:
            self._send(f"G {max_new}")
        dec = StreamDecoder(tok)
        ids = []
        t0 = time.time()
        while True:
            line = self._read_line()
            if line.isdigit():
                tid = int(line)
                ids.append(tid)
                s = dec.feed(tid)
                if s:
                    print(s, end="", flush=True)
            elif line.startswith("END"):
                break
            else:
                print(f"\n[card] {line}", file=sys.stderr)
        tail = dec.flush()
        if tail:
            print(tail, end="", flush=True)
        print()
        dt = time.time() - t0
        rate = len(ids) / dt if dt > 0 else 0.0
        stopped = " (stopped at EOS)" if ids and ids[-1] == tok.eos_id else ""
        print(f"[gen] {len(ids)} tokens in {dt:.1f} s "
              f"({rate:.2f} tok/s){stopped}", flush=True)
        return ids

    def close(self):
        try:
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
USER_BLOCK = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
TURN_CLOSER = "<|im_end|>\n"


def repl(tok: Tokenizer, args) -> int:
    client = CardClient(host=args.host)
    try:
        client.wait_ready()
        client.set_eos(tok.eos_id)
        print(f"EOS set to {tok.eos_id} ({tok.id2tok[tok.eos_id]})")
        print("Type /reset to clear the conversation, /quit to exit.")
        first_turn = True
        while True:
            try:
                text = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/reset":
                try:
                    client.reset()
                except (CardDied, CardError) as e:
                    print(f"[error] {e}", file=sys.stderr)
                    return 1
                first_turn = True
                print("[reset] conversation cleared")
                continue
            if text.startswith("/"):
                print("[?] unknown command (have: /reset /quit)")
                continue

            ids = tok.encode(USER_BLOCK.format(text))
            try:
                if first_turn:
                    client.reset()          # fresh context for turn 1
                    first_turn = False
                client.ingest(ids)
                print("Bonsai> ", end="", flush=True)
                client.generate(tok, args.max_tokens)
                # the emitted non-EOS tokens are already in card state;
                # append the turn closer so the next user block lines up
                client.ingest(tok.encode(TURN_CLOSER), verbose=False)
            except KeyboardInterrupt:
                print("\n[interrupted] card state may be inconsistent; "
                      "use /reset before continuing", file=sys.stderr)
                # drain nothing; the card is still streaming -> safest is reset
                try:
                    client.reset()
                    first_turn = True
                except (CardDied, CardError):
                    pass
            except CardError as e:
                print(f"[card error] {e} — try /reset", file=sys.stderr)
            except CardDied as e:
                print(f"[fatal] {e}", file=sys.stderr)
                return 1
    except (CardDied, CardError) as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


# ---------------------------------------------------------------------------
# Self test (no ssh)
# ---------------------------------------------------------------------------
ROUNDTRIP_STRINGS = [
    "Hello, world!",
    "The capital of France is",
    "你好，世界！这是一段中文测试。",
    "abc123 def-456",
    "a  b   c",
    "line one\nline two\n\nline four\n",
    "🙂",
    "def foo(x):\n    return x + 1  # indented code\n",
    "trailing spaces   ",
    "mixed 中英文 text with 123 numbers and emoji 🙂 end",
]


def selftest(args) -> int:
    print(f"reading tokenizer from {args.gguf} ...", flush=True)
    t0 = time.time()
    meta = read_tokenizer_kv(args.gguf)
    tok = Tokenizer(meta)
    print(f"  loaded in {time.time() - t0:.1f} s")
    print(f"  tokens : {len(tok.id2tok):,} (expect 248,320)")
    print(f"  merges : {len(tok.ranks):,} (expect 247,587)")
    print(f"  model  : {tok.model!r}   pre: {tok.pre!r}")
    print(f"  eos_id : {tok.eos_id} ({tok.id2tok[tok.eos_id]!r})   "
          f"bos_id: {tok.bos_id}")

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""))

    # hard vector
    vec = tok.encode("The capital of France is", allow_special=False)
    check('encode("The capital of France is") == [760, 6511, 314, 9338, 369]',
          vec == [760, 6511, 314, 9338, 369], f"got {vec}")
    check('decode([11751]) == " Paris"',
          tok.decode([11751]) == " Paris", f"got {tok.decode([11751])!r}")

    # special token ids
    for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        tid = tok.vocab.get(name)
        print(f"  vocab id of {name}: {tid}")
        check(f"{name} present in vocab", tid is not None)
    if tok.vocab.get("<|im_start|>") is not None:
        enc = tok.encode("<|im_start|>user\nhi<|im_end|>\n")
        check("special tokens encode atomically",
              enc[0] == tok.vocab["<|im_start|>"]
              and tok.vocab["<|im_end|>"] in enc, f"got {enc}")

    # round trips
    for s in ROUNDTRIP_STRINGS:
        rt = tok.decode(tok.encode(s, allow_special=False))
        check(f"round-trip {s[:32]!r}{'...' if len(s) > 32 else ''}", rt == s,
              "" if rt == s else f"got {rt!r}")

    # incremental streaming decode must equal batch decode
    ids = tok.encode("你好，世界！🙂 stream", allow_special=False)
    sd = StreamDecoder(tok)
    streamed = "".join(sd.feed(i) for i in ids) + sd.flush()
    check("incremental decode == batch decode",
          streamed == "你好，世界！🙂 stream", f"got {streamed!r}")

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    # the Windows console may be GBK; utf-8 with replacement keeps Chinese /
    # emoji output (and the test names) from crashing print()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run tokenizer tests only (no ssh)")
    ap.add_argument("--gguf", default=GGUF_PATH, help="local GGUF path")
    ap.add_argument("--host", default=SSH_HOST, help="ssh target")
    ap.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
                    help="max tokens per reply (G <max>)")
    ap.add_argument("--verify-md5", action="store_true",
                    help="after the upload, md5 the model on the card and "
                         "compare with the local file (slow: BusyBox md5sum "
                         "is scalar on the card)")
    args = ap.parse_args(argv)
    if args.verify_md5:
        os.environ["BONSAI_VERIFY_MD5"] = "1"

    if args.selftest:
        return selftest(args)

    # The card-side chat server UNLINKS /tmp/bonsai.gguf after loading it
    # (the card's /tmp is tmpfs with no swap; the file copy + the in-memory
    # copy would otherwise double-book 14.5 of 15.5 GB and the 2.6.38 OOM
    # killer fires). So the model file survives at most ONE chat session and
    # must be re-uploaded when missing. Check first, upload if needed.
    if not ensure_card_files(args.host, args.gguf):
        return 1

    try:
        meta = read_tokenizer_kv(args.gguf)
        tok = Tokenizer(meta)
    except (OSError, KeyError, ValueError) as e:
        print(f"[fatal] tokenizer load failed: {e}", file=sys.stderr)
        return 1
    print(f"tokenizer ready: {len(tok.id2tok):,} tokens, "
          f"{len(tok.ranks):,} merges")
    return repl(tok, args)


if __name__ == "__main__":
    sys.exit(main())
