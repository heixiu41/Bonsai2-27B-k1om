"""Card gate for the codes-only repack (34 -> 32 bytes per block, in place).

Modeled on scripts/gate_drow.py:
  1. correctness: top-1 over "The capital of France is" must be 11751 (' Paris')
  2. speed: 64 greedy tokens vs the 3.73 tok/s baseline
  3. memory: steady-state VmRSS must stay under the 13.5 GB gate

No hard-link backup of the model: chat mode only frees the tmpfs when the
unlink drops the last name, and a backup link kept the 7.2 GB alive and
OOM-killed the session (documented in gate_drow.py).
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat  # noqa: E402
from chat import CardClient, Tokenizer, ensure_card_files, \
    read_tokenizer_kv, GGUF_PATH, SSH_HOST  # noqa: E402

REMOTE = ("export LD_LIBRARY_PATH=/tmp && "
          "/tmp/bonsai_k1om_chat /tmp/bonsai.gguf chat")
PROMPT = "The capital of France is"
EXPECT_TOP1 = 11751
BASELINE_TOK_S = 3.73            # README baseline for this engine
RSS_GATE_GIB = 13.5

assert ensure_card_files(SSH_HOST, GGUF_PATH), "card file prep failed"

base = chat._ssh_base(SSH_HOST)


def rss_gib():
    pid = chat._run_detached(["ssh", *base, SSH_HOST, "pidof bonsai_k1om_chat"],
                             capture_output=True, text=True).stdout.strip()
    if not pid:
        return None
    out = chat._run_detached(
        ["ssh", *base, SSH_HOST, f"grep VmRSS /proc/{pid}/status"],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1048576.0
    return None


c = CardClient(host=SSH_HOST, remote_cmd=REMOTE)
try:
    c.wait_ready()
    tok = Tokenizer(read_tokenizer_kv(GGUF_PATH))
    c.set_eos(tok.eos_id)
    c.ingest(tok.encode(PROMPT), verbose=False)
    c._send("G 1")
    first = int(c._read_line())
    assert c._read_line().startswith("END")

    t0 = time.time()
    c._send("G 64")
    n = 0
    while True:
        line = c._read_line()
        if line.isdigit():
            n += 1
        elif line.startswith("END"):
            break
    dt = time.time() - t0
    rate = n / dt

    rss = rss_gib()
    ok_top = first == EXPECT_TOP1
    ok_speed = rate > BASELINE_TOK_S
    ok_mem = rss is not None and rss <= RSS_GATE_GIB

    print(f"top-1  : {first} ({ascii(tok.id2tok[first])}) expect {EXPECT_TOP1}"
          f"  -> {'MATCH' if ok_top else 'FAIL'}")
    print(f"decode : {n} tok in {dt:.1f}s = {rate:.2f} tok/s "
          f"(baseline {BASELINE_TOK_S}, {rate / BASELINE_TOK_S:.3f}x)  "
          f"-> {'IMPROVED' if ok_speed else 'NO GAIN'}")
    print(f"memory : VmRSS {rss:.2f} GiB (gate {RSS_GATE_GIB})  "
          f"-> {'OK' if ok_mem else 'FAIL'}")
    ok = ok_top and ok_mem
    print(f"GATE   : {'PASS' if ok else 'FAIL'}   "
          f"(top-1 {ok_top}, memory {ok_mem}, speed {ok_speed})")
    sys.exit(0 if ok else 1)
finally:
    c.close()
