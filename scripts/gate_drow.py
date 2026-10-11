"""Card gate for the drow-precompute build (single session):
load /tmp/bonsai_k1om_chat_drow, forward gate (top-1 == 11751 ' Paris'),
decode speed vs today's known baseline (3.499 tok/s, same card/day/binary-class;
isolated-kernel A/B 1.142x already proven by src/probes/pq2drow.c).

NOTE: no hard-link backup of the model! The chat binary's unlink-after-load
only frees tmpfs when the inode has no other name; a backup link kept it
alive and OOM-killed the session (19.5 GB > 15.5 GB).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat
from chat import CardClient, Tokenizer, ensure_card_files, \
    read_tokenizer_kv, GGUF_PATH, SSH_HOST

REMOTE = ("export LD_LIBRARY_PATH=/tmp && "
          "/tmp/bonsai_k1om_chat_drow /tmp/bonsai.gguf chat")
PROMPT = "The capital of France is"
EXPECT_TOP1 = 11751
BASELINE_TOK_S = 3.499                   # measured today, same card

assert ensure_card_files(SSH_HOST, GGUF_PATH), "card file prep failed"

c = CardClient(host=SSH_HOST, remote_cmd=REMOTE)
try:
    c.wait_ready()
    tok = Tokenizer(read_tokenizer_kv(GGUF_PATH))
    c.set_eos(tok.eos_id)
    c.ingest(tok.encode(PROMPT), verbose=False)
    c._send("G 1")
    first = int(c._read_line())
    assert c._read_line().startswith("END")
    t2 = time.time()
    c._send("G 64")
    n = 0
    while True:
        line = c._read_line()
        if line.isdigit():
            n += 1
        elif line.startswith("END"):
            break
    dt = time.time() - t2
    rate = n / dt
    print(f"top-1={first} ({ascii(tok.id2tok[first])}) expect {EXPECT_TOP1} "
          f"-> {'MATCH' if first == EXPECT_TOP1 else 'FAIL'}", flush=True)
    print(f"decode: {n} tok in {dt:.1f}s = {rate:.2f} tok/s "
          f"(baseline {BASELINE_TOK_S}, ratio {rate / BASELINE_TOK_S:.3f}x)",
          flush=True)
    ok = first == EXPECT_TOP1
    print(f"GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)
finally:
    c.close()
