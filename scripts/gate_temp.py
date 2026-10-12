"""Card gate for the temperature-sampling build:
1. greedy path unchanged: top-1('The capital of France is') == 11751
2. decode speed sanity vs 3.7 tok/s class
3. sampling smoke: 'G 32 0.8' twice -> all ids in range, streams differ
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat
from chat import CardClient, Tokenizer, ensure_card_files, \
    read_tokenizer_kv, GGUF_PATH, SSH_HOST

REMOTE = ("export LD_LIBRARY_PATH=/tmp && "
          "/tmp/bonsai_k1om_chat /tmp/bonsai.gguf chat")
PROMPT = "The capital of France is"
EXPECT_TOP1 = 11751
N_VOCAB = 248320

assert ensure_card_files(SSH_HOST, GGUF_PATH), "card file prep failed"

c = CardClient(host=SSH_HOST, remote_cmd=REMOTE)
try:
    c.wait_ready()
    tok = Tokenizer(read_tokenizer_kv(GGUF_PATH))
    c.set_eos(tok.eos_id)

    # --- gate 1: greedy correctness + speed ---
    c.ingest(tok.encode(PROMPT), verbose=False)
    c._send("G 1")
    first = int(c._read_line())
    assert c._read_line().startswith("END")
    t0 = time.time()
    c._send("G 48")
    n = 0
    while True:
        line = c._read_line()
        if line.isdigit():
            n += 1
        elif line.startswith("END"):
            break
    rate = n / (time.time() - t0)
    greedy_ok = first == EXPECT_TOP1
    print(f"[greedy] top-1={first} ({ascii(tok.id2tok[first])}) "
          f"{'MATCH' if greedy_ok else 'FAIL'}; "
          f"decode {rate:.2f} tok/s", flush=True)

    # --- gate 2: temperature sampling smoke ---
    runs = []
    for r in range(2):
        c._send("R")
        assert c._read_line() == "OK"
        c.ingest(tok.encode(PROMPT), verbose=False)
        c._send("G 32 0.8")
        ids = []
        while True:
            line = c._read_line()
            if line.isdigit():
                ids.append(int(line))
            elif line.startswith("END"):
                break
        assert all(0 <= i < N_VOCAB for i in ids), f"token id out of range: {ids}"
        runs.append(ids)
        print(f"[temp=0.8 run {r + 1}] {len(ids)} ids, first 8: {ids[:8]}",
              flush=True)
    differ = runs[0] != runs[1]
    print(f"[temp=0.8] two runs differ: {differ}", flush=True)

    ok = greedy_ok and differ and rate > 3.0
    print(f"GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)
finally:
    c.close()
