"""One-shot driver: exercises chat.py's tokenizer + card client end to end."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# locate chat.py in the repo's scripts/
repo_scripts = r"C:\Users\Administrator\Desktop\Bonsai2-27B-k1om\scripts"
sys.path.insert(0, repo_scripts)
import chat

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

question = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 24

print(f"[*] question: {question!r}")
if not chat.ensure_card_files(chat.SSH_HOST, chat.GGUF_PATH):
    sys.exit(1)
meta = chat.read_tokenizer_kv(chat.GGUF_PATH)
tok = chat.Tokenizer(meta)
print(f"[*] tokenizer ready; eos={tok.eos_id}")

# reuse the REPL's own turn-building helpers if exposed, else replicate:
im_start = tok.id2tok.index("<|im_start|>")
im_end = tok.id2tok.index("<|im_end|>")
block = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(block)
print(f"[*] prompt block: {len(ids)} tokens")

client = chat.CardClient()
client.wait_ready()
client._send(f"E {tok.eos_id}")
client._expect_ok()
client._send("R")
client._expect_ok()
print(f"[*] ingesting prompt ...", flush=True)
client._send("P " + " ".join(map(str, ids)))
client._expect_ok()
print(f"[*] generating (max {max_tokens}) ...", flush=True)
client._send(f"G {max_tokens}")

dec = chat.StreamDecoder(tok) if hasattr(chat, "StreamDecoder") else None
gen = []
import time
t0 = time.time()
while True:
    line = client._read_line()
    if line.startswith("END"):
        print(f"\n[*] {line}  ({time.time()-t0:.1f}s)")
        break
    if line.isdigit():
        gen.append(int(line))
        if dec is not None:
            sys.stdout.write(dec.feed(int(line)))
        else:
            sys.stdout.write(tok.decode([int(line)]))
        sys.stdout.flush()
    else:
        print(f"\n[card] {line}")
print(f"[*] decoded: {tok.decode(gen)!r}")
client._send("R")
client._expect_ok()
client.p.kill()
print("[*] OK")
