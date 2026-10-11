#!/bin/sh
export LD_LIBRARY_PATH=/tmp
echo "=== Q8_0 Benchmark Start ==="
for t in 244 122 60 32 16 4; do
  echo "--- THREADS=$t ---"
  OMP_NUM_THREADS=$t /tmp/llama2_k1om /tmp/mistral7b_q80.bin -z /tmp/tokenizer.bin -n 16 -i 'hi' 2>&1 | grep 'tok/s'
  echo ""
done
echo "=== DONE ==="
