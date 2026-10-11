#!/bin/sh
export LD_LIBRARY_PATH=/tmp
for t in 1 4 16 32 60 122 244; do
  echo "THREADS=$t"
  OMP_NUM_THREADS=$t /tmp/llama2_k1om /tmp/mistral7b_q80.bin -z /tmp/tokenizer.bin -n 12 -i 'hi' 2>&1 | grep 'tok/s'
done
