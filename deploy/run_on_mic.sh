#!/bin/sh
cd /tmp
export LD_LIBRARY_PATH=/tmp
export OMP_NUM_THREADS=1
echo "=== Testing with 1 thread ==="
./llama2_k1om mistral7b_q80.bin -n 16 -i hi -z tokenizer.bin 2>&1
echo "RC=$?"
echo ""
echo "=== dmesg tail ==="
dmesg | tail -20
