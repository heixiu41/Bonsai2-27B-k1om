# llama2-knc-k1om run script (run inference on Xeon Phi 7120P)
# Usage: powershell -ExecutionPolicy Bypass -File .\deploy\run_k1om.ps1
#        [-Threads 244] [-Model mistral7b_q80.bin] [-Prompt "..."] [-Tokens 128]
param(
    [int]$Threads = 244,
    [string]$Model  = "stories110M.bin",
    [string]$Prompt = "Once upon a time",
    [string]$Tokens = "128"
)

$ErrorActionPreference = "Stop"
$KEY    = Join-Path $env:USERPROFILE ".ssh\id_rsa_mic"
$MIC_IP = "192.168.1.100"
$REMOTE = "/tmp"

$cmd = "export LD_LIBRARY_PATH=$REMOTE; export OMP_NUM_THREADS=$Threads; $REMOTE/llama2_k1om $REMOTE/$Model -z $REMOTE/tokenizer.bin -n $Tokens -i '$Prompt'"

Write-Host "=== Run on 7120P ==="
Write-Host "CMD: $cmd"
Write-Host ""

ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@$MIC_IP $cmd
exit $LASTEXITCODE
