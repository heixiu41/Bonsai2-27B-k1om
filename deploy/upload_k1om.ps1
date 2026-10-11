# llama2-knc-k1om deploy script: upload binary + model to Xeon Phi 7120P
# Usage: powershell -ExecutionPolicy Bypass -File .\deploy\upload_k1om.ps1
#        [-Bin <path>] [-Model <model.bin>]
# 7120P: IP=192.168.1.100, SSH key %USERPROFILE%\.ssh\id_rsa_mic
# NOTE: BusyBox has no SFTP; scp needs the -O (legacy) flag.

param(
    [string]$Bin   = "",   # default: build\llama2_k1om
    [string]$Model = ""    # default: models\stories110M.bin
)

$ErrorActionPreference = "Stop"
$PROJ   = Split-Path -Parent $PSScriptRoot
$KEY    = Join-Path $env:USERPROFILE ".ssh\id_rsa_mic"
$MIC_IP = "192.168.1.100"
$REMOTE = "/tmp"

if (-not $Bin)   { $Bin = Join-Path $PROJ "build\llama2_k1om" }
if (-not $Model) { $Model = Join-Path $PROJ "models\stories110M.bin" }

foreach ($f in @($Bin, $Model)) {
    if (-not (Test-Path $f)) { Write-Error "File not found: $f"; exit 1 }
}

$TOKENIZER = Join-Path $PROJ "models\tokenizer.bin"
$OMP_LIB   = Join-Path $PROJ "toolchain\k1om_mkl_lib\libiomp5.so"   # needed on first deploy

Write-Host "=== Upload to 7120P ($MIC_IP) ==="
Write-Host "Binary  : $Bin"
Write-Host "Model   : $Model"

scp -O -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -i $KEY `
    $Bin, $Model, $TOKENIZER root@${MIC_IP}:${REMOTE}/
if ($LASTEXITCODE -ne 0) { Write-Error "Upload failed"; exit 1 }

if (Test-Path $OMP_LIB) {
    Write-Host "First deploy: uploading OpenMP runtime libiomp5.so ..."
    scp -O -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -i $KEY `
        $OMP_LIB root@${MIC_IP}:${REMOTE}/
}

Write-Host ""
Write-Host "Upload done. To run remotely:"
Write-Host "  ssh -i `"$KEY`" root@$MIC_IP `"export LD_LIBRARY_PATH=$REMOTE; OMP_NUM_THREADS=244 $REMOTE/llama2_k1om $REMOTE/stories110M.bin -z $REMOTE/tokenizer.bin -i 'Once upon a time'`""
