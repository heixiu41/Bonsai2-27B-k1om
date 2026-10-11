# run.ps1 -- build the INSTRUMENTED k1om binary and run it on the card.
#
# The shipped sources in ..\src\ contain no probes. This directory holds
# instrumented copies (model.c, main_k1om.c) plus prof2.h, which add region
# timers behind -DBONSAI_PROF2. Build without that macro and these files
# behave exactly like the shipped ones.
#
# Usage:  .\run.ps1 [-Threads 244] [-Extra ""]
param(
    [int]$Threads = 244,
    [string]$Extra = ""
)
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
$here = $PSScriptRoot
$src  = Join-Path $here '..\src'
$out  = 'D:\zigtest\k1om\prof2'

Write-Host "--- building instrumented (BONSAI_PROF2) from $here" -ForegroundColor Cyan
$args = @('-mmic','-O2','-std=c11','-qopenmp','-DBONSAI_PROF2',
          '-D_FILE_OFFSET_BITS=64','-D_LARGEFILE_SOURCE')
if ($Extra) { $args += $Extra.Split(' ') }
$args += @("$here\src\main_k1om.c", "$here\src\model.c", "$src\gguf.c", '-I', "$here\src", '-I', $src,
           '-o', $out, '-lm')
& $icc @args
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED exit=$LASTEXITCODE" -ForegroundColor Red; exit 1 }

$key = "$env:USERPROFILE\.ssh\id_rsa_mic"
$kh  = "-o UserKnownHostsFile=`"$env:USERPROFILE\.ssh\known_hosts`" -o StrictHostKeyChecking=no -o LogLevel=ERROR"
scp -O -c chacha20-poly1305@openssh.com -i "$key" $kh $out root@192.168.1.100:/tmp/ | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "SCP FAILED" -ForegroundColor Red; exit 1 }

Write-Host "--- running on card, $Threads threads (budget goes to stderr)" -ForegroundColor Cyan
ssh -i "$key" $kh root@192.168.1.100 "chmod +x /tmp/prof2 && export LD_LIBRARY_PATH=/tmp && /tmp/prof2 /tmp/bonsai.gguf forward $Threads 2>&1"
