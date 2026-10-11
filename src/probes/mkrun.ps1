# mkrun.ps1 -- cross-compile a k1om probe, upload it, run it on the card.
#   .\mkrun.ps1 probe_a            # default flags: -mmic -O2 -std=c11
#   .\mkrun.ps1 probe_a "-O0 -no-vec"
#
# $Name is resolved relative to THIS script's directory, so the probes work
# wherever the bonsai2 tree is checked out.  Core sources (model.c etc.) live in
# ..\ , so pass a path for those, e.g. .\mkrun.ps1 ..\test_forward
param(
    [Parameter(Mandatory=$true)][string]$Name,
    [string]$Extra = ""
)
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
$here = $PSScriptRoot
if (-not $here) { $here = (Get-Location).Path }
$out  = "D:\zigtest\k1om\$(Split-Path $Name -Leaf)"

$args = @('-mmic','-O2','-std=c11','-qopenmp')
if ($Extra) { $args += $Extra.Split(' ') }
$args += @("$here\$Name.c", '-o', $out, '-lm')

Write-Host "--- compiling $Name : $($args -join ' ')" -ForegroundColor Cyan
& $icc @args 2>&1 | Where-Object { $_ -notmatch 'unrecognized OpenMP' -and $_ -notmatch '^\s*#pragma omp' -and $_ -notmatch '^\s*\^' } | Select-Object -Last 30
if ($LASTEXITCODE -ne 0) { Write-Host "COMPILE FAILED exit=$LASTEXITCODE" -ForegroundColor Red; exit 1 }

$key = "$env:USERPROFILE\.ssh\id_rsa_mic"
$kh  = "-o UserKnownHostsFile=`"$env:USERPROFILE\.ssh\known_hosts`" -o StrictHostKeyChecking=no -o LogLevel=ERROR"
Write-Host "--- uploading" -ForegroundColor Cyan
scp -O -c chacha20-poly1305@openssh.com -i "$key" $kh $out root@192.168.1.100:/tmp/ 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "SCP FAILED exit=$LASTEXITCODE" -ForegroundColor Red; exit 1 }

Write-Host "--- running $Name on card" -ForegroundColor Cyan
ssh -i "$key" $kh root@192.168.1.100 "chmod +x /tmp/$Name && export LD_LIBRARY_PATH=/tmp && /tmp/$Name"
