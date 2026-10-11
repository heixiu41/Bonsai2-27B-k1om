<#
build_rma.ps1 -- build the SCIF RMA transfer pair (card sink + Windows sender).

  card : tools/scif_rma_sink      icc -mmic ... -lscif   -> EM_K1OM (0xB5)
  host : tools/scif_rma_send.exe  zig cc + MPSS Windows SDK header/import lib

With -Probe it also builds the feasibility probes (rma_probe_sink / _send).

Everything is anchored to the MPSS 3.8.6 install, not to remembered flags:
  card sysroot : "C:\Program Files\Intel\MPSS"          (MIC_SYSROOT)
  card header  : ...\k1om-mpss-linux\usr\include\scif.h
  host header  : "C:\Program Files\Intel\MPSS\sdk\include\scif.h"
  host import  : "C:\Program Files\Intel\MPSS\sdk\lib\lib_uSCIF.lib"
#>
param([switch]$Probe)

# Native tools write warnings to stderr; PowerShell 5.1 turns those into error
# records, so keep 'Continue' and check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = 'Continue'

$tools = $PSScriptRoot
$icc   = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$zig   = 'D:\zig\zig-x86_64-windows-0.17.0\zig.exe'
$mpss  = 'C:\Program Files\Intel\MPSS'
$sdkinc = Join-Path $mpss 'sdk\include'
$usclib = Join-Path $mpss 'sdk\lib\lib_uSCIF.lib'

foreach ($p in @($icc, $zig, (Join-Path $sdkinc 'scif.h'), $usclib)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing build dependency: $p" }
}

$env:MIC_SYSROOT         = $mpss
$env:MIC_LD_LIBRARY_PATH = Join-Path $mpss 'k1om-mpss-linux\usr\lib64'

function Invoke-Native([string]$exe, [string[]]$arglist, [string]$what) {
    $out = & $exe @arglist 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $out | ForEach-Object { Write-Host "  $_" }
        throw "$what failed (exit $code)"
    }
    $out | Where-Object { "$_" -match '\berror\b' } | ForEach-Object { Write-Host "  $_" }
}

function Build-Card([string]$src, [string]$out) {
    Write-Host "icc   $src -> $out"
    Invoke-Native $icc @('-mmic', '-O2', '-std=c11', (Join-Path $tools $src),
                         '-o', (Join-Path $tools $out), '-lscif', '-lpthread') "icc $src"
    $b = [System.IO.File]::ReadAllBytes((Join-Path $tools $out))
    $em = [BitConverter]::ToUInt16($b, 18)
    if ($em -ne 0xB5) { throw "$out is not a k1om ELF (e_machine=0x$('{0:X}' -f $em))" }
}

function Build-Host([string]$src, [string]$out) {
    Write-Host "zig   $src -> $out"
    Invoke-Native $zig @('cc', '-O2', "-I$sdkinc", '-o', (Join-Path $tools $out),
                         (Join-Path $tools $src), $usclib) "zig cc $src"
}

Build-Card 'scif_rma_sink.c' 'scif_rma_sink'
Build-Host 'scif_rma_send.c' 'scif_rma_send.exe'

if ($Probe) {
    Build-Card 'rma_probe_sink.c' 'rma_probe_sink'
    Build-Host 'rma_probe_send.c' 'rma_probe_send.exe'
    Build-Card 'rma_flushbench.c' 'rma_flushbench'
}

Get-ChildItem (Join-Path $tools 'scif_rma_*'), (Join-Path $tools 'rma_probe_*') -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -ne '.c' -and $_.Extension -ne '.h' } |
    Select-Object Name, Length | Format-Table -AutoSize
