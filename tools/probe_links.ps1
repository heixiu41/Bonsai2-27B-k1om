# probe_links.ps1 -- compile intr_probe.c once per TRY_<name> switch and record
# whether the k1om LINK succeeds.  The linker is the only ground truth on KNC:
# several intrinsics are declared in zmmintrin.h but lower to runtime symbols
# the MPSS libraries do not export.
param(
    [string]$Src  = "$PSScriptRoot\..\src\probes\intr_probe.c",
    [string]$OutDir = 'D:\zigtest\k1om\probe',
    [switch]$Only,
    [string[]]$Names
)

$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT         = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$candidates = @(
    'permutevar_epi32','permutexvar_epi32','mask_permutexvar_epi32','mask_permutevar_epi32',
    'shuffle_epi8','swizzle_epi32','shuffle_epi32','permute4f128_epi32','shuffle_i32x4',
    'mask_permutevar_ps','permutevar_ps',
    'unpacklo_epi8','unpackhi_epi8','unpacklo_epi16','unpackhi_epi16',
    'unpacklo_epi32','unpackhi_epi32','unpacklo_epi64','unpackhi_epi64',
    'cvtepu8_epi32','cvtepu16_epi32','cvtepu8_epi16','cvtepi8_epi32','cvtepi16_epi32',
    'slli_epi16','srli_epi16','slli_epi32','srli_epi32','srai_epi32','srlv_epi32','sllv_epi32',
    'or_si512','xor_si512','andnot_si512','sub_epi32','mullo_epi32','mullo_epi16',
    'madd_epi16','maddubs_epi16','abs_epi32',
    'mask_blend_epi32','mask_mov_epi32','mask_set1_epi32','maskz_set1_epi32','movm_epi32',
    'maskz_load_epi32','mask_load_epi32','store_epi32',
    'cvtepi32_ps','cvt_roundepi32_ps',
    # round 2
    'loadu_si512','maskz_loadu_epi32','mask_loadu_epi32','storeu_si512',
    'extload_epi32','extload_epi32_u8','mask_extload_epi32_u8',
    'extloadunpacklo_epi32','extloadunpackhi_epi32','extloadunpacklo_u8','extloadunpackhi_u8','mask_extloadunpacklo_epi32','stream_load_si512',
    'loadu_ps','storeu_ps','mask_storeu_epi32',
    'alignr_epi32','permutex2var_epi32','maskz_permutexvar_epi32','maskz_permutevar_epi32',
    'permutevar_ps','mask_permutevar_ps','maskz_permutevar_ps',
    'sub_epi8','set1_epi8','set1_epi16','setzero_si512','set_epi32','set_epi64',
    'load_epi32','maskz_mov_epi32','max_epi32','min_epi32',
    'cmpeq_epi32_mask','cmplt_epi32_mask','i32extgather_epi32','slli_epi64',
    'add_epi32','bslli_epi128','bsrli_epi128'
)
if ($Only) { $candidates = $Names }

$results = @()
$pass = @()
$fail = @()
foreach ($n in $candidates) {
    $exe = Join-Path $OutDir "p_$n"
    $err = & $icc -mmic -O2 -std=c11 -D__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES "-DTRY_$n" $Src -o $exe -lm 2>&1 | Out-String
    $code = $LASTEXITCODE
    $undef = ($err -split "`n" | Select-String -Pattern 'undefined reference to|error:' | ForEach-Object { $_.Line.Trim() }) -join ' ; '
    if ($code -eq 0) {
        $pass += $n
        $results += [pscustomobject]@{ Name=$n; Link='PASS'; Detail='' }
    } else {
        $fail += $n
        if (-not $undef) { $undef = ($err -split "`n" | Where-Object { $_.Trim() } | Select-Object -Last 3) -join ' ; ' }
        $results += [pscustomobject]@{ Name=$n; Link='FAIL'; Detail=$undef }
    }
}

$results | Format-Table -AutoSize -Wrap
""
"PASS ($($pass.Count)): " + ($pass -join ', ')
"FAIL ($($fail.Count)): " + ($fail -join ', ')
""
"PASS_LIST=" + ($pass -join ' ')
$pass -join ' ' | Out-File -Encoding ascii (Join-Path $OutDir 'pass.txt')

