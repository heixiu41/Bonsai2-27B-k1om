# pq2isa_scan.ps1 -- probe every candidate k1om intrinsic individually.
# For each test: write a one-function .c file, compile with -mmic -S, and look
# at the generated assembly.  A real instruction appears as a mnemonic; an
# unimplemented intrinsic appears as `call _mm512_...` (undefined at link time).
# Results are written to D:\zigtest\k1om\isa_scan.txt
$icc = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\bin\intel64_mic\icc.exe'
$env:MIC_SYSROOT = 'C:\Program Files\Intel\MPSS'
$env:MIC_LD_LIBRARY_PATH = 'C:\Program Files\Intel\MPSS\k1om-mpss-linux\usr\lib64'
$tmp = 'D:\zigtest\k1om\scan'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$out = 'D:\zigtest\k1om\isa_scan.txt'

# name => function body (C)
$tests = [ordered]@{
 'i32_permutevar'      = '__m512i f(__m512i a,__m512i b){return _mm512_permutevar_epi32(a,b);}'
 'i32_mask_permutevar' = '__m512i f(__m512i a,__m512i b){return _mm512_mask_permutevar_epi32(a,(__mmask16)0xFFFF,a,b);}'
 'i32_permute4f128'    = '__m512i f(__m512i a){return _mm512_permute4f128_epi32(a,0x1B);}'
 'ps_permute4f128'     = '__m512 f(__m512 a){return _mm512_permute4f128_ps(a,0x1B);}'
 'i32_swizzle'         = '__m512i f(__m512i a){return _mm512_swizzle_epi32(a,(_MM_SWIZZLE_ENUM)0x0000);}'
 'ps_swizzle'          = '__m512 f(__m512 a){return _mm512_swizzle_ps(a,(_MM_SWIZZLE_ENUM)0x0000);}'
 'i32_shuffle'         = '__m512i f(__m512i a){return _mm512_shuffle_epi32(a,(_MM_PERM_ENUM)0x1B);}'
 'ps_shuffle'          = '__m512 f(__m512 a,__m512 b){return _mm512_shuffle_ps(a,b,0x1B);}'
 'i32_alignr'          = '__m512i f(__m512i a,__m512i b){return _mm512_alignr_epi32(a,b,3);}'
 'i64_alignr'          = '__m512i f(__m512i a,__m512i b){return _mm512_alignr_epi64(a,b,3);}'
 'i8_alignr'           = '__m512i f(__m512i a,__m512i b){return _mm512_alignr_epi8(a,b,3);}'
 'i32_ternarylogic'    = '__m512i f(__m512i a,__m512i b){return _mm512_ternarylogic_epi32(a,b,a,0xCA);}'
 'i32_shuffle_i32x4'   = '__m512i f(__m512i a,__m512i b){return _mm512_shuffle_i32x4(a,b,0x1B);}'
 'i32_permutexvar'     = '__m512i f(__m512i a,__m512i b){return _mm512_permutexvar_epi32(a,b);}'
 'i32_permutex2var'    = '__m512i f(__m512i a,__m512i b){return _mm512_permutex2var_epi32(a,b,a);}'
 'i16_permutexvar'     = '__m512i f(__m512i a,__m512i b){return _mm512_permutexvar_epi16(a,b);}'
 'i8_permutexvar'      = '__m512i f(__m512i a,__m512i b){return _mm512_permutexvar_epi8(a,b);}'
 'i128_bslli'          = '__m512i f(__m512i a){return _mm512_bslli_epi128(a,3);}'
 'i128_bsrli'          = '__m512i f(__m512i a){return _mm512_bsrli_epi128(a,3);}'
 'i32_rolv'            = '__m512i f(__m512i a,__m512i b){return _mm512_rolv_epi32(a,b);}'
 'i32_ror'             = '__m512i f(__m512i a){return _mm512_ror_epi32(a,3);}'
 'i32_sllv'            = '__m512i f(__m512i a,__m512i b){return _mm512_sllv_epi32(a,b);}'
 'i32_srlv'            = '__m512i f(__m512i a,__m512i b){return _mm512_srlv_epi32(a,b);}'
 'i32_slli'            = '__m512i f(__m512i a){return _mm512_slli_epi32(a,3);}'
 'i32_srli'            = '__m512i f(__m512i a){return _mm512_srli_epi32(a,3);}'
 'i32_srai'            = '__m512i f(__m512i a){return _mm512_srai_epi32(a,3);}'
 'i16_slli'            = '__m512i f(__m512i a){return _mm512_slli_epi16(a,3);}'
 'i16_srli'            = '__m512i f(__m512i a){return _mm512_srli_epi16(a,3);}'
 'i16_sllv'            = '__m512i f(__m512i a,__m512i b){return _mm512_sllv_epi16(a,b);}'
 'i32_mullo'           = '__m512i f(__m512i a,__m512i b){return _mm512_mullo_epi32(a,b);}'
 'i32_mulhi'           = '__m512i f(__m512i a,__m512i b){return _mm512_mulhi_epi32(a,b);}'
 'i32_mulhi_epu'       = '__m512i f(__m512i a,__m512i b){return _mm512_mulhi_epu32(a,b);}'
 'i16_mullo'           = '__m512i f(__m512i a,__m512i b){return _mm512_mullo_epi16(a,b);}'
 'i16_madd'            = '__m512i f(__m512i a,__m512i b){return _mm512_madd_epi16(a,b);}'
 'i16_maddubs'         = '__m512i f(__m512i a,__m512i b){return _mm512_maddubs_epi16(a,b);}'
 'i16_mulhrs'          = '__m512i f(__m512i a,__m512i b){return _mm512_mulhrs_epi16(a,b);}'
 'i32_add'             = '__m512i f(__m512i a,__m512i b){return _mm512_add_epi32(a,b);}'
 'i32_sub'             = '__m512i f(__m512i a,__m512i b){return _mm512_sub_epi32(a,b);}'
 'i32_and'             = '__m512i f(__m512i a,__m512i b){return _mm512_and_si512(a,b);}'
 'i32_or'              = '__m512i f(__m512i a,__m512i b){return _mm512_or_si512(a,b);}'
 'i32_xor'             = '__m512i f(__m512i a,__m512i b){return _mm512_xor_si512(a,b);}'
 'i32_andnot'          = '__m512i f(__m512i a,__m512i b){return _mm512_andnot_si512(a,b);}'
 'i8_add'              = '__m512i f(__m512i a,__m512i b){return _mm512_add_epi8(a,b);}'
 'i8_sub'              = '__m512i f(__m512i a,__m512i b){return _mm512_sub_epi8(a,b);}'
 'i8_adds'             = '__m512i f(__m512i a,__m512i b){return _mm512_adds_epi8(a,b);}'
 'i8_subs'             = '__m512i f(__m512i a,__m512i b){return _mm512_subs_epi8(a,b);}'
 'i8_adds_epu'         = '__m512i f(__m512i a,__m512i b){return _mm512_adds_epu8(a,b);}'
 'i8_subs_epu'         = '__m512i f(__m512i a,__m512i b){return _mm512_subs_epu8(a,b);}'
 'i8_abs'              = '__m512i f(__m512i a){return _mm512_abs_epi8(a);}'
 'i16_abs'             = '__m512i f(__m512i a){return _mm512_abs_epi16(a);}'
 'i16_add'             = '__m512i f(__m512i a,__m512i b){return _mm512_add_epi16(a,b);}'
 'i16_sub'             = '__m512i f(__m512i a,__m512i b){return _mm512_sub_epi16(a,b);}'
 'i8_avg'              = '__m512i f(__m512i a,__m512i b){return _mm512_avg_epu8(a,b);}'
 'i16_avg'             = '__m512i f(__m512i a,__m512i b){return _mm512_avg_epu16(a,b);}'
 'i8_sad'              = '__m512i f(__m512i a,__m512i b){return _mm512_sad_epu8(a,b);}'
 'i8_dbsad'            = '__m512i f(__m512i a,__m512i b){return _mm512_dbsad_epu8(a,b,0);}'
 'i8_max_epu'          = '__m512i f(__m512i a,__m512i b){return _mm512_max_epu8(a,b);}'
 'i8_min_epu'          = '__m512i f(__m512i a,__m512i b){return _mm512_min_epu8(a,b);}'
 'i16_packs'           = '__m512i f(__m512i a,__m512i b){return _mm512_packs_epi16(a,b);}'
 'i16_packus'          = '__m512i f(__m512i a,__m512i b){return _mm512_packus_epi16(a,b);}'
 'i32_cvtepi32_ps'     = '__m512 f(__m512i a){return _mm512_cvtepi32_ps(a);}'
 'i32_cvtps_epi32'     = '__m512i f(__m512 a){return _mm512_cvtps_epi32(a);}'
 'i32_cvtepu32_ps'     = '__m512 f(__m512i a){return _mm512_cvtepu32_ps(a);}'
 'ph_cvtph_ps'         = '__m512 f(__m256i a){return _mm512_cvtph_ps(a);}'
 'extload_epi32_u8'    = '__m512i f(void const* p){return _mm512_extload_epi32(p,_MM_UPCONV_EPI32_UINT8,_MM_BROADCAST32_NONE,_MM_HINT_NONE);}'
 'extload_ps_u8'       = '__m512 f(void const* p){return _mm512_extload_ps(p,_MM_UPCONV_PS_UINT8,_MM_BROADCAST32_NONE,_MM_HINT_NONE);}'
 'extload_ps_f16'      = '__m512 f(void const* p){return _mm512_extload_ps(p,_MM_UPCONV_PS_FLOAT16,_MM_BROADCAST32_NONE,_MM_HINT_NONE);}'
 'extload_ps_s16'      = '__m512 f(void const* p){return _mm512_extload_ps(p,_MM_UPCONV_PS_SINT16,_MM_BROADCAST32_NONE,_MM_HINT_NONE);}'
 'extloadunpacklo_u8'  = '__m512i f(void const* p,__m512i a){return _mm512_extloadunpacklo_epi32(a,p,_MM_UPCONV_EPI32_UINT8,_MM_HINT_NONE);}'
 'extloadunpackhi_u8'  = '__m512i f(void const* p,__m512i a){return _mm512_extloadunpackhi_epi32(a,p,_MM_UPCONV_EPI32_UINT8,_MM_HINT_NONE);}'
 'gather_epi32_u16'    = '__m512i f(__m512i o,void const* p){return _mm512_i32extgather_epi32(o,p,_MM_UPCONV_EPI32_UINT16,1,_MM_HINT_NONE);}'
 'gather_ps_f16'       = '__m512 f(__m512i o,void const* p){return _mm512_i32extgather_ps(o,p,_MM_UPCONV_PS_FLOAT16,1,_MM_HINT_NONE);}'
 'gather_ps_none'      = '__m512 f(__m512i o,void const* p){return _mm512_i32extgather_ps(o,p,_MM_UPCONV_PS_NONE,1,_MM_HINT_NONE);}'
 'ps_fmadd233'         = '__m512 f(__m512 a,__m512 b){return _mm512_fmadd233_ps(a,b);}'
 'i32_fmadd233'        = '__m512i f(__m512i a,__m512i b){return _mm512_fmadd233_epi32(a,b);}'
 'ps_exp2a23'          = '__m512 f(__m512 a){return _mm512_exp2a23_round_ps(a,_MM_FROUND_CUR_DIRECTION);}'
 'ps_scale'            = '__m512 f(__m512 a,__m512i b){return _mm512_scale_ps(a,b);}'
 'ps_rsqrt23'          = '__m512 f(__m512 a){return _mm512_rsqrt23_ps(a);}'
 'ps_maxabs'           = '__m512 f(__m512 a,__m512 b){return _mm512_maxabs_ps(a,b);}'
 'ps_gmaxabs'          = '__m512 f(__m512 a,__m512 b){return _mm512_gmaxabs_ps(a,b);}'
 'ps_addsets'          = '__m512 f(__m512 a,__m512 b,__mmask16* k){return _mm512_addsets_ps(a,b,k);}'
 'i32_subsetb'         = '__m512i f(__m512i a,__m512i b,__mmask16* k){return _mm512_subsetb_epi32(a,b,k);}'
 'i32_addsetc'         = '__m512i f(__m512i a,__m512i b,__mmask16* k){return _mm512_addsetc_epi32(a,b,k);}'
 'i32_lzcnt'           = '__m512i f(__m512i a){return _mm512_lzcnt_epi32(a);}'
 'i32_conflict'        = '__m512i f(__m512i a){return _mm512_conflict_epi32(a);}'
 'i32_popcnt'          = '__m512i f(__m512i a){return _mm512_popcnt_epi32(a);}'
 'i8_multishift'       = '__m512i f(__m512i a,__m512i b){return _mm512_multishift_epi64_epi8(a,b);}'
 'i8_shuffle_epi8'     = '__m512i f(__m512i a,__m512i b){return _mm512_shuffle_epi8(a,b);}'
 'i8_unpacklo'         = '__m512i f(__m512i a,__m512i b){return _mm512_unpacklo_epi8(a,b);}'
 'i16_unpacklo'        = '__m512i f(__m512i a,__m512i b){return _mm512_unpacklo_epi16(a,b);}'
 'i32_unpacklo'        = '__m512i f(__m512i a,__m512i b){return _mm512_unpacklo_epi32(a,b);}'
 'i8_broadcastb'       = '__m512i f(__m128i a){return _mm512_broadcastb_epi8(a);}'
 'i32_cmpgt_mask'      = '__mmask16 f(__m512i a,__m512i b){return _mm512_cmpgt_epi32_mask(a,b);}'
 'i8_cmpgt_mask'       = '__mmask64 f(__m512i a,__m512i b){return _mm512_cmpgt_epi8_mask(a,b);}'
 'ps_load'             = '__m512 f(void const* p){return _mm512_load_ps(p);}'
 'i32_load'            = '__m512i f(void const* p){return _mm512_load_epi32(p);}'
 'i32_store'           = 'void f(void* p,__m512i a){_mm512_store_epi32(p,a);}'
 'i32_set1_epi8'       = '__m512i f(void){return _mm512_set1_epi8(3);}'
 'i32_set1_epi16'      = '__m512i f(void){return _mm512_set1_epi16(3);}'
}

$res = @()
foreach($k in $tests.Keys){
  $src = "#include <immintrin.h>`n#pragma intel optimization_level 0`n" + $tests[$k] + "`n"
  $cf  = Join-Path $tmp 'one.c'
  $sf  = Join-Path $tmp 'one.s'
  [System.IO.File]::WriteAllText($cf, $src)
  if(Test-Path $sf){ Remove-Item $sf -Force }
  $o = & $icc -mmic -O1 -std=c11 -D__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES -S $cf -o $sf 2>&1 | Out-String
  $status = '?'
  $detail = ''
  if($o -match 'error #13393' -or $o -match 'catastrophic error'){
     $status = 'REJECTED-AT-COMPILE'
     $mm = [regex]::Match($o,'Opcode unsupported on target architecture: (\S+)')
     if($mm.Success){ $detail = $mm.Groups[1].Value }
  } elseif($o -match 'error'){
     $status = 'COMPILE-ERROR'
     $detail = ([regex]::Match($o,'error[^\r\n]*')).Value
  } else {
     $s = Get-Content $sf -Raw
     $calls = [regex]::Matches($s,'call\s+(_mm512_\w+)') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
     $mn = [regex]::Matches($s,'^\s+(v?[a-z][a-z0-9]+)\s','Multiline') | ForEach-Object { $_.Groups[1].Value } |
           Where-Object { $_ -notmatch '^(push|pop|mov|sub|add|ret|lea|nop|leave|and|or|cmp|jmp|j|test|xor|not|shl|shr|sar|imul|idiv|call|inc|dec|endbr|vmov|vzero)' } | Sort-Object -Unique
     if($calls.Count -gt 0){ $status = 'NOT-IMPLEMENTED(call)' ; $detail = ($calls -join ',') }
     else { $status = 'OK' ; $detail = ($mn -join ' ') }
  }
  $line = ('{0,-22} {1,-24} {2}' -f $k, $status, $detail)
  $res += $line
  Write-Output $line
}
$res | Set-Content -Path $out
Write-Output ("`nwrote " + $out)
