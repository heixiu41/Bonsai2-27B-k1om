/*
 * pq2_0.h — PQ2_0 ternary codec and KNC (k1om) matmul kernels.
 *
 * Part of the Ternary Bonsai 2 27B port to the Intel Xeon Phi 7120P.
 *
 * PQ2_0 layout (verified against PrismML-Eng/llama.cpp ggml-common.h and
 * ggml-quants.c dequantize_row_pq2_0):
 *
 *   #define QK_PQ2_0 128
 *   typedef struct { ggml_half d; uint8_t qs[32]; } block_pq2_0;   // 34 B
 *
 *   34 bytes per 128 weights = 2.125 bpw.
 *   code -> value is  {0:-1, 1:0, 2:+1, 3:+2}, i.e.  w = (q - 1) * d
 *
 * KNC notes (Xeon Phi 7120P, k1om):
 *   - k1om has NO SSE/AVX/AVX2; the vector path must use IMCI _mm512_*.
 *   - _mm512_load_ps / vmovaps requires 64-byte alignment. Any buffer fed to it
 *     must come from _mm_malloc(..., 64) or be offset by a multiple of 64 bytes.
 *   - There is no integer SIMD unit and no ternary/int8 dot product, so the
 *     scalar 2-bit unpack below is the practical decoder. popcount-based
 *     bit-slicing is reserved for the prefill (compute-bound) path.
 */
#ifndef BONSAI_PQ2_0_H
#define BONSAI_PQ2_0_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_PQ2_0        128
#define PQ2_0_BLOCK_BYTES 34

/* ------------------------------------------------------------------ */
/* fp16 -> fp32 (KNC has no hardware conversion for arbitrary halves,  */
/* so a small table-free routine; KNC does have vcvtph2ps but it is    */
/* available only via the IMCI F16C subset, which is not guaranteed on */
/* all steppings. This software path is always correct.                */
/* ------------------------------------------------------------------ */
static inline float bonsai_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                              /* +-0 */
        } else {
            /* subnormal: normalize */
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x400) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);     /* inf / nan */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v;
    v.u = bits;
    return v.f;
}

/*
 * Dequantise one PQ2_0 block (128 weights) into `out[0..127]`.
 * Byte layout inside the block: d at [0..1] (fp16 LE), codes at [2..33].
 * Bit order matches ggml: element j uses byte j/4, shift (j%4)*2.
 */
static inline void pq2_0_dequant_block(const uint8_t *blk, float *out) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    for (int j = 0; j < QK_PQ2_0; ++j) {
        const int q = (qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
        out[j] = (float)(q - 1) * d;                  /* {0:-1,1:0,2:+1,3:+2} */
    }
}

/*
 * Dequantise a whole PQ2_0 matrix into an fp32 buffer.
 *
 * `n_elems` must be a multiple of QK_PQ2_0. `out` must be 64-byte aligned if
 * it will be consumed by the IMCI kernels below.
 *
 * Returns the number of bytes written, or 0 on a size mismatch.
 */
static inline size_t pq2_0_dequant_matrix(const uint8_t *src, float *out,
                                          size_t n_elems) {
    if (n_elems % QK_PQ2_0) return 0;
    const size_t nblk = n_elems / QK_PQ2_0;
    for (size_t b = 0; b < nblk; ++b) {
        pq2_0_dequant_block(src + b * PQ2_0_BLOCK_BYTES, out + b * QK_PQ2_0);
    }
    return nblk * QK_PQ2_0 * sizeof(float);
}

/*
 * PQ2_0 matrix-vector product WITHOUT materialising the weights.
 *
 *   W is (d, n) stored as d*n/128 PQ2_0 blocks, row-major over rows of n.
 *   Computes y[i] = sum_j W[i][j] * x[j]   for i in [0, d)
 *
 * This is the streaming decoder: it keeps the model in its 2.125 bpw form and
 * expands 128 weights at a time into a small stack buffer that the IMCI kernel
 * consumes. Peak extra memory is one block per thread.
 *
 * NOTE ON ALIGNMENT: qs is read as bytes (no alignment requirement). The
 * `wf` staging buffer is declared aligned so _mm512_load_ps is safe.
 */
#ifdef __MIC__
#include <immintrin.h>

/* ======================================================================
 * Vectorised PQ2_0 block decoder for KNC (IMCI).
 *
 * Link-probed instruction budget (src/bonsai/intr_probe.c driven by
 * tools/probe_links.ps1: compiled once per -DTRY_<name> so the k1om LINKER
 * decides).  On this card zmmintrin.h is not a contract:
 *
 *   LINKABLE -- actually resolved by the MPSS libraries:
 *     _mm512_extload_epi32 (incl. _MM_UPCONV_EPI32_UINT8),
 *     _mm512_mask_extload_epi32, _mm512_load_epi32, _mm512_mask_load_epi32,
 *     _mm512_extloadunpacklo/hi_epi32 (incl. _MM_UPCONV_EPI32_UINT8),
 *     _mm512_permutevar_epi32, _mm512_mask_permutevar_epi32,
 *     _mm512_swizzle_epi32, _mm512_shuffle_epi32, _mm512_permute4f128_epi32,
 *     _mm512_alignr_epi32, _mm512_slli/srli/srai_epi32, _mm512_sllv/srlv_epi32,
 *     _mm512_and/or/xor/andnot_si512, _mm512_add/sub/mullo/abs/min/max_epi32,
 *     _mm512_mask_blend/mov/set1_epi32, _mm512_cmpeq/cmplt_epi32_mask,
 *     _mm512_i32extgather_epi32, _mm512_set1/set_epi8/16, _mm512_setzero_si512.
 *
 *   ALIGNMENT / BOUNDARY BEHAVIOUR (measured on the card, src/bonsai/pq2load2.c
 *   and pq2load3.c -- the aligned-only forms SEGFAULT, they never silently
 *   mis-read; the unaligned forms silently ZERO-FILL at a 64-byte boundary):
 *     _mm512_load_epi32 / _mm512_extload_epi32(NONE)   need 64-byte alignment
 *     _mm512_extload_epi32(UINT8)                     needs 16-byte alignment
 *     _mm512_extloadunpacklo_epi32 NONE (dword elem)  needs 4-byte alignment
 *         (offsets 1,2,3 fault; 0 and 4 work)
 *     _mm512_extloadunpacklo_epi32 UINT8              ANY byte offset, BUT it
 *         only delivers bytes [p .. end of the 64-byte aligned block containing
 *         p]; everything past that boundary comes from `old`.  With old = 0 the
 *         tail is silently ZERO -- that is what corrupted the first version of
 *         this decoder for every block with (qs & 63) > 32.
 *     _mm512_extloadunpackhi_epi32(old, p + 64, UINT8)
 *         supplies exactly that tail: it upconverts the aligned block at
 *         ((p+64) & ~63) and shifts it left by (64 - ((p+64) & 63)) lanes, i.e.
 *         into precisely the lanes the lo form left alone.
 *   lo(p) followed by hi(p+64) is therefore the correct arbitrary-offset 16-byte
 *   read; verified at byte offsets 1, 3, 53, 63 (and 2, 52, 69, 79).
 *   NOTE: a PQ2_0 payload is qs = blk+2 with a 34-byte block stride, so qs is
 *   only ever 2-byte aligned -- the NONE (packed 64-byte) form is unusable and
 *   the UINT8 (byte-per-dword) form is the only one that works, which is why
 *   this decoder widens to one byte per dword after loading.
 *
 *   DECLARED IN THE HEADER BUT **ABSENT FROM THE RUNTIME** (link fails with
 *   `undefined reference to ...`):
 *     _mm512_cvtepi32_ps   -> lowers to `_mm512_cvt_roundepi32_ps` (blocker)
 *     _mm512_unpacklo/hi_epi8/16/32/64   -- the ENTIRE unpack family
 *     _mm512_shuffle_epi8                -- vpshufb
 *     _mm512_slli_epi16, _mm512_srli_epi16
 *     _mm512_mullo_epi16, _mm512_madd_epi16, _mm512_maddubs_epi16
 *     _mm512_movm_epi32, _mm512_maskz_set1_epi32, _mm512_maskz_mov_epi32,
 *     _mm512_maskz_load_epi32, _mm512_permutexvar_epi32,
 *     _mm512_shuffle_i32x4, _mm512_permutex2var_epi32,
 *     every *loadu* / *storeu* form (k1om has no unaligned plain load).
 *
 *   REJECTED BY THE COMPILER ITSELF on this target:
 *     _mm512_cvtepu8_epi32 / _mm512_cvtepi8_epi32 / _mm512_cvtepu16_epi32
 *     ("catastrophic error: Function contains unsupported data types or
 *      intrinsics on target").
 *
 * So no textbook 2-bit expansion recipe survives here: no unpack, no vpshufb,
 * no byte->dword convert, no unaligned vector load, no int->float convert.
 * This decoder uses only the proven set:
 *
 *   1. _mm512_extloadunpacklo_epi32(zero, p, UINT8, hint) followed by
 *      _mm512_extloadunpackhi_epi32(that, p + 64, UINT8, hint) reads the 16
 *      bytes at the arbitrary address p as 16 dwords, dword i = p[i].
 *      (Either half alone is wrong: lo zero-fills past the 64-byte aligned
 *      block containing p, and hi supplies exactly that tail.)  Called twice,
 *      at qs and qs+16, to get all 32 code bytes.  This is the only working
 *      byte->dword widening on k1om; the aligned _mm512_extload_epi32 cannot
 *      be used because qs = blk+2 is never 16-byte aligned, and the NONE
 *      (packed) form of the unpack load needs 4-byte alignment, which qs does
 *      not have either.
 *   2. _mm512_permutevar_epi32(index, data) is a full cross-lane 16-entry
 *      dword gather -- measured result[i] = data[index[i] & 15] (operand order
 *      is INDEX FIRST, unlike AVX-512's _mm512_permutexvar_epi32).  Used for
 *      (a) broadcasting the 4 code bytes of a packed group to 4 lanes each, and
 *      (b) the {-1,0,1,2} float LUT that replaces the missing int->float
 *          conversion: one permutevar, no arithmetic, no bit trick.
 *   3. _mm512_srlv_epi32 with the constant shift {0,2,4,6} x4 pulls the 2-bit
 *      field out of each of the 4 code bytes.  Element 16m+e is field (e%4) of
 *      code byte 4m+(e/4); because the LUT has 16 entries holding
 *      (float)((i & 3) - 1), the usual `& 3` is not needed.
 *   4. _mm512_mul_ps applies the fp16 scale d.
 *
 * Cost: 4 loads + 8 x (permutevar + srlv + permutevar + mul_ps) + 8 stores per
 * 128 weights = 4 vector ops + 1 store per 16 floats, versus ~5 scalar ops per
 * weight for pq2_0_dequant_block.  Element-for-element exact (see
 * src/bonsai/pq2imci.c), 5.6x-11x faster single-threaded.
 *
 * CAVEAT: the lo/hi pair reads the two 64-byte aligned blocks bracketing the
 * payload, so the decoder touches memory outside the 34-byte block it decodes.
 * MEASURED on the card against a PROT_NONE guard page: the footprint is
 * [blk, blk+81], i.e. up to 48 bytes PAST the end of the block, and the
 * threshold is identical for all 64 source alignments (qs&63 swept 0..63), so
 * it does not depend on where the block starts. It also reads up to 61 bytes
 * BEFORE the block, so a block placed at offset 0 of a tight allocation is
 * equally exposed.
 *
 * The over-read is a pure READ: poisoning those bytes (0x00 vs 0xA5) changes
 * 0 of 128 decoded weights. So the risk is a SIGSEGV, never silent corruption.
 *
 * gguf.c covers this by allocating file_size + 64 and zeroing the tail; see the
 * note in gguf.h before removing that slack. Any other caller passing a
 * weights buffer must leave >= 64 bytes of readable slack.
 * ====================================================================== */

/* Broadcast the 4 code bytes of group m to 4 lanes each, in element order:
 *   { 4m x4, 4m+1 x4, 4m+2 x4, 4m+3 x4 }
 * (dword 0 is the LAST argument of _mm512_set_epi32) */
static inline __m512i pq2_imci_idx(int m) {
    const int a = 4 * m;
    return _mm512_set_epi32(a+3,a+3,a+3,a+3, a+2,a+2,a+2,a+2,
                            a+1,a+1,a+1,a+1,  a,  a,  a,  a);
}

/* field (lane % 4) -> {0,2,4,6} repeated x4 */
#define PQ2_IMCI_SHIFT() \
    _mm512_set_epi32(6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0)

/* 16-entry float LUT as raw IEEE-754 bit patterns: entry i = (float)((i&3)-1) */
#define PQ2_IMCI_LUT_BITS() \
    _mm512_set_epi32((int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u, \
                     (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u, \
                     (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u, \
                     (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u)

/* 16 bytes at the arbitrary address p -> 16 dwords, dword i = p[i].
 * lo() zero-fills past the 64-byte aligned block containing p; hi(p+64)
 * upconverts the next block and shifts it into exactly those lanes. */
#define PQ2_IMCI_LOAD16(p) \
    _mm512_extloadunpackhi_epi32( \
        _mm512_extloadunpacklo_epi32(_mm512_setzero_si512(), \
            (const void *)(p), _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE), \
        (const void *)((const uint8_t *)(p) + 64), \
        _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE)

/*
 * Dequantise one PQ2_0 block (128 weights) into out[0..127] with IMCI.
 * `out` must be 64-byte aligned (_mm512_store_ps).  `blk` has no alignment
 * requirement.  Element-for-element identical to pq2_0_dequant_block.
 */
static inline void pq2_0_dequant_block_imci(const uint8_t *blk, float *out) {
    const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
    const uint8_t *qs = blk + 2;
    const __m512i  lut = PQ2_IMCI_LUT_BITS();
    const __m512i  shf = PQ2_IMCI_SHIFT();
    const __m512   vd  = _mm512_set1_ps(d);
    const __m512i  c0  = PQ2_IMCI_LOAD16(qs);        /* dword i = qs[i],     i<16 */
    const __m512i  c1  = PQ2_IMCI_LOAD16(qs + 16);   /* dword i = qs[16+i],  i<16 */
    int m;
    for (m = 0; m < 4; ++m) {          /* weights   0.. 63 from code bytes  0..15 */
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
        const __m512i v = _mm512_srlv_epi32(b, shf);
        const __m512  g = _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut));
        _mm512_store_ps(out + 16 * m, _mm512_mul_ps(g, vd));
    }
    for (m = 0; m < 4; ++m) {          /* weights  64..127 from code bytes 16..31 */
        const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
        const __m512i v = _mm512_srlv_epi32(b, shf);
        const __m512  g = _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut));
        _mm512_store_ps(out + 64 + 16 * m, _mm512_mul_ps(g, vd));
    }
}

/* ------------------------------------------------------------------------
 * IMCI PQ2_0 kernel -- DISABLED BY DEFAULT.
 *
 * ROOT CAUSE OF THE "EXACTLY HALF" DEFECT (found and fixed, see the 8-FMA body
 * below): the original body issued only FOUR 16-wide FMAs per block, reading
 * wf+0/16/32/48 -- that is 64 of a block's 128 weights. wf+64/80/96/112 were
 * never read, so the dequantised weight block was half unused and EVERY result
 * came out exactly 2x too small. The block loop advances j by QK_PQ2_0 (128),
 * not 64, so nothing overlaps or compensates: exactly half of every row is
 * simply skipped. Measured on the card with all weights -1.0:
 *     per-accumulator lane value  -40   (40 blocks x 1 x -1; correct -80)
 *     per-accumulator scalar sum  -640  (correct -1280)
 *     row total                  -2560  (correct -5120)
 *
 * Why the earlier diagnostics pointed the wrong way:
 *   - pq2diag.c's "single 128-weight block is handled correctly, each
 *     reduce(load(wf+0/16/32/48)) == -16" is true but vacuous: those four
 *     loads are exactly the 64 weights the kernel *does* cover, so the test
 *     cannot see the missing half. Its "-1280 per accumulator expected" note
 *     was the correct expectation; the -640 it measured was the bug.
 *   - pq2vec.c's half_test() is itself buggy (H5 confirmed): `w` is allocated
 *     for ONE row (row_bytes) but the loop uses `row = w + i * row_bytes` for
 *     i in [0,d), so it reads far past its allocation. The "-40960 / -135631 /
 *     -373223" spread was heap out-of-bounds garbage, not a kernel property.
 *     Stack-vs-heap staging was never the variable (H4 is refuted).
 *   - H1/H2/H3/H6 are all refuted: the dequantiser is exact, the accumulators
 *     are live and correct, _mm512_reduce_add_ps is correct here (its only
 *     oddity is that it is constant-folded when fed a load of a constant
 *     array, which made earlier reduction-only probes pass), and no other
 *     accumulator is aliased or overwritten.
 *
 * The fix keeps the four independent accumulators (same dependency structure,
 * same final add tree, same _mm512_reduce_add_ps) but gives each one TWO
 * 16-wide FMAs per block, covering all 128 weights:
 *     sum0 += wf[  0.. 15]*x[j+  0.. 15]  and  wf[ 64.. 79]*x[j+ 64.. 79]
 *     sum1 += wf[ 16.. 31]*x[j+ 16.. 31]  and  wf[ 80.. 95]*x[j+ 80.. 95]
 *     sum2 += wf[ 32.. 47]*x[j+ 32.. 47]  and  wf[ 96..111]*x[j+ 96..111]
 *     sum3 += wf[ 48.. 63]*x[j+ 48.. 63]  and  wf[112..127]*x[j+112..127]
 *
 * Verified element-by-element against pq2_0_matvec_ref on the card over random
 * PQ2_0 data (random signed/subnormal fp16 scales, random 2-bit codes) at
 * n = 128..17408 and d = 1..64: max abs diff <= 2.2e-2 on |y|max ~2.7e4
 * (~8e-7 relative, pure float summation-order noise), 0 failing cases.
 *
 * This kernel is the DEFAULT on k1om (see the #ifndef below). Define
 * BONSAI_FORCE_SCALAR_PQ2_0 to fall back to the scalar reference, which is also
 * what the host build always uses.
 * ------------------------------------------------------------------------ */
/* ==========================================================================
 * F3 decoder + B2 scale builder -- the current default on k1om.
 * (Replaced the V12 decoder after a same-binary A/B, src/probes/pq2ab.c.)
 *
 * Same currency argument as V12 (every 512-bit op costs ~1.63 ns, integer and
 * FP share one pipe): F3 keeps V12's two scale-folded LUTs (pq2_lut_a serves
 * field 0 from the unshifted index, pq2_lut_b serves field 1; one srli by 4
 * serves fields 2/3) but restructures the block loop to ONE partial-sum
 * vector per block with pointer-increment addressing, and removes the
 * per-block SCALAR fp16 scale conversion:
 *
 *   B2 (pq2_build_drow below): one _mm512_i32extgather_epi32 with the UINT16
 *   upconvert fetches 16 raw fp16 scales at stride 34; ~8 vector ALU ops
 *   expand them to fp32 with a bit formula that is exact for normals; a mask
 *   detects subnormal / inf / nan exponents and repairs those 16 lanes
 *   through bonsai_fp16_to_fp32. The per-row scale vector (drow) is built
 *   once per row in a thread-private stack buffer inside the omp loop.
 *
 * MEASURED on the card (same binary, both orders, best-of-5, real shape mix
 * 5120x17408 x2 + 17408x5120 + 5120x10240 + 5120x6144):
 *   244 threads: V12 = 103.9 G w/s, F3 = 140.4 G w/s  -> 1.35x
 *   1 thread:    V12 =  0.91 G w/s, F3 =  1.65 G w/s  -> 1.81x
 * Exactness: 33,024 row-level checks per decoder (64 source alignments x
 * 128 unit vectors + row sum), forced subnormal / min-normal / max-normal
 * scales including the gather tail (drow bit-exact vs the scalar converter,
 * repair path exercised), multi-block odd/even rows with subnormal/inf/nan
 * scales, matvec-level vs pq2_0_matvec_ref at 1 and 244 threads:
 * 0 mismatches everywhere.
 *
 * CAVEAT: F3 changes the FMA pairing relative to V12 (single accumulator,
 * per-block partial sum), so expect ~1e-7 relative summation-order drift vs
 * the host oracle. top-1 alone does not detect that -- the l_out-63
 * hidden-state landmark must pass too.
 * ====================================================================== */

/* Table A: entry j = (j & 3) - 1, i.e. field 0 taken from the unshifted index. */
static inline __m512 pq2_lut_a(void) {
    return _mm512_castsi512_ps(_mm512_set_epi32(
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,0,(int)0xBF800000u));
}
/* Table B: entry j = ((j >> 2) & 3) - 1, i.e. field 1 from the same index. */
static inline __m512 pq2_lut_b(void) {
    return _mm512_castsi512_ps(_mm512_set_epi32(
        (int)0x40000000u,(int)0x40000000u,(int)0x40000000u,(int)0x40000000u,
        (int)0x3F800000u,(int)0x3F800000u,(int)0x3F800000u,(int)0x3F800000u,
        0,0,0,0,
        (int)0xBF800000u,(int)0xBF800000u,(int)0xBF800000u,(int)0xBF800000u));
}

/* Permute the activation so the 8 gathers consume it in the order they produce
 * weights. Must run once per matvec call, NOT per row.
 *
 * Sizing: 17408 floats = 68 KB on this thread's stack. That is the model's
 * widest folded input; `model.c` validates every width reaching the Hadamard
 * against this bound. The OpenMP stack is sufficient for it (verified by the
 * end-to-end run, which would fault otherwise). */
#define PQ2_XP_MAX 17408

/* Defined further down, but the F3 kernel falls back to it for a non-multiple-
 * of-128 width, so it needs a forward declaration here. */
static inline void pq2_0_matvec_ref(float *y, const float *x,
                                    const uint8_t *w, int n, int d);

static inline void pq2_build_xp(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    for (int b = 0; b < nblk; ++b)
        for (int h = 0; h < 2; ++h)
            for (int f = 0; f < 4; ++f)
                for (int l = 0; l < 16; ++l)
                    xp[((size_t)b * 2 + h) * 4 * 16 + f * 16 + l] =
                        x[(size_t)b * QK_PQ2_0 + 64 * h + 4 * l + f];
}

/* Per-block form of the same permutation, plus an omp wrapper distributing
 * blocks over threads. Bit-identical output: each block's 128 floats are
 * written by exactly one thread, in the same scalar order. Used by model.c's
 * mm_stage() so xp is built ONCE per staged input -- a staged input can feed
 * several matvecs (attn_qkv+attn_gate, attn_q/k/v, ffn_gate+up), and the
 * kernel used to re-run this serial ~50-170 us permutation on every mm_run
 * (401x per token, ~28 ms of serial master time per the static analysis). */
static inline void pq2_build_xp_block(const float *x, float *xp) {
    for (int h = 0; h < 2; ++h)
        for (int f = 0; f < 4; ++f)
            for (int l = 0; l < 16; ++l)
                xp[(h * 4 + f) * 16 + l] = x[64 * h + 4 * l + f];
}
static inline void pq2_build_xp_mt(const float *x, float *xp, int n) {
    const int nblk = n / QK_PQ2_0;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nblk; ++b)
        pq2_build_xp_block(x + (size_t)b * QK_PQ2_0, xp + (size_t)b * QK_PQ2_0);
}

/* B2 scale builder: drow[b] = (float)fp16 scale of block b, for a whole row.
 * One gather per 16 blocks (stride 34, UINT16 upconvert), a bit formula that
 * is exact for normal fp16, and a mask-detected scalar repair for subnormal /
 * inf / nan scales. `drow` must be 64-byte aligned (_mm512_store_ps).
 * Returns the number of 16-block groups that took the repair path. */
#define PQ2_B2_SCALE_OFF() _mm512_set_epi32( \
    15*17, 14*17, 13*17, 12*17, 11*17, 10*17, 9*17, 8*17, \
     7*17,  6*17,  5*17,  4*17,  3*17,  2*17, 1*17,     0)
static inline int pq2_build_drow(const uint8_t *row, int nblk, float *drow) {
    const __m512i off = PQ2_B2_SCALE_OFF();
    const __m512i m0 = _mm512_set1_epi32(0x8000);
    const __m512i m1 = _mm512_set1_epi32(0x7FFF);
    const __m512i m2 = _mm512_set1_epi32(0x7C00);
    const __m512i k  = _mm512_set1_epi32(0x38000000);
    const __m512i z  = _mm512_setzero_si512();
    const __m512i f  = _mm512_set1_epi32(0x7C00);
    int b = 0, repaired = 0;
    for (; b + 16 <= nblk; b += 16) {
        const __m512i h = _mm512_i32extgather_epi32(off, row + (size_t)34 * b,
                              _MM_UPCONV_EPI32_UINT16, 2, _MM_HINT_NONE);
        const __m512i e = _mm512_and_si512(h, m2);
        if ((__mmask16)(_mm512_cmpeq_epi32_mask(e, z) | _mm512_cmpeq_epi32_mask(e, f))) {
            ++repaired;
            for (int i = 0; i < 16; ++i) {
                uint16_t hh; memcpy(&hh, row + (size_t)34 * (b + i), 2);
                drow[b + i] = bonsai_fp16_to_fp32(hh);
            }
            continue;
        }
        const __m512i bits = _mm512_or_si512(
            _mm512_slli_epi32(_mm512_and_si512(h, m0), 16),
            _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(h, m1), 13), k));
        _mm512_store_ps(drow + b, _mm512_castsi512_ps(bits));
    }
    for (; b < nblk; ++b) {
        uint16_t h; memcpy(&h, row + (size_t)34 * b, 2);
        drow[b] = bonsai_fp16_to_fp32(h);
    }
    return repaired;
}

#ifndef BONSAI_FORCE_SCALAR_PQ2_0
/* Kernel body with the permuted activation PREBUILT by the caller (mm_stage
 * builds it once per staged input; the 2-3 matvecs sharing that input then
 * skip the build entirely). xp must be 64-byte aligned and hold n floats;
 * n must be a multiple of QK_PQ2_0 (checked by the wrapper/caller). */
static inline void pq2_0_matvec_knc_xp(float *y, const float *xp,
                                       const uint8_t *w, int n, int d) {
    const int nn = n;                       /* multiple of QK_PQ2_0 */
    const size_t rb = ((size_t)n + QK_PQ2_0 - 1) / QK_PQ2_0 * PQ2_0_BLOCK_BYTES;
    const __m512 l0b = pq2_lut_a(), l1b = pq2_lut_b();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        /* Per-row scale vector, thread-private on the stack: this loop is
         * omp parallel for, so a shared drow would be written by all threads
         * at once. n <= PQ2_XP_MAX is guaranteed by model.c's width check, so
         * nn/QK_PQ2_0 <= PQ2_XP_MAX/QK_PQ2_0 entries suffice. */
        float drow[PQ2_XP_MAX / QK_PQ2_0] __attribute__((aligned(64)));
        const uint8_t *row = w + (size_t)i * rb;
        pq2_build_drow(row, nn / QK_PQ2_0, drow);
        __m512 s = _mm512_setzero_ps();
        const uint8_t *blk = row;
        const float *x0 = xp;
        const float *dp = drow;
        int b = nn >> 7;
        do {
            const __m512 vd = _mm512_set1_ps(*dp);
            const __m512i l0 = _mm512_castps_si512(_mm512_mul_ps(l0b, vd));
            const __m512i l1 = _mm512_castps_si512(_mm512_mul_ps(l1b, vd));
            const __m512i c0 = PQ2_IMCI_LOAD16(blk + 2);
            const __m512i c1 = PQ2_IMCI_LOAD16(blk + 18);
            const __m512i e0 = _mm512_srli_epi32(c0, 4);
            const __m512i e1 = _mm512_srli_epi32(c1, 4);
            __m512 t = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l0)), _mm512_load_ps(x0));
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c0, l1)), _mm512_load_ps(x0 + 16), t);
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l0)), _mm512_load_ps(x0 + 32), t);
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e0, l1)), _mm512_load_ps(x0 + 48), t);
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l0)), _mm512_load_ps(x0 + 64), t);
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(c1, l1)), _mm512_load_ps(x0 + 80), t);
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l0)), _mm512_load_ps(x0 + 96), t);
            t = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(e1, l1)), _mm512_load_ps(x0 +112), t);
            s = _mm512_add_ps(s, t);
            blk += PQ2_0_BLOCK_BYTES;
            x0  += QK_PQ2_0;
            ++dp;
        } while (--b);
        y[i] = _mm512_reduce_add_ps(s);
    }
}

static inline void pq2_0_matvec_knc(float *y, const float *x,
                                    const uint8_t *w, int n, int d) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;

    /* Permuted activation, built ONCE here (serially, before the region) and
     * shared read-only by all threads of the omp parallel for below. Do NOT
     * move the build into the loop: 244 threads would write the same buffer
     * at once. 68 KB on the master's stack is fine (verified end-to-end).
     * model.c's grouped path builds xp in mm_stage instead and calls
     * pq2_0_matvec_knc_xp() directly. */
    float xp[PQ2_XP_MAX] __attribute__((aligned(64)));
    if (nn != n) {
        /* A tail would need the un-permuted activation; fall back to the scalar
         * reference for the whole call rather than mixing conventions. No tensor
         * in this model reaches this path (all folded inputs are 5120/6144/
         * 17408, every one a multiple of 128). */
        pq2_0_matvec_ref(y, x, w, n, d);
        return;
    }
    pq2_build_xp(x, xp, n);
    pq2_0_matvec_knc_xp(y, xp, w, n, d);
}

/* The previous (pre-V12) IMCI kernel, kept verbatim for reference and for
 * pq2test-style A/B comparisons. Deliberately `#if 0` rather than a macro
 * switch: it is a *second* definition of pq2_0_matvec_knc, so enabling it would
 * collide with the V12 kernel above, and its body sits inside the same
 * __MIC__ / BONSAI_FORCE_SCALAR_PQ2_0 nest. To A/B it, copy the body into a
 * uniquely named function in a benchmark file instead of un-gating this. */
#if 0
static inline void pq2_0_matvec_knc_v1(float *y, const float *x,
                                       const uint8_t *w, int n, int d) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;   /* full blocks */
    /* Row stride must use the CEILING block count, not n/QK. If n is not a
     * multiple of QK_PQ2_0 the row occupies ceil(n/QK) blocks, and the tail
     * path would otherwise index row i's tail at (n/QK)*34 -- which is row
     * i+1's FIRST block, and reads past the buffer entirely for the last row.
     * Every tensor in this model has n % 128 == 0 so the tail never executes,
     * but the addressing is fixed here so the code is correct if that changes. */
    const size_t rb = ((size_t)n + QK_PQ2_0 - 1) / QK_PQ2_0 * PQ2_0_BLOCK_BYTES;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        const uint8_t *row = w + (size_t)i * rb;
        float wf[QK_PQ2_0] __attribute__((aligned(64)));
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        int j = 0;
        for (; j < nn; j += QK_PQ2_0) {
            /* Vectorised IMCI decoder: 4 vector ops + 1 store per 16 weights,
             * measured 5.6x (memory-streaming) to 11x (cache-resident) faster
             * than pq2_0_dequant_block, and exact against it element-for-element
             * over every source alignment (src/bonsai/pq2imci.c). */
            pq2_0_dequant_block_imci(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            /* 4 independent accumulators break the FMA dependency chain;
             * this mirrors the winning pattern in matmul_q80 (DEBUG_LOG 12.8).
             * Each accumulator takes TWO 16-wide FMAs per block so that all
             * QK_PQ2_0 (128) weights of the block are consumed -- covering only
             * wf+0/16/32/48 (64 weights) is what caused the half-value bug. */
            sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf),       _mm512_load_ps(x + j),       sum0);
            sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 16),  _mm512_load_ps(x + j + 16),  sum1);
            sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 32),  _mm512_load_ps(x + j + 32),  sum2);
            sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 48),  _mm512_load_ps(x + j + 48),  sum3);
            sum0 = _mm512_fmadd_ps(_mm512_load_ps(wf + 64),  _mm512_load_ps(x + j + 64),  sum0);
            sum1 = _mm512_fmadd_ps(_mm512_load_ps(wf + 80),  _mm512_load_ps(x + j + 80),  sum1);
            sum2 = _mm512_fmadd_ps(_mm512_load_ps(wf + 96),  _mm512_load_ps(x + j + 96),  sum2);
            sum3 = _mm512_fmadd_ps(_mm512_load_ps(wf + 112), _mm512_load_ps(x + j + 112), sum3);
        }
        __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        float acc = _mm512_reduce_add_ps(sum);
        for (; j < n; ++j) {
            /* tail (only if n is not a multiple of 128) -- scalar */
            const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
            const float dd = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int q = (blk[2 + ((j % QK_PQ2_0) >> 2)] >> ((j & 3) * 2)) & 0x3;
            acc += (float)(q - 1) * dd * x[j];
        }
        y[i] = acc;
    }
}
#endif /* 0 -- previous kernel, reference only */
#else  /* BONSAI_FORCE_SCALAR_PQ2_0 -- known-good baseline */
#define pq2_0_matvec_knc pq2_0_matvec_ref
#endif /* BONSAI_FORCE_SCALAR_PQ2_0 */
#else
/* Host build: no IMCI, so the scalar version is the only implementation. */
#define pq2_0_matvec_knc pq2_0_matvec_ref
#endif /* __MIC__ */

/* ------------------------------------------------------------------ */
/* Scalar reference implementation.                                    */
/*                                                                     */
/* Deliberately compiled on ALL targets, including k1om. The card build */
/* needs it reachable so pq2test can run both kernels over the same     */
/* bytes on the same machine and attribute a mismatch to the right one. */
/* ------------------------------------------------------------------ */
static inline void pq2_0_matvec_ref(float *y, const float *x,
                                    const uint8_t *w, int n, int d) {
    const int nn = (n / QK_PQ2_0) * QK_PQ2_0;
    /* Ceiling block count, so the tail of row i lands inside row i. Using
     * n/QK_PQ2_0 as the stride makes a row of a non-multiple-of-128 tensor
     * overlap the next row's first block (and read past the buffer on the
     * last row). See the matching note in pq2_0_matvec_knc. */
    const size_t rb = ((size_t)n + QK_PQ2_0 - 1) / QK_PQ2_0 * PQ2_0_BLOCK_BYTES;
    /* Rows are independent and each writes only y[i], so the outer loop
     * parallelises cleanly. Without this the scalar path runs ~50x slower than
     * the (incorrect) IMCI kernel and inference becomes unusable: measured
     * 0.004 tok/s serial vs 0.19 tok/s with the SIMD path on a 7.2 GB model. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        const uint8_t *row = w + (size_t)i * rb;
        float wf[QK_PQ2_0];
        float acc = 0.0f;
        int j = 0;
        for (; j < nn; j += QK_PQ2_0) {
            pq2_0_dequant_block(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, wf);
            for (int k = 0; k < QK_PQ2_0; ++k) acc += wf[k] * x[j + k];
        }
        for (; j < n; ++j) {
            const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
            const float dd = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            const int q = (blk[2 + ((j % QK_PQ2_0) >> 2)] >> ((j & 3) * 2)) & 0x3;
            acc += (float)(q - 1) * dd * x[j];
        }
        y[i] = acc;
    }
}

/* ------------------------------------------------------------------ */
/* Hadamard activation transform                                       */
/*                                                                     */
/* The model's folded weights are stored in a rotated basis. The       */
/* activation transform is  x' = H . (s * x)  with H a 1024-block      */
/* normalised Sylvester-Walsh-Hadamard matrix:                         */
/*   H[r][c] = (parity(r & c) ? -1 : +1) / sqrt(1024)                  */
/* `signs` has one +-1 entry per element of x.                         */
/* ------------------------------------------------------------------ */

#ifdef __MIC__
/* KNC (IMCI) vector helpers for the Hadamard transform. Every lane performs
 * exactly the same single add/sub/mul on the same element as the scalar code,
 * so the results are BIT-IDENTICAL to the scalar path (no reassociation).
 *
 * _mm512_load_ps/_mm512_store_ps REQUIRE 64-byte alignment and k1om has no
 * working unaligned load/store intrinsics (loadu/storeu do not link), so the
 * callers below must prove alignment before taking these paths. */

/* FWHT butterfly with the h >= 16 stages vectorised: there the inner j loop
 * covers contiguous 16-float spans, and i and j+h offsets are multiples of h
 * (>= 16 floats = 64 B), so every access stays aligned iff `a` is 64-byte
 * aligned. Stages with h < 16 stay scalar (the span is too short to fill a
 * vector). PRECONDITION: `a` is 64-byte aligned. */
static inline void fwht_inplace_mic(float *a, int n) {
    int h;
    for (h = 1; h < n && h < 16; h <<= 1) {
        for (int i = 0; i < n; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                const float u = a[j];
                const float v = a[j + h];
                a[j]     = u + v;
                a[j + h] = u - v;
            }
        }
    }
    for (; h < n; h <<= 1) {
        for (int i = 0; i < n; i += (h << 1)) {
            for (int j = i; j < i + h; j += 16) {
                const __m512 u = _mm512_load_ps(a + j);
                const __m512 v = _mm512_load_ps(a + j + h);
                _mm512_store_ps(a + j,     _mm512_add_ps(u, v));
                _mm512_store_ps(a + j + h, _mm512_sub_ps(u, v));
            }
        }
    }
}

/* p[k] *= q[k] for k in [0, n); n must be a multiple of 16.
 * PRECONDITION: `p` and `q` are 64-byte aligned. */
static inline void vecmul_inplace_mic(float *p, const float *q, int n) {
    for (int k = 0; k < n; k += 16)
        _mm512_store_ps(p + k, _mm512_mul_ps(_mm512_load_ps(p + k),
                                             _mm512_load_ps(q + k)));
}

/* p[k] *= c for k in [0, n); n must be a multiple of 16.
 * PRECONDITION: `p` is 64-byte aligned. */
static inline void vecscale_inplace_mic(float *p, int n, float c) {
    const __m512 vc = _mm512_set1_ps(c);
    for (int k = 0; k < n; k += 16)
        _mm512_store_ps(p + k, _mm512_mul_ps(_mm512_load_ps(p + k), vc));
}
#endif /* __MIC__ */

/* In-place fast Walsh-Hadamard transform on one block (unnormalised).
 * Costs n*log2(n) add/sub; for n=1024 that is 10240 ops per block. */
static inline void fwht_inplace(float *a, int n) {
#ifdef __MIC__
    /* _mm512_load_ps faults on a misaligned base, and embed_row's `out`
     * buffer is one caller whose alignment we cannot rely on. Dispatch once
     * here: aligned -> vectorised stages, otherwise the scalar butterfly for
     * ALL stages. The scalar path below is the reference and stays shared. */
    if (((uintptr_t)a & 63u) == 0) {
        fwht_inplace_mic(a, n);
        return;
    }
#endif
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                const float u = a[j];
                const float v = a[j + h];
                a[j]     = u + v;
                a[j + h] = u - v;
            }
        }
    }
}

/*
 * x' = H . (s * x), applied on the last dimension, block-wise.
 * `signs` may be NULL (identity signs, used by the inverse embedding path).
 *
 * PRECONDITION: `block > 0` and `block` divides `n`. The loop below steps
 * `base += block` and transforms `[base, base+block)` unconditionally, so if
 * `block` does not divide `n` the final iteration writes past `x[n-1]`.
 * Measured: n=10, block=8 wrote 6 of 6 elements past the end.
 * `model.c` validates this at open time; this check makes the header safe on
 * its own and is cheap (once per call, not per element).
 *
 * The result is identical to multiplying by the explicit H matrix but costs
 * O(n log n) instead of O(n^2). With n = 17408 (the widest input) that is
 * ~17x cheaper than the dense form llama.cpp materialises.
 */
static inline int hadamard_forward(float *x, const float *signs,
                                   int n, int block, int inverse) {
    if (block <= 0 || n <= 0 || (n % block) != 0) return -1;
    const float norm = 1.0f / sqrtf((float)block);
#ifdef __MIC__
    /* Per-call vector-path gate for the element-wise loops below. All blocks
     * share x's 64-byte alignment only because `base` steps by `block`
     * floats: that preserves alignment iff block is a multiple of 16 floats
     * (64 B). The precondition above only forces block to divide n (nothing
     * forces block == 1024), so test (block & 15) explicitly instead of
     * assuming it. `signs` is indexed at the same `base` offsets, so it needs
     * the same alignment test. fold_x is 64-byte aligned via aligned_f32()
     * and block == 1024 in practice, so this gate costs one branch per call
     * and is true on every hot path; embed_row's unaligned `out` falls back
     * to scalar. fwht_inplace() re-checks p per block on its own. */
    const int vec_ok = ((block & 15) == 0) &&
                       (((uintptr_t)x & 63u) == 0) &&
                       (signs == NULL || ((uintptr_t)signs & 63u) == 0);
#endif
    /* The sign multiply and the per-block Hadamard are independent for each
     * `base`, so the OUTER loop is parallel and the butterfly inside
     * fwht_inplace() stays serial (it has a true loop-carried dependency).
     *
     * Measured on the card: this function was 9.5% of a token (0.330 s/token at
     * 122 threads) and its cost was FLAT from 1 to 244 threads, i.e. it was pure
     * serial glue. Parallelising it is worth roughly +40% end-to-end once the
     * GDN recurrence is also parallel (see docs/K1OM_TIME_BUDGET.md).
     *
     * Each `base` touches only x[base .. base+block), and the blocks tile
     * [0,n) exactly (n % block == 0 is checked above), so there is no
     * cross-iteration aliasing. */
    #pragma omp parallel for schedule(static)
    for (int base = 0; base < n; base += block) {
        float *p = x + base;
        if (signs) {
#ifdef __MIC__
            if (vec_ok) {
                vecmul_inplace_mic(p, signs + base, block);
            } else
#endif
            {
                for (int k = 0; k < block; ++k) p[k] *= signs[base + k];
            }
        }
        if (!inverse) {
            /* forward: H = (1/sqrt(N)) * WHT */
            fwht_inplace(p, block);
#ifdef __MIC__
            if (vec_ok) {
                vecscale_inplace_mic(p, block, norm);
            } else
#endif
            {
                for (int k = 0; k < block; ++k) p[k] *= norm;
            }
        } else {
            /* inverse: H is symmetric and H*H = I, so applying H again
             * inverts it. (No sign flip: the signs already live in the
             * stored rows.) */
            fwht_inplace(p, block);
#ifdef __MIC__
            if (vec_ok) {
                vecscale_inplace_mic(p, block, norm);
            } else
#endif
            {
                for (int k = 0; k < block; ++k) p[k] *= norm;
            }
        }
    }
    return 0;
}

#endif /* BONSAI_PQ2_0_H */
