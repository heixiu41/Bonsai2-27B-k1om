/* Instruction-cost baseline and variant comparison for the PQ2_0 decoder.
 *
 * Purpose: find out WHAT specifically costs, before changing anything. The
 * project has already been burned once by optimising on a guess (a bandwidth
 * number that was wrong by 59x), so this measures rather than argues.
 *
 * Everything here runs on the card. Variants are compared with an identical
 * harness, identical input, single thread, best-of-N.
 *
 * Variants:
 *   V0  pq2_0_dequant_block_imci      -- the shipped decoder
 *   V1  dequant + read back + 8 FMAs  -- what mm_folded actually does now
 *                                        (isolates the store/load round trip)
 *   V2  register-resident dot product -- decode straight into the FMA
 *                                        accumulators, never touching memory
 *   V3  V2 but with a "no broadcast" extraction attempt (see notes)
 *
 * The metric that matters for the real workload is weights/second for a full
 * row decode + dot product, because that is what the matvec does per token.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#include "pq2_0.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- V1: decode into `out`, then load it back and accumulate (current path) */
static float v1_row(const uint8_t *row, const float *x, int n) {
    float out[QK_PQ2_0] __attribute__((aligned(64)));
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        pq2_0_dequant_block_imci(row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES, out);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(out),      _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(out + 16), _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(out + 32), _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(out + 48), _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(_mm512_load_ps(out + 64), _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(_mm512_load_ps(out + 80), _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(_mm512_load_ps(out + 96), _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(_mm512_load_ps(out +112), _mm512_load_ps(x + j +112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V2: decode + FMA with no memory round trip -------------------------
 * The 8 decoded float vectors live in registers and are consumed by the FMA
 * directly. Nothing is stored. This removes 8 stores + 8 loads per block.
 */
static float v2_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    const __m512i shf = PQ2_IMCI_SHIFT();

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            g[m] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            g[4 + m] = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V3: V2 with the result LUT permute REPLACED by integer arithmetic ----
 * Isolates the cost of the second vpermd (the 16-entry float LUT). The code
 * value k in {0,1,2,3} maps to k-1, so instead of gathering from a table we do
 * (k - 1) in integer and convert. KNC has no _mm512_cvtepi32_ps, so the
 * conversion is the IEEE-754 bias form: for 0 <= k <= 3,
 *     (float)k  ==  bitcast_ps(0x4B000000 + k) - 8388608.0f
 * which holds because adding k to the exponent-2^23 pattern moves the mantissa
 * by exactly k.  Applying it as (k-1) means using bias 0x4AFFFFFF + k, and the
 * result is exact for k-1 in {-1,0,1,2}.
 * If this beats V2, the LUT permute was the cost; if it loses, KNC's integer
 * ALU is the slow path (as an earlier measurement suggested).
 */
static float v3_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i shf = PQ2_IMCI_SHIFT();
    const __m512i bias = _mm512_set1_epi32((int)0x4AFFFFFFu);
    const __m512 magic = _mm512_set1_ps(8388608.0f);

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
            const __m512i v = _mm512_srlv_epi32(b, shf);       /* k in 0..3 */
            const __m512  f = _mm512_sub_ps(
                _mm512_castsi512_ps(_mm512_add_epi32(v, bias)), magic);
            g[m] = _mm512_mul_ps(f, vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            const __m512  f = _mm512_sub_ps(
                _mm512_castsi512_ps(_mm512_add_epi32(v, bias)), magic);
            g[4 + m] = _mm512_mul_ps(f, vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V4: one permute per vector instead of two --------------------------
 * Instead of (broadcast byte) then (shift) then (LUT gather), broadcast and
 * shift in one step is not possible, but we CAN drop the explicit broadcast:
 * keep the codes in their natural positions and use a shift vector whose
 * entries vary per dword so that one srlv + one permute suffices.
 * This variant is a probe: it uses only ONE permute by folding the broadcast
 * into the LUT gather, i.e. permutevar(v, lut) where v has been built so that
 * its low 4 bits already select the right LUT entry. Not correct unless the
 * code bit positions line up; used purely to price a single permute.
 */
static float v4_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i shf = PQ2_IMCI_SHIFT();
    const __m512i lut = PQ2_IMCI_LUT_BITS();

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        /* one permute: broadcast the byte AND its 4 shifted copies into the
         * lanes, by using an idx vector that repeats each source dword 4x and
         * applying srlv with the per-lane shift afterwards. This is V2's shape
         * minus the second permute -- the LUT lookup is replaced by arithmetic
         * (same as V3) so the ONLY difference from V3 is that here we skip the
         * separate LUT gather and keep V3's integer bias conversion. */
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c0);
            const __m512i v = _mm512_srlv_epi32(b, shf);
            (void)lut;
            g[m] = v;   /* integer codes, scaled later -- see the note */
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i b = _mm512_permutevar_epi32(pq2_imci_idx(m), c1);
            g[4 + m] = _mm512_castsi512_ps(_mm512_srlv_epi32(b, shf));
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
        (void)vd;
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V5: no decode at all -- price the FMA network alone ---------------
 * Reads the packed block but feeds the raw bytes straight to the FMA as if
 * they were floats. Wrong numerically; its ONLY purpose is to measure the
 * floor: 4 loads + 8 FMA per block with no extraction work. The gap between
 * this and V2 is the cost of the decoder itself.
 */
static float v5_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const __m512 a = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 2));
        const __m512 b = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 18));
        const __m512 c = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 2));
        const __m512 e = _mm512_castsi512_ps(PQ2_IMCI_LOAD16(blk + 18));
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(c, _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(e, _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(a, _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(b, _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(c, _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(e, _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V6: no broadcast permute at all -----------------------------------
 * KEY INSIGHT this exploits: PQ2_0 packs 4 code bytes into each 32-bit dword
 * of qs, little-endian. `_mm512_extload_epi32(qs, UINT8)` gives dword i =
 * qs[4i] | qs[4i+1]<<8 | qs[4i+2]<<16 | qs[4i+3]<<24 -- i.e. LANE i holds
 * exactly code bytes 4i..4i+3, which are the 4 codes of output group i.
 *
 * So instead of broadcasting each byte to 4 lanes (one vpermd per group, 8 per
 * block), we can extract group m's byte from every dword at once:
 *
 *     w = srlv(dword, 8*m)          -- brings byte m down to bits 0..7
 *     w = w & 3                     -- needed? no: the LUT masks internally
 *     g = permutevar(w, lut)        -- lut[i] = (i&3)-1, so &3 happens here
 *
 * The LUT's own `(i & 15)` index mask makes the AND unnecessary: we need
 * (i & 15) & 3 == i & 3, and for shifts of 8*m with m<=3 only bits 0..5 can be
 * non-zero, so i & 15 is well defined.
 *
 * Instruction count per block, versus V2's 16 permutes + 8 shifts:
 *   2 extloads + 8 x (srlv + permutevar + mul) + 8 FMA + 4 x load x  = still
 *   8 permutes and 8 srlv, but NO per-group broadcast permute.
 * Because each extracted vector is itself a full 16-lane group in the correct
 * output order, the broadcast is free.
 */
static float v6_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut = PQ2_IMCI_LUT_BITS();
    /* lane i shifts by 8*(i&3) -> byte (i&3) of that lane's dword */
    const __m512i sh0 = _mm512_set_epi32(24,16,8,0, 24,16,8,0, 24,16,8,0, 24,16,8,0);

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        /* each dword = 4 consecutive code bytes, in output order */
        const __m512i c0 = _mm512_extload_epi32((const void *)qs,
                              _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE, 0);
        const __m512i c1 = _mm512_extload_epi32((const void *)(qs + 16),
                              _MM_UPCONV_EPI32_UINT8, _MM_HINT_NONE, 0);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            /* rotate the shift vector left by one dword each step, so lane i
             * always shifts by 8*((i+m)&3): no per-m constant rebuild. */
            const __m512i sh = (m == 0) ? sh0 :
                _mm512_alignr_epi32(sh0, sh0, 4 * m);
            const __m512i w = _mm512_srlv_epi32(c0, sh);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(w, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i sh = (m == 0) ? sh0 :
                _mm512_alignr_epi32(sh0, sh0, 4 * m);
            const __m512i w = _mm512_srlv_epi32(c1, sh);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(w, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V7: byte broadcast by vpmulld instead of vpermd --------------------
 * Packing a code byte into 4 dwords so that each of 4 groups gets a shifted
 * copy is a broadcast. The current decoder does it with one vpermd per group
 * (8 per block). An alternative that uses no permute at all:
 *
 *     b  = byte value already in the low byte of each dword (from LOAD16)
 *     b <<= (2*q)                        -- position the wanted 2-bit field
 *     v  = (b & 0xFF) * 0x01010101       -- replicate that byte 4x inside the dword
 *     g  = permutevar(v, lut)            -- still one LUT permute
 *
 * i.e. per group: 1 shift + 1 vpmulld, versus the current 1 broadcast permute.
 * So it trades 8 vpermd for 8 vpmulld.
 *
 * THE OPEN QUESTION THIS MEASURES: is KNC's _mm512_mullo_epi32 cheaper than
 * _mm512_permutevar_epi32? Both are plausibly microcoded. Do not assume --
 * that is exactly the kind of guess this project has already been burned by.
 *
 * NOTE: this variant is only a PRICING probe for the multiply. The output
 * ordering is deliberately the same shape as V2 so the cost comparison is
 * meaningful, but the values are NOT expected to match V1 -- read the timing
 * only. See v8 for the correct version.
 */
static float v7_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i shf  = PQ2_IMCI_SHIFT();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    const __m512i bmask= _mm512_set1_epi32(0xFF);

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            /* each lane holds one code byte in its low byte; replicate it 4x
             * inside the dword, then extract field (lane%4) with the shift */
            const __m512i t = _mm512_mullo_epi32(
                _mm512_and_si512(c0, bmask), mulc);
            const __m512i v = _mm512_srlv_epi32(t, shf);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(
                _mm512_and_si512(c1, bmask), mulc);
            const __m512i v = _mm512_srlv_epi32(t, shf);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(v, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V7b: price vpmulld and vpsrlvd in isolation -----------------------
 * No LUT permute at all: just the multiply + shift chain, accumulated. This
 * isolates "what does vpmulld cost" from "what does the LUT permute cost".
 * Wrong numerically; timing only.
 */
static float v7b_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i shf  = PQ2_IMCI_SHIFT();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    const __m512i bmask= _mm512_set1_epi32(0xFF);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const uint8_t *qs = blk + 2;
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(_mm512_and_si512(c0, bmask), mulc);
            g[m] = _mm512_castsi512_ps(_mm512_srlv_epi32(t, shf));
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i t = _mm512_mullo_epi32(_mm512_and_si512(c1, bmask), mulc);
            g[4 + m] = _mm512_castsi512_ps(_mm512_srlv_epi32(t, shf));
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V8: the CORRECT vpmulld decoder (the candidate for production) -------
 *
 * Layout derivation. In the shipped decoder (V1) each lane holds ONE code byte
 * (via LOAD16), group m broadcasts byte m to 4 lanes, and SHIFT[l] = (l%4)*2
 * selects field (l%4) of that byte. So output lane i takes
 *     code = (qs[i] >> 2*(i%4)) & 3
 * i.e. each lane's OWN byte, field (lane index mod 4).
 *
 * With vpmulld we broadcast by replication instead of permutation, so the
 * wanted field must first be moved to the low byte of that lane's dword:
 *
 *     low2 = (dword >> (2*q)) & 3          q = lane%4   [+8*m for the dword]
 *     t    = vpmulld(low2, 0x01010101)     replicate that byte 4x in the dword
 *     g    = permutevar(t, lut)            LUT[i&3] = (i&3)-1 selects byte i&3
 *
 * Checking the dataflow: lane i of `g` reads byte (i&3) of t, and t's byte j
 * equals low2 of its own lane, so g[i] = (qs[i] >> 2*(i&3)) & 3 - 1 -- exactly
 * what V1 produces, expressed with a multiply instead of a broadcast permute.
 *
 * Note the AND is required here (unlike V1, where the LUT's &15 masked the
 * garbage): after the MULTIPLY the higher bits of the byte are replicated too,
 * so without the mask they would leak into the byte the LUT selects.
 */
static float v8_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    /* lane i shifts by 2*(i%4) + 8*m: the field index within its own byte, plus
     * the byte-group offset. Precomputed for m = 0..3. NOTE: shifting
     * 0x3F3F3F06 right by 2 is fine (it stays positive), but a left shift could
     * overflow a signed int, so build these as explicit constants. */
    const __m512i shv[4] = {
        _mm512_set_epi32( 6, 4, 2, 0,  6, 4, 2, 0,  6, 4, 2, 0,  6, 4, 2, 0),
        _mm512_set_epi32(14,12,10, 8, 14,12,10, 8, 14,12,10, 8, 14,12,10, 8),
        _mm512_set_epi32(22,20,18,16, 22,20,18,16, 22,20,18,16, 22,20,18,16),
        _mm512_set_epi32(30,28,26,24, 30,28,26,24, 30,28,26,24, 30,28,26,24),
    };

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i k = _mm512_srlv_epi32(c0, shv[m]);   /* field at low bits */
            const __m512i t = _mm512_mullo_epi32(k, mulc);     /* replicate in dword */
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i k = _mm512_srlv_epi32(c1, shv[m]);
            const __m512i t = _mm512_mullo_epi32(k, mulc);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- element-wise verification of a decoder against the scalar reference ---
 * The whole point: a row-sum comparison can hide per-lane errors (and the
 * extloadunpacklo zero-fill bug nearly shipped because only a result-vs-
 * reference check caught it). So dequantise through the variant of interest
 * into a buffer and compare EVERY weight against the scalar decoder, over all
 * 64 byte alignments of the source block.
 *
 * To reuse the row kernels (which fuse the FMA) we instead reconstruct each
 * weight by differencing unit vectors: run the kernel with x = e_j and read the
 * result, which equals weight_j * d. That is 128 kernel calls per block, too
 * slow for 64 alignments. Simpler and sufficient: compare v8_row against
 * v1_row element-wise by using x = e_j for a sample of j, and compare the full
 * ROW SUM for all j. We do both:
 *   - all 128 weights of a handful of blocks, via unit vectors
 *   - row sums over many rows (random x), which catches any lane permutation
 *     error because a permutation changes the sum against a non-constant x
 */
static int verify_v8(const uint8_t *block_area, size_t area_len, const float *x, int n) {
    (void)area_len;
    int bad = 0;
    /* (a) unit-vector probe: every weight of the first block */
    float xv[QK_PQ2_0] __attribute__((aligned(64)));
    for (int jj = 0; jj < QK_PQ2_0; ++jj) {
        for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
        xv[jj] = 1.0f;
        const float a = v1_row(block_area, xv, QK_PQ2_0);
        const float b = v8_row(block_area, xv, QK_PQ2_0);
        if (fabsf(a - b) > 1e-6f) {
            if (bad < 8) printf("  unit-vector mismatch at j=%d: V1=%.7g V8=%.7g\n", jj, a, b);
            ++bad;
        }
    }
    /* (b) row sums with a non-constant x -- a lane permutation would show here */
    const float a = v1_row(block_area, x, n);
    const float b = v8_row(block_area, x, n);
    if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) {
        printf("  row-sum mismatch: V1=%.7g V8=%.7g\n", a, b);
        ++bad;
    }
    return bad;
}

/* ---- V9: vpslld + vpmulld instead of the broadcast permute ----------------
 *
 * The shipped decoder (V1) spends one BROADCAST vpermd per group of 16 weights
 * to spread a code byte across 4 lanes. Measured, the two vpermd ops per group
 * are ~62% of the decoder cost and vpmulld is cheaper than vpermd, so V9
 * replaces the broadcast with integer arithmetic while keeping the LUT permute.
 *
 * Per group m, physical lane l in [0,16):
 *
 *     byte b   = 4m + l/4                      (the code byte this group needs)
 *     shift_l  = 8 * ((b - (l & 3)) mod 4)     (place byte b at position l&3)
 *     pre      = slli(word, shift_l)
 *     t        = mullo(pre, 0x01010101)         (replicate that byte across the dword)
 *     g[m]     = LUT(permutevar(t, lut))
 *
 * Why the shift is what it is: the LUT gather reads byte (lane & 3) of the
 * permuted dword, so the byte this group needs must be sitting at position
 * (lane & 3). Byte b starts at position b modulo the dword, hence the shift.
 *
 * THE BUG THE PREVIOUS ATTEMPT (v8_row) MADE, for contrast: it used
 * 2*(l&3) + 8m, which selects byte (q - 2m) mod 8 rather than byte m, so only
 * group m=0 came out right. That was caught by element-wise verification, not
 * by reading the code.
 */
static float v9_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    /* Scalar shift amounts, one per group. _mm512_slli_epi32 on KNC takes a
     * SCALAR immediate (the vector form _mm512_sllv_epi32 does not link), but
     * that is sufficient: each group m needs ONE shift amount, not one per lane.
     *
     * Why this is correct: after slli(word, 8m), byte position (l&3) of the
     * dword holds the byte that was at position (l&3 - m) mod 4. The LUT reads
     * byte (l&3), and for output lane 16m + l we need field (l&3) of byte
     * 4m + l/4 -- i.e. exactly the field that byte (l&3) of the shifted,
     * replicated dword carries. Byte selection is handled by l/4, which the
     * permutation of the load already provides. */
    const int shm[4] = { 0, 8, 16, 24 };

    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_slli_epi32(c0, shm[m]);
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_slli_epi32(c1, shm[m]);
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- V9spec: the V9 design implemented LITERALLY, i.e. with the four
 * PER-LANE shift vectors the doc prints (a single scalar shift 8*m is a
 * different design; see v9_row above). Written by a separate agent so that
 * both readings of the spec are measured by the same harness.
 *
 *     shift_l = 8 * ((l&3) - m) mod 4, lane-ordered as printed:
 *       m=0:  0  8 16 24 | ...
 *       m=1: 24  0  8 16 | ...
 *       m=2: 16 24  0  8 | ...
 *       m=3:  8 16 24  0 | ...
 *     pre = sllv(c, shift)   -- per lane, so the VARIABLE shift form
 *     t   = mullo(pre, 0x01010101)
 *     g   = mul_ps(permutevar_epi32(t, lut), vd)
 *
 * NOTE: _mm512_sllv_epi32 DOES link on this card (measured) -- the claim in
 * v9_row's comment above that the vector form does not link is wrong.
 */
static inline __m512i v9spec_shift(int m) {
    /* _mm512_set_epi32's LAST argument is lane 0, so each group of four is
     * written reversed relative to the lane-ordered doc listing. */
    switch (m) {
    case 0:  return _mm512_set_epi32(24,16,8,0, 24,16,8,0, 24,16,8,0, 24,16,8,0);
    case 1:  return _mm512_set_epi32(16,8,0,24, 16,8,0,24, 16,8,0,24, 16,8,0,24);
    case 2:  return _mm512_set_epi32(8,0,24,16, 8,0,24,16, 8,0,24,16, 8,0,24,16);
    default: return _mm512_set_epi32(0,24,16,8, 0,24,16,8, 0,24,16,8, 0,24,16,8);
    }
}

static float v9spec_row(const uint8_t *row, const float *x, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    const __m512i lut  = PQ2_IMCI_LUT_BITS();
    const __m512i mulc = _mm512_set1_epi32(0x01010101);
    for (int j = 0; j < n; j += QK_PQ2_0) {
        const uint8_t *blk = row + (size_t)(j / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
        const float d = bonsai_fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
        const uint8_t *qs = blk + 2;
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i c0 = PQ2_IMCI_LOAD16(qs);
        const __m512i c1 = PQ2_IMCI_LOAD16(qs + 16);
        __m512 g[8];
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_sllv_epi32(c0, v9spec_shift(m));
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        for (int m = 0; m < 4; ++m) {
            const __m512i pre = _mm512_sllv_epi32(c1, v9spec_shift(m));
            const __m512i t   = _mm512_mullo_epi32(pre, mulc);
            g[4 + m] = _mm512_mul_ps(
                _mm512_castsi512_ps(_mm512_permutevar_epi32(t, lut)), vd);
        }
        s0 = _mm512_fmadd_ps(g[0], _mm512_load_ps(x + j),      s0);
        s1 = _mm512_fmadd_ps(g[1], _mm512_load_ps(x + j + 16), s1);
        s2 = _mm512_fmadd_ps(g[2], _mm512_load_ps(x + j + 32), s2);
        s3 = _mm512_fmadd_ps(g[3], _mm512_load_ps(x + j + 48), s3);
        s0 = _mm512_fmadd_ps(g[4], _mm512_load_ps(x + j + 64), s0);
        s1 = _mm512_fmadd_ps(g[5], _mm512_load_ps(x + j + 80), s1);
        s2 = _mm512_fmadd_ps(g[6], _mm512_load_ps(x + j + 96), s2);
        s3 = _mm512_fmadd_ps(g[7], _mm512_load_ps(x + j + 112), s3);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    return _mm512_reduce_add_ps(s);
}

/* ---- generic element-wise verification, parameterised by the decoder ------
 * (verify_v8 below is the original single-target version; this one lets the
 * harness check any candidate without duplicating the logic) */
static int verify_vs(float (*cand)(const uint8_t *, const float *, int),
                     const uint8_t *block_area, const float *x, int n) {
    int bad = 0;
    float xv[QK_PQ2_0] __attribute__((aligned(64)));
    for (int jj = 0; jj < QK_PQ2_0; ++jj) {
        for (int t = 0; t < QK_PQ2_0; ++t) xv[t] = 0.0f;
        xv[jj] = 1.0f;
        const float a = v1_row(block_area, xv, QK_PQ2_0);
        const float b = cand(block_area, xv, QK_PQ2_0);
        if (fabsf(a - b) > 1e-6f) ++bad;
    }
    const float a = v1_row(block_area, x, n);
    const float b = cand(block_area, x, n);
    if (fabsf(a - b) > 1e-4f * (fabsf(a) + 1.0f)) ++bad;
    return bad;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 5120;
    const int d = argc > 2 ? atoi(argv[2]) : 512;   /* rows */
    const int reps = argc > 3 ? atoi(argv[3]) : 5;

    const size_t row_bytes = (size_t)(n / QK_PQ2_0) * PQ2_0_BLOCK_BYTES;
    uint8_t *w = (uint8_t *)aligned_alloc(64, ((row_bytes * d) + 127) & ~(size_t)63);
    float *x = (float *)aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    if (!w || !x) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* deterministic pseudo-random, valid PQ2_0 blocks with finite scales */
    uint32_t st = 12345u;
    for (size_t i = 0; i < row_bytes * d; ++i) {
        st = st * 1103515245u + 12345u;
        w[i] = (uint8_t)(st >> 16);
    }
    for (size_t r = 0; r < d; ++r)
        for (int b = 0; b < n / QK_PQ2_0; ++b) {
            uint8_t *blk = w + r * row_bytes + (size_t)b * PQ2_0_BLOCK_BYTES;
            blk[0] = 0x00; blk[1] = 0x3C;               /* fp16 scale = 1.0 */
        }
    for (int i = 0; i < n; ++i) x[i] = (float)((int)(i % 7) - 3) * 0.25f;

    printf("=== PQ2_0 decoder / dot-product variants (single thread) ===\n");
    printf("n=%d  rows=%d  row_bytes=%zu  weights/row=%d  total=%zu\n\n",
           n, d, row_bytes, n, (size_t)n * d);

    /* --- correctness first: V1, V2, V3 must agree (V4/V5 are deliberate
     * non-numeric probes used only for pricing, see their comments) --- */
    {
        double a = 0, b = 0, c = 0;
        for (int r = 0; r < d; ++r) {
            const uint8_t *row = w + (size_t)r * row_bytes;
            a += v1_row(row, x, n);
            b += v2_row(row, x, n);
            c += v3_row(row, x, n);
        }
        printf("V1 total = %.6f\nV2 total = %.6f\nV3 total = %.6f\n", a, b, c);
        printf("max|V1-V2| = %.3e  %s\n", fabs(a - b), fabs(a - b) < 1e-3 ? "AGREE" : "DISAGREE");
        printf("max|V1-V3| = %.3e  %s   (V3 is the int-bias probe)\n\n",
               fabs(a - c), fabs(a - c) < 1e-3 ? "AGREE" : "DISAGREE");
    }

    /* --- element-wise verification of the candidate decoder (V8) -----------
     * Over ALL 64 byte alignments of the source block, compare V8 against V1
     * weight by weight using unit vectors, plus the row sum with a
     * non-constant x (which catches any lane permutation). */
    {
        printf("=== V8 (vpmulld decoder) vs V1: element-wise, 64 alignments ===\n");
        uint8_t *area = (uint8_t *)aligned_alloc(64, 64 + 64 + 256);
        if (!area) { fprintf(stderr, "alloc failed\n"); return 1; }
        /* a randomized but fixed block */
        uint32_t s2 = 987654321u;
        uint8_t blk[QK_PQ2_0 * 2 + 34];
        for (unsigned i = 0; i < sizeof blk; ++i) {
            s2 = s2 * 1103515245u + 12345u;
            blk[i] = (uint8_t)(s2 >> 16);
        }
        blk[0] = 0x00; blk[1] = 0x3C;   /* fp16 scale 1.0 */

        int total_bad = 0, checked = 0;
        for (int off = 0; off < 64; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
            /* x must be 64-byte aligned; length covers one block */
            total_bad += verify_v8(area + off, 512, x, QK_PQ2_0);
            ++checked;
        }
        printf("  blocks checked = %d alignments x (128 unit vectors + 1 row sum)\n", checked);
        printf("  MISMATCHES = %d   %s\n\n", total_bad,
               total_bad == 0 ? "*** V8 IS EXACT ***" : "*** V8 IS WRONG ***");

        /* --- the actual candidate: V9 --- */
        int v9_bad = 0;
        int v9_per_align[64];
        for (int off = 0; off < 64; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
            v9_per_align[off] = verify_vs(v9_row, area + off, x, QK_PQ2_0);
            v9_bad += v9_per_align[off];
        }
        printf("=== V9 (vpslld+vpmulld decoder) vs V1: element-wise, 64 alignments ===\n");
        printf("  MISMATCHES = %d / 64 alignments\n", v9_bad);
        if (v9_bad) {
            printf("  alignments with mismatches:");
            for (int o = 0; o < 64; ++o) if (v9_per_align[o]) printf(" %d(%d)", o, v9_per_align[o]);
            printf("\n");
        }
        printf("  %s\n\n", v9_bad == 0 ? "*** V9 IS EXACT ***" : "*** V9 IS WRONG ***");

        /* --- the same design with the doc's PER-LANE shift vectors --- */
        int v9s_bad = 0;
        int v9s_per_align[64];
        for (int off = 0; off < 64; ++off) {
            memset(area, 0, 64 + 64 + 256);
            memcpy(area + off, blk, PQ2_0_BLOCK_BYTES);
            v9s_per_align[off] = verify_vs(v9spec_row, area + off, x, QK_PQ2_0);
            v9s_bad += v9s_per_align[off];
        }
        printf("=== V9spec (doc's per-lane shift vectors) vs V1: 64 alignments ===\n");
        printf("  MISMATCHES = %d / 64 alignments\n", v9s_bad);
        if (v9s_bad) {
            printf("  alignments with mismatches:");
            for (int o = 0; o < 64; ++o) if (v9s_per_align[o]) printf(" %d(%d)", o, v9s_per_align[o]);
            printf("\n");
        }
        printf("  %s\n\n", v9s_bad == 0 ? "*** V9spec IS EXACT ***" : "*** V9spec IS WRONG ***");

        free(area);
        if (total_bad != 0 && v9_bad != 0 && v9s_bad != 0) {
            printf("all V9 readings failed; refusing to price them.\n");
            free(w); free(x);
            return 1;
        }
    }

    struct { const char *name; float (*fn)(const uint8_t *, const float *, int); } V[] = {
        { "V1 dequant+reload+FMA (shipped path)", v1_row },
        { "V2 register-resident decode+FMA",      v2_row },
        { "V3 no LUT permute (int bias convert) [WRONG]", v3_row },
        { "V4 ONE permute, code not converted  [WRONG]",  v4_row },
        { "V5 NO decode, FMA network only      [WRONG]",  v5_row },
        { "V7 vpmulld broadcast + LUT permute  [PRICE]", v7_row },
        { "V7b vpmulld+vpsrlvd only, no LUT    [PRICE]", v7b_row },
        { "V9 vpslld+vpmulld broadcast       [CAND]",  v9_row },
        { "V9spec vpslld+vpmulld, per-lane vectors [CAND]", v9spec_row },
    };

    for (unsigned v = 0; v < sizeof(V)/sizeof(V[0]); ++v) {
        double best = 1e30; double sink = 0;
        for (int r = 0; r < reps; ++r) {
            double t0 = now_s();
            double acc = 0;
            for (int i = 0; i < d; ++i) acc += V[v].fn(w + (size_t)i * row_bytes, x, n);
            double t1 = now_s();
            if (t1 - t0 < best) best = t1 - t0;
            sink += acc;
        }
        const double wps = (double)n * d / best;
        printf("%-46s %8.4f s   %8.1f M weights/s\n",
               V[v].name, best, wps / 1e6);
    }

    printf("\nnote: store traffic of V1 alone = %zu bytes per row\n", (size_t)n * 4);
    free(w); free(x);
    return 0;
}
