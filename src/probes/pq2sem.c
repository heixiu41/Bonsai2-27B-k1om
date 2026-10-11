/* Empirically pin down the two intrinsic semantics that V8's derivation got
 * wrong. Both were ASSUMED and never measured:
 *
 *   1. _mm512_permutevar_epi32(idx, data) -- which argument is the index, and
 *      what does lane i of the result equal?
 *   2. _mm512_set_epi32(...) -- is the first (leftmost) argument lane 15?
 *   3. _mm512_srlv_epi32(data, counts) -- per-lane variable shift direction.
 *
 * Method: make every lane of the data vector carry a unique value, make the
 * index vector's lane i equal i, and read the result. Then vary one lane of
 * the index at a time to see which data lane is selected.
 *
 * Earlier versions of this project's probes segfaulted from unaligned vector
 * stores and from reading past buffers, so: all store targets are 64-byte
 * aligned, the data buffer is generously oversized, and this program performs
 * no out-of-range loads.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

static void dump(const char *tag, __m512i v) {
    int l[16] __attribute__((aligned(64)));
    _mm512_store_epi32(l, v);
    printf("%-30s :", tag);
    for (int i = 0; i < 16; ++i) printf(" %4d", l[i]);
    printf("\n");
}

int main(void) {
    printf("=== 1. _mm512_permutevar_epi32(idx, data) ===\n\n");

    /* data lane i = 1000 + i, so a result lane shows which data lane it came from */
    const __m512i data = _mm512_set_epi32(
        1015,1014,1013,1012, 1011,1010,1009,1008,
        1007,1006,1005,1004, 1003,1002,1001,1000);
    dump("data lane i = 1000+i", data);

    /* idx lane i = i  (in build order, so the leftmost arg is lane 15) */
    const __m512i idn = _mm512_set_epi32(
        15,14,13,12, 11,10,9,8, 7,6,5,4, 3,2,1,0);
    dump("idx lane i = i", idn);

    dump("permutevar(idx, data)", _mm512_permutevar_epi32(idn, data));
    dump("permutevar(data, idx)", _mm512_permutevar_epi32(data, idn));

    /* Now select data lane 3 into every result lane */
    const __m512i all3 = _mm512_set1_epi32(3);
    dump("permutevar(all-3, data)", _mm512_permutevar_epi32(all3, data));

    /* And test the masking behaviour: 16-entry wrap (KNC uses only 4 bits) */
    const __m512i all19 = _mm512_set1_epi32(19);   /* 19 & 15 == 3 */
    dump("permutevar(all-19, data)", _mm512_permutevar_epi32(all19, data));

    printf("\n=== 2. _mm512_set_epi32 byte/argument order ===\n\n");
    /* argument order: set_epi32(e15,...,e0) per Intel docs */
    const __m512i s = _mm512_set_epi32(15,14,13,12, 11,10,9,8, 7,6,5,4, 3,2,1,0);
    int ls[16] __attribute__((aligned(64)));
    _mm512_store_epi32(ls, s);
    printf("set_epi32(15,...,0) -> lane values:");
    for (int i = 0; i < 16; ++i) printf(" %d", ls[i]);
    printf("\n  => lane 0 = %d (so the LAST argument is lane 0)\n", ls[0]);

    printf("\n=== 3. _mm512_srlv_epi32(data, counts) ===\n\n");
    /* lane i holds 0x100 * (i+1); shift each lane by its own index */
    const __m512i d3 = _mm512_set_epi32(
        16*16,15*16,14*16,13*16, 12*16,11*16,10*16,9*16,
        8*16,7*16,6*16,5*16, 4*16,3*16,2*16,1*16);
    const __m512i c3 = _mm512_set_epi32(
        15,14,13,12, 11,10,9,8, 7,6,5,4, 3,2,1,0);
    dump("data (lane i = 16*(i+1))", d3);
    dump("counts lane i = i", c3);
    dump("srlv(data, counts)", _mm512_srlv_epi32(d3, c3));
    printf("  (lane i should be 16*(i+1) >> i if the shift is per-lane right)\n");

    printf("\n=== 4. what the shipped decoder's LUT actually selects ===\n\n");
    /* Reproduce the LUT bit patterns exactly as pq2_0.h builds them. */
    const __m512i lut = _mm512_set_epi32(
        (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u,
        (int)0x40000000u,(int)0x3F800000u,(int)0x00000000u,(int)0xBF800000u);
    /* index = lane number, so result lane i selects LUT lane i */
    dump("lut as ints (bit patterns)", lut);
    __m512 f;
    _mm512_store_ps((float *)ls, _mm512_castsi512_ps(lut));
    printf("lut as floats            :");
    for (int i = 0; i < 16; ++i) printf(" %4.0f", ((float *)ls)[i]);
    printf("\n  => LUT lane i = %g, and index (i & 3) selects...\n",
           ((float *)ls)[0]);
    /* apply: index vector = 0,1,2,3 repeating (lane order matters) */
    const __m512i idx4 = _mm512_set_epi32(
        3,2,1,0, 3,2,1,0, 3,2,1,0, 3,2,1,0);
    f = _mm512_castsi512_ps(_mm512_permutevar_epi32(idx4, lut));
    _mm512_store_ps((float *)ls, f);
    printf("permutevar(0,1,2,3 x4, lut) ->");
    for (int i = 0; i < 16; ++i) printf(" %4.0f", ((float *)ls)[i]);
    printf("\n  expected for code->value {0:-1,1:0,2:+1,3:+2}: -1 0 1 2 repeating\n");
    printf("\n=== 5. _mm512_mullo_epi32: does k * 0x01010101 replicate the byte? ===\n\n");
    {
        /* lane i = i (0..15); expect lane i -> i * 0x01010101 */
        const __m512i kk = _mm512_set_epi32(
            15,14,13,12, 11,10,9,8, 7,6,5,4, 3,2,1,0);
        const __m512i mc = _mm512_set1_epi32(0x01010101);
        dump("k lane i = i", kk);
        dump("mullo(k, 0x01010101)", _mm512_mullo_epi32(kk, mc));
        int lp[16] __attribute__((aligned(64)));
        _mm512_store_epi32(lp, _mm512_mullo_epi32(kk, mc));
        printf("  lane 0 = 0x%08X (expect 0x00000000)\n", (unsigned)lp[0]);
        printf("  lane 1 = 0x%08X (expect 0x01010101)\n", (unsigned)lp[1]);
        printf("  lane 3 = 0x%08X (expect 0x03030303)\n", (unsigned)lp[3]);
        printf("  lane 15 = 0x%08X (expect 0x0F0F0F0F)\n", (unsigned)lp[15]);
        printf("  => mullo replicates the byte: %s\n",
               lp[1] == 0x01010101 && lp[3] == 0x03030303 && lp[15] == 0x0F0F0F0F
               ? "CONFIRMED" : "REFUTED");
    }

    printf("\n=== 6. does mullo see garbage above the field? ===\n\n");
    {
        /* What the decoder does: dword holds a full byte; shift by 2q leaves
         * the 2-bit field at the bottom but ALSO higher bits of the byte. */
        const __m512i d = _mm512_set1_epi32(0x000000B4);   /* byte 0xB4 */
        const __m512i sh2 = _mm512_set1_epi32(2);
        const __m512i sh6 = _mm512_set1_epi32(6);
        const __m512i mc = _mm512_set1_epi32(0x01010101);
        int l2[16] __attribute__((aligned(64)));
        _mm512_store_epi32(l2, _mm512_srlv_epi32(d, sh2));
        printf("  0xB4 >> 2            = 0x%08X  (low2 = %u)\n",
               (unsigned)l2[0], (unsigned)(l2[0] & 3));
        _mm512_store_epi32(l2, _mm512_mullo_epi32(_mm512_srlv_epi32(d, sh2), mc));
        printf("  (0xB4>>2) * 0x01010101 = 0x%08X   low byte = 0x%02X\n",
               (unsigned)l2[0], (unsigned)(l2[0] & 0xFF));
        _mm512_store_epi32(l2, _mm512_mullo_epi32(_mm512_srlv_epi32(d, sh6), mc));
        printf("  (0xB4>>6) * 0x01010101 = 0x%08X   low byte = 0x%02X\n",
               (unsigned)l2[0], (unsigned)(l2[0] & 0xFF));
        printf("  NOTE: the LUT masks with &15, and byte (i&3) of the product is\n");
        printf("        exactly k, so only k matters -- high bits must NOT appear\n");
        printf("        in the low byte of ANY byte position.\n");
    }

    return 0;
}
