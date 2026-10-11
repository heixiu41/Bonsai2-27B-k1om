/* audit_t5_fp16.c -- dump every bonsai_fp16_to_fp32(h) result as raw f32 bits,
 * for exhaustive comparison against numpy's half->float oracle.
 *
 *   zig cc -O2 -std=c11 -o D:\zigtest\t5.exe audit_t5_fp16.c -I . -lm
 *   D:\zigtest\t5.exe > D:\zigtest\fp16_bits.txt
 */
#include "pq2_0.h"
#include <stdio.h>
#include <string.h>

int main(void)
{
    for (uint32_t h = 0; h < 65536u; ++h) {
        const float f = bonsai_fp16_to_fp32((uint16_t)h);
        uint32_t u;
        memcpy(&u, &f, 4);
        printf("%08x\n", u);
    }
    return 0;
}
