/* audit_t6_arith.c -- index-arithmetic / UB consequences in model.c and pq2_0.h.
 *
 *   zig cc -O2 -std=c11 -o D:\zigtest\t6.exe audit_t6_arith.c -I . -lm
 *   D:\zigtest\t6.exe            (safe demos)
 *   D:\zigtest\t6.exe divide     (shows int division by zero traps)
 *   D:\zigtest\t6.exe convhist   (shows the (ck-2) size_t underflow)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";

    if (!strcmp(mode, "convhist")) {
        /* model.c:303 calloc(L*(ck-1)*conv_dim) with ck=ssm_d_conv from the GGUF.
         * ck==1 -> calloc(0) (non-NULL on glibc) and then model.c:539 does
         *   memcpy(hist + (size_t)(ck-2)*conv_dim, s->qkv, conv_dim*4)
         * with ck-2 == -1 converted to size_t. */
        const int ck = 1, conv_dim = 10240;
        const size_t off = (size_t)(ck - 2) * (size_t)conv_dim;   /* exactly model.c:539 */
        printf("t6 convhist: ssm_d_conv=%d -> hist + %zu floats, i.e. %zu bytes "
               "BEFORE the buffer (%.1f KB of heap corruption)\n",
               ck, off, off * sizeof(float), (double)off * 4.0 / 1024.0);
        printf("             (ck==0 aborts earlier: calloc(L*(0-1)*conv_dim) fails)\n");
        return 0;
    }

    if (!strcmp(mode, "divide")) {
        /* model.c:267  m->head_v_dim = m->value_dim / m->ssm_dt_rank;
         * model.c:479  const int rep = vh / kh;   (kh = ssm_group)
         * both run before any validation of ssm_dt_rank / ssm_group. */
        volatile int dt_rank = 0;                 /* as a malformed GGUF would give */
        const int value_dim = 6144;
        printf("t6 divide: about to execute value_dim / ssm_dt_rank with ssm_dt_rank=%d\n",
               (int)dt_rank);
        fflush(stdout);
        printf("  result = %d\n", value_dim / (int)dt_rank);
        printf("  (unreachable: SIGFPE)\n");
        return 0;
    }

    /* default: the shift/mask and offset arithmetic used by the decoder */
    printf("t6 arith:\n");
    printf("  shift amounts (j&3)*2 for j=0..3: ");
    for (int j = 0; j < 4; ++j) printf("%d ", (j & 3) * 2);
    printf("(< 32, no UB)\n");
    printf("  bit index 2 + (j%%128)/4 max = %d (block[33], the last code byte)\n",
           2 + (127 % 128) / 4);
    printf("  _mm512_load_ps(x + j + k) byte offsets for j multiple of 128:\n   ");
    {
        const int ks[8] = { 0, 16, 32, 48, 64, 80, 96, 112 };
        for (int i = 0; i < 8; ++i) printf("k=%d:%d%s", ks[i], ks[i] * 4,
                                           (ks[i] * 4) % 64 ? " MISALIGNED" : "");
        printf("  -> all multiples of 64 bytes\n");
    }
    printf("  PQ2_0 block stride 34 B; j stride 512 B; (n/128) int arithmetic max "
           "= 17408/128*34 = %d (fits int)\n", 17408 / 128 * 34);
    return 0;
}
