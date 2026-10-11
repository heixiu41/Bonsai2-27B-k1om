/* audit_t4_had.c -- Hadamard transform audit (pq2_0.h:404-457).
 *
 *   zig cc -O2 -std=c11 -o D:\zigtest\t4.exe audit_t4_had.c -I . -lm
 *
 * Checks:
 *   1. fwht_inplace(n) == explicit Sylvester matrix M[r][c] = parity(r&c)
 *      (the exact convention pq2_0.h:410 documents, H[r][c] = parity(r&c)/sqrt(N))
 *   2. normalized H is orthogonal: ||H x|| == ||x||, and H(H x) == x
 *   3. the `inverse` flag is inert: H(x, inverse=1) is bit-identical to
 *      H(x, inverse=0)  (pq2_0.h:445-455 -- both branches are the same code)
 *   4. the signs branch really computes H . (s * x)
 */
#include "pq2_0.h"
#include <stdio.h>
#include <string.h>

static int parity(unsigned v) { int p = 0; while (v) { p ^= (int)(v & 1u); v >>= 1; } return p; }

static long fail_parity = 0, cases_parity = 0;
static void test_fwht_matrix(void)
{
    for (int n = 1; n <= 32; n <<= 1) {
        float x[64], y[64], want[64];
        for (int c = 0; c < n; ++c) x[c] = (float)((c * 37) % 11) - 5.0f;
        for (int r = 0; r < n; ++r) {
            float s = 0.0f;
            for (int c = 0; c < n; ++c) s += (parity((unsigned)(r & c)) ? -1.0f : 1.0f) * x[c];
            want[r] = s;
        }
        memcpy(y, x, sizeof(float) * n);
        fwht_inplace(y, n);
        for (int r = 0; r < n; ++r) {
            cases_parity++;
            if (memcmp(&y[r], &want[r], 4) != 0) {
                if (fail_parity < 5)
                    printf("   fwht n=%d row %d: got %a want %a\n",
                           n, r, (double)y[r], (double)want[r]);
                fail_parity++;
            }
        }
    }
    printf("(1) fwht_inplace == parity matrix M[r][c]=(-1)^popcount(r&c): %ld/%ld values match\n",
           cases_parity - fail_parity, cases_parity);
}

static void test_hadamard_normalisation(void)
{
    enum { N = 1024 };
    static float x[N], y[N], z[N];
    for (int i = 0; i < N; ++i) x[i] = (float)((i * 2654435761u) % 1000) / 500.0f - 1.0f;
    double n0 = 0.0, n1 = 0.0;
    for (int i = 0; i < N; ++i) n0 += (double)x[i] * x[i];

    memcpy(y, x, sizeof x);
    hadamard_forward(y, NULL, N, N, 0);
    for (int i = 0; i < N; ++i) n1 += (double)y[i] * y[i];
    printf("(2a) ||x||^2=%.9f  ||H x||^2=%.9f  (relative %.3g)\n",
           n0, n1, (n1 - n0) / n0);

    memcpy(z, y, sizeof y);
    hadamard_forward(z, NULL, N, N, 0);
    double maxd = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = (double)z[i] - x[i];
        if (d < 0) d = -d;
        if (d > maxd) maxd = d;
    }
    printf("(2b) max |H(H x) - x| = %.3g\n", maxd);

    /* the documented entry values: row r of a basis vector e_c is parity(r&c)/32 */
    long bad = 0, tested = 0;
    static float e[N], o[N];
    for (int t = 0; t < 24; ++t) {
        const int r = (int)((t * 61u) % (unsigned)N), c = (int)((t * 397u) % (unsigned)N);
        memset(e, 0, sizeof e);
        e[c] = 1.0f;
        memcpy(o, e, sizeof e);
        hadamard_forward(o, NULL, N, N, 0);
        const float want = parity((unsigned)(r & c)) ? -0.03125f : 0.03125f;
        tested++;
        if (memcmp(&o[r], &want, 4) != 0) { bad++;
            printf("   H[%d][%d] = %a want %a\n", r, c, (double)o[r], (double)want); }
    }
    printf("(2c) sampled H[r][c] == parity(r&c)/sqrt(1024): %ld bad of %ld\n", bad, tested);

    /* inverse flag inertness */
    float a[N], b[N];
    memcpy(a, x, sizeof x); hadamard_forward(a, NULL, N, N, 0);
    memcpy(b, x, sizeof x); hadamard_forward(b, NULL, N, N, 1);
    printf("(3) inverse=1 output identical to inverse=0: %s\n",
           memcmp(a, b, sizeof a) == 0 ? "YES (flag is inert)" : "NO");
}

static void test_signs_branch(void)
{
    enum { N = 8 };
    float x[N], y[N], want[N], signs[N];
    for (int i = 0; i < N; ++i) { x[i] = (float)i - 3.0f; signs[i] = (i % 3) ? 1.0f : -1.0f; }
    for (int r = 0; r < N; ++r) {
        float s = 0.0f;
        for (int c = 0; c < N; ++c)
            s += (parity((unsigned)(r & c)) ? -1.0f : 1.0f) * (signs[c] * x[c]);
        want[r] = s / sqrtf((float)N);
    }
    memcpy(y, x, sizeof x);
    hadamard_forward(y, signs, N, N, 0);
    long bad = 0;
    for (int r = 0; r < N; ++r) {
        float d = y[r] - want[r]; if (d < 0) d = -d;
        if (d > 1e-6f) bad++;
    }
    printf("(4) signs branch H.(s*x): %ld mismatches of %d\n", bad, N);
}

/* hadamard_forward walks [base, base+block) for every base < n, so if
 * block does not divide n the last block runs past the end of the n-element
 * buffer. model.c never validates that had_block divides n. */
static void test_block_divides_n(void)
{
    static float buf[16];
    for (int i = 0; i < 16; ++i) buf[i] = 1.0f;
    const int n = 10, block = 8;              /* 8 does not divide 10 */
    hadamard_forward(buf, NULL, n, block, 0);
    int clobbered = 0;
    for (int i = n; i < 16; ++i) if (buf[i] != 1.0f) clobbered++;
    printf("(5) hadamard_forward(n=10, block=8): %d of %d elements PAST index n-1 "
           "were written (guard against block !| n is absent in model.c)\n",
           clobbered, 16 - n);
}

int main(void)
{
    printf("T4 Hadamard audit\n");
    test_fwht_matrix();
    test_hadamard_normalisation();
    test_signs_branch();
    test_block_divides_n();
    return 0;
}
