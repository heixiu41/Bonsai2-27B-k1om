/* typeaccept.c -- does the loader accept a tensor type that model.c cannot
 * decode?  (GGML_TYPE_PTQ1_0 = 143 is given the same 34-byte block size as
 * PQ2_0 in bonsai_tensor_nbytes, but no code in model.c decodes it.) */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "gguf.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "D:/zigtest/review/revgguf/ptq1_0.gguf";
    bonsai_gguf *g = bonsai_gguf_load(path);
    if (!g) { printf("load: NULL (clean error)\n"); return 1; }
    printf("load: SUCCESS  n_tensors=%llu\n", (unsigned long long)g->tensor_len);
    for (uint64_t i = 0; i < g->tensor_len; ++i) {
        const bonsai_tensor *t = &g->tensors[i];
        printf("  %-12s type=%u (%s) dims={%llu,%llu} nbytes=%llu n_elements=%llu\n",
               t->name, t->type, bonsai_type_name(t->type),
               (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
               (unsigned long long)t->nbytes, (unsigned long long)t->n_elements);
    }
    return 0;
}
