/*
 * test_gguf.c - host self-test for the Bonsai 2 k1om GGUF loader.
 *
 * Built with `zig cc` on the development machine and with the k1om toolchain on
 * the card. It parses the real 7.2 GB model and checks:
 *   - header fields against the values the official `llama-gguf r` reports
 *   - geometry against the published config
 *   - every tensor lies inside the data region
 *   - tensor shapes and computed byte sizes match the on-disk layout exactly
 */
#include "gguf.h"
#include "pq2_0.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...)                                            \
    do {                                                                 \
        if (cond) {                                                      \
            printf("  OK    " fmt "\n", ##__VA_ARGS__);                   \
        } else {                                                         \
            printf("  FAIL  " fmt "\n", ##__VA_ARGS__);                   \
            ++failures;                                                  \
        }                                                                \
    } while (0)

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
                                  : "D:/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf";

    printf("=== bonsai gguf loader self-test ===\n%s\n\n", path);
    if (bonsai_gguf_selftest(path) != 0) {
        printf("\nselftest reported problems\n");
        return 1;
    }

    printf("\n=== cross-checks against the official reader ===\n");
    bonsai_gguf *g = bonsai_gguf_load(path);
    if (!g) { printf("reload failed\n"); return 1; }

    /* Values reported by `llama-gguf.exe <file> r` */
    CHECK(g->version == 3, "version 3 (got %u)", g->version);
    CHECK(g->alignment == 32, "alignment 32 (got %llu)",
          (unsigned long long)g->alignment);
    CHECK(g->data_offset == 11120992ULL, "data offset 11120992 (got %llu)",
          (unsigned long long)g->data_offset);
    CHECK(g->n_kv == 49, "n_kv 49 (got %llu)", (unsigned long long)g->n_kv);
    CHECK(g->tensor_len == 851, "n_tensors 851 (got %llu)",
          (unsigned long long)g->tensor_len);

    /* Published architecture */
    CHECK(g->n_layer == 64, "block_count 64 (got %d)", g->n_layer);
    CHECK(g->n_embd == 5120, "embedding_length 5120 (got %d)", g->n_embd);
    CHECK(g->n_ff == 17408, "feed_forward_length 17408 (got %d)", g->n_ff);
    CHECK(g->n_vocab == 248320, "vocab 248320 (got %d)", g->n_vocab);
    CHECK(g->n_head == 24, "head_count 24 (got %d)", g->n_head);
    CHECK(g->n_head_kv == 4, "head_count_kv 4 (got %d)", g->n_head_kv);
    CHECK(g->head_dim == 256, "key_length 256 (got %d)", g->head_dim);
    CHECK(g->n_rot == 64, "rope.dimension_count 64 (got %d)", g->n_rot);
    CHECK(g->ssm_d_conv == 4, "ssm.conv_kernel 4 (got %d)", g->ssm_d_conv);
    CHECK(g->ssm_d_state == 128, "ssm.state_size 128 (got %d)", g->ssm_d_state);
    CHECK(g->ssm_dt_rank == 48, "ssm.time_step_rank 48 (got %d)", g->ssm_dt_rank);
    CHECK(g->ssm_n_group == 16, "ssm.group_count 16 (got %d)", g->ssm_n_group);
    CHECK(g->full_attn_interval == 4, "full_attention_interval 4 (got %d)",
          g->full_attn_interval);
    CHECK(g->had_block == 1024, "hadamard block_size 1024 (got %d)", g->had_block);
    CHECK(g->had_sign_width_count == 3 &&
          g->had_sign_widths[0] == 5120 && g->had_sign_widths[1] == 6144 &&
          g->had_sign_widths[2] == 17408,
          "hadamard sign widths [5120 6144 17408]");
    CHECK(g->rope_sections[0] == 11 && g->rope_sections[1] == 11 &&
          g->rope_sections[2] == 10 && g->rope_sections[3] == 0,
          "rope sections [11 11 10 0] (got [%lld %lld %lld %lld])",
          (long long)g->rope_sections[0], (long long)g->rope_sections[1],
          (long long)g->rope_sections[2], (long long)g->rope_sections[3]);

    printf("\n=== tensor shape spot-checks ===\n");
    struct { const char *name; uint64_t d0, d1; uint32_t type; } want[] = {
        { "token_embd.weight",           5120, 248320, GGML_TYPE_PQ2_0 },
        { "output.weight",               5120, 248320, GGML_TYPE_PQ2_0 },
        { "blk.0.attn_qkv.weight",       5120,  10240, GGML_TYPE_PQ2_0 },
        { "blk.0.attn_gate.weight",      5120,   6144, GGML_TYPE_PQ2_0 },
        { "blk.0.ssm_out.weight",        6144,   5120, GGML_TYPE_PQ2_0 },
        { "blk.0.ffn_gate.weight",       5120,  17408, GGML_TYPE_PQ2_0 },
        { "blk.0.ffn_down.weight",      17408,   5120, GGML_TYPE_PQ2_0 },
        { "blk.0.ssm_conv1d.weight",        4,  10240, GGML_TYPE_F32  },
        { "blk.0.ssm_alpha.weight",      5120,     48, GGML_TYPE_BF16 },
        { "blk.3.attn_q.weight",         5120,  12288, GGML_TYPE_PQ2_0 },
        { "blk.3.attn_k.weight",         5120,   1024, GGML_TYPE_PQ2_0 },
    };
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i) {
        const bonsai_tensor *t = bonsai_tensor_get(g, want[i].name);
        if (!t) { printf("  FAIL  %s missing\n", want[i].name); ++failures; continue; }
        int ok = (t->dims[0] == want[i].d0) && (t->dims[1] == want[i].d1) &&
                 (t->type == want[i].type);
        CHECK(ok, "%-24s dims={%llu,%llu} %s", want[i].name,
              (unsigned long long)t->dims[0], (unsigned long long)t->dims[1],
              bonsai_type_name(t->type));
    }

    printf("\n=== PQ2_0 byte accounting ===\n");
    {
        const bonsai_tensor *t = bonsai_tensor_get(g, "blk.0.attn_qkv.weight");
        uint64_t expect = t->n_elements / 128 * 34;
        CHECK(t->nbytes == expect,
              "attn_qkv: %llu elements -> %llu bytes (2.125 bpw)",
              (unsigned long long)t->n_elements, (unsigned long long)t->nbytes);
    }
    {
        uint64_t total = 0, ternary = 0;
        for (uint64_t i = 0; i < g->tensor_len; ++i) {
            total += g->tensors[i].nbytes;
            if (g->tensors[i].type == GGML_TYPE_PQ2_0) ternary += 1;
        }
        printf("  total tensor bytes = %llu\n", (unsigned long long)total);
        printf("  PQ2_0 tensor count = %llu\n", (unsigned long long)ternary);
        CHECK(total + g->data_offset == g->file_size,
              "regions tile the file exactly (data_offset + sum == file_size)");
    }

    printf("\n=== fp16 decode sanity ===\n");
    {
        /* Bit patterns produced by numpy's float16 view, so the expected
         * values are authoritative rather than hand-computed. */
        CHECK(bonsai_fp16_to_fp32(0x0000) == 0.0f, "fp16 0x0000 -> 0");
        CHECK(bonsai_fp16_to_fp32(0x3C00) == 1.0f, "fp16 0x3C00 -> 1");
        CHECK(bonsai_fp16_to_fp32(0xC000) == -2.0f, "fp16 0xC000 -> -2");
        float v = bonsai_fp16_to_fp32(0x6059);   /* 556.5, a real PQ2_0 scale */
        CHECK(v == 556.5f, "fp16 0x6059 -> %.4f (expect 556.5)", v);
        float w = bonsai_fp16_to_fp32(0x21E3);   /* ~0.0114975, a typical scale */
        CHECK(w > 0.01149f && w < 0.01150f,
              "fp16 0x21E3 -> %.9f (expect ~0.011497)", w);
        /* subnormal: 0x0001 = 2^-24 */
        float s = bonsai_fp16_to_fp32(0x0001);
        CHECK(s > 5.9e-8f && s < 6.0e-8f, "fp16 0x0001 -> %.3e (expect 2^-24)", s);
    }

    printf("\n=== PQ2_0 decode sanity ===\n");
    {
        uint8_t blk[PQ2_0_BLOCK_BYTES];
        memset(blk, 0, sizeof(blk));
        blk[0] = 0x00; blk[1] = 0x3C;          /* d = 1.0 */
        blk[2] = 0xE4;                          /* codes 0,1,2,3 in slots 0..3 */
        float out[QK_PQ2_0];
        pq2_0_dequant_block(blk, out);
        CHECK(out[0] == -1.0f && out[1] == 0.0f && out[2] == 1.0f && out[3] == 2.0f,
              "code->value {0:-1, 1:0, 2:+1, 3:+2} -> %.1f %.1f %.1f %.1f",
              out[0], out[1], out[2], out[3]);
    }

    bonsai_gguf_free(g);
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
