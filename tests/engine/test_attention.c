/*
 * test_attention.c — Phase 2-F: MHSA test (TDD Red)
 *
 * Test targets:
 *   - Multi-Head Self-Attention (n_heads=4)
 *   - head_dim=5 (Tiny) — Requires SIMD padding as it's not a multiple of 4
 *   - Self-attention with identical Q=K=V input
 *   - Zero query → output becomes weighted sum of value context
 *   - Softmax normalization
 *
 * Compile:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_attention.c \
 *       src/engine/common/attention.c src/engine/common/activations.c -o test_attention -lm
 */

#include "unity.h"
#include "attention.h"
#include <math.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* --- Basic tests --- */

void test_attention_output_finite(void) {
    /* Tiny: n_heads=4, head_dim=5, C2=20, freq=128 */
    int n_heads = 4;
    int head_dim = 5;
    int c2 = n_heads * head_dim;  /* =20 */
    int seq_len = 8;  /* Short sequence for verification */

    float input[8 * 20];
    for (int i = 0; i < seq_len * c2; i++) input[i] = 0.1f;

    /* Q,K,V weights: [C2 × C2] */
    float W_q[20 * 20], W_k[20 * 20], W_v[20 * 20], W_o[20 * 20];
    float b_q[20] = {0}, b_k[20] = {0}, b_v[20] = {0}, b_o[20] = {0};

    for (int i = 0; i < c2 * c2; i++) {
        W_q[i] = (i % (c2 + 1) == 0) ? 1.0f : 0.0f;
        W_k[i] = W_q[i];
        W_v[i] = W_q[i];
        W_o[i] = W_q[i];
    }

    FeMhsaWeights weights = {
        .W_q = W_q, .b_q = b_q,
        .W_k = W_k, .b_k = b_k,
        .W_v = W_v, .b_v = b_v,
        .W_o = W_o, .b_o = b_o,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .c2 = c2
    };

    float output[8 * 20];
    float attn_buf[4 * 8 * 8];
    float scratch[4 * 8 * 20];

    fe_mhsa(&weights, input, output, attn_buf, scratch, seq_len);

    for (int i = 0; i < seq_len * c2; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- Uniform input test --- */

void test_attention_uniform_input(void) {
    /* Q=K=V all the same → attention weights uniform → output=input */
    int n_heads = 4;
    int head_dim = 5;
    int c2 = 20;
    int seq_len = 4;

    float input[4 * 20];
    for (int i = 0; i < seq_len * c2; i++) input[i] = 1.0f;

    float W_q[400], W_k[400], W_v[400], W_o[400];
    float b_q[20] = {0}, b_k[20] = {0}, b_v[20] = {0}, b_o[20] = {0};

    /* Identity matrix */
    memset(W_q, 0, sizeof(W_q));
    memset(W_k, 0, sizeof(W_k));
    memset(W_v, 0, sizeof(W_v));
    memset(W_o, 0, sizeof(W_o));
    for (int i = 0; i < c2; i++) {
        W_q[i * c2 + i] = 1.0f;
        W_k[i * c2 + i] = 1.0f;
        W_v[i * c2 + i] = 1.0f;
        W_o[i * c2 + i] = 1.0f;
    }

    FeMhsaWeights weights = {
        .W_q = W_q, .b_q = b_q,
        .W_k = W_k, .b_k = b_k,
        .W_v = W_v, .b_v = b_v,
        .W_o = W_o, .b_o = b_o,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .c2 = c2
    };

    float output[4 * 20];
    float attn_buf[4 * 4 * 4];
    float scratch[4 * 4 * 20];

    fe_mhsa(&weights, input, output, attn_buf, scratch, seq_len);

    /* Uniform input + identity matrix → Q=K=V=input
     * attention weight = softmax(Q·K^T/sqrt(d)) = uniform distribution
     * output = attention · V = average of input = input itself (since all are the same) */
    for (int i = 0; i < seq_len * c2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, 1.0f, output[i]);
    }
}

/* --- Zero input test --- */

void test_attention_zero_input(void) {
    int n_heads = 4;
    int head_dim = 5;
    int c2 = 20;
    int seq_len = 4;

    float input[4 * 20] = {0};

    float W_q[400] = {0}, W_k[400] = {0}, W_v[400] = {0}, W_o[400] = {0};
    float b_q[20] = {0}, b_k[20] = {0}, b_v[20] = {0}, b_o[20] = {0};

    FeMhsaWeights weights = {
        .W_q = W_q, .b_q = b_q,
        .W_k = W_k, .b_k = b_k,
        .W_v = W_v, .b_v = b_v,
        .W_o = W_o, .b_o = b_o,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .c2 = c2
    };

    float output[4 * 20];
    float attn_buf[4 * 4 * 4];
    float scratch[4 * 4 * 20];

    fe_mhsa(&weights, input, output, attn_buf, scratch, seq_len);

    for (int i = 0; i < seq_len * c2; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, output[i]);
    }
}

/* --- softmax normalization test --- */

void test_softmax_normalization(void) {
    float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[8];

    fe_softmax(input, output, 8);

    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += output[i];
        TEST_ASSERT_TRUE(output[i] > 0.0f);
        TEST_ASSERT_TRUE(output[i] < 1.0f);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, sum);
}

void test_softmax_equal_inputs(void) {
    float input[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[4];

    fe_softmax(input, output, 4);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.25f, output[i]);
    }
}

void test_softmax_numerical_stability(void) {
    /* Large values don't overflow (max subtraction trick) */
    float input[4] = {1000.0f, 1001.0f, 1002.0f, 1003.0f};
    float output[4];

    fe_softmax(input, output, 4);

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
        TEST_ASSERT_TRUE(output[i] >= 0.0f);
        sum += output[i];
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, sum);
}

void test_softmax_extreme_values(void) {
    float input[4] = {-88.0f, 0.0f, 88.0f, 0.0f};
    float output[4];

    fe_softmax(input, output, 4);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += output[i];
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, sum);
}

/* --- head_dim=5 (not multiple of 4) test --- */

void test_attention_head_dim_5_simd_alignment(void) {
    /* head_dim=5 is not SIMD 4-element aligned. Verify padding is handled correctly. */
    int n_heads = 4;
    int head_dim = 5;
    int c2 = 20;
    int seq_len = 4;

    float input[4 * 20];
    for (int i = 0; i < seq_len * c2; i++) {
        input[i] = (float)(i % 7) * 0.1f;
    }

    float W_q[400], W_k[400], W_v[400], W_o[400];
    float b_q[20] = {0}, b_k[20] = {0}, b_v[20] = {0}, b_o[20] = {0};

    memset(W_q, 0, sizeof(W_q));
    memset(W_k, 0, sizeof(W_k));
    memset(W_v, 0, sizeof(W_v));
    memset(W_o, 0, sizeof(W_o));
    for (int i = 0; i < c2; i++) {
        W_q[i * c2 + i] = 1.0f;
        W_k[i * c2 + i] = 1.0f;
        W_v[i * c2 + i] = 1.0f;
        W_o[i * c2 + i] = 1.0f;
    }

    FeMhsaWeights weights = {
        .W_q = W_q, .b_q = b_q,
        .W_k = W_k, .b_k = b_k,
        .W_v = W_v, .b_v = b_v,
        .W_o = W_o, .b_o = b_o,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .c2 = c2
    };

    float output[4 * 20];
    float attn_buf[4 * 4 * 4];
    float scratch[4 * 4 * 20];

    fe_mhsa(&weights, input, output, attn_buf, scratch, seq_len);

    for (int i = 0; i < seq_len * c2; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }

    /* For identity weights (W=I), output matches attention-weighted average of V values.
     * Numerically verify that the 5th dimension (index 4,9,14,19) is correctly reflected in output. */
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < c2; d++) {
            /* Identity W + uniform attention means output ≈ column average of input */
            float col_avg = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                col_avg += input[t * c2 + d];
            }
            col_avg /= (float)seq_len;
            /* Softmax is not perfectly uniform so allow somewhat wider tolerance */
            TEST_ASSERT_FLOAT_WITHIN(0.15f, col_avg, output[s * c2 + d]);
        }
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_attention_output_finite);
    RUN_TEST(test_attention_uniform_input);
    RUN_TEST(test_attention_zero_input);
    RUN_TEST(test_softmax_normalization);
    RUN_TEST(test_softmax_equal_inputs);
    RUN_TEST(test_softmax_numerical_stability);
    RUN_TEST(test_softmax_extreme_values);
    RUN_TEST(test_attention_head_dim_5_simd_alignment);

    return UNITY_END();
}
