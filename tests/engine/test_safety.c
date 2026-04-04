/*
 * test_safety.c — E1-E7 Safety Tests (GPT-5.4 Review Findings)
 *
 * E1: GRU hidden_size upper limit validation
 * E3: compression n<=0 guard
 * E4: GRU reset size<=0 guard
 * E5: attention parameter consistency validation
 * E6: softmax NaN/Inf guard
 * E7: setup_weights termination validation (indirect)
 */

#include "unity.h"
#include "gru.h"
#include "compression.h"
#include "attention.h"
#include "fastenhancer.h"
#include "tiny_48k.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

/* ---- E1: GRU hidden_size upper limit check ---- */

void test_gru_step_rejects_oversized_hidden(void) {
    float dummy_weights[512];
    float input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hidden_backup[4];

    memset(dummy_weights, 0, sizeof(dummy_weights));
    memcpy(hidden_backup, hidden, sizeof(hidden));

    FeGruWeights w = {
        .W_z = dummy_weights, .U_z = dummy_weights, .b_z = dummy_weights,
        .W_r = dummy_weights, .U_r = dummy_weights, .b_r = dummy_weights,
        .W_n = dummy_weights, .U_n = dummy_weights,
        .b_in_n = dummy_weights, .b_hn_n = dummy_weights,
        .input_size = 4,
        .hidden_size = FE_GRU_MAX_HIDDEN + 1
    };

    fe_gru_step(&w, input, hidden);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(hidden_backup[i], hidden[i]);
    }
}

void test_gru_step_rejects_zero_hidden(void) {
    float dummy[16] = {0};
    float hidden[1] = {5.0f};

    FeGruWeights w = {
        .W_z = dummy, .U_z = dummy, .b_z = dummy,
        .W_r = dummy, .U_r = dummy, .b_r = dummy,
        .W_n = dummy, .U_n = dummy,
        .b_in_n = dummy, .b_hn_n = dummy,
        .input_size = 1,
        .hidden_size = 0
    };

    fe_gru_step(&w, dummy, hidden);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, hidden[0]);
}

void test_gru_step_rejects_negative_hidden(void) {
    float dummy[16] = {0};
    float hidden[1] = {7.0f};

    FeGruWeights w = {
        .W_z = dummy, .U_z = dummy, .b_z = dummy,
        .W_r = dummy, .U_r = dummy, .b_r = dummy,
        .W_n = dummy, .U_n = dummy,
        .b_in_n = dummy, .b_hn_n = dummy,
        .input_size = 1,
        .hidden_size = -1
    };

    fe_gru_step(&w, dummy, hidden);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, hidden[0]);
}

void test_gru_step_accepts_max_hidden(void) {
    /* FE_GRU_MAX_HIDDEN should be accepted */
    int hs = FE_GRU_MAX_HIDDEN;
    int is = hs;
    int mat_sz = hs * is;

    float* W = (float*)calloc(mat_sz, sizeof(float));
    float* b = (float*)calloc(hs, sizeof(float));
    float* input = (float*)calloc(is, sizeof(float));
    float* hidden = (float*)calloc(hs, sizeof(float));

    TEST_ASSERT_NOT_NULL(W);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_NOT_NULL(input);
    TEST_ASSERT_NOT_NULL(hidden);

    FeGruWeights w = {
        .W_z = W, .U_z = W, .b_z = b,
        .W_r = W, .U_r = W, .b_r = b,
        .W_n = W, .U_n = W,
        .b_in_n = b, .b_hn_n = b,
        .input_size = is,
        .hidden_size = hs
    };

    fe_gru_step(&w, input, hidden);

    for (int i = 0; i < hs; i++) {
        TEST_ASSERT_FALSE(isnan(hidden[i]));
    }

    free(W);
    free(b);
    free(input);
    free(hidden);
}

/* ---- E3: compression n<=0 guard ---- */

void test_compress_rejects_negative_n(void) {
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {99.0f, 99.0f, 99.0f, 99.0f};
    fe_power_compress(in, out, -1, 0.3f);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out[0]);
}

void test_compress_rejects_negative_n_zero_exp(void) {
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {99.0f, 99.0f, 99.0f, 99.0f};
    fe_power_compress(in, out, -1, 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out[0]);
}

void test_decompress_rejects_negative_n(void) {
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {99.0f, 99.0f, 99.0f, 99.0f};
    fe_power_decompress(in, out, -1, 0.3f);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out[0]);
}

void test_compress_complex_rejects_negative_n(void) {
    float re[4] = {1.0f}, im[4] = {1.0f};
    float out_re[4] = {99.0f}, out_im[4] = {99.0f};
    fe_power_compress_complex(re, im, out_re, out_im, -1, 0.3f);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out_re[0]);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out_im[0]);
}

void test_decompress_complex_rejects_negative_n(void) {
    float re[4] = {1.0f}, im[4] = {1.0f};
    float out_re[4] = {99.0f}, out_im[4] = {99.0f};
    fe_power_decompress_complex(re, im, out_re, out_im, -1, 0.3f);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out_re[0]);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, out_im[0]);
}

/* ---- E4: GRU reset size<=0 guard ---- */

void test_gru_reset_rejects_zero_size(void) {
    float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    fe_gru_reset_hidden(hidden, 0);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, hidden[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, hidden[3]);
}

void test_gru_reset_rejects_negative_size(void) {
    float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    fe_gru_reset_hidden(hidden, -1);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, hidden[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, hidden[3]);
}

/* ---- E5: attention parameter consistency ---- */

void test_mhsa_rejects_inconsistent_c2(void) {
    float dummy[256];
    float input[20];
    float output[20];
    float attn_buf[100];
    float scratch[400];

    memset(dummy, 0, sizeof(dummy));
    memset(input, 0, sizeof(input));
    memset(scratch, 0, sizeof(scratch));
    memset(attn_buf, 0, sizeof(attn_buf));
    for (int i = 0; i < 20; i++) output[i] = 99.0f;

    FeMhsaWeights w = {
        .W_q = dummy, .b_q = dummy,
        .W_k = dummy, .b_k = dummy,
        .W_v = dummy, .b_v = dummy,
        .W_o = dummy, .b_o = dummy,
        .n_heads = 4,
        .head_dim = 3,
        .c2 = 20
    };

    fe_mhsa(&w, input, output, attn_buf, scratch, 5);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, output[0]);
}

void test_mhsa_rejects_zero_seq_len(void) {
    float dummy[16];
    float input[1] = {0};
    float output[1] = {99.0f};
    float attn_buf[1] = {0};
    float scratch[4] = {0};

    memset(dummy, 0, sizeof(dummy));

    FeMhsaWeights w = {
        .W_q = dummy, .b_q = dummy,
        .W_k = dummy, .b_k = dummy,
        .W_v = dummy, .b_v = dummy,
        .W_o = dummy, .b_o = dummy,
        .n_heads = 1, .head_dim = 1, .c2 = 1
    };

    fe_mhsa(&w, input, output, attn_buf, scratch, 0);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, output[0]);
}

void test_mhsa_rejects_negative_seq_len(void) {
    float dummy[16];
    float input[1] = {0};
    float output[1] = {99.0f};
    float attn_buf[1] = {0};
    float scratch[4] = {0};

    memset(dummy, 0, sizeof(dummy));

    FeMhsaWeights w = {
        .W_q = dummy, .b_q = dummy,
        .W_k = dummy, .b_k = dummy,
        .W_v = dummy, .b_v = dummy,
        .W_o = dummy, .b_o = dummy,
        .n_heads = 1, .head_dim = 1, .c2 = 1
    };

    fe_mhsa(&w, input, output, attn_buf, scratch, -1);
    TEST_ASSERT_EQUAL_FLOAT(99.0f, output[0]);
}

/* ---- E6: softmax NaN/Inf guard ---- */

void test_softmax_handles_nan_input(void) {
    float input[4] = {1.0f, NAN, 2.0f, 0.5f};
    float output[4];

    fe_softmax(input, output, 4);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
        TEST_ASSERT_TRUE(output[i] >= 0.0f);
    }

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += output[i];
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum);
}

void test_softmax_handles_inf_input(void) {
    float input[4] = {1.0f, INFINITY, -INFINITY, 2.0f};
    float output[4];

    fe_softmax(input, output, 4);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
        TEST_ASSERT_TRUE(output[i] >= 0.0f);
    }

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += output[i];
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum);
}

void test_softmax_handles_all_nan(void) {
    float input[3] = {NAN, NAN, NAN};
    float output[3];

    fe_softmax(input, output, 3);

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

void test_softmax_handles_all_neg_inf(void) {
    float input[3] = {-INFINITY, -INFINITY, -INFINITY};
    float output[3];

    fe_softmax(input, output, 3);

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* ---- E7: setup_weights termination validation (indirect) ---- */

void test_weight_count_matches_config(void) {
    TEST_ASSERT_EQUAL_INT(FE_TOTAL_WEIGHTS, fe_weight_count(FE_MODEL_TINY));
}

void test_create_with_correct_weight_count_succeeds(void) {
    float* weights = (float*)calloc(FE_TOTAL_WEIGHTS, sizeof(float));
    TEST_ASSERT_NOT_NULL(weights);

    FeState* state = fe_create(FE_MODEL_TINY, weights, FE_TOTAL_WEIGHTS);
    TEST_ASSERT_NOT_NULL(state);

    fe_destroy(state);
    free(weights);
}

void test_create_with_wrong_weight_count_fails(void) {
    float* weights = (float*)calloc(FE_TOTAL_WEIGHTS + 1, sizeof(float));
    TEST_ASSERT_NOT_NULL(weights);

    FeState* state = fe_create(FE_MODEL_TINY, weights, FE_TOTAL_WEIGHTS + 1);
    TEST_ASSERT_NULL(state);

    free(weights);
}

int main(void) {
    UNITY_BEGIN();

    /* E1: GRU hidden_size */
    RUN_TEST(test_gru_step_rejects_oversized_hidden);
    RUN_TEST(test_gru_step_rejects_zero_hidden);
    RUN_TEST(test_gru_step_rejects_negative_hidden);
    RUN_TEST(test_gru_step_accepts_max_hidden);

    /* E3: compression n<=0 */
    RUN_TEST(test_compress_rejects_negative_n);
    RUN_TEST(test_compress_rejects_negative_n_zero_exp);
    RUN_TEST(test_decompress_rejects_negative_n);
    RUN_TEST(test_compress_complex_rejects_negative_n);
    RUN_TEST(test_decompress_complex_rejects_negative_n);

    /* E4: GRU reset */
    RUN_TEST(test_gru_reset_rejects_zero_size);
    RUN_TEST(test_gru_reset_rejects_negative_size);

    /* E5: attention */
    RUN_TEST(test_mhsa_rejects_inconsistent_c2);
    RUN_TEST(test_mhsa_rejects_zero_seq_len);
    RUN_TEST(test_mhsa_rejects_negative_seq_len);

    /* E6: softmax */
    RUN_TEST(test_softmax_handles_nan_input);
    RUN_TEST(test_softmax_handles_inf_input);
    RUN_TEST(test_softmax_handles_all_nan);
    RUN_TEST(test_softmax_handles_all_neg_inf);

    /* E7: weight count */
    RUN_TEST(test_weight_count_matches_config);
    RUN_TEST(test_create_with_correct_weight_count_succeeds);
    RUN_TEST(test_create_with_wrong_weight_count_fails);

    return UNITY_END();
}
