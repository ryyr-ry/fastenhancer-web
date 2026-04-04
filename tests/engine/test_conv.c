/*
 * test_conv.c — Phase 2-D: Convolution Tests (TDD Red)
 *
 * Test targets:
 *   - Conv1d (standard 1D convolution)
 *   - StridedConv1d (stride=4, for Encoder)
 *   - ConvTranspose1d (stride=4, for Decoder)
 *   - Expected output with known weights
 *   - BatchNorm fused (y = x*scale + bias form)
 *
 * Compile:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_conv.c \
 *       src/engine/common/conv.c -o test_conv -lm
 */

#include "unity.h"
#include "conv.h"
#include "tiny_48k.h"
#include <math.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* --- Conv1d basic tests --- */

void test_conv1d_identity_kernel(void) {
    /* kernel=[1.0], input=[1,2,3,4] → output=[1,2,3,4] */
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[1] = {1.0f};
    float bias = 0.0f;
    float output[4];

    fe_conv1d(input, weight, &bias, output,
              /*in_len=*/4, /*in_channels=*/1, /*out_channels=*/1,
              /*kernel_size=*/1, /*stride=*/1, /*padding=*/0);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, input[i], output[i]);
    }
}

void test_conv1d_with_bias(void) {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[1] = {1.0f};
    float bias = 0.5f;
    float output[4];

    fe_conv1d(input, weight, &bias, output,
              4, 1, 1, 1, 1, 0);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, input[i] + 0.5f, output[i]);
    }
}

void test_conv1d_known_kernel3(void) {
    /* kernel=[1,2,1], input=[1,0,1,0,1], padding=1 → verify convolution result */
    float input[5] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    float weight[3] = {1.0f, 2.0f, 1.0f};
    float bias = 0.0f;
    float output[5];

    fe_conv1d(input, weight, &bias, output,
              5, 1, 1, 3, 1, 1);

    /* Manual calculation:
     * out[0] = 0*1 + 1*2 + 0*1 = 2
     * out[1] = 1*1 + 0*2 + 1*1 = 2
     * out[2] = 0*1 + 1*2 + 0*1 = 2
     * out[3] = 1*1 + 0*2 + 1*1 = 2
     * out[4] = 0*1 + 1*2 + 0*1 = 2
     */
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, output[i]);
    }
}

void test_conv1d_all_zeros_input(void) {
    float input[8] = {0};
    float weight[3] = {1.0f, 2.0f, 3.0f};
    float bias = 0.0f;
    float output[8];

    fe_conv1d(input, weight, &bias, output,
              8, 1, 1, 3, 1, 1);

    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, output[i]);
    }
}

/* --- StridedConv1d (Encoder) --- */

void test_strided_conv1d_stride4_kernel8(void) {
    /* FastEnhancer Tiny Encoder layer 1: kernel=8, stride=4 */
    float input[32];
    for (int i = 0; i < 32; i++) input[i] = (float)i;

    float weight[8];
    for (int i = 0; i < 8; i++) weight[i] = 1.0f / 8.0f;
    float bias = 0.0f;

    /* Output length: (32 - 8) / 4 + 1 = 7 */
    float output[7];

    fe_strided_conv1d(input, weight, &bias, output,
                      32, 1, 1, 8, 4);

    for (int i = 0; i < 7; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }

    /* Averaging kernel result: each output is the mean of 8 consecutive samples */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.5f, output[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 7.5f, output[1]);
}

void test_strided_conv1d_output_length(void) {
    float input[24];
    float weight[8];
    float bias = 0.0f;
    float output[16];
    memset(input, 0, sizeof(input));
    memset(weight, 0, sizeof(weight));
    weight[0] = 1.0f;

    int out_len = fe_strided_conv1d(input, weight, &bias, output,
                                    24, 1, 1, 8, 4);

    /* (24 - 8) / 4 + 1 = 5 */
    TEST_ASSERT_EQUAL_INT(5, out_len);
}

void test_strided_conv1d_multichannel(void) {
    /* Input: 2ch × 16 samples, kernel: 3×2ch→4ch, stride=4 */
    int in_ch = 2, out_ch = 4, kernel_size = 3, stride = 4, in_len = 16;
    float input[2 * 16];
    float weight[4 * 2 * 3];
    float bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[4 * 4];

    for (int i = 0; i < in_ch * in_len; i++) input[i] = 1.0f;
    for (int i = 0; i < out_ch * in_ch * kernel_size; i++) weight[i] = 0.1f;

    int out_len = fe_strided_conv1d(input, weight, bias, output,
                                    in_len, in_ch, out_ch, kernel_size, stride);

    TEST_ASSERT_TRUE(out_len > 0);
    for (int i = 0; i < out_ch * out_len; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
    }
}

/* --- ConvTranspose1d (Decoder) --- */

void test_conv_transpose1d_stride4(void) {
    /* Input length 5, stride=4 → output length = (5-1)*4 + 8 - 2*0 = 24 (padding=0) */
    float input[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float weight[8];
    for (int i = 0; i < 8; i++) weight[i] = 1.0f;
    float bias = 0.0f;

    int expected_out_len = (5 - 1) * 4 + 8;  /* = 24 */
    float output[32];

    int out_len = fe_conv_transpose1d(input, weight, &bias, output,
                                      5, 1, 1, 8, 4, /*padding=*/0);

    TEST_ASSERT_EQUAL_INT(expected_out_len, out_len);

    /* First impulse response: entire weight vector */
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, output[i]);
    }
}

void test_conv_transpose1d_is_upsampling(void) {
    /* ConvTranspose1d with stride=4 performs 4x upsampling */
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[1] = {1.0f};
    float bias = 0.0f;
    float output[16];

    int out_len = fe_conv_transpose1d(input, weight, &bias, output,
                                      4, 1, 1, 1, 4, /*padding=*/0);

    /* kernel_size=1, stride=4 → output stretches input by 4x (zero insertion) */
    TEST_ASSERT_EQUAL_INT(13, out_len);  /* (4-1)*4 + 1 = 13 */
}

/* --- ConvTranspose1d padding test (3B-0a) --- */

void test_conv_transpose1d_padding_basic(void) {
    /* padding=1: trim padding from both ends of full output
     * in_len=2, kernel=4, stride=2, padding=1
     * Full: (2-1)*2+4 = 6
     * Padded: 6-2*1 = 4
     *
     * Input=[1,0], Weight=[1,2,3,4], bias=0
     * Full output: [1,2,3,4,0,0] (input[0] scattered to positions 0..3)
     * Trim padding=1: out[1..4] = [2,3,4,0]
     */
    float input[2] = {1.0f, 0.0f};
    float weight[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bias = 0.0f;
    float output[8];

    int out_len = fe_conv_transpose1d(input, weight, &bias, output,
                                      2, 1, 1, 4, 2, /*padding=*/1);

    TEST_ASSERT_EQUAL_INT(4, out_len);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, output[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f, output[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, output[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output[3]);
}

void test_conv_transpose1d_padding2_decoder_postnet(void) {
    /* Decoder PostNet dimension test:
     * in_len=128, kernel=8, stride=4, padding=2
     * Output: (128-1)*4+8-2*2 = 508+4 = 512 */
    int in_len = 128, in_ch = 1, out_ch = 1;
    int kernel_size = 8, stride = 4, padding = 2;

    float input[128];
    float weight[8];
    float bias = 0.0f;
    float output[600];

    for (int i = 0; i < in_len; i++) input[i] = 1.0f;
    for (int i = 0; i < kernel_size; i++) weight[i] = 1.0f / kernel_size;

    int out_len = fe_conv_transpose1d(input, weight, &bias, output,
                                      in_len, in_ch, out_ch,
                                      kernel_size, stride, padding);

    TEST_ASSERT_EQUAL_INT(512, out_len);

    for (int i = 0; i < out_len; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

void test_conv_transpose1d_padding_multichannel(void) {
    /* Multichannel + padding: in_ch=2, out_ch=2, padding=1 */
    int in_len = 4, in_ch = 2, out_ch = 2;
    int kernel = 4, stride = 2, padding = 1;
    int expected_out_len = (in_len - 1) * stride + kernel - 2 * padding;
    /* = 3*2+4-2 = 8 */

    float input[2 * 4];
    float weight[2 * 2 * 4];
    float bias[2] = {0.1f, 0.2f};
    float output[2 * 8];

    for (int i = 0; i < 2 * 4; i++) input[i] = (float)(i + 1) * 0.1f;
    for (int i = 0; i < 2 * 2 * 4; i++) weight[i] = 0.1f;

    int out_len = fe_conv_transpose1d(input, weight, bias, output,
                                      in_len, in_ch, out_ch,
                                      kernel, stride, padding);

    TEST_ASSERT_EQUAL_INT(expected_out_len, out_len);
    for (int i = 0; i < out_ch * out_len; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- BatchNorm fusion test --- */

void test_conv1d_batchnorm_fused(void) {
    /* BatchNorm fused: y = x*scale + bias
     * weight=1.0, conv_bias=0, bn_scale=2.0, bn_bias=1.0
     * → final output = conv_out * 2.0 + 1.0 */
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[1] = {1.0f};
    float bn_scale = 2.0f;
    float bn_bias = 1.0f;
    float output[4];

    fe_conv1d_bn(input, weight, &bn_scale, &bn_bias, output,
                 4, 1, 1, 1, 1, 0);

    for (int i = 0; i < 4; i++) {
        float expected = input[i] * 2.0f + 1.0f;
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected, output[i]);
    }
}

/* --- Encoder→Decoder symmetry --- */

void test_encoder_decoder_dimension_consistency(void) {
    /* Encoder PreNet: StridedConv1d(2→C1, k=8, s=4, pad=2) on F_BINS=512
     * output_len = (512 + 2*2 - 8) / 4 + 1 = (512+4-8)/4+1 = 508/4+1 = 127+1 = 128 = F1
     * Decoder PostNet: ConvTranspose1d(C1→2, k=8, s=4, pad=2) on F1=128
     * output_len = (128-1)*4+8-2*2 = 127*4+4 = 512 = FREQ_BINS */
    float enc_in[2 * 512];
    float enc_out[FE_C1 * FE_F1];
    float dec_out[2 * 512];
    float enc_w[FE_C1 * 2 * FE_ENC_K0];
    float enc_bn_s[FE_C1];
    float enc_bn_b[FE_C1];
    float dec_w[FE_C1 * 2 * FE_ENC_K0];
    float dec_b[2];

    memset(enc_in, 0, sizeof(enc_in));
    memset(enc_w, 0, sizeof(enc_w));
    memset(dec_w, 0, sizeof(dec_w));
    memset(dec_b, 0, sizeof(dec_b));
    for (int i = 0; i < FE_C1; i++) { enc_bn_s[i] = 1.0f; enc_bn_b[i] = 0.0f; }

    enc_in[0] = 1.0f;

    fe_conv1d_bn(enc_in, enc_w, enc_bn_s, enc_bn_b,
                 enc_out, 512, 2, FE_C1, FE_ENC_K0, FE_STRIDE, FE_ENC_PRE_PAD);

    fe_conv_transpose1d(enc_out, dec_w, dec_b,
                        dec_out, FE_F1, FE_C1, 2,
                        FE_ENC_K0, FE_STRIDE, FE_ENC_PRE_PAD);

    /* Verify dimensions are correctly restored (no crash = dimensional consistency) */
    for (int i = 0; i < 2 * 512; i++) {
        TEST_ASSERT_FALSE(isnan(dec_out[i]));
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_conv1d_identity_kernel);
    RUN_TEST(test_conv1d_with_bias);
    RUN_TEST(test_conv1d_known_kernel3);
    RUN_TEST(test_conv1d_all_zeros_input);
    RUN_TEST(test_strided_conv1d_stride4_kernel8);
    RUN_TEST(test_strided_conv1d_output_length);
    RUN_TEST(test_strided_conv1d_multichannel);
    RUN_TEST(test_conv_transpose1d_stride4);
    RUN_TEST(test_conv_transpose1d_is_upsampling);
    RUN_TEST(test_conv_transpose1d_padding_basic);
    RUN_TEST(test_conv_transpose1d_padding2_decoder_postnet);
    RUN_TEST(test_conv_transpose1d_padding_multichannel);
    RUN_TEST(test_conv1d_batchnorm_fused);
    RUN_TEST(test_encoder_decoder_dimension_consistency);

    return UNITY_END();
}
