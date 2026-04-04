/*
 * test_compression.c — Phase 2-G: パワー圧縮テスト (TDD Red)
 *
 * 検証対象:
 *   - mag^0.3 パワー圧縮 (ダイナミックレンジ縮小)
 *   - 微小値の安定性 (カテゴリ1: 数値安定性)
 *   - NaN/Inf入力の安全性 (カテゴリ2: エッジケース)
 *   - ゼロ入力
 *   - 逆圧縮 (mag^(1/0.3)) の整合性
 *
 * コンパイル:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_compression.c \
 *       src/engine/common/compression.c -o test_compression -lm
 */

#include "unity.h"
#include "compression.h"
#include <math.h>
#include <float.h>
#include <string.h>

#define FREQ_BINS 513

void setUp(void) {}
void tearDown(void) {}

/* --- 基本テスト --- */

void test_compression_unit_value(void) {
    /* mag=1.0 → 1.0^0.3 = 1.0 */
    float input[1] = {1.0f};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, output[0]);
}

void test_compression_known_values(void) {
    float input[3] = {0.0f, 0.5f, 2.0f};
    float output[3];
    fe_power_compress(input, output, 3, 0.3f);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, powf(0.5f, 0.3f), output[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, powf(2.0f, 0.3f), output[2]);
}

void test_compression_monotonic(void) {
    float values[10] = {0.0f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 50.0f, 100.0f};
    float outputs[10];
    fe_power_compress(values, outputs, 10, 0.3f);

    for (int i = 1; i < 10; i++) {
        TEST_ASSERT_TRUE(outputs[i] >= outputs[i - 1]);
    }
}

void test_compression_reduces_dynamic_range(void) {
    /* 入力範囲: [0.01, 100] → 圧縮後の範囲は狭くなるはず */
    float lo_in = 0.01f;
    float hi_in = 100.0f;
    float lo_out, hi_out;
    fe_power_compress(&lo_in, &lo_out, 1, 0.3f);
    fe_power_compress(&hi_in, &hi_out, 1, 0.3f);

    float range_in = hi_in / lo_in;         /* = 10000 */
    float range_out = hi_out / lo_out;
    TEST_ASSERT_TRUE(range_out < range_in);
}

/* --- 微小値の安定性 (カテゴリ1) --- */

void test_compression_tiny_value(void) {
    float input[1] = {1e-20f};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_FALSE(isinf(output[0]));
    TEST_ASSERT_TRUE(output[0] >= 0.0f);
}

void test_compression_denormal(void) {
    float input[1] = {FLT_MIN * 0.1f};  /* デノーマル値 */
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_FALSE(isinf(output[0]));
    TEST_ASSERT_TRUE(output[0] >= 0.0f);
}

void test_compression_flt_min(void) {
    float input[1] = {FLT_MIN};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_FALSE(isinf(output[0]));
    TEST_ASSERT_TRUE(output[0] >= 0.0f);
}

/* --- NaN/Inf入力 (カテゴリ2) --- */

void test_compression_nan_input(void) {
    float input[1] = {NAN};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    /* NaN入力は0.0fにクランプされるべき */
    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, output[0]);
}

void test_compression_inf_input(void) {
    float input[1] = {INFINITY};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_FALSE(isinf(output[0]));
    TEST_ASSERT_TRUE(output[0] >= 0.0f);
}

void test_compression_negative_input(void) {
    /* magnitudeは非負のはずだが、安全のため */
    float input[1] = {-1.0f};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_TRUE(output[0] >= 0.0f);
}

/* --- ゼロ入力 --- */

void test_compression_zero(void) {
    float input[1] = {0.0f};
    float output[1];
    fe_power_compress(input, output, 1, 0.3f);

    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, output[0]);
}

/* --- バッチ処理 --- */

void test_compression_full_spectrum(void) {
    float input[FREQ_BINS];
    float output[FREQ_BINS];

    for (int i = 0; i < FREQ_BINS; i++) {
        input[i] = (float)i / (float)FREQ_BINS;
    }

    fe_power_compress(input, output, FREQ_BINS, 0.3f);

    for (int i = 0; i < FREQ_BINS; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
        TEST_ASSERT_TRUE(output[i] >= 0.0f);
    }
}

/* --- 逆圧縮テスト --- */

void test_decompression_roundtrip(void) {
    float input[5] = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f};
    float compressed[5];
    float decompressed[5];

    fe_power_compress(input, compressed, 5, 0.3f);
    fe_power_decompress(compressed, decompressed, 5, 0.3f);

    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(input[i] * 1e-4f + 1e-6f, input[i], decompressed[i]);
    }
}

void test_decompression_zero(void) {
    float input[1] = {0.0f};
    float output[1];
    fe_power_decompress(input, output, 1, 0.3f);
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, output[0]);
}

/* --- 複素スペクトル圧縮テスト (3B-0b) --- */

void test_complex_compression_unit_circle(void) {
    /* 単位円上の点: mag=1 → scale=1^(-0.7)=1 → 変化なし */
    float re[1] = {0.6f};
    float im[1] = {0.8f};   /* mag = sqrt(0.36+0.64) = 1.0 */
    float out_re[1], out_im[1];

    fe_power_compress_complex(re, im, out_re, out_im, 1, 0.3f);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.6f, out_re[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.8f, out_im[0]);
}

void test_complex_compression_known_values(void) {
    /* mag=2, phase=0: re=2, im=0
     * scale = 2^(-0.7) ≈ 0.6156
     * out_re = 2 * 0.6156 ≈ 1.2311
     * out_im = 0 */
    float re[1] = {2.0f};
    float im[1] = {0.0f};
    float out_re[1], out_im[1];

    fe_power_compress_complex(re, im, out_re, out_im, 1, 0.3f);

    float expected_scale = powf(2.0f, -0.7f);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 2.0f * expected_scale, out_re[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, out_im[0]);
}

void test_complex_compression_magnitude_check(void) {
    /* 圧縮後のmag = original_mag^0.3 であることを検証 */
    float re[4] = {3.0f, 0.0f, 1.0f, -2.0f};
    float im[4] = {4.0f, 5.0f, 1.0f,  3.0f};
    float out_re[4], out_im[4];

    fe_power_compress_complex(re, im, out_re, out_im, 4, 0.3f);

    for (int i = 0; i < 4; i++) {
        float orig_mag = sqrtf(re[i] * re[i] + im[i] * im[i]);
        float comp_mag = sqrtf(out_re[i] * out_re[i] + out_im[i] * out_im[i]);
        float expected_mag = powf(orig_mag, 0.3f);
        TEST_ASSERT_FLOAT_WITHIN(expected_mag * 1e-4f + 1e-6f,
                                 expected_mag, comp_mag);
    }
}

void test_complex_compression_preserves_phase(void) {
    /* 圧縮は位相を保持する: atan2(out_im, out_re) == atan2(im, re) */
    float re[3] = {1.0f, -3.0f, 0.0f};
    float im[3] = {2.0f,  4.0f, 7.0f};
    float out_re[3], out_im[3];

    fe_power_compress_complex(re, im, out_re, out_im, 3, 0.3f);

    for (int i = 0; i < 3; i++) {
        float orig_phase = atan2f(im[i], re[i]);
        float comp_phase = atan2f(out_im[i], out_re[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, orig_phase, comp_phase);
    }
}

void test_complex_compression_zero_magnitude(void) {
    /* mag=0: NaN/Infにならないこと */
    float re[1] = {0.0f};
    float im[1] = {0.0f};
    float out_re[1], out_im[1];

    fe_power_compress_complex(re, im, out_re, out_im, 1, 0.3f);

    TEST_ASSERT_FALSE(isnan(out_re[0]));
    TEST_ASSERT_FALSE(isnan(out_im[0]));
    TEST_ASSERT_FALSE(isinf(out_re[0]));
    TEST_ASSERT_FALSE(isinf(out_im[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, out_re[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, out_im[0]);
}

void test_complex_compression_roundtrip(void) {
    /* compress → decompress = 元の値 */
    float re[5] = {1.0f, -2.0f, 0.5f, 3.0f, 0.0f};
    float im[5] = {0.0f,  1.5f, 0.5f, -4.0f, 2.0f};
    float comp_re[5], comp_im[5];
    float dec_re[5], dec_im[5];

    fe_power_compress_complex(re, im, comp_re, comp_im, 5, 0.3f);
    fe_power_decompress_complex(comp_re, comp_im, dec_re, dec_im, 5, 0.3f);

    for (int i = 0; i < 5; i++) {
        float mag = sqrtf(re[i] * re[i] + im[i] * im[i]);
        float tol = mag * 1e-4f + 1e-6f;
        TEST_ASSERT_FLOAT_WITHIN(tol, re[i], dec_re[i]);
        TEST_ASSERT_FLOAT_WITHIN(tol, im[i], dec_im[i]);
    }
}

void test_complex_compression_nan_input(void) {
    /* NaN入力は安全に処理 */
    float re[2] = {NAN, 1.0f};
    float im[2] = {1.0f, NAN};
    float out_re[2], out_im[2];

    fe_power_compress_complex(re, im, out_re, out_im, 2, 0.3f);

    for (int i = 0; i < 2; i++) {
        TEST_ASSERT_FALSE(isnan(out_re[i]));
        TEST_ASSERT_FALSE(isnan(out_im[i]));
        TEST_ASSERT_FALSE(isinf(out_re[i]));
        TEST_ASSERT_FALSE(isinf(out_im[i]));
    }
}

void test_complex_compression_inf_input_safe(void) {
    float re[2] = {INFINITY, 1.0f};
    float im[2] = {0.0f, -INFINITY};
    float out_re[2], out_im[2];

    fe_power_compress_complex(re, im, out_re, out_im, 2, 0.3f);

    TEST_ASSERT_FALSE(isnan(out_re[0]));
    TEST_ASSERT_FALSE(isnan(out_im[0]));
    TEST_ASSERT_FALSE(isinf(out_re[0]));
    TEST_ASSERT_FALSE(isinf(out_im[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, out_re[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, out_im[0]);

    TEST_ASSERT_FALSE(isnan(out_re[1]));
    TEST_ASSERT_FALSE(isnan(out_im[1]));
    TEST_ASSERT_FALSE(isinf(out_re[1]));
    TEST_ASSERT_FALSE(isinf(out_im[1]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, out_re[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, out_im[1]);
}

void test_complex_compression_batch_512(void) {
    /* 512 bins (48kHz STFT Nyquist除去後) の一括処理 */
    float re[512], im[512];
    float out_re[512], out_im[512];

    for (int i = 0; i < 512; i++) {
        re[i] = sinf((float)i * 0.1f);
        im[i] = cosf((float)i * 0.1f);
    }

    fe_power_compress_complex(re, im, out_re, out_im, 512, 0.3f);

    for (int i = 0; i < 512; i++) {
        TEST_ASSERT_FALSE(isnan(out_re[i]));
        TEST_ASSERT_FALSE(isnan(out_im[i]));
        TEST_ASSERT_FALSE(isinf(out_re[i]));
        TEST_ASSERT_FALSE(isinf(out_im[i]));
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_compression_unit_value);
    RUN_TEST(test_compression_known_values);
    RUN_TEST(test_compression_monotonic);
    RUN_TEST(test_compression_reduces_dynamic_range);
    RUN_TEST(test_compression_tiny_value);
    RUN_TEST(test_compression_denormal);
    RUN_TEST(test_compression_flt_min);
    RUN_TEST(test_compression_nan_input);
    RUN_TEST(test_compression_inf_input);
    RUN_TEST(test_compression_negative_input);
    RUN_TEST(test_compression_zero);
    RUN_TEST(test_compression_full_spectrum);
    RUN_TEST(test_decompression_roundtrip);
    RUN_TEST(test_decompression_zero);
    RUN_TEST(test_complex_compression_unit_circle);
    RUN_TEST(test_complex_compression_known_values);
    RUN_TEST(test_complex_compression_magnitude_check);
    RUN_TEST(test_complex_compression_preserves_phase);
    RUN_TEST(test_complex_compression_zero_magnitude);
    RUN_TEST(test_complex_compression_roundtrip);
    RUN_TEST(test_complex_compression_nan_input);
    RUN_TEST(test_complex_compression_inf_input_safe);
    RUN_TEST(test_complex_compression_batch_512);

    return UNITY_END();
}
