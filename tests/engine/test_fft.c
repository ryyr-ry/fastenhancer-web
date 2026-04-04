/*
 * test_fft.c — Phase 2-B: 1024点FFTテスト (TDD Red)
 *
 * 検証対象:
 *   - 1024点 Radix-2 DIT FFT / iFFT
 *   - インパルス応答 (時間域デルタ → 周波数域フラット)
 *   - Parseval定理 (時間域エネルギー = 周波数域エネルギー)
 *   - ラウンドトリップ (FFT → iFFT → 元信号)
 *   - twiddle factor 精度 (カテゴリ1)
 *
 * コンパイル:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_fft.c \
 *       src/engine/common/fft.c -o test_fft -lm
 */

#include "unity.h"
#include "fft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define N_FFT 1024

static float real_buf[N_FFT];
static float imag_buf[N_FFT];
static float real_ref[N_FFT];
static float imag_ref[N_FFT];

void setUp(void) {
    memset(real_buf, 0, sizeof(real_buf));
    memset(imag_buf, 0, sizeof(imag_buf));
    memset(real_ref, 0, sizeof(real_ref));
    memset(imag_ref, 0, sizeof(imag_ref));
}

void tearDown(void) {}

/* --- インパルス応答 --- */

void test_fft_impulse_response(void) {
    real_buf[0] = 1.0f;
    fe_fft(real_buf, imag_buf, N_FFT);

    for (int k = 0; k < N_FFT; k++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, real_buf[k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, imag_buf[k]);
    }
}

/* --- DC信号 --- */

void test_fft_dc_signal(void) {
    for (int i = 0; i < N_FFT; i++) real_buf[i] = 1.0f;
    fe_fft(real_buf, imag_buf, N_FFT);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, (float)N_FFT, real_buf[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, imag_buf[0]);
    for (int k = 1; k < N_FFT; k++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, real_buf[k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, imag_buf[k]);
    }
}

/* --- 純正弦波 --- */

void test_fft_single_frequency(void) {
    int bin = 32;
    for (int n = 0; n < N_FFT; n++) {
        real_buf[n] = cosf(2.0f * (float)M_PI * bin * n / N_FFT);
    }
    fe_fft(real_buf, imag_buf, N_FFT);

    float mag_at_bin = sqrtf(real_buf[bin] * real_buf[bin] + imag_buf[bin] * imag_buf[bin]);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, (float)N_FFT / 2.0f, mag_at_bin);

    for (int k = 2; k < N_FFT - 1; k++) {
        if (k == bin || k == N_FFT - bin) continue;
        float mag = sqrtf(real_buf[k] * real_buf[k] + imag_buf[k] * imag_buf[k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-2f, 0.0f, mag);
    }
}

/* --- Parseval定理 --- */

void test_fft_parseval_theorem(void) {
    for (int n = 0; n < N_FFT; n++) {
        real_buf[n] = sinf(2.0f * (float)M_PI * 7.0f * n / N_FFT) +
                      0.5f * cosf(2.0f * (float)M_PI * 100.0f * n / N_FFT);
    }

    double time_energy = 0.0;
    for (int n = 0; n < N_FFT; n++) {
        time_energy += (double)real_buf[n] * (double)real_buf[n];
    }

    fe_fft(real_buf, imag_buf, N_FFT);

    double freq_energy = 0.0;
    for (int k = 0; k < N_FFT; k++) {
        freq_energy += (double)real_buf[k] * (double)real_buf[k] +
                       (double)imag_buf[k] * (double)imag_buf[k];
    }
    freq_energy /= (double)N_FFT;

    TEST_ASSERT_FLOAT_WITHIN((float)(time_energy * 1e-5), (float)time_energy, (float)freq_energy);
}

/* --- ラウンドトリップ (FFT → iFFT) --- */

void test_fft_roundtrip(void) {
    for (int n = 0; n < N_FFT; n++) {
        real_buf[n] = sinf(2.0f * (float)M_PI * 50.0f * n / N_FFT) +
                      0.3f * cosf(2.0f * (float)M_PI * 200.0f * n / N_FFT);
        real_ref[n] = real_buf[n];
    }

    fe_fft(real_buf, imag_buf, N_FFT);
    fe_ifft(real_buf, imag_buf, N_FFT);

    for (int n = 0; n < N_FFT; n++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, real_ref[n], real_buf[n]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, imag_buf[n]);
    }
}

void test_fft_roundtrip_complex_input(void) {
    for (int n = 0; n < N_FFT; n++) {
        real_buf[n] = cosf(2.0f * (float)M_PI * 10.0f * n / N_FFT);
        imag_buf[n] = sinf(2.0f * (float)M_PI * 10.0f * n / N_FFT);
        real_ref[n] = real_buf[n];
        imag_ref[n] = imag_buf[n];
    }

    fe_fft(real_buf, imag_buf, N_FFT);
    fe_ifft(real_buf, imag_buf, N_FFT);

    for (int n = 0; n < N_FFT; n++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, real_ref[n], real_buf[n]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, imag_ref[n], imag_buf[n]);
    }
}

/* --- twiddle factor精度 (カテゴリ1) --- */

void test_fft_twiddle_precision(void) {
    const float* tw_re = fe_fft_get_twiddle_re();
    const float* tw_im = fe_fft_get_twiddle_im();
    TEST_ASSERT_NOT_NULL(tw_re);
    TEST_ASSERT_NOT_NULL(tw_im);

    for (int k = 0; k < N_FFT / 2; k++) {
        /* twiddle factorsはdouble精度で計算後floatにキャストされるため、
           テスト側もdouble精度で計算してfloatに変換する */
        float expected_re = (float)cos(2.0 * M_PI * (double)k / (double)N_FFT);
        float expected_im = (float)(-sin(2.0 * M_PI * (double)k / (double)N_FFT));
        TEST_ASSERT_FLOAT_WITHIN(1e-7f, expected_re, tw_re[k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-7f, expected_im, tw_im[k]);
    }
}

/* --- ゼロ入力 --- */

void test_fft_all_zeros(void) {
    fe_fft(real_buf, imag_buf, N_FFT);
    for (int k = 0; k < N_FFT; k++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, real_buf[k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, imag_buf[k]);
    }
}

/* --- 線形性 --- */

void test_fft_linearity(void) {
    float real_a[N_FFT], imag_a[N_FFT];
    float real_b[N_FFT], imag_b[N_FFT];
    float real_sum[N_FFT], imag_sum[N_FFT];

    for (int n = 0; n < N_FFT; n++) {
        real_a[n] = cosf(2.0f * (float)M_PI * 5.0f * n / N_FFT);
        real_b[n] = sinf(2.0f * (float)M_PI * 30.0f * n / N_FFT);
        real_sum[n] = real_a[n] + real_b[n];
        imag_a[n] = imag_b[n] = imag_sum[n] = 0.0f;
    }

    fe_fft(real_a, imag_a, N_FFT);
    fe_fft(real_b, imag_b, N_FFT);
    fe_fft(real_sum, imag_sum, N_FFT);

    for (int k = 0; k < N_FFT; k++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, real_a[k] + real_b[k], real_sum[k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, imag_a[k] + imag_b[k], imag_sum[k]);
    }
}

/* --- 共役対称性 (実数入力) --- */

void test_fft_conjugate_symmetry(void) {
    for (int n = 0; n < N_FFT; n++) {
        real_buf[n] = sinf(2.0f * (float)M_PI * 17.0f * n / N_FFT);
    }
    fe_fft(real_buf, imag_buf, N_FFT);

    for (int k = 1; k < N_FFT / 2; k++) {
        int conj_k = N_FFT - k;
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, real_buf[k], real_buf[conj_k]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, -imag_buf[k], imag_buf[conj_k]);
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_fft_impulse_response);
    RUN_TEST(test_fft_dc_signal);
    RUN_TEST(test_fft_single_frequency);
    RUN_TEST(test_fft_parseval_theorem);
    RUN_TEST(test_fft_roundtrip);
    RUN_TEST(test_fft_roundtrip_complex_input);
    RUN_TEST(test_fft_twiddle_precision);
    RUN_TEST(test_fft_all_zeros);
    RUN_TEST(test_fft_linearity);
    RUN_TEST(test_fft_conjugate_symmetry);

    return UNITY_END();
}
