/*
 * test_stft.c — Phase 2-C: STFT/iSTFT Tests (TDD Red)
 *
 * Verification targets:
 *   - Hann window correctness (symmetry, endpoints, energy)
 *   - STFT → iSTFT round-trip (COLA condition)
 *   - Streaming continuity (overlap-add across frames)
 *
 * Compile:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_stft.c \
 *       src/engine/common/stft.c src/engine/common/fft.c -o test_stft -lm
 */

#include "unity.h"
#include "stft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define N_FFT 1024
#define HOP_SIZE 512
#define FREQ_BINS 513  /* n_fft/2 + 1 */

static FeStftState stft_state;

void setUp(void) {
    int rc = fe_stft_init(&stft_state, N_FFT, HOP_SIZE);
    TEST_ASSERT_EQUAL_INT(0, rc);
}

void tearDown(void) {}

/* --- Hann Window Tests --- */

void test_hann_window_endpoints(void) {
    const float* window = fe_stft_get_window(&stft_state);
    TEST_ASSERT_NOT_NULL(window);
    TEST_ASSERT_FLOAT_WITHIN(1e-7f, 0.0f, window[0]);
    /* Periodic Hann window: w[N-1] ≈ 0 but not exactly zero (∝ 1/N²) */
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, window[N_FFT - 1]);
}

void test_hann_window_peak(void) {
    const float* window = fe_stft_get_window(&stft_state);
    float peak = window[N_FFT / 2];
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, peak);
}

void test_hann_window_symmetry(void) {
    const float* window = fe_stft_get_window(&stft_state);
    /* Periodic Hann window is not strictly symmetric, but approximately so (error ∝ 1/N) */
    for (int i = 0; i < N_FFT / 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.01f, window[i], window[N_FFT - 1 - i]);
    }
}

void test_hann_window_nonnegative(void) {
    const float* window = fe_stft_get_window(&stft_state);
    for (int i = 0; i < N_FFT; i++) {
        TEST_ASSERT_TRUE(window[i] >= 0.0f);
        TEST_ASSERT_TRUE(window[i] <= 1.0f);
    }
}

void test_hann_window_cola_condition(void) {
    const float* window = fe_stft_get_window(&stft_state);

    /* COLA condition: for Hann window with hop=n_fft/2, the sum of adjacent windows is constant */
    for (int i = 0; i < HOP_SIZE; i++) {
        float sum = window[i] + window[i + HOP_SIZE];
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, sum);
    }
}

/* --- STFT Basic Tests --- */

void test_stft_impulse(void) {
    float input[HOP_SIZE];
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;

    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];

    fe_stft_reset(&stft_state);
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);

    int has_nonzero = 0;
    for (int k = 0; k < FREQ_BINS; k++) {
        TEST_ASSERT_FALSE(isnan(spec_real[k]));
        TEST_ASSERT_FALSE(isnan(spec_imag[k]));
        if (fabsf(spec_real[k]) > 1e-10f || fabsf(spec_imag[k]) > 1e-10f) {
            has_nonzero = 1;
        }
    }
    TEST_ASSERT_TRUE(has_nonzero);
}

void test_stft_dc_produces_energy_at_bin0(void) {
    float input[HOP_SIZE];
    for (int i = 0; i < HOP_SIZE; i++) input[i] = 1.0f;

    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];

    fe_stft_reset(&stft_state);
    /* Fill the buffer with the first frame */
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);
    /* Stable output on the second frame */
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);

    float mag_dc = sqrtf(spec_real[0] * spec_real[0] + spec_imag[0] * spec_imag[0]);
    TEST_ASSERT_TRUE(mag_dc > 0.1f);
}

/* --- Round-trip (STFT → iSTFT) --- */

void test_stft_roundtrip_sine(void) {
    float input[HOP_SIZE];
    float prev_input[HOP_SIZE];
    float output[HOP_SIZE];
    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];

    fe_stft_reset(&stft_state);

    /* Warm-up: process several frames to reach steady state */
    int warmup_frames = 4;
    for (int f = 0; f < warmup_frames; f++) {
        for (int i = 0; i < HOP_SIZE; i++) {
            input[i] = sinf(2.0f * (float)M_PI * 1000.0f *
                            (f * HOP_SIZE + i) / 48000.0f);
        }
        fe_stft_forward(&stft_state, input, spec_real, spec_imag);
        fe_stft_inverse(&stft_state, spec_real, spec_imag, output);
    }

    /* Save the next frame's input, then advance one more frame */
    for (int i = 0; i < HOP_SIZE; i++) {
        prev_input[i] = sinf(2.0f * (float)M_PI * 1000.0f *
                        (warmup_frames * HOP_SIZE + i) / 48000.0f);
    }
    fe_stft_forward(&stft_state, prev_input, spec_real, spec_imag);
    fe_stft_inverse(&stft_state, spec_real, spec_imag, output);

    for (int i = 0; i < HOP_SIZE; i++) {
        input[i] = sinf(2.0f * (float)M_PI * 1000.0f *
                        ((warmup_frames + 1) * HOP_SIZE + i) / 48000.0f);
    }
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);
    fe_stft_inverse(&stft_state, spec_real, spec_imag, output);

    /* overlap-add has 1-frame latency: output corresponds to prev_input */
    float max_error = 0.0f;
    for (int i = 0; i < HOP_SIZE; i++) {
        float err = fabsf(output[i] - prev_input[i]);
        if (err > max_error) max_error = err;
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, max_error);
}

void test_stft_roundtrip_white_noise(void) {
    float output[HOP_SIZE];
    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];

    fe_stft_reset(&stft_state);

    /* Simple PRNG */
    unsigned int seed = 42;
    float prev_input[HOP_SIZE];
    int total_frames = 8;

    for (int f = 0; f < total_frames; f++) {
        float input[HOP_SIZE];
        for (int i = 0; i < HOP_SIZE; i++) {
            seed = seed * 1103515245 + 12345;
            input[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
        }
        /* overlap-add 1-frame latency: output corresponds to prev_input */
        if (f == total_frames - 2) {
            memcpy(prev_input, input, sizeof(input));
        }
        fe_stft_forward(&stft_state, input, spec_real, spec_imag);
        fe_stft_inverse(&stft_state, spec_real, spec_imag, output);
    }

    float max_error = 0.0f;
    for (int i = 0; i < HOP_SIZE; i++) {
        float err = fabsf(output[i] - prev_input[i]);
        if (err > max_error) max_error = err;
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, max_error);
}

/* --- Streaming Continuity --- */

void test_stft_streaming_continuity(void) {
    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];
    float output[HOP_SIZE];
    float reconstructed[HOP_SIZE * 12];
    int total_frames = 12;

    fe_stft_reset(&stft_state);

    for (int f = 0; f < total_frames; f++) {
        float input[HOP_SIZE];
        for (int i = 0; i < HOP_SIZE; i++) {
            float t = (float)(f * HOP_SIZE + i) / 48000.0f;
            input[i] = sinf(2.0f * (float)M_PI * 440.0f * t);
        }
        fe_stft_forward(&stft_state, input, spec_real, spec_imag);
        fe_stft_inverse(&stft_state, spec_real, spec_imag, output);
        memcpy(&reconstructed[f * HOP_SIZE], output, HOP_SIZE * sizeof(float));
    }

    /* Verify no discontinuities at frame boundaries after stabilization (accounting for 1-frame latency) */
    for (int f = 6; f < total_frames - 1; f++) {
        int boundary = (f + 1) * HOP_SIZE;
        float diff = fabsf(reconstructed[boundary] - reconstructed[boundary - 1]);
        /* Max adjacent sample difference for a 440Hz sine wave: 2π*440/48000 ≈ 0.058 */
        TEST_ASSERT_TRUE(diff < 0.1f);
    }
}

/* --- Zero Input --- */

void test_stft_all_zeros(void) {
    float input[HOP_SIZE];
    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];
    float output[HOP_SIZE];
    memset(input, 0, sizeof(input));

    fe_stft_reset(&stft_state);
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);
    fe_stft_inverse(&stft_state, spec_real, spec_imag, output);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, output[i]);
    }
}

/* --- Reset --- */

void test_stft_reset_clears_state(void) {
    float input[HOP_SIZE];
    float spec_real[FREQ_BINS];
    float spec_imag[FREQ_BINS];
    float output1[HOP_SIZE], output2[HOP_SIZE];

    for (int i = 0; i < HOP_SIZE; i++) input[i] = 1.0f;

    fe_stft_reset(&stft_state);
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);
    fe_stft_inverse(&stft_state, spec_real, spec_imag, output1);

    fe_stft_reset(&stft_state);
    fe_stft_forward(&stft_state, input, spec_real, spec_imag);
    fe_stft_inverse(&stft_state, spec_real, spec_imag, output2);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-7f, output1[i], output2[i]);
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_hann_window_endpoints);
    RUN_TEST(test_hann_window_peak);
    RUN_TEST(test_hann_window_symmetry);
    RUN_TEST(test_hann_window_nonnegative);
    RUN_TEST(test_hann_window_cola_condition);
    RUN_TEST(test_stft_impulse);
    RUN_TEST(test_stft_dc_produces_energy_at_bin0);
    RUN_TEST(test_stft_roundtrip_sine);
    RUN_TEST(test_stft_roundtrip_white_noise);
    RUN_TEST(test_stft_streaming_continuity);
    RUN_TEST(test_stft_all_zeros);
    RUN_TEST(test_stft_reset_clears_state);

    return UNITY_END();
}
