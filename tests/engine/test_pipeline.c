/*
 * test_pipeline.c — Phase 3B-D: HPF/AGC Preprocessing Pipeline Tests
 *
 * HPF: 2nd-order Butterworth high-pass filter (80Hz @ 48kHz)
 * AGC: RMS-based automatic gain control
 *
 * Compile:
 *   gcc -I tests/engine/unity -I src/engine -I src/engine/common \
 *       -I src/engine/configs \
 *       tests/engine/unity/unity.c tests/engine/test_pipeline.c \
 *       src/engine/pipeline.c \
 *       -o test_pipeline -lm
 */

#include "unity.h"
#include "pipeline.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define PI_F 3.14159265f
#define FRAME_LEN 512

void setUp(void) {}
void tearDown(void) {}

/* ============================================================
 * HPF Tests
 * ============================================================ */

/* Verify that DC component (constant input) is removed */
void test_hpf_removes_dc(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];

    /* Process 20 frames to let the filter settle */
    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < FRAME_LEN; i++) buf[i] = 0.5f;
        fe_hpf_process(&hpf, buf, FRAME_LEN);
    }

    /* After settling: DC component should be almost completely removed */
    float max_abs = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) {
        float a = fabsf(buf[i]);
        if (a > max_abs) max_abs = a;
    }
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, max_abs);
}

/* Verify that a high-frequency sine wave (5kHz) passes through nearly unchanged */
void test_hpf_passes_high_freq(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];
    float ref[FRAME_LEN];

    /* Stabilize over 20 frames */
    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            int sample = f * FRAME_LEN + i;
            buf[i] = 0.5f * sinf(2.0f * PI_F * 5000.0f * sample / 48000.0f);
        }
        fe_hpf_process(&hpf, buf, FRAME_LEN);
    }

    /* Next frame: compute energy ratio between input and output */
    float energy_in = 0.0f, energy_out = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) {
        int sample = 20 * FRAME_LEN + i;
        ref[i] = 0.5f * sinf(2.0f * PI_F * 5000.0f * sample / 48000.0f);
        buf[i] = ref[i];
    }
    for (int i = 0; i < FRAME_LEN; i++) energy_in += ref[i] * ref[i];
    fe_hpf_process(&hpf, buf, FRAME_LEN);
    for (int i = 0; i < FRAME_LEN; i++) energy_out += buf[i] * buf[i];

    float gain_db = 10.0f * log10f(energy_out / energy_in);
    /* 5kHz is far above the 80Hz cutoff -> gain should be within -0.5dB */
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 0.0f, gain_db);
}

/* Verify that a low-frequency sine wave (20Hz) is significantly attenuated */
void test_hpf_attenuates_low_freq(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];

    /* Stabilize over 50 frames (more needed since 20Hz has a long period) */
    for (int f = 0; f < 50; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            int sample = f * FRAME_LEN + i;
            buf[i] = 0.5f * sinf(2.0f * PI_F * 20.0f * sample / 48000.0f);
        }
        fe_hpf_process(&hpf, buf, FRAME_LEN);
    }

    /* Measurement frame */
    float energy_in = 0.0f, energy_out = 0.0f;
    float ref[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) {
        int sample = 50 * FRAME_LEN + i;
        ref[i] = 0.5f * sinf(2.0f * PI_F * 20.0f * sample / 48000.0f);
        buf[i] = ref[i];
    }
    for (int i = 0; i < FRAME_LEN; i++) energy_in += ref[i] * ref[i];
    fe_hpf_process(&hpf, buf, FRAME_LEN);
    for (int i = 0; i < FRAME_LEN; i++) energy_out += buf[i] * buf[i];

    /* 20Hz is below the 80Hz cutoff -> at least -12dB attenuation (2nd-order filter) */
    float gain_db = 10.0f * log10f((energy_out + 1e-20f) / (energy_in + 1e-20f));
    TEST_ASSERT_TRUE_MESSAGE(gain_db < -12.0f,
        "20Hz not sufficiently attenuated (2nd-order Butterworth HPF 80Hz)");
}

/* Verify that state returns to initial conditions after reset */
void test_hpf_reset(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) buf[i] = 0.3f;
    fe_hpf_process(&hpf, buf, FRAME_LEN);

    float first_out = buf[0];

    fe_hpf_reset(&hpf);

    for (int i = 0; i < FRAME_LEN; i++) buf[i] = 0.3f;
    fe_hpf_process(&hpf, buf, FRAME_LEN);

    TEST_ASSERT_FLOAT_WITHIN(1e-7f, first_out, buf[0]);
}

/* Verify that no NaN/Inf values appear after processing 1000 frames */
void test_hpf_stability(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);
    float buf[FRAME_LEN];

    for (int f = 0; f < 1000; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            buf[i] = 0.8f * sinf(2.0f * PI_F * 200.0f * (f * FRAME_LEN + i) / 48000.0f);
        }
        fe_hpf_process(&hpf, buf, FRAME_LEN);
        for (int i = 0; i < FRAME_LEN; i++) {
            TEST_ASSERT_FALSE(isnan(buf[i]));
            TEST_ASSERT_FALSE(isinf(buf[i]));
        }
    }
}

/* ============================================================
 * AGC Tests
 * ============================================================ */

/* Verify that a quiet signal is amplified */
void test_agc_amplifies_quiet(void) {
    FeAgcState agc;
    fe_agc_init(&agc);

    float buf[FRAME_LEN];
    float input_rms = 0.01f;

    /* Process 30 frames to let the AGC settle */
    for (int f = 0; f < 30; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            buf[i] = input_rms * sinf(2.0f * PI_F * 1000.0f * (f * FRAME_LEN + i) / 48000.0f);
        }
        fe_agc_process(&agc, buf, FRAME_LEN);
    }

    /* Measure the output RMS */
    float rms = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) rms += buf[i] * buf[i];
    rms = sqrtf(rms / FRAME_LEN);

    /* Output should be greater than the input RMS (~0.007) */
    TEST_ASSERT_TRUE_MESSAGE(rms > input_rms * 1.5f,
        "AGC did not amplify quiet signal");
}

/* Verify that a loud signal is attenuated */
void test_agc_attenuates_loud(void) {
    FeAgcState agc;
    fe_agc_init(&agc);

    float buf[FRAME_LEN];
    float amp = 0.9f;

    for (int f = 0; f < 30; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            buf[i] = amp * sinf(2.0f * PI_F * 1000.0f * (f * FRAME_LEN + i) / 48000.0f);
        }
        fe_agc_process(&agc, buf, FRAME_LEN);
    }

    float rms = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) rms += buf[i] * buf[i];
    rms = sqrtf(rms / FRAME_LEN);

    float input_rms = amp / sqrtf(2.0f);
    TEST_ASSERT_TRUE_MESSAGE(rms < input_rms * 0.7f,
        "AGC did not attenuate loud signal");
}

/* Verify that gain does not become infinite on silent input */
void test_agc_silence_safe(void) {
    FeAgcState agc;
    fe_agc_init(&agc);

    float buf[FRAME_LEN];
    memset(buf, 0, sizeof(buf));

    for (int f = 0; f < 50; f++) {
        fe_agc_process(&agc, buf, FRAME_LEN);
        for (int i = 0; i < FRAME_LEN; i++) {
            TEST_ASSERT_FALSE(isnan(buf[i]));
            TEST_ASSERT_FALSE(isinf(buf[i]));
            TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, buf[i]);
        }
    }
}

/* Verify that state returns to initial conditions after reset */
void test_agc_reset(void) {
    FeAgcState agc;
    fe_agc_init(&agc);

    float buf1[FRAME_LEN], buf2[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) {
        buf1[i] = 0.3f * sinf(2.0f * PI_F * 500.0f * i / 48000.0f);
        buf2[i] = buf1[i];
    }

    fe_agc_process(&agc, buf1, FRAME_LEN);

    /* Process several frames then reset */
    float tmp[FRAME_LEN];
    for (int f = 0; f < 10; f++) {
        for (int i = 0; i < FRAME_LEN; i++)
            tmp[i] = 0.3f * sinf(2.0f * PI_F * 500.0f * ((f + 1) * FRAME_LEN + i) / 48000.0f);
        fe_agc_process(&agc, tmp, FRAME_LEN);
    }

    fe_agc_reset(&agc);
    fe_agc_process(&agc, buf2, FRAME_LEN);

    for (int i = 0; i < FRAME_LEN; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, buf1[i], buf2[i]);
    }
}

/* 1000-frame stability */
void test_agc_stability(void) {
    FeAgcState agc;
    fe_agc_init(&agc);
    float buf[FRAME_LEN];

    for (int f = 0; f < 1000; f++) {
        float amp = 0.1f + 0.4f * sinf(0.01f * f);
        for (int i = 0; i < FRAME_LEN; i++) {
            buf[i] = amp * sinf(2.0f * PI_F * 440.0f * (f * FRAME_LEN + i) / 48000.0f);
        }
        fe_agc_process(&agc, buf, FRAME_LEN);
        for (int i = 0; i < FRAME_LEN; i++) {
            TEST_ASSERT_FALSE(isnan(buf[i]));
            TEST_ASSERT_FALSE(isinf(buf[i]));
        }
    }
}

/* ============================================================
 * Integration Tests: HPF -> AGC -> Inference Pipeline Enable/Disable Toggling
 * (using fe_set_hpf/fe_set_agc from fastenhancer.h)
 * ============================================================ */

#include "fastenhancer.h"
#include "tiny_48k.h"

/* Random weight helper for tests (same as test_inference.c) */
static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
}

static void fix_bn_scales_pipeline(float* w) {
    int p = 0;
    p += FE_C1 * 2 * FE_ENC_K0;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    }
    p += FE_F2 * FE_F1;
    p += FE_C2 * FE_C1;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 1.0f; p += FE_C2;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 0.0f; p += FE_C2;
    for (int b = 0; b < FE_RF_BLOCKS; b++) {
        p += FE_W_GRU + FE_W_GRU_FC;
        if (b == 0) p += FE_W_PE;
        p += FE_W_MHSA;
    }
    p += FE_C1 * FE_C2;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    p += FE_F1 * FE_F2;
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * 2 * FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    }
    p += FE_C1 * 2 * FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
}

static FeState* create_state_for_pipeline(void) {
    int n = fe_weight_count(FE_MODEL_TINY);
    float* w = (float*)malloc(n * sizeof(float));
    unsigned int s = 42;
    for (int i = 0; i < n; i++) w[i] = lcg_float(&s);
    fix_bn_scales_pipeline(w);
    FeState* state = fe_create(FE_MODEL_TINY, w, n);
    free(w);
    return state;
}

/* HPF disabled by default: output should differ between enabled and disabled */
void test_integration_hpf_changes_output(void) {
    float dc_input[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) dc_input[i] = 0.5f;

    float out_no_hpf[FRAME_LEN], out_hpf[FRAME_LEN];

    /* HPF disabled */
    FeState* s1 = create_state_for_pipeline();
    fe_process_frame(s1, dc_input, out_no_hpf);
    fe_destroy(s1);

    /* HPF enabled */
    FeState* s2 = create_state_for_pipeline();
    fe_set_hpf(s2, 1);
    fe_process_frame(s2, dc_input, out_hpf);
    fe_destroy(s2);

    int differ = 0;
    for (int i = 0; i < FRAME_LEN; i++) {
        if (fabsf(out_no_hpf[i] - out_hpf[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "Output unchanged with HPF enabled/disabled: HPF is not connected to the pipeline");
}

/* AGC disabled by default: output should differ between enabled and disabled */
void test_integration_agc_changes_output(void) {
    float input[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) {
        input[i] = 0.01f * sinf(2.0f * PI_F * 1000.0f * i / 48000.0f);
    }

    float out_no_agc[FRAME_LEN], out_agc[FRAME_LEN];

    FeState* s1 = create_state_for_pipeline();
    for (int f = 0; f < 8; f++) {
        fe_process_frame(s1, input, out_no_agc);
    }
    fe_destroy(s1);

    FeState* s2 = create_state_for_pipeline();
    fe_set_agc(s2, 1);
    for (int f = 0; f < 8; f++) {
        fe_process_frame(s2, input, out_agc);
    }
    fe_destroy(s2);

    int differ = 0;
    for (int i = 0; i < FRAME_LEN; i++) {
        if (fabsf(out_no_agc[i] - out_agc[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "Output unchanged with AGC enabled/disabled: AGC is not connected to the pipeline");
}

/* Verify that HPF/AGC are disabled by default */
void test_integration_default_disabled(void) {
    float input[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) {
        input[i] = 0.5f * sinf(2.0f * PI_F * 440.0f * i / 48000.0f);
    }

    float out_default[FRAME_LEN], out_explicit_off[FRAME_LEN];

    FeState* s1 = create_state_for_pipeline();
    fe_process_frame(s1, input, out_default);
    fe_destroy(s1);

    FeState* s2 = create_state_for_pipeline();
    fe_set_hpf(s2, 0);
    fe_set_agc(s2, 0);
    fe_process_frame(s2, input, out_explicit_off);
    fe_destroy(s2);

    for (int i = 0; i < FRAME_LEN; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-7f, out_default[i], out_explicit_off[i]);
    }
}

int main(void) {
    UNITY_BEGIN();

    /* HPF unit tests */
    RUN_TEST(test_hpf_removes_dc);
    RUN_TEST(test_hpf_passes_high_freq);
    RUN_TEST(test_hpf_attenuates_low_freq);
    RUN_TEST(test_hpf_reset);
    RUN_TEST(test_hpf_stability);

    /* AGC unit tests */
    RUN_TEST(test_agc_amplifies_quiet);
    RUN_TEST(test_agc_attenuates_loud);
    RUN_TEST(test_agc_silence_safe);
    RUN_TEST(test_agc_reset);
    RUN_TEST(test_agc_stability);

    /* Integration tests */
    RUN_TEST(test_integration_hpf_changes_output);
    RUN_TEST(test_integration_agc_changes_output);
    RUN_TEST(test_integration_default_disabled);

    return UNITY_END();
}
