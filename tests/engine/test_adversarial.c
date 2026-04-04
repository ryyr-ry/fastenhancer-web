/*
 * test_adversarial.c — 敵対的入力パターンテスト
 *
 * test_edge_cases.c がカバーしない攻撃的シナリオを検証:
 *   - 処理中リセットの安全性
 *   - 無音→大音量→無音の急変
 *   - 非正規化浮動小数点 (subnormal)
 *   - FLT_MAX入力
 *   - 複数create/destroyサイクル（メモリリーク兆候）
 *   - 長時間無音後の復帰
 *   - 振幅ランプアップ（漸増入力）
 *   - 高エネルギーノイズ連続
 */

#include "unity.h"
#include "fastenhancer.h"
#include "exports.h"
#include "tiny_48k.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HOP FE_HOP_SIZE

static FeState* state;
static float dummy_weights[FE_TOTAL_WEIGHTS];

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fffu) / 32768.0f - 0.5f) * 2.0f;
}

void setUp(void) {
    memset(dummy_weights, 0, sizeof(dummy_weights));
    state = fe_create(FE_MODEL_TINY, dummy_weights, FE_TOTAL_WEIGHTS);
    TEST_ASSERT_NOT_NULL(state);
}

void tearDown(void) {
    if (state) {
        fe_destroy(state);
        state = NULL;
    }
}

static void assert_all_finite(const float* buf, int n) {
    for (int i = 0; i < n; i++) {
        TEST_ASSERT_FALSE_MESSAGE(isnan(buf[i]), "NaN detected");
        TEST_ASSERT_FALSE_MESSAGE(isinf(buf[i]), "Inf detected");
    }
}

/* --- 処理中リセットの安全性 --- */

void test_adv_reset_mid_stream(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);
    unsigned int seed = 99u;

    for (int f = 0; f < 50; f++) {
        for (int i = 0; i < HOP; i++) in[i] = lcg_float(&seed) * 0.3f;
        fe_process(state, in, out);
    }

    fe_reset(state);

    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < HOP; i++) in[i] = lcg_float(&seed) * 0.3f;
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- 無音→爆音→無音の急変 --- */

void test_adv_silence_burst_silence(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    /* 100フレーム無音 */
    memset(in, 0, HOP * sizeof(float));
    for (int f = 0; f < 100; f++) {
        fe_process(state, in, out);
    }

    /* 5フレーム大音量 */
    for (int f = 0; f < 5; f++) {
        for (int i = 0; i < HOP; i++)
            in[i] = (i % 2 == 0) ? 0.99f : -0.99f;
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);

    /* 再び100フレーム無音 */
    memset(in, 0, HOP * sizeof(float));
    for (int f = 0; f < 100; f++) {
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- 非正規化浮動小数点 (subnormal) --- */

void test_adv_subnormal_input(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    for (int i = 0; i < HOP; i++)
        in[i] = FLT_MIN * 0.5f; /* subnormal value */

    for (int f = 0; f < 10; f++) {
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- FLT_MAX入力 --- */

void test_adv_flt_max_input(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    for (int i = 0; i < HOP; i++)
        in[i] = (i % 2 == 0) ? FLT_MAX : -FLT_MAX;

    int ret = fe_process(state, in, out);
    TEST_ASSERT_EQUAL_INT(0, ret);
    /* FLT_MAXは入力サニタイズでクランプされるため出力は有限値 */
    assert_all_finite(out, HOP);
}

/* --- 複数create/destroyサイクル --- */

void test_adv_create_destroy_cycles(void) {
    float input[HOP];
    float output[HOP];

    /* setUp()で作られたstateを先に解放 */
    fe_destroy(state);
    state = NULL;

    for (int cycle = 0; cycle < 50; cycle++) {
        FeState* s = fe_create(FE_MODEL_TINY, dummy_weights, FE_TOTAL_WEIGHTS);
        TEST_ASSERT_NOT_NULL_MESSAGE(s, "create failed in cycle");

        memset(input, 0, sizeof(input));
        int ret = fe_process(s, input, output);
        TEST_ASSERT_EQUAL_INT(0, ret);

        fe_destroy(s);
    }

    /* tearDown()でstate=NULLなのでdouble freeしない */
}

/* --- 長時間無音後の復帰 --- */

void test_adv_long_silence_then_signal(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    /* 1000フレーム無音（GRU隠れ状態が安定するか） */
    memset(in, 0, HOP * sizeof(float));
    for (int f = 0; f < 1000; f++) {
        fe_process(state, in, out);
    }

    /* 信号復帰 */
    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < HOP; i++)
            in[i] = 0.2f * sinf(2.0f * 3.14159f * 440.0f * (float)(f * HOP + i) / 48000.0f);
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- 振幅ランプアップ --- */

void test_adv_amplitude_ramp(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    for (int f = 0; f < 200; f++) {
        float amplitude = (float)f / 200.0f; /* 0.0 → 1.0 */
        for (int i = 0; i < HOP; i++)
            in[i] = amplitude * sinf(2.0f * 3.14159f * 1000.0f * (float)(f * HOP + i) / 48000.0f);
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- 高エネルギーノイズ連続 --- */

void test_adv_sustained_noise(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);
    unsigned int seed = 777u;

    for (int f = 0; f < 500; f++) {
        for (int i = 0; i < HOP; i++)
            in[i] = lcg_float(&seed) * 0.9f;
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- 交互NaN/正常入力 --- */

void test_adv_alternating_nan_normal(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    for (int f = 0; f < 20; f++) {
        if (f % 2 == 0) {
            for (int i = 0; i < HOP; i++) in[i] = NAN;
        } else {
            for (int i = 0; i < HOP; i++)
                in[i] = 0.1f * sinf(2.0f * 3.14159f * 440.0f * (float)(f * HOP + i) / 48000.0f);
        }
        int ret = fe_process(state, in, out);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }
    assert_all_finite(out, HOP);
}

/* --- 連続リセット --- */

void test_adv_rapid_resets(void) {
    float* in = fe_get_input_ptr(state);
    float* out = fe_get_output_ptr(state);

    for (int i = 0; i < HOP; i++) in[i] = 0.1f;

    for (int cycle = 0; cycle < 100; cycle++) {
        fe_process(state, in, out);
        fe_reset(state);
    }

    int ret = fe_process(state, in, out);
    TEST_ASSERT_EQUAL_INT(0, ret);
    assert_all_finite(out, HOP);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_adv_reset_mid_stream);
    RUN_TEST(test_adv_silence_burst_silence);
    RUN_TEST(test_adv_subnormal_input);
    RUN_TEST(test_adv_flt_max_input);
    RUN_TEST(test_adv_create_destroy_cycles);
    RUN_TEST(test_adv_long_silence_then_signal);
    RUN_TEST(test_adv_amplitude_ramp);
    RUN_TEST(test_adv_sustained_noise);
    RUN_TEST(test_adv_alternating_nan_normal);
    RUN_TEST(test_adv_rapid_resets);

    return UNITY_END();
}
