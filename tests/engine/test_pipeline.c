/*
 * test_pipeline.c — Phase 3B-D: HPF/AGC前処理パイプライン テスト
 *
 * HPF: 2次Butterworth高域通過フィルタ (80Hz @ 48kHz)
 * AGC: RMSベース自動ゲイン制御
 *
 * コンパイル:
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
 * HPF テスト
 * ============================================================ */

/* DC成分(定数入力)が除去されること */
void test_hpf_removes_dc(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];

    /* 20フレーム分処理してフィルタを安定させる */
    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < FRAME_LEN; i++) buf[i] = 0.5f;
        fe_hpf_process(&hpf, buf, FRAME_LEN);
    }

    /* 安定後: DC成分はほぼ完全に除去されるはず */
    float max_abs = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) {
        float a = fabsf(buf[i]);
        if (a > max_abs) max_abs = a;
    }
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, max_abs);
}

/* 高周波正弦波(5kHz)がほぼそのまま通過すること */
void test_hpf_passes_high_freq(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];
    float ref[FRAME_LEN];

    /* 20フレームで安定化 */
    for (int f = 0; f < 20; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            int sample = f * FRAME_LEN + i;
            buf[i] = 0.5f * sinf(2.0f * PI_F * 5000.0f * sample / 48000.0f);
        }
        fe_hpf_process(&hpf, buf, FRAME_LEN);
    }

    /* 次のフレーム: 入力と出力のエネルギー比を計算 */
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
    /* 5kHzは80Hzカットオフのはるか上 → ゲインは-0.5dB以内 */
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 0.0f, gain_db);
}

/* 低周波正弦波(20Hz)が大幅に減衰すること */
void test_hpf_attenuates_low_freq(void) {
    FeHpfState hpf;
    fe_hpf_init(&hpf);

    float buf[FRAME_LEN];

    /* 50フレームで安定化 (20Hz = 長い周期なので多めに) */
    for (int f = 0; f < 50; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            int sample = f * FRAME_LEN + i;
            buf[i] = 0.5f * sinf(2.0f * PI_F * 20.0f * sample / 48000.0f);
        }
        fe_hpf_process(&hpf, buf, FRAME_LEN);
    }

    /* 計測フレーム */
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

    /* 20Hzは80Hzカットオフの下 → -12dB以上減衰 (2次フィルタ) */
    float gain_db = 10.0f * log10f((energy_out + 1e-20f) / (energy_in + 1e-20f));
    TEST_ASSERT_TRUE_MESSAGE(gain_db < -12.0f,
        "20Hzが十分に減衰されていない (2次ButterworthでHPF 80Hz)");
}

/* リセット後に初期状態に戻ること */
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

/* 1000フレーム処理してNaN/Infが出ないこと */
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
 * AGC テスト
 * ============================================================ */

/* 静かな信号が増幅されること */
void test_agc_amplifies_quiet(void) {
    FeAgcState agc;
    fe_agc_init(&agc);

    float buf[FRAME_LEN];
    float input_rms = 0.01f;

    /* 30フレーム処理してAGCを安定させる */
    for (int f = 0; f < 30; f++) {
        for (int i = 0; i < FRAME_LEN; i++) {
            buf[i] = input_rms * sinf(2.0f * PI_F * 1000.0f * (f * FRAME_LEN + i) / 48000.0f);
        }
        fe_agc_process(&agc, buf, FRAME_LEN);
    }

    /* 出力RMSを計測 */
    float rms = 0.0f;
    for (int i = 0; i < FRAME_LEN; i++) rms += buf[i] * buf[i];
    rms = sqrtf(rms / FRAME_LEN);

    /* 入力RMS(~0.007) より大きくなっているはず */
    TEST_ASSERT_TRUE_MESSAGE(rms > input_rms * 1.5f,
        "AGCが静かな信号を増幅していない");
}

/* 大きな信号が減衰されること */
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
        "AGCが大きな信号を減衰していない");
}

/* 無音入力でゲインが無限大にならないこと */
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

/* リセット後に初期状態に戻ること */
void test_agc_reset(void) {
    FeAgcState agc;
    fe_agc_init(&agc);

    float buf1[FRAME_LEN], buf2[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) {
        buf1[i] = 0.3f * sinf(2.0f * PI_F * 500.0f * i / 48000.0f);
        buf2[i] = buf1[i];
    }

    fe_agc_process(&agc, buf1, FRAME_LEN);

    /* 数フレーム処理してリセット */
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

/* 1000フレーム安定性 */
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
 * 統合テスト: HPF→AGC→推論パイプラインの有効/無効切替
 * (fastenhancer.hのfe_set_hpf/fe_set_agcを使用)
 * ============================================================ */

#include "fastenhancer.h"
#include "tiny_48k.h"

/* テスト用のランダム重みヘルパー (test_inference.cと同様) */
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

/* HPFデフォルト無効: 有効/無効で出力が変わること */
void test_integration_hpf_changes_output(void) {
    float dc_input[FRAME_LEN];
    for (int i = 0; i < FRAME_LEN; i++) dc_input[i] = 0.5f;

    float out_no_hpf[FRAME_LEN], out_hpf[FRAME_LEN];

    /* HPF無効 */
    FeState* s1 = create_state_for_pipeline();
    fe_process_frame(s1, dc_input, out_no_hpf);
    fe_destroy(s1);

    /* HPF有効 */
    FeState* s2 = create_state_for_pipeline();
    fe_set_hpf(s2, 1);
    fe_process_frame(s2, dc_input, out_hpf);
    fe_destroy(s2);

    int differ = 0;
    for (int i = 0; i < FRAME_LEN; i++) {
        if (fabsf(out_no_hpf[i] - out_hpf[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "HPF有効/無効で出力が変わらない: HPFがパイプラインに接続されていない");
}

/* AGCデフォルト無効: 有効/無効で出力が変わること */
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
        "AGC有効/無効で出力が変わらない: AGCがパイプラインに接続されていない");
}

/* デフォルトでHPF/AGCが無効であること */
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

    /* HPF単体テスト */
    RUN_TEST(test_hpf_removes_dc);
    RUN_TEST(test_hpf_passes_high_freq);
    RUN_TEST(test_hpf_attenuates_low_freq);
    RUN_TEST(test_hpf_reset);
    RUN_TEST(test_hpf_stability);

    /* AGC単体テスト */
    RUN_TEST(test_agc_amplifies_quiet);
    RUN_TEST(test_agc_attenuates_loud);
    RUN_TEST(test_agc_silence_safe);
    RUN_TEST(test_agc_reset);
    RUN_TEST(test_agc_stability);

    /* 統合テスト */
    RUN_TEST(test_integration_hpf_changes_output);
    RUN_TEST(test_integration_agc_changes_output);
    RUN_TEST(test_integration_default_disabled);

    return UNITY_END();
}
