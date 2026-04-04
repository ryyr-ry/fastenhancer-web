/*
 * test_inference.c — Phase 3B-C: 推論パイプライン テスト
 *
 * ランダム非ゼロ重み（BNスケール=1.0）を使用し、パイプラインの配線が
 * 正しく接続されていることを検証する。ゼロ重みでは全てゼロが通過して
 * しまい配線ミスを検出できないため、意味のあるテストにならない。
 *
 * コンパイル:
 *   gcc -I tests/engine/unity -I src/engine/common -I src/engine/configs \
 *       -I src/engine tests/engine/unity/unity.c tests/engine/test_inference.c \
 *       src/engine/fastenhancer.c \
 *       src/engine/common/fft.c src/engine/common/stft.c \
 *       src/engine/common/conv.c src/engine/common/gru.c \
 *       src/engine/common/attention.c src/engine/common/activations.c \
 *       src/engine/common/compression.c \
 *       -o test_inference -lm
 */

#include "unity.h"
#include "fastenhancer.h"
#include "tiny_48k.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI_F 3.14159265f

/* 決定論的擬似乱数 (LCG) */
static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
}

/* BNスケール=1.0, BNバイアス=0.0 に上書きする。
 * 重みバッファのレイアウトはsetup_weights()と完全一致が必要。 */
static void fix_bn_scales(float* w) {
    int p = 0;

    /* Encoder PreNet: conv_w[C1*2*K0] + bn_s[C1] + bn_b[C1] */
    p += FE_C1 * 2 * FE_ENC_K0;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;

    /* Encoder Blocks */
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    }

    /* RNNFormer PreNet: freq_w + conv_w + bn_s + bn_b */
    p += FE_F2 * FE_F1;
    p += FE_C2 * FE_C1;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 1.0f; p += FE_C2;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 0.0f; p += FE_C2;

    /* RNNFormer Blocks (BNなし — GRU/FC/PE/MHSA) */
    for (int b = 0; b < FE_RF_BLOCKS; b++) {
        p += FE_W_GRU + FE_W_GRU_FC;
        if (b == 0) p += FE_W_PE;
        p += FE_W_MHSA;
    }

    /* RNNFormer PostNet: conv_w + bn_s + bn_b + freq_w */
    p += FE_C1 * FE_C2;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    p += FE_F1 * FE_F2;

    /* Decoder Blocks */
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        /* skip 1×1 conv + BN */
        p += FE_C1 * 2 * FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
        /* 3×3 conv + BN */
        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
    }

    /* Decoder PostNet: skip 1×1 conv + BN + deconv_w + deconv_b */
    p += FE_C1 * 2 * FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f; p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f; p += FE_C1;
}

/* ランダム非ゼロ重み（BNスケール修正済み）のFeStateを生成 */
static FeState* create_random_state(unsigned int seed) {
    int n = fe_weight_count(FE_MODEL_TINY);
    float* w = (float*)malloc(n * sizeof(float));
    unsigned int s = seed;
    for (int i = 0; i < n; i++) w[i] = lcg_float(&s);
    fix_bn_scales(w);
    FeState* state = fe_create(FE_MODEL_TINY, w, n);
    free(w);
    return state;
}

/* ゼロ重みのFeState */
static FeState* create_zero_state(void) {
    int n = fe_weight_count(FE_MODEL_TINY);
    float* w = (float*)calloc(n, sizeof(float));
    FeState* state = fe_create(FE_MODEL_TINY, w, n);
    free(w);
    return state;
}

/* 正弦波入力を生成 */
static void gen_sine(float* buf, int len, float freq, float amp) {
    for (int i = 0; i < len; i++) {
        buf[i] = amp * sinf(2.0f * PI_F * freq * (float)i / 48000.0f);
    }
}

void setUp(void) {}
void tearDown(void) {}

/* ======== ライフサイクルテスト ======== */

void test_fe_create_destroy(void) {
    FeState* state = create_random_state(42);
    TEST_ASSERT_NOT_NULL(state);
    fe_destroy(state);
}

void test_fe_create_null_on_invalid_model(void) {
    float w[1] = {0.0f};
    TEST_ASSERT_NULL(fe_create(99, w, 1));
}

void test_fe_create_null_on_wrong_weight_count(void) {
    float w[1] = {0.0f};
    TEST_ASSERT_NULL(fe_create(FE_MODEL_TINY, w, 1));
}

void test_fe_get_hop_size(void) {
    FeState* state = create_random_state(42);
    TEST_ASSERT_EQUAL_INT(512, fe_get_hop_size(state));
    fe_destroy(state);
}

/* ======== 配線検証: 非ゼロ重み → 非ゼロ出力 ======== */

void test_nonzero_weights_produce_nonzero_output(void) {
    FeState* state = create_random_state(42);
    float input[512], output[512];
    gen_sine(input, 512, 1000.0f, 0.5f);

    fe_process_frame(state, input, output);

    int nonzero = 0;
    for (int i = 0; i < 512; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
        if (fabsf(output[i]) > 1e-10f) nonzero++;
    }
    TEST_ASSERT_MESSAGE(nonzero > 0,
        "非ゼロ重み+非ゼロ入力なのに全出力がゼロ: パイプライン断線の可能性");

    fe_destroy(state);
}

/* ======== 入力依存性: 異なる入力 → 異なる出力 ======== */

void test_different_inputs_different_outputs(void) {
    FeState* s1 = create_random_state(42);
    FeState* s2 = create_random_state(42);

    float in_a[512], in_b[512], out_a[512], out_b[512];
    gen_sine(in_a, 512, 1000.0f, 0.5f);
    gen_sine(in_b, 512, 2000.0f, 0.5f);

    fe_process_frame(s1, in_a, out_a);
    fe_process_frame(s2, in_b, out_b);

    int differ = 0;
    for (int i = 0; i < 512; i++) {
        if (fabsf(out_a[i] - out_b[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "異なる入力なのに同一出力: モデルが入力を無視している");

    fe_destroy(s1);
    fe_destroy(s2);
}

/* ======== 重み依存性: 異なる重み → 異なる出力 ======== */

void test_different_weights_different_outputs(void) {
    FeState* s1 = create_random_state(42);
    FeState* s2 = create_random_state(12345);

    float input[512], out_a[512], out_b[512];
    gen_sine(input, 512, 1000.0f, 0.5f);

    fe_process_frame(s1, input, out_a);
    fe_process_frame(s2, input, out_b);

    int differ = 0;
    for (int i = 0; i < 512; i++) {
        if (fabsf(out_a[i] - out_b[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "異なる重みなのに同一出力: 重みがパイプラインに接続されていない");

    fe_destroy(s1);
    fe_destroy(s2);
}

/* ======== GRU状態蓄積: 同一入力でも2フレーム目は異なる ======== */

void test_gru_state_accumulates_across_frames(void) {
    FeState* state = create_random_state(42);

    float input[512], out1[512], out2[512];
    gen_sine(input, 512, 500.0f, 0.3f);

    fe_process_frame(state, input, out1);
    fe_process_frame(state, input, out2);

    int differ = 0;
    for (int i = 0; i < 512; i++) {
        if (fabsf(out1[i] - out2[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "2フレーム目が1フレーム目と完全一致: GRU隠れ状態が蓄積されていない");

    fe_destroy(state);
}

/* ======== リセット: 状態を元に戻す ======== */

void test_reset_restores_initial_state(void) {
    FeState* state = create_random_state(42);

    float input[512], first[512], after_reset[512], dummy[512];
    gen_sine(input, 512, 800.0f, 0.4f);

    fe_process_frame(state, input, first);
    for (int f = 0; f < 5; f++) fe_process_frame(state, input, dummy);

    fe_reset(state);
    fe_process_frame(state, input, after_reset);

    for (int i = 0; i < 512; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, first[i], after_reset[i]);
    }

    fe_destroy(state);
}

/* ======== 決定性: 同一重み+入力 → 同一出力 ======== */

void test_deterministic_same_weights_same_input(void) {
    float input[512], out_a[512], out_b[512];
    gen_sine(input, 512, 1000.0f, 0.5f);

    FeState* a = create_random_state(42);
    fe_process_frame(a, input, out_a);
    fe_destroy(a);

    FeState* b = create_random_state(42);
    fe_process_frame(b, input, out_b);
    fe_destroy(b);

    for (int i = 0; i < 512; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-7f, out_a[i], out_b[i]);
    }
}

/* ======== 100フレーム数値安定性 ======== */

void test_100_frames_numerically_stable(void) {
    FeState* state = create_random_state(42);
    float input[512], output[512];

    for (int frame = 0; frame < 100; frame++) {
        gen_sine(input, 512, 440.0f, 0.2f);
        fe_process_frame(state, input, output);

        for (int i = 0; i < 512; i++) {
            TEST_ASSERT_FALSE_MESSAGE(isnan(output[i]),
                "NaN検出: 100フレーム以内に数値爆発");
            TEST_ASSERT_FALSE_MESSAGE(isinf(output[i]),
                "Inf検出: 100フレーム以内に数値爆発");
        }
    }

    fe_destroy(state);
}

/* ======== インスタンス独立性 ======== */

void test_instances_independent(void) {
    FeState* s1 = create_random_state(42);
    FeState* s2 = create_random_state(42);

    float sine[512], zero[512] = {0};
    float out_sine[512], out_zero[512];
    gen_sine(sine, 512, 1000.0f, 0.5f);

    fe_process_frame(s1, sine, out_sine);
    fe_process_frame(s2, zero, out_zero);

    /* s1は非ゼロ入力→非ゼロ出力、s2はゼロ入力 */
    int differ = 0;
    for (int i = 0; i < 512; i++) {
        if (fabsf(out_sine[i] - out_zero[i]) > 1e-8f) differ++;
    }
    TEST_ASSERT_MESSAGE(differ > 0,
        "異なるインスタンスに異なる入力を与えたのに同一出力: メモリ共有の疑い");

    fe_destroy(s1);
    fe_destroy(s2);
}

/* ======== ゼロ重み+ゼロ入力 → ゼロ出力 (エッジケース) ======== */

void test_zero_weights_zero_input_gives_zero(void) {
    FeState* state = create_zero_state();
    float input[512] = {0}, output[512];

    fe_process_frame(state, input, output);

    for (int i = 0; i < 512; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output[i]);
    }

    fe_destroy(state);
}

int main(void) {
    UNITY_BEGIN();

    /* ライフサイクル */
    RUN_TEST(test_fe_create_destroy);
    RUN_TEST(test_fe_create_null_on_invalid_model);
    RUN_TEST(test_fe_create_null_on_wrong_weight_count);
    RUN_TEST(test_fe_get_hop_size);

    /* パイプライン配線検証 (ランダム非ゼロ重み) */
    RUN_TEST(test_nonzero_weights_produce_nonzero_output);
    RUN_TEST(test_different_inputs_different_outputs);
    RUN_TEST(test_different_weights_different_outputs);
    RUN_TEST(test_gru_state_accumulates_across_frames);

    /* 状態管理 */
    RUN_TEST(test_reset_restores_initial_state);
    RUN_TEST(test_deterministic_same_weights_same_input);

    /* 安定性 */
    RUN_TEST(test_100_frames_numerically_stable);
    RUN_TEST(test_instances_independent);

    /* エッジケース */
    RUN_TEST(test_zero_weights_zero_input_gives_zero);

    return UNITY_END();
}
