/*
 * test_edge_cases.c — Phase 2-H: エッジケース入力テスト (TDD Red)
 *
 * 検証対象 (カテゴリ2: エッジケース入力):
 *   - 全ゼロ入力
 *   - 最大振幅 (±1.0f交互)
 *   - DC offset (定数入力)
 *   - NaN注入
 *   - Inf注入
 *   - 1サンプルインパルス
 *
 * これらはfe_process()（統合パイプライン）に対するテスト。
 * 個別モジュールのエッジケースは各モジュールテストでカバー。
 *
 * コンパイル:
 *   gcc -I tests/engine/unity -I src/engine/common -I src/engine \
 *       tests/engine/unity/unity.c tests/engine/test_edge_cases.c \
 *       src/engine/fastenhancer.c src/engine/common/*.c -o test_edge_cases -lm
 */

#include "unity.h"
#include "fastenhancer.h"
#include "exports.h"
#include "tiny_48k.h"
#include <math.h>
#include <string.h>

#define HOP_SIZE 512

static FeState* state;

/* 全ゼロだが正しいサイズの重みでパイプラインの安定性を検証。
 * 実際の推論精度テストは test_inference.c が担当。 */
static float dummy_weights[FE_TOTAL_WEIGHTS];

void setUp(void) {
    memset(dummy_weights, 0, sizeof(dummy_weights));
    /* BN scale を 1.0 に設定しないと出力が全ゼロ固定になる。
     * ここではクラッシュ耐性が目的なので全ゼロのまま。 */
    state = fe_create(FE_MODEL_TINY, dummy_weights, FE_TOTAL_WEIGHTS);
    TEST_ASSERT_NOT_NULL(state);
}

void tearDown(void) {
    if (state) {
        fe_destroy(state);
        state = NULL;
    }
}

/* --- 全ゼロ入力 --- */

void test_edge_all_zeros(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);
    memset(input, 0, HOP_SIZE * sizeof(float));

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, output[i]);
    }
}

/* --- 最大振幅 --- */

void test_edge_max_amplitude(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < HOP_SIZE; i++) {
        input[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- DC offset (定数入力) --- */

void test_edge_dc_offset(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < HOP_SIZE; i++) input[i] = 0.5f;

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- DC offset連続処理 --- */

void test_edge_dc_offset_sustained(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < HOP_SIZE; i++) input[i] = 0.5f;
        int ret = fe_process(state, input, output);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- NaN注入 --- */

void test_edge_nan_first_sample(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < HOP_SIZE; i++) input[i] = 0.1f;
    input[0] = NAN;

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    /* NaN注入後も出力全体がNaNやInfに汚染されないこと */
    int nan_count = 0;
    int inf_count = 0;
    for (int i = 0; i < HOP_SIZE; i++) {
        if (isnan(output[i])) nan_count++;
        if (isinf(output[i])) inf_count++;
    }
    TEST_ASSERT_EQUAL_INT(0, nan_count);
    TEST_ASSERT_EQUAL_INT(0, inf_count);
}

void test_edge_nan_all_samples(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < HOP_SIZE; i++) input[i] = NAN;

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- Inf注入 --- */

void test_edge_positive_inf(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < HOP_SIZE; i++) input[i] = 0.1f;
    input[0] = INFINITY;

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

void test_edge_negative_inf(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    for (int i = 0; i < HOP_SIZE; i++) input[i] = 0.1f;
    input[0] = -INFINITY;

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- 1サンプルインパルス --- */

void test_edge_single_impulse(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    memset(input, 0, HOP_SIZE * sizeof(float));
    input[0] = 1.0f;

    int ret = fe_process(state, input, output);
    TEST_ASSERT_EQUAL_INT(0, ret);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

/* --- init/destroy --- */

void test_edge_init_returns_valid_state(void) {
    /* setUp already creates state */
    TEST_ASSERT_NOT_NULL(state);

    float* input_ptr = fe_get_input_ptr(state);
    float* output_ptr = fe_get_output_ptr(state);
    TEST_ASSERT_NOT_NULL(input_ptr);
    TEST_ASSERT_NOT_NULL(output_ptr);

    int hop = fe_get_hop_size(state);
    TEST_ASSERT_EQUAL_INT(HOP_SIZE, hop);
}

void test_edge_reset_clears_hidden_state(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    /* いくつかフレームを処理して内部状態を変更 */
    for (int f = 0; f < 10; f++) {
        for (int i = 0; i < HOP_SIZE; i++) input[i] = 0.3f;
        fe_process(state, input, output);
    }

    /* リセット */
    fe_reset(state);

    /* リセット後のゼロ入力は初回と同じ出力になるはず */
    memset(input, 0, HOP_SIZE * sizeof(float));
    fe_process(state, input, output);

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, output[i]);
    }
}

/* --- 連続フレーム安定性 --- */

void test_edge_many_frames_no_crash(void) {
    float* input = fe_get_input_ptr(state);
    float* output = fe_get_output_ptr(state);

    unsigned int seed = 42;
    for (int frame = 0; frame < 500; frame++) {
        for (int i = 0; i < HOP_SIZE; i++) {
            seed = seed * 1103515245 + 12345;
            input[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
        }
        int ret = fe_process(state, input, output);
        TEST_ASSERT_EQUAL_INT(0, ret);
    }

    for (int i = 0; i < HOP_SIZE; i++) {
        TEST_ASSERT_FALSE(isnan(output[i]));
        TEST_ASSERT_FALSE(isinf(output[i]));
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_edge_all_zeros);
    RUN_TEST(test_edge_max_amplitude);
    RUN_TEST(test_edge_dc_offset);
    RUN_TEST(test_edge_dc_offset_sustained);
    RUN_TEST(test_edge_nan_first_sample);
    RUN_TEST(test_edge_nan_all_samples);
    RUN_TEST(test_edge_positive_inf);
    RUN_TEST(test_edge_negative_inf);
    RUN_TEST(test_edge_single_impulse);
    RUN_TEST(test_edge_init_returns_valid_state);
    RUN_TEST(test_edge_reset_clears_hidden_state);
    RUN_TEST(test_edge_many_frames_no_crash);

    return UNITY_END();
}
