/*
 * test_gru.c — Phase 2-E: GRUテスト (TDD Red)
 *
 * 検証対象:
 *   - ゼロ入力 → ゼロ出力
 *   - 既知重みでの期待出力
 *   - 1000ステップ安定性 (カテゴリ1: 数値安定性)
 *   - 隠れ状態リセット
 *   - 更新ゲート/リセットゲートの動作
 *
 * コンパイル:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_gru.c \
 *       src/engine/common/gru.c src/engine/common/activations.c -o test_gru -lm
 */

#include "unity.h"
#include "gru.h"
#include <math.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* --- ゼロ入力テスト --- */

void test_gru_zero_input_zero_hidden(void) {
    /* 入力=0, 隠れ状態=0 → 出力は0（バイアスにも依存するが、
     * ゼロ初期化されたGRUでは z=sigmoid(0)=0.5, r=sigmoid(0)=0.5,
     * n=tanh(0)=0, h=(1-0.5)*0 + 0.5*0 = 0 となる。
     * ただしバイアスが全てゼロの場合。 */
    int hidden_size = 20;  /* Tiny C2=20 */
    int input_size = 20;

    float input[20] = {0};
    float hidden[20] = {0};
    /* ゼロ重み・ゼロバイアス */
    float W_z[20 * 20] = {0};
    float U_z[20 * 20] = {0};
    float b_z[20] = {0};
    float W_r[20 * 20] = {0};
    float U_r[20 * 20] = {0};
    float b_r[20] = {0};
    float W_n[20 * 20] = {0};
    float U_n[20 * 20] = {0};
    float b_in_n[20] = {0};
    float b_hn_n[20] = {0};

    FeGruWeights weights = {
        .W_z = W_z, .U_z = U_z, .b_z = b_z,
        .W_r = W_r, .U_r = U_r, .b_r = b_r,
        .W_n = W_n, .U_n = U_n, .b_in_n = b_in_n, .b_hn_n = b_hn_n,
        .input_size = input_size,
        .hidden_size = hidden_size
    };

    fe_gru_step(&weights, input, hidden);

    for (int i = 0; i < hidden_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, hidden[i]);
    }
}

/* --- 既知重みでの手計算テスト --- */

void test_gru_known_weights_single_unit(void) {
    /* hidden_size=1, input_size=1 のミニGRUで手計算と照合 */
    int hidden_size = 1;
    int input_size = 1;

    float input[1] = {1.0f};
    float hidden[1] = {0.0f};

    float W_z[1] = {0.5f};
    float U_z[1] = {0.5f};
    float b_z[1] = {0.0f};
    float W_r[1] = {0.5f};
    float U_r[1] = {0.5f};
    float b_r[1] = {0.0f};
    float W_n[1] = {1.0f};
    float U_n[1] = {1.0f};
    float b_in_n[1] = {0.0f};
    float b_hn_n[1] = {0.0f};

    FeGruWeights weights = {
        .W_z = W_z, .U_z = U_z, .b_z = b_z,
        .W_r = W_r, .U_r = U_r, .b_r = b_r,
        .W_n = W_n, .U_n = U_n, .b_in_n = b_in_n, .b_hn_n = b_hn_n,
        .input_size = input_size,
        .hidden_size = hidden_size
    };

    fe_gru_step(&weights, input, hidden);

    /* 手計算:
     * z = sigmoid(0.5*1 + 0.5*0 + 0) = sigmoid(0.5) ≈ 0.6225
     * r = sigmoid(0.5*1 + 0.5*0 + 0) = sigmoid(0.5) ≈ 0.6225
     * n = tanh(1.0*1 + 0.6225*(1.0*0 + 0)) = tanh(1.0) ≈ 0.7616
     * h = (1-0.6225)*0.7616 + 0.6225*0 ≈ 0.2878
     */
    float z = 1.0f / (1.0f + expf(-0.5f));
    float n = tanhf(1.0f);
    float expected_h = (1.0f - z) * n;

    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected_h, hidden[0]);
}

/* --- 1000ステップ安定性テスト (カテゴリ1) --- */

void test_gru_stability_1000_steps(void) {
    int hidden_size = 20;
    int input_size = 20;

    float input[20] = {0};
    float hidden[20] = {0};

    /* 小さなランダム重み */
    float W_z[400], U_z[400], b_z[20];
    float W_r[400], U_r[400], b_r[20];
    float W_n[400], U_n[400], b_in_n[20], b_hn_n[20];

    unsigned int seed = 12345;
    for (int i = 0; i < 400; i++) {
        seed = seed * 1103515245 + 12345;
        float val = ((float)(seed >> 16) / 32768.0f - 1.0f) * 0.1f;
        W_z[i] = val; U_z[i] = val;
        W_r[i] = val; U_r[i] = val;
        W_n[i] = val; U_n[i] = val;
    }
    for (int i = 0; i < 20; i++) {
        b_z[i] = 0.0f; b_r[i] = 0.0f; b_in_n[i] = 0.0f; b_hn_n[i] = 0.0f;
    }

    FeGruWeights weights = {
        .W_z = W_z, .U_z = U_z, .b_z = b_z,
        .W_r = W_r, .U_r = U_r, .b_r = b_r,
        .W_n = W_n, .U_n = U_n, .b_in_n = b_in_n, .b_hn_n = b_hn_n,
        .input_size = input_size,
        .hidden_size = hidden_size
    };

    for (int step = 0; step < 1000; step++) {
        fe_gru_step(&weights, input, hidden);
    }

    /* 1000ステップ後、全隠れ状態が有限かつ発散していないこと */
    for (int i = 0; i < hidden_size; i++) {
        TEST_ASSERT_FALSE(isnan(hidden[i]));
        TEST_ASSERT_FALSE(isinf(hidden[i]));
        TEST_ASSERT_TRUE(fabsf(hidden[i]) < 1.0f);
    }
}

/* --- 10000ステップ安定性テスト (カテゴリ1: 長時間) --- */

void test_gru_stability_10000_steps(void) {
    int hidden_size = 20;
    int input_size = 20;

    float input[20] = {0};
    float hidden[20] = {0};

    float W_z[400] = {0}, U_z[400] = {0}, b_z[20] = {0};
    float W_r[400] = {0}, U_r[400] = {0}, b_r[20] = {0};
    float W_n[400] = {0}, U_n[400] = {0}, b_in_n[20] = {0}, b_hn_n[20] = {0};

    /* 対角要素のみ小さな値 */
    for (int i = 0; i < 20; i++) {
        W_z[i * 20 + i] = 0.01f; U_z[i * 20 + i] = 0.01f;
        W_r[i * 20 + i] = 0.01f; U_r[i * 20 + i] = 0.01f;
        W_n[i * 20 + i] = 0.01f; U_n[i * 20 + i] = 0.01f;
    }

    FeGruWeights weights = {
        .W_z = W_z, .U_z = U_z, .b_z = b_z,
        .W_r = W_r, .U_r = U_r, .b_r = b_r,
        .W_n = W_n, .U_n = U_n, .b_in_n = b_in_n, .b_hn_n = b_hn_n,
        .input_size = input_size,
        .hidden_size = hidden_size
    };

    for (int step = 0; step < 10000; step++) {
        fe_gru_step(&weights, input, hidden);
    }

    for (int i = 0; i < hidden_size; i++) {
        TEST_ASSERT_FALSE(isnan(hidden[i]));
        TEST_ASSERT_FALSE(isinf(hidden[i]));
        TEST_ASSERT_TRUE(fabsf(hidden[i]) < 1.0f);
    }
}

/* --- 隠れ状態リセット --- */

void test_gru_reset_hidden(void) {
    int hidden_size = 20;
    float hidden[20];
    for (int i = 0; i < hidden_size; i++) hidden[i] = 99.0f;

    fe_gru_reset_hidden(hidden, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-10f, 0.0f, hidden[i]);
    }
}

/* --- 非ゼロ入力での動作確認 --- */

void test_gru_nonzero_input_produces_nonzero_output(void) {
    int hidden_size = 4;
    int input_size = 4;

    float input[4] = {1.0f, 0.5f, 0.3f, 0.1f};
    float hidden[4] = {0};

    float W_z[16], U_z[16], b_z[4];
    float W_r[16], U_r[16], b_r[4];
    float W_n[16], U_n[16], b_in_n[4], b_hn_n[4];

    for (int i = 0; i < 16; i++) {
        W_z[i] = 0.1f; U_z[i] = 0.1f;
        W_r[i] = 0.1f; U_r[i] = 0.1f;
        W_n[i] = 0.5f; U_n[i] = 0.5f;
    }
    for (int i = 0; i < 4; i++) {
        b_z[i] = 0.0f; b_r[i] = 0.0f; b_in_n[i] = 0.0f; b_hn_n[i] = 0.0f;
    }

    FeGruWeights weights = {
        .W_z = W_z, .U_z = U_z, .b_z = b_z,
        .W_r = W_r, .U_r = U_r, .b_r = b_r,
        .W_n = W_n, .U_n = U_n, .b_in_n = b_in_n, .b_hn_n = b_hn_n,
        .input_size = input_size,
        .hidden_size = hidden_size
    };

    fe_gru_step(&weights, input, hidden);

    int has_nonzero = 0;
    for (int i = 0; i < hidden_size; i++) {
        if (fabsf(hidden[i]) > 1e-6f) has_nonzero = 1;
        TEST_ASSERT_FALSE(isnan(hidden[i]));
    }
    TEST_ASSERT_TRUE(has_nonzero);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_gru_zero_input_zero_hidden);
    RUN_TEST(test_gru_known_weights_single_unit);
    RUN_TEST(test_gru_stability_1000_steps);
    RUN_TEST(test_gru_stability_10000_steps);
    RUN_TEST(test_gru_reset_hidden);
    RUN_TEST(test_gru_nonzero_input_produces_nonzero_output);

    return UNITY_END();
}
