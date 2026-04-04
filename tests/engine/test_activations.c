/*
 * test_activations.c — Phase 2-A: 活性化関数テスト (TDD Red)
 *
 * 検証対象:
 *   - sigmoid 多項式近似 (4-5次ミニマックス近似、[-8,8]範囲)
 *   - SiLU (= x * sigmoid(x))
 *   - 飽和域の精度 (カテゴリ1: 数値安定性)
 *
 * コンパイル:
 *   gcc -I tests/engine/unity -I src/engine/common \
 *       tests/engine/unity/unity.c tests/engine/test_activations.c \
 *       src/engine/common/activations.c -o test_activations -lm
 */

#include "unity.h"
#include "activations.h"
#include <math.h>
#include <float.h>

void setUp(void) {}
void tearDown(void) {}

/* --- sigmoid テスト --- */

void test_sigmoid_at_zero(void) {
    float result = fe_sigmoid(0.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, result);
}

void test_sigmoid_positive(void) {
    float result = fe_sigmoid(2.0f);
    float expected = 1.0f / (1.0f + expf(-2.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected, result);
}

void test_sigmoid_negative(void) {
    float result = fe_sigmoid(-2.0f);
    float expected = 1.0f / (1.0f + expf(2.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected, result);
}

void test_sigmoid_symmetry(void) {
    for (float x = 0.1f; x <= 8.0f; x += 0.5f) {
        float pos = fe_sigmoid(x);
        float neg = fe_sigmoid(-x);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, pos + neg);
    }
}

void test_sigmoid_range_within_approximation_domain(void) {
    for (float x = -8.0f; x <= 8.0f; x += 0.25f) {
        float result = fe_sigmoid(x);
        float exact = 1.0f / (1.0f + expf(-x));
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, exact, result);
    }
}

void test_sigmoid_saturation_positive(void) {
    float result = fe_sigmoid(20.0f);
    float exact = 1.0f / (1.0f + expf(-20.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, exact, result);
    TEST_ASSERT_TRUE(result > 0.99f);
    TEST_ASSERT_TRUE(result <= 1.0f);
}

void test_sigmoid_saturation_negative(void) {
    float result = fe_sigmoid(-20.0f);
    float exact = 1.0f / (1.0f + expf(20.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, exact, result);
    TEST_ASSERT_TRUE(result < 0.01f);
    TEST_ASSERT_TRUE(result >= 0.0f);
}

void test_sigmoid_extreme_positive(void) {
    float result = fe_sigmoid(88.0f);
    TEST_ASSERT_TRUE(result >= 0.0f && result <= 1.0f);
    TEST_ASSERT_FALSE(isnan(result));
    TEST_ASSERT_FALSE(isinf(result));
}

void test_sigmoid_extreme_negative(void) {
    float result = fe_sigmoid(-88.0f);
    TEST_ASSERT_TRUE(result >= 0.0f && result <= 1.0f);
    TEST_ASSERT_FALSE(isnan(result));
    TEST_ASSERT_FALSE(isinf(result));
}

void test_sigmoid_nan_input(void) {
    float result = fe_sigmoid(NAN);
    TEST_ASSERT_TRUE(result >= 0.0f && result <= 1.0f);
}

void test_sigmoid_inf_input(void) {
    float pos = fe_sigmoid(INFINITY);
    float neg = fe_sigmoid(-INFINITY);
    TEST_ASSERT_TRUE(pos >= 0.0f && pos <= 1.0f);
    TEST_ASSERT_TRUE(neg >= 0.0f && neg <= 1.0f);
}

void test_sigmoid_monotonic(void) {
    float prev = fe_sigmoid(-8.0f);
    for (float x = -7.5f; x <= 8.0f; x += 0.5f) {
        float curr = fe_sigmoid(x);
        TEST_ASSERT_TRUE(curr >= prev);
        prev = curr;
    }
}

/* --- SiLU テスト --- */

void test_silu_at_zero(void) {
    float result = fe_silu(0.0f);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, result);
}

void test_silu_positive(void) {
    float x = 2.0f;
    float expected = x / (1.0f + expf(-x));
    float result = fe_silu(x);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected, result);
}

void test_silu_negative(void) {
    float x = -2.0f;
    float expected = x / (1.0f + expf(-x));
    float result = fe_silu(x);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected, result);
}

void test_silu_definition(void) {
    for (float x = -5.0f; x <= 5.0f; x += 0.5f) {
        float silu_val = fe_silu(x);
        float manual = x * fe_sigmoid(x);
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, manual, silu_val);
    }
}

void test_silu_large_positive(void) {
    float x = 20.0f;
    float result = fe_silu(x);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, x, result);
    TEST_ASSERT_FALSE(isnan(result));
    TEST_ASSERT_FALSE(isinf(result));
}

void test_silu_large_negative(void) {
    float x = -20.0f;
    float result = fe_silu(x);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, result);
    TEST_ASSERT_FALSE(isnan(result));
    TEST_ASSERT_FALSE(isinf(result));
}

void test_silu_minimum_exists(void) {
    float min_val = fe_silu(0.0f);
    for (float x = -5.0f; x <= 5.0f; x += 0.01f) {
        float val = fe_silu(x);
        if (val < min_val) min_val = val;
    }
    TEST_ASSERT_TRUE(min_val < 0.0f);
    TEST_ASSERT_TRUE(min_val > -0.4f);
}

void test_silu_nan_input(void) {
    float result = fe_silu(NAN);
    TEST_ASSERT_FALSE(isinf(result));
    TEST_ASSERT_FALSE(isnan(result));
}

void test_silu_inf_input(void) {
    float result = fe_silu(-INFINITY);
    TEST_ASSERT_FALSE(isnan(result));
    TEST_ASSERT_FALSE(isinf(result));
}

/* --- バッチ処理テスト (SIMD向け) --- */

void test_sigmoid_batch(void) {
    float input[8] = {-4.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f, 8.0f};
    float output[8];
    fe_sigmoid_batch(input, output, 8);
    for (int i = 0; i < 8; i++) {
        float expected = 1.0f / (1.0f + expf(-input[i]));
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected, output[i]);
    }
}

void test_silu_batch(void) {
    float input[8] = {-4.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f, 8.0f};
    float output[8];
    fe_silu_batch(input, output, 8);
    for (int i = 0; i < 8; i++) {
        float expected = input[i] / (1.0f + expf(-input[i]));
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected, output[i]);
    }
}

void test_silu_batch_nan_and_negative_inf_safe(void) {
    float input[4] = {NAN, -INFINITY, INFINITY, 0.0f};
    float output[4];

    fe_silu_batch(input, output, 4);

    TEST_ASSERT_FALSE(isnan(output[0]));
    TEST_ASSERT_FALSE(isinf(output[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output[0]);

    TEST_ASSERT_FALSE(isnan(output[1]));
    TEST_ASSERT_FALSE(isinf(output[1]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, output[1]);

    TEST_ASSERT_FALSE(isnan(output[2]));
    TEST_ASSERT_TRUE(output[2] > 0.0f);
}

void test_sigmoid_batch_non_multiple_of_4(void) {
    float input[5] = {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float output[5];
    fe_sigmoid_batch(input, output, 5);
    for (int i = 0; i < 5; i++) {
        float expected = 1.0f / (1.0f + expf(-input[i]));
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected, output[i]);
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_sigmoid_at_zero);
    RUN_TEST(test_sigmoid_positive);
    RUN_TEST(test_sigmoid_negative);
    RUN_TEST(test_sigmoid_symmetry);
    RUN_TEST(test_sigmoid_range_within_approximation_domain);
    RUN_TEST(test_sigmoid_saturation_positive);
    RUN_TEST(test_sigmoid_saturation_negative);
    RUN_TEST(test_sigmoid_extreme_positive);
    RUN_TEST(test_sigmoid_extreme_negative);
    RUN_TEST(test_sigmoid_nan_input);
    RUN_TEST(test_sigmoid_inf_input);
    RUN_TEST(test_sigmoid_monotonic);
    RUN_TEST(test_silu_at_zero);
    RUN_TEST(test_silu_positive);
    RUN_TEST(test_silu_negative);
    RUN_TEST(test_silu_definition);
    RUN_TEST(test_silu_large_positive);
    RUN_TEST(test_silu_large_negative);
    RUN_TEST(test_silu_minimum_exists);
    RUN_TEST(test_silu_nan_input);
    RUN_TEST(test_silu_inf_input);
    RUN_TEST(test_sigmoid_batch);
    RUN_TEST(test_silu_batch);
    RUN_TEST(test_silu_batch_nan_and_negative_inf_safe);
    RUN_TEST(test_sigmoid_batch_non_multiple_of_4);

    return UNITY_END();
}
