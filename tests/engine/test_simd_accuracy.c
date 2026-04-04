/*
 * test_simd_accuracy.c — SIMD最適化の数値精度検証
 *
 * SIMD関数(fast_exp, fast_sigmoid, fast_tanh, matvec_add, softmax)が
 * 参照スカラー実装と十分に近い結果を返すことを検証する。
 * 
 * さらに、フルパイプラインで1000フレーム処理し、
 * 出力の有界性・安定性・非退化を確認する。
 */

#include "unity.h"
#include "simd.h"
#include "activations.h"
#include "attention.h"
#include "fastenhancer.h"
#include "tiny_48k.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define PI_F 3.14159265f

void setUp(void) {}
void tearDown(void) {}

/* ======== 1. fast_expf 精度検証 ======== */

void test_fast_expf_vs_libm_over_range(void) {
    /* [-20, 20] の範囲で 10001点を比較 */
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    float worst_x = 0.0f;
    int n_points = 10001;

    for (int i = 0; i < n_points; i++) {
        float x = -20.0f + 40.0f * (float)i / (float)(n_points - 1);
        float ref = expf(x);
        float approx = fe_fast_expf(x);
        float abs_err = fabsf(approx - ref);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_x = x;
        }
        if (ref > 1e-10f) {
            float rel_err = abs_err / ref;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
    }

    /* 相対誤差 < 0.1% を要求 */
    char msg[256];
    snprintf(msg, sizeof(msg),
        "fe_fast_expf max_rel_err=%.6e at x=%.4f (abs=%.6e)",
        max_rel_err, worst_x, max_abs_err);
    TEST_ASSERT_TRUE_MESSAGE(max_rel_err < 1e-3f, msg);
}

/* ======== 2. f32x4_fast_exp 精度検証 (SSE2/WASM/scalar) ======== */

void test_f32x4_fast_exp_vs_libm(void) {
    float max_rel_err = 0.0f;
    int n_tests = 2500;

    for (int i = 0; i < n_tests; i++) {
        float base = -20.0f + 40.0f * (float)i / (float)(n_tests - 1);
        float in_arr[4] = {base, base + 0.1f, base + 0.2f, base + 0.3f};
        float out_arr[4];

        f32x4 v = f32x4_load(in_arr);
        f32x4 r = f32x4_fast_exp(v);
        f32x4_store(out_arr, r);

        for (int j = 0; j < 4; j++) {
            float ref = expf(in_arr[j]);
            if (ref > 1e-10f) {
                float rel = fabsf(out_arr[j] - ref) / ref;
                if (rel > max_rel_err) max_rel_err = rel;
            }
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "f32x4_fast_exp max_rel_err=%.6e", max_rel_err);
    TEST_ASSERT_TRUE_MESSAGE(max_rel_err < 1e-3f, msg);
}

/* ======== 3. f32x4_fast_sigmoid 精度検証 ======== */

void test_f32x4_fast_sigmoid_vs_scalar(void) {
    float max_abs_err = 0.0f;
    int n_tests = 2500;

    for (int i = 0; i < n_tests; i++) {
        float base = -20.0f + 40.0f * (float)i / (float)(n_tests - 1);
        float in_arr[4] = {base, base + 0.1f, base + 0.2f, base + 0.3f};
        float out_arr[4];

        f32x4 v = f32x4_load(in_arr);
        f32x4 r = f32x4_fast_sigmoid(v);
        f32x4_store(out_arr, r);

        for (int j = 0; j < 4; j++) {
            /* 参照: libm sigmoid */
            float ref = 1.0f / (1.0f + expf(-in_arr[j]));
            float err = fabsf(out_arr[j] - ref);
            if (err > max_abs_err) max_abs_err = err;
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "f32x4_fast_sigmoid max_abs_err=%.6e", max_abs_err);
    /* sigmoid出力は[0,1]、絶対誤差 < 0.001 を要求 */
    TEST_ASSERT_TRUE_MESSAGE(max_abs_err < 1e-3f, msg);
}

/* ======== 4. fe_sigmoid_batch SIMD vs スカラー fe_sigmoid ======== */

void test_sigmoid_batch_matches_scalar(void) {
    float max_err = 0.0f;
    int n = 256;
    float* input = (float*)malloc(n * sizeof(float));
    float* batch_out = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        input[i] = -10.0f + 20.0f * (float)i / (float)(n - 1);
    }

    fe_sigmoid_batch(input, batch_out, n);

    for (int i = 0; i < n; i++) {
        float scalar = fe_sigmoid(input[i]);
        float err = fabsf(batch_out[i] - scalar);
        if (err > max_err) max_err = err;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "sigmoid_batch vs scalar max_err=%.6e", max_err);
    /* fast_sigmoid近似とfe_sigmoid(libm)の差は許容するが限定的 */
    TEST_ASSERT_TRUE_MESSAGE(max_err < 5e-3f, msg);

    free(input);
    free(batch_out);
}

/* ======== 5. fe_matvec_add 精度検証 (SIMD vs 参照スカラー) ======== */

void test_matvec_add_vs_scalar_reference(void) {
    int rows = 60, cols = 20;
    float* W = (float*)malloc(rows * cols * sizeof(float));
    float* x = (float*)malloc(cols * sizeof(float));
    float* out_simd = (float*)calloc(rows, sizeof(float));
    float* out_ref = (float*)calloc(rows, sizeof(float));

    /* 決定論的データ */
    unsigned int seed = 12345;
    for (int i = 0; i < rows * cols; i++) {
        seed = seed * 1103515245u + 12345u;
        W[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.1f;
    }
    for (int i = 0; i < cols; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.5f;
    }

    /* SIMD版 */
    fe_matvec_add(W, x, out_simd, rows, cols);

    /* 参照スカラー実装 */
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += W[r * cols + c] * x[c];
        }
        out_ref[r] = sum;
    }

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        float err = fabsf(out_simd[r] - out_ref[r]);
        if (err > max_err) max_err = err;
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "matvec_add max_err=%.6e", max_err);
    TEST_ASSERT_TRUE_MESSAGE(max_err < 1e-5f, msg);

    free(W); free(x); free(out_simd); free(out_ref);
}

/* ======== 6. softmax SIMD精度検証 ======== */

void test_softmax_simd_vs_reference(void) {
    /* 典型的なattentionスコア範囲でテスト */
    int sizes[] = {24, 36, 48};
    float max_err_all = 0.0f;

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        float* input = (float*)malloc(n * sizeof(float));
        float* out_simd = (float*)malloc(n * sizeof(float));
        float* out_ref = (float*)malloc(n * sizeof(float));

        /* 決定論的データ: attention-like scores */
        unsigned int seed = 42 + s;
        for (int i = 0; i < n; i++) {
            seed = seed * 1103515245u + 12345u;
            input[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 4.0f;
        }

        /* SIMD softmax (テスト対象) */
        fe_softmax(input, out_simd, n);

        /* 参照: libm expf使用のスカラーsoftmax */
        float max_val = -FLT_MAX;
        for (int i = 0; i < n; i++) {
            if (input[i] > max_val) max_val = input[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            out_ref[i] = expf(input[i] - max_val);
            sum += out_ref[i];
        }
        for (int i = 0; i < n; i++) {
            out_ref[i] /= sum;
        }

        for (int i = 0; i < n; i++) {
            float err = fabsf(out_simd[i] - out_ref[i]);
            if (err > max_err_all) max_err_all = err;
        }

        /* softmax出力の和が1.0に近いことも確認 */
        float sum_check = 0.0f;
        for (int i = 0; i < n; i++) sum_check += out_simd[i];
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, sum_check);

        free(input); free(out_simd); free(out_ref);
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "softmax SIMD vs ref max_err=%.6e", max_err_all);
    TEST_ASSERT_TRUE_MESSAGE(max_err_all < 1e-3f, msg);
}

/* ======== 7. フルパイプライン1000フレーム安定性 ======== */

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
}

static void fix_bn_scales(float* w, int n) {
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
    (void)n;
}

void test_1000_frames_no_divergence(void) {
    int wc = fe_weight_count(FE_MODEL_TINY);
    float* w = (float*)malloc(wc * sizeof(float));
    unsigned int seed = 42;
    for (int i = 0; i < wc; i++) w[i] = lcg_float(&seed);
    fix_bn_scales(w, wc);

    FeState* state = fe_create(FE_MODEL_TINY, w, wc);
    TEST_ASSERT_NOT_NULL(state);
    free(w);

    float input[512], output[512];
    float max_output = 0.0f;
    int nan_count = 0, inf_count = 0;

    for (int frame = 0; frame < 1000; frame++) {
        /* 多様な入力パターン */
        if (frame % 4 == 0) {
            /* 正弦波 */
            for (int i = 0; i < 512; i++)
                input[i] = 0.3f * sinf(2.0f * PI_F * 440.0f * (float)(frame * 512 + i) / 48000.0f);
        } else if (frame % 4 == 1) {
            /* ホワイトノイズ */
            for (int i = 0; i < 512; i++) {
                seed = seed * 1103515245u + 12345u;
                input[i] = ((float)((seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.2f;
            }
        } else if (frame % 4 == 2) {
            /* 無音 */
            memset(input, 0, sizeof(input));
        } else {
            /* インパルス */
            memset(input, 0, sizeof(input));
            input[0] = 0.5f;
        }

        fe_process_frame(state, input, output);

        for (int i = 0; i < 512; i++) {
            if (output[i] != output[i]) nan_count++;
            else if (!isfinite(output[i])) inf_count++;
            else {
                float a = fabsf(output[i]);
                if (a > max_output) max_output = a;
            }
        }
    }

    fe_destroy(state);

    char msg[256];
    snprintf(msg, sizeof(msg),
        "1000 frames: nan=%d, inf=%d, max_abs=%.6f",
        nan_count, inf_count, max_output);

    TEST_ASSERT_EQUAL_INT_MESSAGE(0, nan_count, msg);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, inf_count, msg);
    /* 出力が有界であること (音声信号は通常[-1,1]だがNNの出力はもう少し大きくなりうる) */
    TEST_ASSERT_TRUE_MESSAGE(max_output < 100.0f,
        "出力が100を超えた: 数値発散の疑い");
    /* 出力が全てゼロでないこと (退化チェック) */
    TEST_ASSERT_TRUE_MESSAGE(max_output > 1e-8f,
        "1000フレームの全出力がゼロ: パイプライン死亡");
}

/* ======== 8. 決定性再現テスト: 2回実行で完全一致 ======== */

void test_full_pipeline_deterministic(void) {
    int wc = fe_weight_count(FE_MODEL_TINY);
    float* w = (float*)malloc(wc * sizeof(float));
    unsigned int seed = 777;
    for (int i = 0; i < wc; i++) w[i] = lcg_float(&seed);
    fix_bn_scales(w, wc);

    float input[512];
    for (int i = 0; i < 512; i++)
        input[i] = 0.4f * sinf(2.0f * PI_F * 1000.0f * (float)i / 48000.0f);

    /* 1回目: 10フレーム処理 */
    FeState* s1 = fe_create(FE_MODEL_TINY, w, wc);
    float out1[10][512];
    for (int f = 0; f < 10; f++) {
        fe_process_frame(s1, input, out1[f]);
    }
    fe_destroy(s1);

    /* 2回目: 同一入力で10フレーム処理 */
    FeState* s2 = fe_create(FE_MODEL_TINY, w, wc);
    float out2[10][512];
    for (int f = 0; f < 10; f++) {
        fe_process_frame(s2, input, out2[f]);
    }
    fe_destroy(s2);

    free(w);

    /* ビット完全一致を要求 */
    for (int f = 0; f < 10; f++) {
        for (int i = 0; i < 512; i++) {
            char msg[128];
            snprintf(msg, sizeof(msg), "frame %d, sample %d", f, i);
            TEST_ASSERT_EQUAL_FLOAT_MESSAGE(out1[f][i], out2[f][i], msg);
        }
    }
}

/* ======== 9. 長期安定性: 出力振幅の有界性 (1000フレーム後) ======== */

void test_output_bounded_after_1000_frames(void) {
    int wc = fe_weight_count(FE_MODEL_TINY);
    float* w = (float*)malloc(wc * sizeof(float));
    unsigned int seed = 99;
    for (int i = 0; i < wc; i++) w[i] = lcg_float(&seed);
    fix_bn_scales(w, wc);

    FeState* state = fe_create(FE_MODEL_TINY, w, wc);
    free(w);

    float input[512], output[512];
    float max_out_early = 0.0f;
    float max_out_late = 0.0f;

    for (int frame = 0; frame < 1000; frame++) {
        for (int i = 0; i < 512; i++)
            input[i] = 0.3f * sinf(2.0f * PI_F * 800.0f * (float)(frame * 512 + i) / 48000.0f);
        fe_process_frame(state, input, output);

        for (int i = 0; i < 512; i++) {
            float a = fabsf(output[i]);
            if (frame < 10) {
                if (a > max_out_early) max_out_early = a;
            }
            if (frame >= 990) {
                if (a > max_out_late) max_out_late = a;
            }
        }
    }

    fe_destroy(state);

    char msg[256];
    snprintf(msg, sizeof(msg),
        "early max=%.6f, late max=%.6f, ratio=%.2f",
        max_out_early, max_out_late,
        max_out_early > 0 ? max_out_late / max_out_early : 0.0f);

    /* 後半の出力が初期の100倍以上に膨らんでいたら発散 */
    if (max_out_early > 1e-8f) {
        TEST_ASSERT_TRUE_MESSAGE(max_out_late < max_out_early * 100.0f, msg);
    }
    /* 絶対値で有界 */
    TEST_ASSERT_TRUE_MESSAGE(max_out_late < 100.0f, msg);
}

int main(void) {
    UNITY_BEGIN();

    /* SIMD関数個別精度 */
    RUN_TEST(test_fast_expf_vs_libm_over_range);
    RUN_TEST(test_f32x4_fast_exp_vs_libm);
    RUN_TEST(test_f32x4_fast_sigmoid_vs_scalar);
    RUN_TEST(test_sigmoid_batch_matches_scalar);
    RUN_TEST(test_matvec_add_vs_scalar_reference);
    RUN_TEST(test_softmax_simd_vs_reference);

    /* フルパイプライン検証 */
    RUN_TEST(test_1000_frames_no_divergence);
    RUN_TEST(test_full_pipeline_deterministic);
    RUN_TEST(test_output_bounded_after_1000_frames);

    return UNITY_END();
}
