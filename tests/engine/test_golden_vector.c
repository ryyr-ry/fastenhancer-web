/*
 * test_golden_vector.c — PyTorch推論出力とCエンジン出力のgolden vector比較
 *
 * scripts/golden_vectors_streaming.py で生成した
 * streaming STFT + PyTorch model_forward() 出力と
 * Cエンジンの fe_process() 出力をフレーム単位で比較する。
 *
 * 双方とも streaming STFT（zero-init overlap）を使用するため
 * STFT framing 差異はなく、NN pipeline の数値一致のみをテストする。
 *
 * golden_input.bin / golden_output.bin: raw float32 LE, N_FRAMES * HOP_SIZE floats
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "unity/unity.h"
#include "../../src/engine/exports.h"

#define GOLDEN_N_FRAMES   40
#define GOLDEN_HOP_SIZE   512
#define GOLDEN_N_SAMPLES  (GOLDEN_N_FRAMES * GOLDEN_HOP_SIZE)

#if defined(FE_USE_BASE_48K)
#define GOLDEN_MODEL_ID       FE_MODEL_BASE
#define WEIGHTS_PATH         "weights/fe_base_48k.bin"
#define GOLDEN_INPUT_PATH    "tests/golden_base/golden_input.bin"
#define GOLDEN_OUTPUT_PATH   "tests/golden_base/golden_output.bin"
#elif defined(FE_USE_SMALL_48K)
#define GOLDEN_MODEL_ID       FE_MODEL_SMALL
#define WEIGHTS_PATH         "weights/fe_small_48k.bin"
#define GOLDEN_INPUT_PATH    "tests/golden_small/golden_input.bin"
#define GOLDEN_OUTPUT_PATH   "tests/golden_small/golden_output.bin"
#else
#define GOLDEN_MODEL_ID       FE_MODEL_TINY
#define WEIGHTS_PATH         "weights/fe_tiny_48k.bin"
#define GOLDEN_INPUT_PATH    "tests/golden/golden_input.bin"
#define GOLDEN_OUTPUT_PATH   "tests/golden/golden_output.bin"
#endif

/* 許容閾値: streaming STFT golden (float32 NN近似実装の精度限界)
 * MSE: CとPyTorchの絶対誤差。1e-10以下で十分な精度。
 * SNR: Cの多項式sigmoid近似 vs PyTorchの厳密sigmoid、
 *      および Cの radix-2 FFT vs ライブラリFFT の差が
 *      40フレームのGRU隠れ状態を通じて蓄積し、~60dB程度になる。
 *      55dBに設定（10dBの安全マージン）。 */
#define MSE_THRESHOLD_OVERALL  1.0e-10f
#define MSE_THRESHOLD_STEADY   1.0e-10f  /* フレーム5以降 */
#define SNR_THRESHOLD_DB       55.0f

void setUp(void) {}
void tearDown(void) {}

static uint8_t* read_file_u8(const char* path, int* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { *out_len = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (buf) {
        size_t rd = fread(buf, 1, (size_t)sz, f);
        *out_len = (int)rd;
    }
    fclose(f);
    return buf;
}

static float* read_file_f32(const char* path, int* out_count) {
    int byte_len = 0;
    uint8_t* raw = read_file_u8(path, &byte_len);
    if (!raw) { *out_count = 0; return NULL; }
    *out_count = byte_len / (int)sizeof(float);
    return (float*)raw;
}

/* フレーム単位MSE計算 */
static float compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)(sum / n);
}

/* 信号パワー計算 */
static float compute_power(const float* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i] * (double)x[i];
    }
    return (float)(sum / n);
}

void test_golden_vector_files_exist(void) {
    int len = 0;
    float* inp = read_file_f32(GOLDEN_INPUT_PATH, &len);
    TEST_ASSERT_NOT_NULL_MESSAGE(inp,
        "golden_input.bin が見つかりません。先に scripts/golden_vectors_streaming.py を実行してください。");
    TEST_ASSERT_EQUAL_INT_MESSAGE(GOLDEN_N_SAMPLES, len,
        "golden_input.bin のサンプル数が期待値と一致しません");
    free(inp);

    float* out = read_file_f32(GOLDEN_OUTPUT_PATH, &len);
    TEST_ASSERT_NOT_NULL_MESSAGE(out,
        "golden_output.bin が見つかりません。先に scripts/golden_vectors_streaming.py を実行してください。");
    TEST_ASSERT_EQUAL_INT_MESSAGE(GOLDEN_N_SAMPLES, len,
        "golden_output.bin のサンプル数が期待値と一致しません");
    free(out);
}

void test_golden_vector_comparison(void) {
    /* 重み読み込み */
    int wt_len = 0;
    uint8_t* wt_data = read_file_u8(WEIGHTS_PATH, &wt_len);
    TEST_ASSERT_NOT_NULL_MESSAGE(wt_data, "重みファイルが見つかりません");

    FeState* state = fe_init(GOLDEN_MODEL_ID, wt_data, wt_len);
    TEST_ASSERT_NOT_NULL_MESSAGE(state, "fe_init 失敗");

    /* golden vector 読み込み */
    int inp_count = 0, out_count = 0;
    float* golden_in  = read_file_f32(GOLDEN_INPUT_PATH, &inp_count);
    float* golden_out = read_file_f32(GOLDEN_OUTPUT_PATH, &out_count);
    TEST_ASSERT_NOT_NULL_MESSAGE(golden_in, "golden_input.bin 読み込み失敗");
    TEST_ASSERT_NOT_NULL_MESSAGE(golden_out, "golden_output.bin 読み込み失敗");
    TEST_ASSERT_EQUAL_INT(GOLDEN_N_SAMPLES, inp_count);
    TEST_ASSERT_EQUAL_INT(GOLDEN_N_SAMPLES, out_count);

    /* C エンジンでフレーム単位処理 */
    float* c_output = (float*)calloc(GOLDEN_N_SAMPLES, sizeof(float));
    TEST_ASSERT_NOT_NULL(c_output);

    float* in_ptr  = fe_get_input_ptr(state);
    float* out_ptr = fe_get_output_ptr(state);
    int hop = fe_get_hop_size(state);
    TEST_ASSERT_EQUAL_INT_MESSAGE(GOLDEN_HOP_SIZE, hop, "hop_size 不一致");

    for (int f = 0; f < GOLDEN_N_FRAMES; f++) {
        memcpy(in_ptr, golden_in + f * hop, sizeof(float) * hop);
        int ret = fe_process(state, in_ptr, out_ptr);
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, ret, "fe_process 失敗");
        memcpy(c_output + f * hop, out_ptr, sizeof(float) * hop);
    }

    /* === 全フレーム比較 === */
    float overall_mse = compute_mse(c_output, golden_out, GOLDEN_N_SAMPLES);
    float ref_power   = compute_power(golden_out, GOLDEN_N_SAMPLES);
    float snr_db      = (ref_power > 1e-20f)
                      ? 10.0f * log10f(ref_power / (overall_mse + 1e-30f))
                      : -999.0f;

    printf("\n=== Golden Vector 比較結果 ===\n");
    printf("Overall MSE:       %.6e\n", overall_mse);
    printf("Reference power:   %.6e\n", ref_power);
    printf("SNR:               %.2f dB\n", snr_db);

    /* === フレーム単位比較 === */
    float max_frame_mse = 0.0f;
    float max_steady_mse = 0.0f;
    int worst_frame = 0;
    int worst_steady_frame = 0;

    printf("\nフレーム別MSE (先頭10フレーム):\n");
    for (int f = 0; f < GOLDEN_N_FRAMES; f++) {
        float fmse = compute_mse(c_output + f * hop, golden_out + f * hop, hop);
        if (f < 10) {
            printf("  Frame %2d: MSE=%.6e\n", f, fmse);
        }
        if (fmse > max_frame_mse) {
            max_frame_mse = fmse;
            worst_frame = f;
        }
        if (f >= 5 && fmse > max_steady_mse) {
            max_steady_mse = fmse;
            worst_steady_frame = f;
        }
    }
    printf("...\n");
    printf("最悪フレーム:       Frame %d (MSE=%.6e)\n", worst_frame, max_frame_mse);
    printf("定常域最悪フレーム: Frame %d (MSE=%.6e)\n", worst_steady_frame, max_steady_mse);

    /* === 出力が全てNaN/Infでないことを確認 === */
    for (int i = 0; i < GOLDEN_N_SAMPLES; i++) {
        if (c_output[i] != c_output[i] || c_output[i] > 1e30f || c_output[i] < -1e30f) {
            char msg[128];
            snprintf(msg, sizeof(msg), "C output[%d] = %f (NaN/Inf検出)", i, c_output[i]);
            TEST_FAIL_MESSAGE(msg);
        }
    }

    /* === 閾値チェック === */
    char msg[256];
    snprintf(msg, sizeof(msg),
        "Overall MSE %.6e > threshold %.6e", overall_mse, MSE_THRESHOLD_OVERALL);
    TEST_ASSERT_TRUE_MESSAGE(overall_mse < MSE_THRESHOLD_OVERALL, msg);

    snprintf(msg, sizeof(msg),
        "Steady-state MSE %.6e > threshold %.6e (worst frame %d)",
        max_steady_mse, MSE_THRESHOLD_STEADY, worst_steady_frame);
    TEST_ASSERT_TRUE_MESSAGE(max_steady_mse < MSE_THRESHOLD_STEADY, msg);

    snprintf(msg, sizeof(msg),
        "SNR %.2f dB < threshold %.2f dB", snr_db, SNR_THRESHOLD_DB);
    TEST_ASSERT_TRUE_MESSAGE(snr_db > SNR_THRESHOLD_DB, msg);

    /* クリーンアップ */
    free(c_output);
    free(golden_in);
    free(golden_out);
    fe_destroy(state);
    free(wt_data);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_golden_vector_files_exist);
    RUN_TEST(test_golden_vector_comparison);
    return UNITY_END();
}
