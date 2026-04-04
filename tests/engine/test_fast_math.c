/*
 * test_fast_math.c — Phase 3B-F: -ffast-math 差分検証ハーネス
 *
 * このプログラムは deterministic な重みと入力で推論を実行し、
 * 出力系列を標準出力へ書き出す。通常ビルド版と -ffast-math 版を
 * 別々にコンパイルして出力差分を比較する。
 */

#include "fastenhancer.h"
#include "tiny_48k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI_F 3.14159265358979323846f
#define FE_FAST_MATH_FRAMES 16

/* Unity互換のためのスタブ（このファイルは自前main()の独立ハーネス） */
void setUp(void) {}
void tearDown(void) {}

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fffu) / 32768.0f - 0.5f) * 0.02f;
}

static void fix_bn_scales(float* w) {
    int p = 0;

    p += FE_C1 * 2 * FE_ENC_K0;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
    p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
    p += FE_C1;

    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
        p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
        p += FE_C1;
    }

    p += FE_F2 * FE_F1;
    p += FE_C2 * FE_C1;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 1.0f;
    p += FE_C2;
    for (int i = 0; i < FE_C2; i++) w[p + i] = 0.0f;
    p += FE_C2;

    for (int b = 0; b < FE_RF_BLOCKS; b++) {
        p += FE_W_GRU + FE_W_GRU_FC;
        if (b == 0) p += FE_W_PE;
        p += FE_W_MHSA;
    }

    p += FE_C1 * FE_C2;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
    p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
    p += FE_C1;
    p += FE_F1 * FE_F2;

    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        p += FE_C1 * 2 * FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
        p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
        p += FE_C1;

        p += FE_C1 * FE_C1 * FE_ENC_K;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
        p += FE_C1;
        for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
        p += FE_C1;
    }

    p += FE_C1 * 2 * FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 1.0f;
    p += FE_C1;
    for (int i = 0; i < FE_C1; i++) w[p + i] = 0.0f;
    p += FE_C1;
}

static void fill_input_frame(float* input, int frame_index) {
    unsigned int noise_seed = (unsigned int)(1234 + frame_index * 17);

    for (int i = 0; i < FE_HOP_SIZE; i++) {
        const int sample_index = frame_index * FE_HOP_SIZE + i;
        const float tone_a = 0.20f * sinf(2.0f * PI_F * 440.0f * (float)sample_index / 48000.0f);
        const float tone_b = 0.08f * cosf(2.0f * PI_F * 960.0f * (float)sample_index / 48000.0f);
        const float tone_c = 0.03f * sinf(2.0f * PI_F * 3120.0f * (float)sample_index / 48000.0f);
        const float dc = (frame_index % 2 == 0) ? 0.01f : -0.01f;
        const float noise = lcg_float(&noise_seed) * 0.5f;
        input[i] = tone_a + tone_b + tone_c + dc + noise;
    }
}

int main(int argc, char* argv[]) {
    const int total_samples = FE_FAST_MATH_FRAMES * FE_HOP_SIZE;
    const int weight_count = fe_weight_count(FE_MODEL_TINY);
    float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
    FeState* state;
    float input[FE_HOP_SIZE];
    float output[FE_HOP_SIZE];
    unsigned int seed = 42u;
    int compare_mode = 0;
    FILE* ref_fp = NULL;

    if (argc >= 3 && strcmp(argv[1], "-ref") == 0) {
        compare_mode = 1;
        ref_fp = fopen(argv[2], "r");
        if (!ref_fp) {
            fprintf(stderr, "cannot open reference file: %s\n", argv[2]);
            return 1;
        }
    }

    if (!weights) {
        fprintf(stderr, "weights allocation failed\n");
        return 1;
    }

    for (int i = 0; i < weight_count; i++) {
        weights[i] = lcg_float(&seed);
    }
    fix_bn_scales(weights);

    state = fe_create(FE_MODEL_TINY, weights, weight_count);
    free(weights);
    if (!state) {
        fprintf(stderr, "fe_create failed\n");
        return 1;
    }

    fe_set_hpf(state, 1);
    fe_set_agc(state, 1);

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    int sample_count = 0;

    for (int frame = 0; frame < FE_FAST_MATH_FRAMES; frame++) {
        fill_input_frame(input, frame);
        fe_process_frame(state, input, output);

        for (int i = 0; i < FE_HOP_SIZE; i++) {
            if (!isfinite(output[i])) {
                fprintf(stderr, "non-finite output at frame %d sample %d\n", frame, i);
                fe_destroy(state);
                if (ref_fp) fclose(ref_fp);
                return 1;
            }

            if (compare_mode) {
                double ref_val;
                if (fscanf(ref_fp, "%lf", &ref_val) != 1) {
                    fprintf(stderr, "reference file too short at frame %d sample %d\n", frame, i);
                    fe_destroy(state);
                    fclose(ref_fp);
                    return 1;
                }
                double diff = fabs((double)output[i] - ref_val);
                if (diff > max_abs_diff) max_abs_diff = diff;
                sum_abs_diff += diff;
                sum_sq_diff += diff * diff;
                sample_count++;
            } else {
                printf("%.9e\n", output[i]);
            }
        }
    }

    fe_destroy(state);

    if (compare_mode) {
        fclose(ref_fp);
        double mean_diff = sum_abs_diff / sample_count;
        double rmse = sqrt(sum_sq_diff / sample_count);

        printf("samples=%d max_abs_diff=%.12e mean_abs_diff=%.12e rmse=%.12e\n",
               sample_count, max_abs_diff, mean_diff, rmse);

        if (max_abs_diff > 1e-3) {
            fprintf(stderr, "FAIL: max_abs_diff %.12e exceeds threshold 1e-3\n", max_abs_diff);
            return 1;
        }
        if (rmse > 1e-5) {
            fprintf(stderr, "FAIL: rmse %.12e exceeds threshold 1e-5\n", rmse);
            return 1;
        }
        printf("PASS: -ffast-math differences within acceptable thresholds\n");
    } else {
        printf("PASS: all %d output samples are finite\n", total_samples);
    }

    return 0;
}
