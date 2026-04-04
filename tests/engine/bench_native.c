/*
 * bench_native.c — C native benchmark
 */

#include "benchmark_stats.h"
#include "fastenhancer.h"
#if defined(FE_USE_SMALL_48K)
#include "small_48k.h"
#define BENCH_MODEL_ID   FE_MODEL_SMALL
#define BENCH_MODEL_NAME "small"
#elif defined(FE_USE_BASE_48K)
#include "base_48k.h"
#define BENCH_MODEL_ID   FE_MODEL_BASE
#define BENCH_MODEL_NAME "base"
#else
#include "tiny_48k.h"
#define BENCH_MODEL_ID   FE_MODEL_TINY
#define BENCH_MODEL_NAME "tiny"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#define FE_BENCH_WARMUP_FRAMES   100
#define FE_BENCH_MEASURE_FRAMES  1000
#define FE_BENCH_SR              48000.0
#define FE_BENCH_PI              3.14159265358979323846

/* Stub for Unity compatibility (this file is a standalone benchmark with its own main()) */
void setUp(void) {}
void tearDown(void) {}

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
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
    for (int i = 0; i < FE_HOP_SIZE; i++) {
        const int sample_index = frame_index * FE_HOP_SIZE + i;
        input[i] = 0.2f * (float)sin(2.0 * FE_BENCH_PI * 440.0 * (double)sample_index / FE_BENCH_SR);
    }
}

static double now_ms(void) {
#if defined(__EMSCRIPTEN__)
    return emscripten_get_now();
#elif defined(_WIN32)
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);
    return 1000.0 * (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

int main(void) {
    const int weight_count = fe_weight_count(BENCH_MODEL_ID);
    const double budget_ms = 1000.0 * (double)FE_HOP_SIZE / FE_BENCH_SR;
    float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
    FeState* state = NULL;
    float input[FE_HOP_SIZE];
    float output[FE_HOP_SIZE];
    double samples[FE_BENCH_MEASURE_FRAMES];
    FeBenchStats stats;
    unsigned int seed = 42u;

    if (!weights) {
        fprintf(stderr, "weights allocation failed\n");
        return 1;
    }

    for (int i = 0; i < weight_count; i++) {
        weights[i] = lcg_float(&seed);
    }
    fix_bn_scales(weights);

    state = fe_create(BENCH_MODEL_ID, weights, weight_count);
    free(weights);
    if (!state) {
        fprintf(stderr, "fe_create failed\n");
        return 1;
    }

    for (int frame = 0; frame < FE_BENCH_WARMUP_FRAMES; frame++) {
        fill_input_frame(input, frame);
        fe_process_frame(state, input, output);
    }

    for (int frame = 0; frame < FE_BENCH_MEASURE_FRAMES; frame++) {
        const int absolute_frame = FE_BENCH_WARMUP_FRAMES + frame;
        double t0;

        fill_input_frame(input, absolute_frame);
        t0 = now_ms();
        fe_process_frame(state, input, output);
        samples[frame] = now_ms() - t0;

        for (int i = 0; i < FE_HOP_SIZE; i++) {
            if (!isfinite(output[i])) {
                fprintf(stderr, "non-finite output at frame %d sample %d\n", absolute_frame, i);
                fe_destroy(state);
                return 1;
            }
        }
    }

    if (fe_bench_compute_stats(samples, FE_BENCH_MEASURE_FRAMES, &stats) != 0) {
        fprintf(stderr, "failed to compute benchmark stats\n");
        fe_destroy(state);
        return 1;
    }

    printf("model=%s\n", BENCH_MODEL_NAME);
    printf("warmup_frames=%d\n", FE_BENCH_WARMUP_FRAMES);
    printf("measured_frames=%d\n", FE_BENCH_MEASURE_FRAMES);
    printf("median_ms=%.6f\n", stats.median_ms);
    printf("p99_ms=%.6f\n", stats.p99_ms);
    printf("avg_ms=%.6f\n", stats.avg_ms);
    printf("budget_ms=%.6f\n", budget_ms);
    printf("median_utilization=%.6f\n", stats.median_ms / budget_ms);
    printf("p99_utilization=%.6f\n", stats.p99_ms / budget_ms);
    printf("within_budget=%s\n", stats.p99_ms <= budget_ms ? "true" : "false");

    fe_destroy(state);

    if (stats.p99_ms <= budget_ms) {
        printf("PASS: benchmark within budget\n");
        return 0;
    }
    printf("FAIL: P99 %.3fms exceeds budget %.3fms\n", stats.p99_ms, budget_ms);
    return 1;
}
