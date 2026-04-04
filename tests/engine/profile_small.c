/*
 * profile_small.c — Small model P99 spike root cause analysis
 *
 * Records processing time for each frame to identify slow frames
 * and analyze processing time distribution in detail.
 */

#include "fastenhancer.h"

#ifndef FE_USE_SMALL_48K
#define FE_USE_SMALL_48K
#endif
#include "small_48k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

#define PI_F 3.14159265f
#define WARMUP 200
#define MEASURE 2000

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(void) {
    int wc = fe_weight_count(FE_MODEL_SMALL);
    float* w = (float*)malloc(wc * sizeof(float));
    unsigned int seed = 42;
    for (int i = 0; i < wc; i++) w[i] = lcg_float(&seed);

    /* Set BN scale=1 (simplified) */
    for (int i = 0; i < wc; i++) {
        if (i % 100 < 3) w[i] = (i % 3 == 0) ? 1.0f : 0.0f;
    }

    FeState* state = fe_create(FE_MODEL_SMALL, w, wc);
    if (!state) {
        fprintf(stderr, "fe_create failed for model_size=%d\n", FE_MODEL_SMALL);
        free(w);
        return 1;
    }
    free(w);

    float input[512], output[512];
    double* timings = (double*)malloc(MEASURE * sizeof(double));

    /* Warm-up */
    for (int f = 0; f < WARMUP; f++) {
        for (int i = 0; i < 512; i++)
            input[i] = 0.3f * sinf(2.0f * PI_F * 440.0f * (float)(f * 512 + i) / 48000.0f);
        fe_process_frame(state, input, output);
    }

    /* Measurement: Record processing time for each frame */
    for (int f = 0; f < MEASURE; f++) {
        for (int i = 0; i < 512; i++)
            input[i] = 0.3f * sinf(2.0f * PI_F * 440.0f * (float)((WARMUP + f) * 512 + i) / 48000.0f);

        double t0 = get_time_ms();
        fe_process_frame(state, input, output);
        double t1 = get_time_ms();
        timings[f] = t1 - t0;
    }

    fe_destroy(state);

    /* ===== Analysis ===== */

    /* Before sorting: Identify indices of slow frames */
    printf("=== Slow Frames (>5ms) ===\n");
    int slow_count = 0;
    for (int f = 0; f < MEASURE; f++) {
        if (timings[f] > 5.0) {
            printf("  frame %d: %.3f ms\n", f, timings[f]);
            slow_count++;
            if (slow_count > 20) { printf("  ... (more)\n"); break; }
        }
    }
    printf("total slow (>5ms): %d / %d\n\n", slow_count, MEASURE);

    /* Consecutive delay pattern: Check if slow frames occur consecutively */
    printf("=== Consecutive Slow Frames ===\n");
    int max_consecutive = 0, cur_consecutive = 0;
    for (int f = 0; f < MEASURE; f++) {
        if (timings[f] > 5.0) {
            cur_consecutive++;
            if (cur_consecutive > max_consecutive) max_consecutive = cur_consecutive;
        } else {
            cur_consecutive = 0;
        }
    }
    printf("max_consecutive_slow: %d\n\n", max_consecutive);

    /* Histogram */
    printf("=== Histogram ===\n");
    int hist[20] = {0};
    for (int f = 0; f < MEASURE; f++) {
        int bin = (int)(timings[f] / 1.0);
        if (bin >= 20) bin = 19;
        hist[bin]++;
    }
    for (int b = 0; b < 20; b++) {
        if (hist[b] > 0) {
            printf("  %2d-%2dms: %4d |", b, b + 1, hist[b]);
            for (int h = 0; h < hist[b] && h < 80; h++) printf("#");
            printf("\n");
        }
    }

    /* Percentile analysis */
    qsort(timings, MEASURE, sizeof(double), cmp_double);

    printf("\n=== Percentiles ===\n");
    printf("  min:    %.3f ms\n", timings[0]);
    printf("  p10:    %.3f ms\n", timings[MEASURE / 10]);
    printf("  p25:    %.3f ms\n", timings[MEASURE / 4]);
    printf("  median: %.3f ms\n", timings[MEASURE / 2]);
    printf("  p75:    %.3f ms\n", timings[MEASURE * 3 / 4]);
    printf("  p90:    %.3f ms\n", timings[MEASURE * 9 / 10]);
    printf("  p95:    %.3f ms\n", timings[MEASURE * 95 / 100]);
    printf("  p99:    %.3f ms\n", timings[MEASURE * 99 / 100]);
    printf("  max:    %.3f ms\n", timings[MEASURE - 1]);

    double sum = 0.0;
    for (int f = 0; f < MEASURE; f++) sum += timings[f];
    printf("  avg:    %.3f ms\n", sum / MEASURE);

    /* Budget overrun rate */
    int over_budget = 0;
    for (int f = 0; f < MEASURE; f++) {
        if (timings[f] > 10.667) over_budget++;
    }
    printf("\n  over_budget(10.67ms): %d / %d (%.2f%%)\n",
        over_budget, MEASURE, 100.0 * over_budget / MEASURE);

    free(timings);
    return 0;
}
