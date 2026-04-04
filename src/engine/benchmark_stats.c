/*
 * benchmark_stats.c — ベンチマーク統計ヘルパー
 */

#include "benchmark_stats.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define FE_BENCH_MAX_SAMPLES 4096

static int compare_double_asc(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int fe_bench_compute_stats(const double* samples, int count, FeBenchStats* out_stats) {
    double sum = 0.0;
    double sorted[FE_BENCH_MAX_SAMPLES];
    int median_lo;
    int median_hi;
    int p99_rank;

    if (!samples || count <= 0 || !out_stats) return -1;
    if (count > FE_BENCH_MAX_SAMPLES) return -1;

    for (int i = 0; i < count; i++) {
        if (!isfinite(samples[i])) {
            return -1;
        }
        sorted[i] = samples[i];
        sum += samples[i];
    }

    qsort(sorted, (size_t)count, sizeof(double), compare_double_asc);

    median_lo = (count - 1) / 2;
    median_hi = count / 2;
    p99_rank = (int)ceil(0.99 * (double)count);
    if (p99_rank < 1) p99_rank = 1;
    if (p99_rank > count) p99_rank = count;

    out_stats->median_ms = 0.5 * (sorted[median_lo] + sorted[median_hi]);
    out_stats->p99_ms = sorted[p99_rank - 1];
    out_stats->avg_ms = sum / (double)count;

    return 0;
}
