/*
 * benchmark_stats.h — ベンチマーク統計ヘルパー
 */

#ifndef FE_BENCHMARK_STATS_H
#define FE_BENCHMARK_STATS_H

typedef struct {
    double median_ms;
    double p99_ms;
    double avg_ms;
} FeBenchStats;

int fe_bench_compute_stats(const double* samples, int count, FeBenchStats* out_stats);

#endif /* FE_BENCHMARK_STATS_H */
