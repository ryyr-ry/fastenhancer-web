/*
 * test_benchmark_stats.c — Phase 3B-G: ベンチマーク統計ヘルパー テスト
 */

#include "unity.h"
#include "benchmark_stats.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_bench_stats_even_count(void) {
    const double samples[] = { 1.0, 2.0, 3.0, 4.0 };
    FeBenchStats stats;

    TEST_ASSERT_EQUAL_INT(0, fe_bench_compute_stats(samples, 4, &stats));

    TEST_ASSERT_TRUE(fabs(stats.median_ms - 2.5) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.p99_ms - 4.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.avg_ms - 2.5) < 1e-12);
}

void test_bench_stats_odd_count(void) {
    const double samples[] = { 5.0, 1.0, 3.0 };
    FeBenchStats stats;

    TEST_ASSERT_EQUAL_INT(0, fe_bench_compute_stats(samples, 3, &stats));

    TEST_ASSERT_TRUE(fabs(stats.median_ms - 3.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.p99_ms - 5.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.avg_ms - 3.0) < 1e-12);
}

void test_bench_stats_nearest_rank_p99(void) {
    double samples[1000];
    FeBenchStats stats;

    for (int i = 0; i < 1000; i++) {
        samples[i] = (double)(i + 1);
    }

    TEST_ASSERT_EQUAL_INT(0, fe_bench_compute_stats(samples, 1000, &stats));

    TEST_ASSERT_TRUE(fabs(stats.median_ms - 500.5) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.p99_ms - 990.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.avg_ms - 500.5) < 1e-12);
}

void test_bench_stats_null_safe(void) {
    const double samples[] = { 1.0, 2.0, 3.0 };
    FeBenchStats stats = { 123.0, 456.0, 789.0 };

    TEST_ASSERT_EQUAL_INT(-1, fe_bench_compute_stats(NULL, 3, &stats));
    TEST_ASSERT_TRUE(fabs(stats.median_ms - 123.0) < 1e-12);

    TEST_ASSERT_EQUAL_INT(-1, fe_bench_compute_stats(samples, 0, &stats));
    TEST_ASSERT_TRUE(fabs(stats.median_ms - 123.0) < 1e-12);

    TEST_ASSERT_EQUAL_INT(-1, fe_bench_compute_stats(samples, 3, NULL));
}

void test_bench_stats_rejects_nan_sample(void) {
    const double samples[] = { 1.0, NAN, 2.0 };
    FeBenchStats stats = { 123.0, 456.0, 789.0 };

    TEST_ASSERT_EQUAL_INT(-1, fe_bench_compute_stats(samples, 3, &stats));
    TEST_ASSERT_TRUE(fabs(stats.median_ms - 123.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.p99_ms - 456.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.avg_ms - 789.0) < 1e-12);
}

void test_bench_stats_rejects_inf_sample(void) {
    const double samples[] = { 1.0, INFINITY, 2.0 };
    FeBenchStats stats = { 123.0, 456.0, 789.0 };

    TEST_ASSERT_EQUAL_INT(-1, fe_bench_compute_stats(samples, 3, &stats));
    TEST_ASSERT_TRUE(fabs(stats.median_ms - 123.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.p99_ms - 456.0) < 1e-12);
    TEST_ASSERT_TRUE(fabs(stats.avg_ms - 789.0) < 1e-12);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_bench_stats_even_count);
    RUN_TEST(test_bench_stats_odd_count);
    RUN_TEST(test_bench_stats_nearest_rank_p99);
    RUN_TEST(test_bench_stats_null_safe);
    RUN_TEST(test_bench_stats_rejects_nan_sample);
    RUN_TEST(test_bench_stats_rejects_inf_sample);

    return UNITY_END();
}
