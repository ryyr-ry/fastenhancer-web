/*
 * profile_stages.c — Pipeline stage-by-stage profiler
 *
 * Measures processing time of each inference stage individually
 * to identify bottlenecks that should be optimized.
 *
 * Measurement method: Wrap stages inside fe_process_frame,
 * call them repeatedly from external code and collect statistics.
 *
 * Build: gcc -O3 -march=native -ffast-math -DFE_USE_TINY_48K -DFE_MODEL_SIZE=0
 *        -I ... profile_stages.c <all_sources> -o profile_stages.exe -lm
 */

#include "fastenhancer.h"

#if defined(FE_USE_BASE_48K)
#include "base_48k.h"
#elif defined(FE_USE_SMALL_48K)
#include "small_48k.h"
#else
#include "tiny_48k.h"
#endif

#include "stft.h"
#include "conv.h"
#include "gru.h"
#include "attention.h"
#include "activations.h"
#include "compression.h"
#include "simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
static double get_us(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e6;
}
#else
#include <time.h>
static double get_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
#endif

#define WARMUP 200
#define ITERS  1000
#define N_STAGES 12

static const char* stage_names[N_STAGES] = {
    "STFT",
    "PowerCompress",
    "EncPreNet",
    "EncBlocks",
    "RF_PreNet",
    "RF_Blocks",
    "RF_PostNet",
    "DecBlocks",
    "DecPostNet",
    "MaskApply",
    "PowerDecomp",
    "iSTFT"
};

static double stage_totals[N_STAGES];

static float lcg_float(unsigned int* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)((*seed >> 16) & 0x7fff) / 32768.0f - 0.5f) * 0.02f;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(void) {
    int model_id = FE_MODEL_TINY;
#if defined(FE_USE_BASE_48K)
    model_id = FE_MODEL_BASE;
#elif defined(FE_USE_SMALL_48K)
    model_id = FE_MODEL_SMALL;
#endif
    int wc = fe_weight_count(model_id);
    float* w = (float*)malloc(wc * sizeof(float));
    unsigned int seed = 42;
    for (int i = 0; i < wc; i++) w[i] = lcg_float(&seed);

    FeState* state = fe_create(model_id, w, wc);
    if (!state) { fprintf(stderr, "fe_create failed\n"); free(w); return 1; }
    free(w);

    float input[FE_HOP_SIZE], output[FE_HOP_SIZE];
    double frame_times[ITERS];

    /* Warm-up */
    for (int f = 0; f < WARMUP; f++) {
        for (int i = 0; i < FE_HOP_SIZE; i++)
            input[i] = 0.3f * sinf(2.0f * 3.14159265f * 440.0f * (float)(f * FE_HOP_SIZE + i) / 48000.0f);
        fe_process_frame(state, input, output);
    }

    /* Stage-by-stage measurement: Direct stage separation is not possible,
     * so measure overall time while also performing microbenchmarks of individual functions */
    memset(stage_totals, 0, sizeof(stage_totals));

    /* First measure overall frame time */
    for (int f = 0; f < ITERS; f++) {
        for (int i = 0; i < FE_HOP_SIZE; i++)
            input[i] = 0.3f * sinf(2.0f * 3.14159265f * 440.0f *
                (float)((WARMUP + f) * FE_HOP_SIZE + i) / 48000.0f);
        double t0 = get_us();
        fe_process_frame(state, input, output);
        frame_times[f] = get_us() - t0;
    }

    qsort(frame_times, ITERS, sizeof(double), cmp_double);
    double median_us = frame_times[ITERS / 2];
    double p99_us = frame_times[ITERS * 99 / 100];
    printf("=== Full Frame ===\n");
    printf("  median: %.1f us (%.3f ms)\n", median_us, median_us / 1000.0);
    printf("  P99:    %.1f us (%.3f ms)\n", p99_us, p99_us / 1000.0);

    /* ---- Individual stage microbenchmarks ---- */
    printf("\n=== Stage Microbenchmarks (Tiny: C1=%d C2=%d F1=%d F2=%d) ===\n",
           FE_C1, FE_C2, FE_F1, FE_F2);

    /* Test buffers */
    float* buf_512  = (float*)calloc(FE_FREQ_BINS + 1, sizeof(float));
    float* buf_512b = (float*)calloc(FE_FREQ_BINS + 1, sizeof(float));
    float* buf_1024 = (float*)calloc(FE_N_FFT, sizeof(float));
    float* buf_1024b = (float*)calloc(FE_N_FFT, sizeof(float));
    float* buf_cf1  = (float*)calloc(FE_C1 * FE_F1, sizeof(float));
    float* buf_cf1b = (float*)calloc(FE_C1 * FE_F1, sizeof(float));
    float* buf_2cf1 = (float*)calloc(2 * FE_C1 * FE_F1, sizeof(float));
    float* enc_in   = (float*)calloc(2 * FE_FREQ_BINS, sizeof(float));
    float* rf_a     = (float*)calloc(FE_C1 * FE_F2, sizeof(float));
    float* rf_b     = (float*)calloc(FE_C2 * FE_F2, sizeof(float));
    float* rf_c     = (float*)calloc(FE_F2 * FE_C2, sizeof(float));
    float* attn_sc  = (float*)calloc(FE_NUM_HEADS * FE_F2 * FE_F2, sizeof(float));
    float* attn_scr = (float*)calloc(4 * FE_F2 * FE_C2, sizeof(float));

    /* Fill enc_in with dummy data */
    for (int i = 0; i < 2 * FE_FREQ_BINS; i++) enc_in[i] = lcg_float(&seed);
    for (int i = 0; i < FE_C1 * FE_F1; i++) { buf_cf1[i] = lcg_float(&seed); buf_cf1b[i] = lcg_float(&seed); }
    for (int i = 0; i < FE_F2 * FE_C2; i++) rf_c[i] = lcg_float(&seed);

    /* Dummy weights */
    float* dummy_w_enc_pre = (float*)calloc(FE_C1 * 2 * FE_ENC_K0, sizeof(float));
    float* dummy_bn_s = (float*)calloc(FE_C1, sizeof(float));
    float* dummy_bn_b = (float*)calloc(FE_C1, sizeof(float));
    float* dummy_w_k3 = (float*)calloc(FE_C1 * FE_C1 * FE_ENC_K, sizeof(float));
    float* dummy_w_k1_2c = (float*)calloc(FE_C1 * 2 * FE_C1, sizeof(float));
    float* dummy_w_deconv = (float*)calloc(FE_C1 * 2 * FE_ENC_K0, sizeof(float));
    float* dummy_b_deconv = (float*)calloc(2, sizeof(float));
    float* dummy_freq_w = (float*)calloc(FE_F2 * FE_F1, sizeof(float));
    float* dummy_conv_c2c1 = (float*)calloc(FE_C2 * FE_C1 * 1, sizeof(float));
    float* dummy_bn_s2 = (float*)calloc(FE_C2, sizeof(float));
    float* dummy_bn_b2 = (float*)calloc(FE_C2, sizeof(float));

    for (int i = 0; i < FE_C1; i++) { dummy_bn_s[i] = 1.0f; }
    for (int i = 0; i < FE_C2; i++) { dummy_bn_s2[i] = 1.0f; }
    for (int i = 0; i < FE_C1 * 2 * FE_ENC_K0; i++) dummy_w_enc_pre[i] = lcg_float(&seed);
    for (int i = 0; i < FE_C1 * FE_C1 * FE_ENC_K; i++) dummy_w_k3[i] = lcg_float(&seed);

    FeStftState stft_state;
    if (fe_stft_init(&stft_state, FE_N_FFT, FE_HOP_SIZE) != 0) {
        fprintf(stderr, "STFT init failed\n");
        return 1;
    }

    double t0, t1;
    double times[ITERS];

    /* Stage 0: STFT */
    for (int i = 0; i < FE_HOP_SIZE; i++) buf_512[i] = lcg_float(&seed);
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_stft_forward(&stft_state, buf_512, buf_1024, buf_1024b);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us\n", "STFT", times[ITERS/2], times[ITERS*99/100]);

    /* Stage 1: PowerCompress */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_power_compress_complex(buf_1024, buf_1024b, buf_512, buf_512b,
                                  FE_FREQ_BINS, FE_COMPRESS_EXP);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us\n", "PowerCompress", times[ITERS/2], times[ITERS*99/100]);

    /* Stage 2: Encoder PreNet (STRIDED Conv - SCALAR path) */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_conv1d_bn(enc_in, dummy_w_enc_pre, dummy_bn_s, dummy_bn_b,
                     buf_cf1, FE_FREQ_BINS, 2, FE_C1,
                     FE_ENC_K0, FE_STRIDE, FE_ENC_PRE_PAD);
        fe_silu_batch(buf_cf1, buf_cf1, FE_C1 * FE_F1);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us  [stride=%d, SCALAR]\n",
           "EncPreNet", times[ITERS/2], times[ITERS*99/100], FE_STRIDE);

    /* Stage 3: Encoder Block (stride=1, SIMD path) */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_conv1d_bn(buf_cf1, dummy_w_k3, dummy_bn_s, dummy_bn_b,
                     buf_cf1b, FE_F1, FE_C1, FE_C1,
                     FE_ENC_K, 1, FE_ENC_PAD);
        fe_silu_batch(buf_cf1b, buf_cf1b, FE_C1 * FE_F1);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us  [stride=1, SIMD]\n",
           "EncBlock(x1)", times[ITERS/2], times[ITERS*99/100]);

    /* Stage 4: RF PreNet (Linear + Conv1d_BN k=1) */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        for (int r = 0; r < FE_C1; r++) {
            memset(rf_a + r * FE_F2, 0, sizeof(float) * FE_F2);
            fe_matvec_add(dummy_freq_w, buf_cf1 + r * FE_F1,
                          rf_a + r * FE_F2, FE_F2, FE_F1);
        }
        fe_conv1d_bn(rf_a, dummy_conv_c2c1, dummy_bn_s2, dummy_bn_b2,
                     rf_b, FE_F2, FE_C1, FE_C2, 1, 1, 0);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us\n", "RF_PreNet", times[ITERS/2], times[ITERS*99/100]);

    /* Stage 5: RF Block (GRU + MHSA) — This needs GRU weights so estimate from fe_process_frame difference */
    /* GRU step micro-bench: Measure GRU 1 step with dummy weights */
    {
        float gru_wz[FE_C2 * FE_C2], gru_wr[FE_C2 * FE_C2], gru_wn[FE_C2 * FE_C2];
        float gru_uz[FE_C2 * FE_C2], gru_ur[FE_C2 * FE_C2], gru_un[FE_C2 * FE_C2];
        float gru_bz[FE_C2], gru_br[FE_C2], gru_bin[FE_C2], gru_bhn[FE_C2];
        float gru_x[FE_C2], gru_h[FE_C2];
        for (int i = 0; i < FE_C2*FE_C2; i++) {
            gru_wz[i] = lcg_float(&seed); gru_wr[i] = lcg_float(&seed);
            gru_wn[i] = lcg_float(&seed); gru_uz[i] = lcg_float(&seed);
            gru_ur[i] = lcg_float(&seed); gru_un[i] = lcg_float(&seed);
        }
        for (int i = 0; i < FE_C2; i++) {
            gru_bz[i] = lcg_float(&seed); gru_br[i] = lcg_float(&seed);
            gru_bin[i] = lcg_float(&seed); gru_bhn[i] = lcg_float(&seed);
            gru_x[i] = lcg_float(&seed); gru_h[i] = 0.0f;
        }
        FeGruWeights gw = {
            .W_z = gru_wz, .U_z = gru_uz, .b_z = gru_bz,
            .W_r = gru_wr, .U_r = gru_ur, .b_r = gru_br,
            .W_n = gru_wn, .U_n = gru_un, .b_in_n = gru_bin, .b_hn_n = gru_bhn,
            .input_size = FE_C2, .hidden_size = FE_C2
        };

        /* GRU × F2 bins + FC */
        float fc_w[FE_C2 * FE_C2], fc_b[FE_C2], fc_out[FE_C2];
        for (int i = 0; i < FE_C2*FE_C2; i++) fc_w[i] = lcg_float(&seed);
        for (int i = 0; i < FE_C2; i++) fc_b[i] = lcg_float(&seed);

        for (int f = 0; f < ITERS; f++) {
            t0 = get_us();
            for (int fi = 0; fi < FE_F2; fi++) {
                fe_gru_step(&gw, gru_x, gru_h);
                memcpy(fc_out, fc_b, sizeof(float) * FE_C2);
                fe_matvec_add(fc_w, gru_h, fc_out, FE_C2, FE_C2);
            }
            t1 = get_us();
            times[f] = t1 - t0;
        }
        qsort(times, ITERS, sizeof(double), cmp_double);
        printf("  %-15s median=%6.1f us  P99=%6.1f us  [F2=%d bins]\n",
               "GRU+FC(x1blk)", times[ITERS/2], times[ITERS*99/100], FE_F2);

        /* MHSA */
        FeMhsaWeights mw = {0};
        float mhsa_wq[FE_C2*FE_C2], mhsa_wk[FE_C2*FE_C2], mhsa_wv[FE_C2*FE_C2];
        float mhsa_wo[FE_C2*FE_C2], mhsa_bo[FE_C2];
        for (int i = 0; i < FE_C2*FE_C2; i++) {
            mhsa_wq[i] = lcg_float(&seed); mhsa_wk[i] = lcg_float(&seed);
            mhsa_wv[i] = lcg_float(&seed); mhsa_wo[i] = lcg_float(&seed);
        }
        for (int i = 0; i < FE_C2; i++) mhsa_bo[i] = lcg_float(&seed);
        float mhsa_bq[FE_C2], mhsa_bk[FE_C2], mhsa_bv[FE_C2];
        for (int i = 0; i < FE_C2; i++) { mhsa_bq[i] = lcg_float(&seed); mhsa_bk[i] = lcg_float(&seed); mhsa_bv[i] = lcg_float(&seed); }
        mw.W_q = mhsa_wq; mw.b_q = mhsa_bq; mw.W_k = mhsa_wk; mw.b_k = mhsa_bk;
        mw.W_v = mhsa_wv; mw.b_v = mhsa_bv;
        mw.W_o = mhsa_wo; mw.b_o = mhsa_bo;
        mw.n_heads = FE_NUM_HEADS; mw.head_dim = FE_C2 / FE_NUM_HEADS;
        mw.c2 = FE_C2;

        for (int f = 0; f < ITERS; f++) {
            t0 = get_us();
            fe_mhsa(&mw, rf_c, rf_a, attn_sc, attn_scr, FE_F2);
            t1 = get_us();
            times[f] = t1 - t0;
        }
        qsort(times, ITERS, sizeof(double), cmp_double);
        printf("  %-15s median=%6.1f us  P99=%6.1f us  [heads=%d, F2=%d]\n",
               "MHSA(x1blk)", times[ITERS/2], times[ITERS*99/100], FE_NUM_HEADS, FE_F2);
    }

    /* Stage 6: Decoder Block (1×1 Conv + k=3 Conv, stride=1) */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_conv1d_bn(buf_2cf1, dummy_w_k1_2c, dummy_bn_s, dummy_bn_b,
                     buf_cf1b, FE_F1, 2 * FE_C1, FE_C1, 1, 1, 0);
        fe_silu_batch(buf_cf1b, buf_cf1b, FE_C1 * FE_F1);
        fe_conv1d_bn(buf_cf1b, dummy_w_k3, dummy_bn_s, dummy_bn_b,
                     buf_cf1, FE_F1, FE_C1, FE_C1, FE_ENC_K, 1, FE_ENC_PAD);
        fe_silu_batch(buf_cf1, buf_cf1, FE_C1 * FE_F1);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us  [stride=1, SIMD]\n",
           "DecBlock(x1)", times[ITERS/2], times[ITERS*99/100]);

    /* Stage 7: ConvTranspose1d (Decoder PostNet - SCALAR) */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_conv_transpose1d(buf_cf1b, dummy_w_deconv, dummy_b_deconv,
                            enc_in, FE_F1, FE_C1, 2,
                            FE_ENC_K0, FE_STRIDE, FE_ENC_PRE_PAD);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us  [SCALAR]\n",
           "ConvTranspose", times[ITERS/2], times[ITERS*99/100]);

    /* Stage 8: iSTFT */
    for (int f = 0; f < ITERS; f++) {
        t0 = get_us();
        fe_stft_inverse(&stft_state, buf_1024, buf_1024b, output);
        t1 = get_us();
        times[f] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    printf("  %-15s median=%6.1f us  P99=%6.1f us\n", "iSTFT", times[ITERS/2], times[ITERS*99/100]);

    /* Clean up */
    fe_destroy(state);
    free(buf_512); free(buf_512b); free(buf_1024); free(buf_1024b);
    free(buf_cf1); free(buf_cf1b); free(buf_2cf1); free(enc_in);
    free(rf_a); free(rf_b); free(rf_c); free(attn_sc); free(attn_scr);
    free(dummy_w_enc_pre); free(dummy_bn_s); free(dummy_bn_b);
    free(dummy_w_k3); free(dummy_w_k1_2c); free(dummy_w_deconv); free(dummy_b_deconv);
    free(dummy_freq_w); free(dummy_conv_c2c1); free(dummy_bn_s2); free(dummy_bn_b2);

    return 0;
}
