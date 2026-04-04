/*
 * diag_c_intermediates.c — Output intermediate values for each pipeline stage of the C engine
 *
 * Compare with PyTorch intermediate values to identify the first stage where divergence occurs.
 * Using Frame 1 (frame 0 has STFT padding differences).
 */

#include "fastenhancer.h"
#include "tiny_48k.h"
#include "stft.h"
#include "conv.h"
#include "activations.h"
#include "compression.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_stats(const char* name, const float* data, int n) {
    float mn = data[0], mx = data[0], sum2 = 0.0f;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum2 += data[i] * data[i];
    }
    float rms = sqrtf(sum2 / n);
    printf("  %s: n=%d, min=%.6e, max=%.6e, rms=%.6e\n", name, n, mn, mx, rms);
    printf("    first5=[%.6e, %.6e, %.6e, %.6e, %.6e]\n",
           data[0], data[1], data[2], data[3], data[4]);
}

static float* load_weights(const char* path, int* count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    unsigned char hdr[20];
    if (fread(hdr, 1, 20, f) != 20) { fclose(f); return NULL; }

    *count = (int)(hdr[12] | (hdr[13]<<8) | (hdr[14]<<16) | (hdr[15]<<24));
    float* w = (float*)malloc(sizeof(float)*(*count));
    if (!w) { fclose(f); return NULL; }

    unsigned char* raw = (unsigned char*)malloc((*count) * 4);
    if (fread(raw, 1, (*count)*4, f) != (size_t)((*count)*4)) {
        free(w); free(raw); fclose(f); return NULL;
    }
    for (int i = 0; i < *count; i++) {
        unsigned char* p = raw + i*4;
        unsigned int bits = p[0] | (p[1]<<8) | (p[2]<<16) | (p[3]<<24);
        memcpy(&w[i], &bits, 4);
    }
    free(raw);
    fclose(f);
    return w;
}

static float* load_bin(const char* path, int expected_floats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    float* data = (float*)malloc(sizeof(float)*expected_floats);
    int got = (int)fread(data, sizeof(float), expected_floats, f);
    fclose(f);
    if (got != expected_floats) {
        fprintf(stderr, "Expected %d floats, got %d\n", expected_floats, got);
        free(data); return NULL;
    }
    return data;
}

int main(void) {
    printf("=== C Engine Intermediate Diagnostics ===\n\n");

    int wcount;
    float* weights = load_weights("weights/fe_tiny_48k.bin", &wcount);
    if (!weights) { fprintf(stderr, "Failed to load weights\n"); return 1; }
    printf("Weights loaded: %d floats\n", wcount);

    FeState* state = fe_create(FE_MODEL_TINY, weights, wcount);
    if (!state) { fprintf(stderr, "fe_create failed\n"); free(weights); return 1; }
    printf("fe_create succeeded\n\n");

    int n_samples = 40 * FE_HOP_SIZE;
    float* input = load_bin("tests/golden/golden_input.bin", n_samples);
    if (!input) { fe_destroy(state); free(weights); return 1; }

    /* Process Frame 0 (warm up STFT buffer) */
    float output_frame[FE_HOP_SIZE];
    fe_process_frame(state, input, output_frame);
    printf("Frame 0 output:\n");
    print_stats("output", output_frame, FE_HOP_SIZE);

    /* Process Frame 1 and output intermediate values */
    float* frame1_input = input + FE_HOP_SIZE;

    /* Step 1: STFT */
    printf("\n=== Frame 1: STFT ===\n");
    {
        /* Call STFT directly to check intermediate values */
        /* Note: state's internal STFT buffer has already processed frame 0 */
        float spec_re[FE_SPEC_BINS], spec_im[FE_SPEC_BINS];
        /* Cannot reset STFT and process frame 1 independently (streaming) */
        /* Instead, process frame 0 and frame 1 sequentially */
    }

    /* Process Frame 1 via fe_process_frame */
    fe_process_frame(state, frame1_input, output_frame);
    printf("Frame 1 final output:\n");
    print_stats("output", output_frame, FE_HOP_SIZE);

    /* Additional: Frame 2-9 */
    for (int f = 2; f < 10; f++) {
        fe_process_frame(state, input + f * FE_HOP_SIZE, output_frame);
        printf("Frame %d: rms=%.6e, first=%.6e\n", f,
               sqrtf(output_frame[0]*output_frame[0]), output_frame[0]);
    }

    printf("\n=== Weight Inspection ===\n");
    /* Output first elements of Encoder PreNet weights */
    printf("enc_pre_conv_w (first 10):\n  ");
    for (int i = 0; i < 10; i++) printf("%.6e ", weights[i]);
    printf("\n");

    /* BN scale/bias */
    int enc_pre_conv_size = FE_C1 * 2 * FE_ENC_K0;
    printf("enc_pre_bn_s (first 5):\n  ");
    for (int i = 0; i < 5; i++) printf("%.6e ", weights[enc_pre_conv_size + i]);
    printf("\n");
    printf("enc_pre_bn_b (first 5):\n  ");
    for (int i = 0; i < 5; i++) printf("%.6e ", weights[enc_pre_conv_size + FE_C1 + i]);
    printf("\n");

    fe_destroy(state);
    free(weights);
    free(input);
    return 0;
}
