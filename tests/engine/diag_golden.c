#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../src/engine/exports.h"

static uint8_t* read_file(const char* path, int* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { *out_len = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (buf) fread(buf, 1, (size_t)sz, f);
    fclose(f);
    *out_len = (int)sz;
    return buf;
}

int main(void) {
    int wt_len = 0;
    uint8_t* wt = read_file("weights/fe_tiny_48k.bin", &wt_len);
    if (!wt) { printf("No weights\n"); return 1; }

    FeState* state = fe_init(0, wt, wt_len);
    if (!state) { printf("fe_init failed\n"); return 1; }

    int inp_cnt = 0;
    float* golden_in = (float*)read_file("tests/golden/golden_input.bin", &inp_cnt);
    inp_cnt /= (int)sizeof(float);

    float* in_ptr = fe_get_input_ptr(state);
    float* out_ptr = fe_get_output_ptr(state);
    int hop = fe_get_hop_size(state);

    printf("hop=%d, n_frames=%d\n", hop, inp_cnt / hop);

    for (int f = 0; f < 5; f++) {
        memcpy(in_ptr, golden_in + f * hop, sizeof(float) * (size_t)hop);
        fe_process(state, in_ptr, out_ptr);

        float mn = out_ptr[0], mx = out_ptr[0];
        double rms_sum = 0;
        for (int i = 0; i < hop; i++) {
            if (out_ptr[i] < mn) mn = out_ptr[i];
            if (out_ptr[i] > mx) mx = out_ptr[i];
            rms_sum += (double)out_ptr[i] * (double)out_ptr[i];
        }
        float rms = (float)sqrt(rms_sum / hop);
        printf("Frame %d: min=%.6e max=%.6e rms=%.6e first5=[%.6e %.6e %.6e %.6e %.6e]\n",
               f, mn, mx, rms, out_ptr[0], out_ptr[1], out_ptr[2], out_ptr[3], out_ptr[4]);
    }

    free(golden_in);
    fe_destroy(state);
    free(wt);
    return 0;
}
