/* Offline-only staged engine. The real-time binary still compiles fastenhancer.c.
 * Reuse its kernels and accumulation order. GRU state is owned per frequency
 * position; analysis and synthesis context are supplied explicitly per tile. */
#include "fastenhancer.c"

#define FE_OFFLINE_FEATURES (FE_F2 * FE_C2)
#define FE_OFFLINE_SKIPS ((FE_ENC_BLOCKS + 1) * FE_C1 * FE_F1)
#define FE_OFFLINE_ENCODED (2 * FE_SPEC_BINS + FE_OFFLINE_SKIPS + FE_OFFLINE_FEATURES)
#define FE_OFFLINE_MAX_FRAMES 128

FE_PUBLIC_API int fe_offline_info(int field) {
    switch (field) {
        case 0: return FE_OFFLINE_ENCODED;
        case 1: return FE_OFFLINE_FEATURES;
        case 2: return FE_F2;
        case 3: return FE_C2;
        case 4: return FE_RF_BLOCKS;
        default: return -1;
    }
}

/* input = previous hop followed by frames current hops. No recurrent updates. */
FE_PUBLIC_API int fe_offline_encode(FeState* s, const float* input, int frames, float* output) {
    if (!s || !input || !output || frames < 1 || frames > FE_OFFLINE_MAX_FRAMES) return -1;
    memcpy(s->stft.input_buffer + FE_HOP_SIZE, input, sizeof(float) * FE_HOP_SIZE);
    for (int t = 0; t < frames; t++) {
        pipeline_stft_compress(s, input + (t + 1) * FE_HOP_SIZE);
        pipeline_encoder(s);
        pipeline_rf_prenet(s);
        float* dst = output + t * FE_OFFLINE_ENCODED;
        memcpy(dst, s->spec_in_re, sizeof(float) * FE_SPEC_BINS);
        memcpy(dst + FE_SPEC_BINS, s->spec_in_im, sizeof(float) * FE_SPEC_BINS);
        /* These unused legacy Nyquist inputs may overflow for finite PCM.
         * The legacy mask outputs zero there; do not reject a valid frame. */
        dst[FE_FREQ_BINS] = 0.0f;
        dst[FE_SPEC_BINS + FE_FREQ_BINS] = 0.0f;
        memcpy(dst + 2 * FE_SPEC_BINS, s->enc_skip, sizeof(float) * FE_OFFLINE_SKIPS);
        memcpy(dst + 2 * FE_SPEC_BINS + FE_OFFLINE_SKIPS, s->rf_c, sizeof(float) * FE_OFFLINE_FEATURES);
    }
    return 0;
}

/* Process every time step in order for this worker's frequency positions. */
FE_PUBLIC_API int fe_offline_gru(FeState* s, const float* input, int frames, int block, int first, int end, float* output) {
    if (!s || !input || !output || frames < 1 || frames > FE_OFFLINE_MAX_FRAMES
        || block < 0 || block >= FE_RF_BLOCKS || first < 0 || end > FE_F2 || first >= end) return -1;
    const FeRfBlockWeights* rb = &s->w.rf_blocks[block];
    for (int t = 0; t < frames; t++) for (int f = first; f < end; f++) {
        const float* src = input + t * FE_OFFLINE_FEATURES + f * FE_C2;
        float* dst = output + (t * (end - first) + f - first) * FE_C2;
        float* hidden = s->gru_h[block] + f * FE_C2;
        float fc_out[FE_C2];
        memcpy(dst, src, sizeof(float) * FE_C2);
        fe_gru_step(&rb->gru, src, hidden);
        memcpy(fc_out, rb->gru_fc_b, sizeof(float) * FE_C2);
        fe_matvec_add(rb->gru_fc_w, hidden, fc_out, FE_C2, FE_C2);
        fe_vec_add(dst, fc_out, FE_C2);
        if (rb->pe) fe_vec_add(dst, rb->pe + f * FE_C2, FE_C2);
    }
    return 0;
}

FE_PUBLIC_API int fe_offline_attention(FeState* s, const float* input, int frames, int block, float* output) {
    if (!s || !input || !output || frames < 1 || frames > FE_OFFLINE_MAX_FRAMES || block < 0 || block >= FE_RF_BLOCKS) return -1;
    const FeRfBlockWeights* rb = &s->w.rf_blocks[block];
    for (int t = 0; t < frames; t++) {
        memcpy(s->rf_c, input + t * FE_OFFLINE_FEATURES, sizeof(float) * FE_OFFLINE_FEATURES);
        fe_mhsa(&rb->mhsa, s->rf_c, s->rf_a, s->attn_scores, s->attn_scratch, FE_F2);
        fe_vec_add(s->rf_c, s->rf_a, FE_OFFLINE_FEATURES);
        memcpy(output + t * FE_OFFLINE_FEATURES, s->rf_c, sizeof(float) * FE_OFFLINE_FEATURES);
    }
    return 0;
}

/* Return the two iFFT halves separately; the coordinator performs ordered OLA. */
FE_PUBLIC_API int fe_offline_decode(FeState* s, const float* encoded, const float* features, int frames, float* output) {
    if (!s || !encoded || !features || !output || frames < 1 || frames > FE_OFFLINE_MAX_FRAMES) return -1;
    for (int t = 0; t < frames; t++) {
        const float* src = encoded + t * FE_OFFLINE_ENCODED;
        memcpy(s->spec_in_re, src, sizeof(float) * FE_SPEC_BINS);
        memcpy(s->spec_in_im, src + FE_SPEC_BINS, sizeof(float) * FE_SPEC_BINS);
        memcpy(s->enc_skip, src + 2 * FE_SPEC_BINS, sizeof(float) * FE_OFFLINE_SKIPS);
        memcpy(s->rf_c, features + t * FE_OFFLINE_FEATURES, sizeof(float) * FE_OFFLINE_FEATURES);
        pipeline_rf_postnet(s);
        pipeline_decoder(s);
        memset(s->stft.overlap, 0, sizeof(float) * FE_HOP_SIZE);
        /* Preserve the legacy engine's zero-initialized Nyquist output. */
        s->spec_out_re[FE_FREQ_BINS] = 0.0f;
        pipeline_mask_istft(s, output + t * FE_N_FFT);
        /* Avoid an extra +0 before ordered OLA, including signed-zero changes. */
        memcpy(output + t * FE_N_FFT, s->stft.fft_real, sizeof(float) * FE_HOP_SIZE);
        memcpy(output + t * FE_N_FFT + FE_HOP_SIZE, s->stft.overlap, sizeof(float) * FE_HOP_SIZE);
    }
    return 0;
}
