/*
 * fastenhancer.c — FastEnhancer inference pipeline implementation
 *
 * Tensor flow (48kHz Tiny):
 *   Input [512] -> STFT -> [513] -> Nyquist removal -> [512] complex
 *   -> Power compression -> [2,512] -> Encoder -> [C1,F1]
 *   -> RNNFormer PreNet -> [F2,C2] -> RNNFormer Blocks
 *   -> RNNFormer PostNet -> [C1,F1] -> Decoder -> [2,512] mask
 *   -> Complex mask application -> Power decompression -> Nyquist restoration -> iSTFT -> [512]
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
#include "pipeline.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Weight pointer structures (views into the flat weight buffer) ---- */

typedef struct {
    const float* conv_w;
    const float* bn_s;
    const float* bn_b;
} FeConvBnWeights;

typedef struct {
    const float* skip_conv_w;
    const float* skip_bn_s;
    const float* skip_bn_b;
    const float* conv_w;
    const float* bn_s;
    const float* bn_b;
} FeDecBlockWeights;

typedef struct {
    FeGruWeights     gru;
    const float*     gru_fc_w;
    const float*     gru_fc_b;
    const float*     pe;
    FeMhsaWeights    mhsa;
} FeRfBlockWeights;

typedef struct {
    /* Encoder */
    const float*     enc_pre_conv_w;
    const float*     enc_pre_bn_s;
    const float*     enc_pre_bn_b;
    FeConvBnWeights  enc_blocks[FE_ENC_BLOCKS];

    /* RNNFormer PreNet */
    const float*     rf_pre_freq_w;
    const float*     rf_pre_conv_w;
    const float*     rf_pre_bn_s;
    const float*     rf_pre_bn_b;

    /* RNNFormer Blocks */
    FeRfBlockWeights rf_blocks[FE_RF_BLOCKS];

    /* RNNFormer PostNet */
    const float*     rf_post_conv_w;
    const float*     rf_post_bn_s;
    const float*     rf_post_bn_b;
    const float*     rf_post_freq_w;

    /* Decoder */
    FeDecBlockWeights dec_blocks[FE_ENC_BLOCKS];

    /* Decoder PostNet */
    const float*     dec_post_skip_conv_w;
    const float*     dec_post_skip_bn_s;
    const float*     dec_post_skip_bn_b;
    const float*     dec_post_deconv_w;
    const float*     dec_post_deconv_b;
} FeWeights;

/* ---- FeState full definition ---- */

struct FeState {
    FeStftState stft;
    FeHpfState hpf;
    FeAgcState agc;
    int hpf_enabled;
    int agc_enabled;

    /* WASM/FFI input buffer / preprocessing input */
    float frame_input[FE_HOP_SIZE];
    float frame_output[FE_HOP_SIZE];

    /* Spectrum buffers [SPEC_BINS=513] */
    float spec_in_re[FE_SPEC_BINS];
    float spec_in_im[FE_SPEC_BINS];
    float spec_out_re[FE_SPEC_BINS];
    float spec_out_im[FE_SPEC_BINS];

    /* Encoder input [2, FREQ_BINS] = [2, 512] */
    float enc_input[2 * FE_FREQ_BINS];

    /* Encoder/Decoder ping-pong buffers [C1, F1] */
    float buf_a[FE_C1 * FE_F1];
    float buf_b[FE_C1 * FE_F1];

    /* Decoder concat buffer [2*C1, F1] */
    float buf_cat[2 * FE_C1 * FE_F1];

    /* Encoder skip connections [ENC_BLOCKS+1][C1*F1] */
    float enc_skip[FE_ENC_BLOCKS + 1][FE_C1 * FE_F1];

    /* RNNFormer intermediate buffers */
    float rf_a[FE_C1 * FE_F2];
    float rf_b[FE_C2 * FE_F2];
    float rf_c[FE_F2 * FE_C2];

    /* GRU hidden state (preserved across frames) */
    float gru_h[FE_RF_BLOCKS][FE_F2 * FE_C2];

    /* Attention workspace */
    float attn_scratch[4 * FE_F2 * FE_C2];
    float attn_scores[FE_NUM_HEADS * FE_F2 * FE_F2];

    /* Internal copy of weights and pointers */
    float*    weight_data;
    FeWeights w;

    int model_size;
};

/* ---- Helper functions ---- */

/* Apply Linear to the last dimension: [batch, in_dim] -> [batch, out_dim]
 * Uses SIMD matrix-vector product */
static void linear_last_dim(const float* W, const float* input, float* output,
                            int batch, int out_dim, int in_dim) {
    for (int b = 0; b < batch; b++) {
        float* out_b = output + b * out_dim;
        memset(out_b, 0, sizeof(float) * out_dim);
        fe_matvec_add(W, input + b * in_dim, out_b, out_dim, in_dim);
    }
}

/* 2D transpose: [rows, cols] -> [cols, rows] */
static void transpose_2d(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

/* Channel-wise concat: [C, L] + [C, L] -> [2C, L] */
static void concat_channels(const float* a, const float* b, float* out,
                            int channels, int length) {
    int size = channels * length;
    memcpy(out, a, sizeof(float) * size);
    memcpy(out + size, b, sizeof(float) * size);
}

/* ---- Weight pointer initialization ---- */

static int setup_weights(FeState* state) {
    const float* p = state->weight_data;
    FeWeights* w = &state->w;

    /* Encoder PreNet: conv[C1,2,K0] + bn_s[C1] + bn_b[C1] */
    w->enc_pre_conv_w = p; p += FE_C1 * 2 * FE_ENC_K0;
    w->enc_pre_bn_s   = p; p += FE_C1;
    w->enc_pre_bn_b   = p; p += FE_C1;

    /* Encoder Blocks: conv[C1,C1,K] + bn_s[C1] + bn_b[C1] */
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        w->enc_blocks[b].conv_w = p; p += FE_C1 * FE_C1 * FE_ENC_K;
        w->enc_blocks[b].bn_s   = p; p += FE_C1;
        w->enc_blocks[b].bn_b   = p; p += FE_C1;
    }

    /* RNNFormer PreNet */
    w->rf_pre_freq_w = p; p += FE_F2 * FE_F1;
    w->rf_pre_conv_w = p; p += FE_C2 * FE_C1;
    w->rf_pre_bn_s   = p; p += FE_C2;
    w->rf_pre_bn_b   = p; p += FE_C2;

    /* RNNFormer Blocks */
    for (int b = 0; b < FE_RF_BLOCKS; b++) {
        FeRfBlockWeights* rb = &w->rf_blocks[b];

        /* GRU (separated weights) */
        rb->gru.W_z    = p; p += FE_C2 * FE_C2;
        rb->gru.U_z    = p; p += FE_C2 * FE_C2;
        rb->gru.b_z    = p; p += FE_C2;
        rb->gru.W_r    = p; p += FE_C2 * FE_C2;
        rb->gru.U_r    = p; p += FE_C2 * FE_C2;
        rb->gru.b_r    = p; p += FE_C2;
        rb->gru.W_n    = p; p += FE_C2 * FE_C2;
        rb->gru.U_n    = p; p += FE_C2 * FE_C2;
        rb->gru.b_in_n = p; p += FE_C2;
        rb->gru.b_hn_n = p; p += FE_C2;
        rb->gru.input_size  = FE_C2;
        rb->gru.hidden_size = FE_C2;

        /* Linear after GRU */
        rb->gru_fc_w = p; p += FE_C2 * FE_C2;
        rb->gru_fc_b = p; p += FE_C2;

        /* Positional embedding (block 0 only) */
        if (b == 0) {
            rb->pe = p; p += FE_F2 * FE_C2;
        } else {
            rb->pe = NULL;
        }

        /* MHSA */
        rb->mhsa.W_q     = p; p += FE_C2 * FE_C2;
        rb->mhsa.b_q     = p; p += FE_C2;
        rb->mhsa.W_k     = p; p += FE_C2 * FE_C2;
        rb->mhsa.b_k     = p; p += FE_C2;
        rb->mhsa.W_v     = p; p += FE_C2 * FE_C2;
        rb->mhsa.b_v     = p; p += FE_C2;
        rb->mhsa.W_o     = p; p += FE_C2 * FE_C2;
        rb->mhsa.b_o     = p; p += FE_C2;
        rb->mhsa.n_heads  = FE_NUM_HEADS;
        rb->mhsa.head_dim = FE_HEAD_DIM;
        rb->mhsa.c2       = FE_C2;
    }

    /* RNNFormer PostNet */
    w->rf_post_conv_w = p; p += FE_C1 * FE_C2;
    w->rf_post_bn_s   = p; p += FE_C1;
    w->rf_post_bn_b   = p; p += FE_C1;
    w->rf_post_freq_w = p; p += FE_F1 * FE_F2;

    /* Decoder Blocks */
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        FeDecBlockWeights* db = &w->dec_blocks[b];
        db->skip_conv_w = p; p += FE_C1 * 2 * FE_C1;
        db->skip_bn_s   = p; p += FE_C1;
        db->skip_bn_b   = p; p += FE_C1;
        db->conv_w      = p; p += FE_C1 * FE_C1 * FE_ENC_K;
        db->bn_s        = p; p += FE_C1;
        db->bn_b        = p; p += FE_C1;
    }

    /* Decoder PostNet */
    w->dec_post_skip_conv_w = p; p += FE_C1 * 2 * FE_C1;
    w->dec_post_skip_bn_s   = p; p += FE_C1;
    w->dec_post_skip_bn_b   = p; p += FE_C1;
    w->dec_post_deconv_w    = p; p += FE_C1 * 2 * FE_ENC_K0;
    w->dec_post_deconv_b    = p; p += 2;

    /* E7: Verify that weight pointer walk ends exactly at FE_TOTAL_WEIGHTS */
    {
        int consumed = (int)(p - state->weight_data);
        if (consumed != FE_TOTAL_WEIGHTS) {
            memset(&state->w, 0, sizeof(state->w));
            return -1;
        }
    }
    return 0;
}

/* ---- Pipeline steps ---- */

static const float* pipeline_preprocess(FeState* s, const float* input) {
    if (!s->hpf_enabled && !s->agc_enabled) {
        return input;
    }

    memmove(s->frame_input, input, sizeof(float) * FE_HOP_SIZE);

    if (s->hpf_enabled) {
        fe_hpf_process(&s->hpf, s->frame_input, FE_HOP_SIZE);
    }
    if (s->agc_enabled) {
        fe_agc_process(&s->agc, s->frame_input, FE_HOP_SIZE);
    }

    return s->frame_input;
}

/* Step 1-4: STFT -> Nyquist removal -> power compression -> channel separation */
static void pipeline_stft_compress(FeState* s, const float* input) {
    fe_stft_forward(&s->stft, input, s->spec_in_re, s->spec_in_im);

    fe_power_compress_complex(s->spec_in_re, s->spec_in_im,
                              s->spec_in_re, s->spec_in_im,
                              FE_FREQ_BINS, FE_COMPRESS_EXP);

    memcpy(s->enc_input, s->spec_in_re, sizeof(float) * FE_FREQ_BINS);
    memcpy(s->enc_input + FE_FREQ_BINS, s->spec_in_im, sizeof(float) * FE_FREQ_BINS);
}

/* Step 5-6: Encoder (PreNet + Blocks) */
static void pipeline_encoder(FeState* s) {
    const FeWeights* w = &s->w;
    const int cf1 = FE_C1 * FE_F1;

    /* PreNet: Conv1d_BN(2→C1, k=K0, s=STRIDE, pad=PRE_PAD) + SiLU */
    fe_conv1d_bn(s->enc_input, w->enc_pre_conv_w,
                 w->enc_pre_bn_s, w->enc_pre_bn_b,
                 s->buf_a, FE_FREQ_BINS, 2, FE_C1,
                 FE_ENC_K0, FE_STRIDE, FE_ENC_PRE_PAD);
    fe_silu_batch(s->buf_a, s->buf_a, cf1);
    memcpy(s->enc_skip[0], s->buf_a, sizeof(float) * cf1);

    /* Encoder Blocks: Conv1d_BN(C1→C1, k=K, pad=PAD) + SiLU */
    float *src = s->buf_a, *dst = s->buf_b;
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        fe_conv1d_bn(src, w->enc_blocks[b].conv_w,
                     w->enc_blocks[b].bn_s, w->enc_blocks[b].bn_b,
                     dst, FE_F1, FE_C1, FE_C1,
                     FE_ENC_K, 1, FE_ENC_PAD);
        fe_silu_batch(dst, dst, cf1);
        memcpy(s->enc_skip[b + 1], dst, sizeof(float) * cf1);
        float *tmp = src; src = dst; dst = tmp;
    }

    /* Copy encoder output to buf_a if not already there */
    if (src != s->buf_a) {
        memcpy(s->buf_a, src, sizeof(float) * cf1);
    }
}

/* Step 7: RNNFormer PreNet */
static void pipeline_rf_prenet(FeState* s) {
    const FeWeights* w = &s->w;

    /* Linear(F1→F2, bias=False): buf_a [C1,F1] → rf_a [C1,F2] */
    linear_last_dim(w->rf_pre_freq_w, s->buf_a, s->rf_a,
                    FE_C1, FE_F2, FE_F1);

    /* Conv1d_BN(C1→C2, k=1): rf_a [C1,F2] → rf_b [C2,F2] */
    fe_conv1d_bn(s->rf_a, w->rf_pre_conv_w,
                 w->rf_pre_bn_s, w->rf_pre_bn_b,
                 s->rf_b, FE_F2, FE_C1, FE_C2, 1, 1, 0);

    /* Transpose [C2,F2] → [F2,C2] */
    transpose_2d(s->rf_b, s->rf_c, FE_C2, FE_F2);
}

/* Step 8: RNNFormer Block (GRU + Linear + PE + MHSA) */
static void pipeline_rf_block(FeState* s, int block) {
    const FeRfBlockWeights* rb = &s->w.rf_blocks[block];

    /* GRU: process each frequency bin independently + Linear + residual connection */
    for (int f = 0; f < FE_F2; f++) {
        float* x_f = s->rf_c + f * FE_C2;
        float* h_f = s->gru_h[block] + f * FE_C2;

        fe_gru_step(&rb->gru, x_f, h_f);

        float fc_out[FE_C2];
        memcpy(fc_out, rb->gru_fc_b, sizeof(float) * FE_C2);
        fe_matvec_add(rb->gru_fc_w, h_f, fc_out, FE_C2, FE_C2);
        fe_vec_add(x_f, fc_out, FE_C2);
    }

    /* Positional embedding (block 0 only) */
    if (rb->pe) {
        fe_vec_add(s->rf_c, rb->pe, FE_F2 * FE_C2);
    }

    /* MHSA + residual connection */
    fe_mhsa(&rb->mhsa, s->rf_c, s->rf_a,
            s->attn_scores, s->attn_scratch, FE_F2);

    int n = FE_F2 * FE_C2;
    fe_vec_add(s->rf_c, s->rf_a, n);
}

/* Step 9: RNNFormer PostNet */
static void pipeline_rf_postnet(FeState* s) {
    const FeWeights* w = &s->w;

    /* Transpose [F2,C2] → [C2,F2] */
    transpose_2d(s->rf_c, s->rf_b, FE_F2, FE_C2);

    /* Linear(F2→F1, bias=False): rf_b [C2,F2] → buf_b [C2,F1] */
    linear_last_dim(w->rf_post_freq_w, s->rf_b, s->buf_b,
                    FE_C2, FE_F1, FE_F2);

    /* Conv1d_BN(C2→C1, k=1): buf_b [C2,F1] → buf_a [C1,F1] */
    fe_conv1d_bn(s->buf_b, w->rf_post_conv_w,
                 w->rf_post_bn_s, w->rf_post_bn_b,
                 s->buf_a, FE_F1, FE_C2, FE_C1, 1, 1, 0);
}

/* Step 10-11: Decoder (Blocks + PostNet) */
static void pipeline_decoder(FeState* s) {
    const FeWeights* w = &s->w;
    const int cf1 = FE_C1 * FE_F1;

    /* Decoder Blocks (reverse skip: enc_skip[N]..enc_skip[1]) */
    for (int b = 0; b < FE_ENC_BLOCKS; b++) {
        int skip_idx = FE_ENC_BLOCKS - b;
        const FeDecBlockWeights* db = &w->dec_blocks[b];

        concat_channels(s->buf_a, s->enc_skip[skip_idx],
                        s->buf_cat, FE_C1, FE_F1);

        /* 1×1 Conv_BN(2C1→C1) + SiLU */
        fe_conv1d_bn(s->buf_cat, db->skip_conv_w,
                     db->skip_bn_s, db->skip_bn_b,
                     s->buf_b, FE_F1, 2 * FE_C1, FE_C1, 1, 1, 0);
        fe_silu_batch(s->buf_b, s->buf_b, cf1);

        /* 3×3 Conv_BN(C1→C1) + SiLU */
        fe_conv1d_bn(s->buf_b, db->conv_w,
                     db->bn_s, db->bn_b,
                     s->buf_a, FE_F1, FE_C1, FE_C1,
                     FE_ENC_K, 1, FE_ENC_PAD);
        fe_silu_batch(s->buf_a, s->buf_a, cf1);
    }

    /* Decoder PostNet (skip[0]) */
    concat_channels(s->buf_a, s->enc_skip[0],
                    s->buf_cat, FE_C1, FE_F1);

    /* 1×1 Conv_BN(2C1→C1) + SiLU */
    fe_conv1d_bn(s->buf_cat, w->dec_post_skip_conv_w,
                 w->dec_post_skip_bn_s, w->dec_post_skip_bn_b,
                 s->buf_b, FE_F1, 2 * FE_C1, FE_C1, 1, 1, 0);
    fe_silu_batch(s->buf_b, s->buf_b, cf1);

    /* ConvTranspose1d(C1→2, k=K0, s=STRIDE, pad=PRE_PAD) */
    fe_conv_transpose1d(s->buf_b, w->dec_post_deconv_w,
                        w->dec_post_deconv_b,
                        s->buf_a, FE_F1, FE_C1, 2,
                        FE_ENC_K0, FE_STRIDE, FE_ENC_PRE_PAD);
}

/* Step 12-16: Mask application -> power decompression -> Nyquist restoration -> iSTFT */
static void pipeline_mask_istft(FeState* s, float* output) {
    /* Complex mask application: out = compressed_input x mask */
    const float* mask_re = s->buf_a;
    const float* mask_im = s->buf_a + FE_FREQ_BINS;

    int i = 0;
    for (; i + 3 < FE_FREQ_BINS; i += 4) {
        f32x4 ire = f32x4_load(s->spec_in_re + i);
        f32x4 iim = f32x4_load(s->spec_in_im + i);
        f32x4 mre = f32x4_load(mask_re + i);
        f32x4 mim = f32x4_load(mask_im + i);
        f32x4_store(s->spec_out_re + i, f32x4_sub(f32x4_mul(ire, mre), f32x4_mul(iim, mim)));
        f32x4_store(s->spec_out_im + i, f32x4_add(f32x4_mul(ire, mim), f32x4_mul(iim, mre)));
    }
    for (; i < FE_FREQ_BINS; i++) {
        float in_re = s->spec_in_re[i];
        float in_im = s->spec_in_im[i];
        float m_re  = mask_re[i];
        float m_im  = mask_im[i];
        s->spec_out_re[i] = in_re * m_re - in_im * m_im;
        s->spec_out_im[i] = in_re * m_im + in_im * m_re;
    }

    /* Power decompression */
    fe_power_decompress_complex(s->spec_out_re, s->spec_out_im,
                                s->spec_out_re, s->spec_out_im,
                                FE_FREQ_BINS, FE_COMPRESS_EXP);

    /* Nyquist restoration: imag=0 for real-valued output, preserve real part */
    s->spec_out_im[FE_FREQ_BINS] = 0.0f;

    /* iSTFT -> output waveform */
    fe_stft_inverse(&s->stft, s->spec_out_re, s->spec_out_im, output);
}

/* ---- Public API ---- */

int fe_weight_count(int model_size) {
#if defined(FE_USE_BASE_48K)
    if (model_size == FE_MODEL_BASE) {
        return FE_TOTAL_WEIGHTS;
    }
#elif defined(FE_USE_SMALL_48K)
    if (model_size == FE_MODEL_SMALL) {
        return FE_TOTAL_WEIGHTS;
    }
#else
    if (model_size == FE_MODEL_TINY) {
        return FE_TOTAL_WEIGHTS;
    }
#endif
    return -1;
}

FeState* fe_create(int model_size, const float* weights, int weight_count) {
#if defined(FE_USE_BASE_48K)
    if (model_size != FE_MODEL_BASE) return NULL;
#elif defined(FE_USE_SMALL_48K)
    if (model_size != FE_MODEL_SMALL) return NULL;
#else
    if (model_size != FE_MODEL_TINY) return NULL;
#endif
    if (weight_count != FE_TOTAL_WEIGHTS) return NULL;
    if (!weights) return NULL;

    FeState* s = (FeState*)calloc(1, sizeof(FeState));
    if (!s) return NULL;

    s->model_size = model_size;

    /* Copy weight data */
    s->weight_data = (float*)malloc(sizeof(float) * weight_count);
    if (!s->weight_data) {
        free(s);
        return NULL;
    }
    memcpy(s->weight_data, weights, sizeof(float) * weight_count);

    /* Set up weight pointers */
    if (setup_weights(s) != 0) {
        free(s->weight_data);
        free(s);
        return NULL;
    }

    /* Initialize STFT */
    if (fe_stft_init(&s->stft, FE_N_FFT, FE_HOP_SIZE) != 0) {
        free(s->weight_data);
        free(s);
        return NULL;
    }
    fe_hpf_init(&s->hpf);
    fe_agc_init(&s->agc);
    s->hpf_enabled = 0;
    s->agc_enabled = 0;

    return s;
}

void fe_process_frame(FeState* state, const float* input, float* output) {
    const float* processed_input;

    if (!state || !input || !output) return;

    processed_input = pipeline_preprocess(state, input);
    pipeline_stft_compress(state, processed_input);
    pipeline_encoder(state);
    pipeline_rf_prenet(state);

    for (int b = 0; b < FE_RF_BLOCKS; b++) {
        pipeline_rf_block(state, b);
    }

    pipeline_rf_postnet(state);
    pipeline_decoder(state);
    pipeline_mask_istft(state, output);
}

void fe_set_hpf(FeState* state, int enabled) {
    if (!state) return;
    int val = enabled ? 1 : 0;
    if (state->hpf_enabled == val) return;
    state->hpf_enabled = val;
    fe_hpf_reset(&state->hpf);
}

void fe_set_agc(FeState* state, int enabled) {
    if (!state) return;
    int val = enabled ? 1 : 0;
    if (state->agc_enabled == val) return;
    state->agc_enabled = val;
    fe_agc_reset(&state->agc);
}

void fe_reset(FeState* state) {
    if (!state) return;

    fe_stft_reset(&state->stft);
    fe_hpf_reset(&state->hpf);
    fe_agc_reset(&state->agc);
    memset(state->gru_h, 0, sizeof(state->gru_h));
    memset(state->frame_input, 0, sizeof(state->frame_input));
    memset(state->frame_output, 0, sizeof(state->frame_output));
    memset(state->spec_in_re, 0, sizeof(state->spec_in_re));
    memset(state->spec_in_im, 0, sizeof(state->spec_in_im));
    memset(state->spec_out_re, 0, sizeof(state->spec_out_re));
    memset(state->spec_out_im, 0, sizeof(state->spec_out_im));
    memset(state->enc_input, 0, sizeof(state->enc_input));
    memset(state->buf_a, 0, sizeof(state->buf_a));
    memset(state->buf_b, 0, sizeof(state->buf_b));
    memset(state->buf_cat, 0, sizeof(state->buf_cat));
    memset(state->enc_skip, 0, sizeof(state->enc_skip));
    memset(state->rf_a, 0, sizeof(state->rf_a));
    memset(state->rf_b, 0, sizeof(state->rf_b));
    memset(state->rf_c, 0, sizeof(state->rf_c));
    memset(state->attn_scratch, 0, sizeof(state->attn_scratch));
    memset(state->attn_scores, 0, sizeof(state->attn_scores));
}

void fe_destroy(FeState* state) {
    if (!state) return;
    free(state->weight_data);
    free(state);
}

FE_PUBLIC_API int fe_get_hop_size(const FeState* state) {
    return state ? FE_HOP_SIZE : 0;
}

FE_PUBLIC_API float* fe_get_input_ptr(FeState* state) {
    return state ? state->frame_input : NULL;
}

FE_PUBLIC_API float* fe_get_output_ptr(FeState* state) {
    return state ? state->frame_output : NULL;
}

FE_PUBLIC_API int fe_get_n_fft(const FeState* state) {
    return state ? FE_N_FFT : 0;
}
