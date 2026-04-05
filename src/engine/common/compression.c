/*
 * compression.c — Power compression/decompression
 *
 * Compression: mag^exponent (typically exponent=0.3)
 * Decompression: mag^(1/exponent)
 *
 * Safely handles NaN/Inf/negative values/denormals.
 */

#include "compression.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

static const float MAX_MAGNITUDE = 1e10f;
/* Same mag floor as PyTorch CompressedSTFT (during training: mag.clamp(min=eps)) */
static const float MAG_FLOOR = 1e-5f;

/* Bit-manipulation-based finiteness check that works correctly even with -ffast-math */
static inline int fe_is_finite(float x) {
    union { float f; uint32_t u; } conv;
    conv.f = x;
    return (conv.u & 0x7F800000u) != 0x7F800000u;
}

static inline float fe_sanitize(float x) {
    return fe_is_finite(x) ? x : 0.0f;
}

static inline float safe_clamp_input(float x) {
    if (!fe_is_finite(x)) return 0.0f;
    if (x > MAX_MAGNITUDE) return MAX_MAGNITUDE;
    if (x < 0.0f) return 0.0f;
    return x;
}

void fe_power_compress(const float* input, float* output, int n, float exponent) {
    if (n <= 0) return;
    if (exponent <= 0.0f) {
        memset(output, 0, sizeof(float) * n);
        return;
    }
    for (int i = 0; i < n; i++) {
        float x = safe_clamp_input(input[i]);
        if (x == 0.0f) {
            output[i] = 0.0f;
        } else {
            output[i] = powf(x, exponent);
            if (output[i] != output[i]) {
                output[i] = 0.0f;
            } else if (output[i] > MAX_MAGNITUDE) {
                output[i] = MAX_MAGNITUDE;
            }
        }
    }
}

void fe_power_decompress(const float* input, float* output, int n, float exponent) {
    if (n <= 0) return;
    if (exponent <= 0.0f) {
        memset(output, 0, sizeof(float) * n);
        return;
    }
    float inv_exp = 1.0f / exponent;
    for (int i = 0; i < n; i++) {
        float x = safe_clamp_input(input[i]);
        if (x == 0.0f) {
            output[i] = 0.0f;
        } else {
            output[i] = powf(x, inv_exp);
            if (output[i] != output[i]) {
                output[i] = 0.0f;
            } else if (output[i] > MAX_MAGNITUDE) {
                output[i] = MAX_MAGNITUDE;
            }
        }
    }
}

void fe_power_compress_complex(const float* re, const float* im,
                                float* out_re, float* out_im,
                                int n, float exponent) {
    if (n <= 0) return;
    if (exponent <= 0.0f) {
        memset(out_re, 0, sizeof(float) * n);
        memset(out_im, 0, sizeof(float) * n);
        return;
    }
    float half_scale_exp = (exponent - 1.0f) * 0.5f;
    float mag_floor_sq = MAG_FLOOR * MAG_FLOOR;
    for (int i = 0; i < n; i++) {
        float r = fe_sanitize(re[i]);
        float m_v = fe_sanitize(im[i]);
        float mag_sq = r * r + m_v * m_v;
        if (mag_sq < mag_floor_sq) mag_sq = mag_floor_sq;
        float scale = powf(mag_sq, half_scale_exp);
        out_re[i] = r * scale;
        out_im[i] = m_v * scale;
    }
}

void fe_power_decompress_complex(const float* re, const float* im,
                                  float* out_re, float* out_im,
                                  int n, float exponent) {
    if (n <= 0) return;
    if (exponent <= 0.0f) {
        memset(out_re, 0, sizeof(float) * n);
        memset(out_im, 0, sizeof(float) * n);
        return;
    }
    float half_scale_exp = (1.0f / exponent - 1.0f) * 0.5f;
    float mag_floor_sq = MAG_FLOOR * MAG_FLOOR;
    for (int i = 0; i < n; i++) {
        float r = fe_sanitize(re[i]);
        float m_v = fe_sanitize(im[i]);
        float mag_sq = r * r + m_v * m_v;
        if (mag_sq < mag_floor_sq) mag_sq = mag_floor_sq;
        float scale = powf(mag_sq, half_scale_exp);
        float out_r = r * scale;
        float out_m = m_v * scale;
        if (out_r > MAX_MAGNITUDE) out_r = MAX_MAGNITUDE;
        else if (out_r < -MAX_MAGNITUDE) out_r = -MAX_MAGNITUDE;
        if (out_m > MAX_MAGNITUDE) out_m = MAX_MAGNITUDE;
        else if (out_m < -MAX_MAGNITUDE) out_m = -MAX_MAGNITUDE;
        out_re[i] = out_r;
        out_im[i] = out_m;
    }
}
