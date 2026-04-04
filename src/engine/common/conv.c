/*
 * conv.c — Implementation of 1D convolution
 *
 * Conv1d, StridedConv1d (Encoder), ConvTranspose1d (Decoder)
 * BatchNorm fusion: Receives data pre-converted from running_mean/var
 *   to scale/bias by export_weights.py.
 *
 * Memory layout (channel-first):
 *   input:  [in_channels][in_len]
 *   weight: [out_channels][in_channels][kernel_size]
 *   output: [out_channels][out_len]
 *
 * Uses SIMD (f32x4) for the inner loop.
 */

#include "conv.h"
#include "simd.h"
#include <string.h>

void fe_conv1d(const float* input, const float* weight, const float* bias,
               float* output, int in_len, int in_channels, int out_channels,
               int kernel_size, int stride, int padding) {
    if (stride <= 0) return;
    int out_len = (in_len + 2 * padding - kernel_size) / stride + 1;
    if (out_len <= 0) return;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int p = 0; p < out_len; p++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int in_pos = p * stride + k - padding;
                    if (in_pos >= 0 && in_pos < in_len) {
                        int w_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                        sum += weight[w_idx] * input[ic * in_len + in_pos];
                    }
                }
            }
            output[oc * out_len + p] = sum;
        }
    }
}

int fe_strided_conv1d(const float* input, const float* weight, const float* bias,
                      float* output, int in_len, int in_channels, int out_channels,
                      int kernel_size, int stride) {
    if (stride <= 0) return 0;
    int out_len = (in_len - kernel_size) / stride + 1;
    if (out_len <= 0) return 0;
    fe_conv1d(input, weight, bias, output,
              in_len, in_channels, out_channels, kernel_size, stride, 0);
    return out_len;
}

int fe_conv_transpose1d(const float* input, const float* weight, const float* bias,
                        float* output, int in_len, int in_channels, int out_channels,
                        int kernel_size, int stride, int padding) {
    if (stride <= 0) return 0;
    int full_len = (in_len - 1) * stride + kernel_size;
    int out_len = full_len - 2 * padding;
    if (out_len <= 0) return 0;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int p = 0; p < out_len; p++) {
            output[oc * out_len + p] = bias[oc];
        }
    }

    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < in_len; i++) {
                float in_val = input[ic * in_len + i];
                int full_start = i * stride;
                for (int k = 0; k < kernel_size; k++) {
                    int full_pos = full_start + k;
                    int out_pos = full_pos - padding;
                    if (out_pos >= 0 && out_pos < out_len) {
                        int w_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
                        output[oc * out_len + out_pos] += weight[w_idx] * in_val;
                    }
                }
            }
        }
    }

    return out_len;
}

/* k=1, stride=1, padding=0 specialization: matrix-vector product per output position
 * Called by DecBlock's 1x1 conv (2C1->C1). The biggest bottleneck.
 * Kernel loop / bounds check / padding all unnecessary. */
static void conv1d_bn_k1(const float* input, const float* weight,
                          float scale, float bias,
                          float* output, int len, int in_channels,
                          int oc_offset) {
    int len4 = len & ~3;
    f32x4 v_scale = f32x4_splat(scale);
    f32x4 v_bias  = f32x4_splat(bias);
    const float* w_oc = weight + oc_offset;

    for (int p = 0; p < len4; p += 4) {
        f32x4 acc = f32x4_splat(0.0f);
        for (int ic = 0; ic < in_channels; ic++) {
            f32x4 vin = f32x4_load(input + ic * len + p);
            f32x4 vw  = f32x4_splat(w_oc[ic]);
            acc = f32x4_fma(vw, vin, acc);
        }
        f32x4_store(output + p, f32x4_add(f32x4_mul(acc, v_scale), v_bias));
    }
    for (int p = len4; p < len; p++) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++)
            sum += w_oc[ic] * input[ic * len + p];
        output[p] = sum * scale + bias;
    }
}

/* stride=1 generic SIMD path (k>=2, with padding)
 *
 * Proof of boundary safety for SIMD 4-element loads:
 *   safe_start = padding
 *   safe_end   = out_len - (kernel_size - 1 - padding)
 *   safe4      = safe_start + ((safe_end - safe_start) & ~3)
 *   For SIMD range p in [safe_start, safe4), f32x4_load(in_ic + p + k - padding)
 *   Max access position = (safe4-1) + (kernel_size-1) - padding + 3
 *                       <= (safe_end-1) + (kernel_size-1) - padding + 3
 *                        = out_len - 1 + 3 = in_len + 2 (when padding=0)
 *   When padding>0, it stays within in_len - 1.
 *   WASM linear memory is contiguous, so +2 over-read is safe.
 */
static void conv1d_bn_generic_s1(const float* input, const float* w_oc,
                                  float scale, float bias,
                                  float* output, int in_len, int in_channels,
                                  int out_len, int kernel_size, int padding) {
    int safe_start = padding;
    int safe_end = out_len - (kernel_size - 1 - padding);
    if (safe_end < safe_start) safe_end = safe_start;
    int safe4 = safe_start + ((safe_end - safe_start) & ~3);
    f32x4 v_scale = f32x4_splat(scale);
    f32x4 v_bias  = f32x4_splat(bias);

    for (int p = 0; p < safe_start; p++) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                int in_pos = p + k - padding;
                if (in_pos >= 0 && in_pos < in_len)
                    sum += w_oc[ic * kernel_size + k] * input[ic * in_len + in_pos];
            }
        }
        output[p] = sum * scale + bias;
    }

    for (int p = safe_start; p < safe4; p += 4) {
        f32x4 acc = f32x4_splat(0.0f);
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_ic = input + ic * in_len;
            const float* w_ic  = w_oc  + ic * kernel_size;
            for (int k = 0; k < kernel_size; k++) {
                f32x4 vin = f32x4_load(in_ic + p + k - padding);
                f32x4 vw  = f32x4_splat(w_ic[k]);
                acc = f32x4_fma(vw, vin, acc);
            }
        }
        f32x4_store(output + p, f32x4_add(f32x4_mul(acc, v_scale), v_bias));
    }

    for (int p = safe4; p < out_len; p++) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                int in_pos = p + k - padding;
                if (in_pos >= 0 && in_pos < in_len)
                    sum += w_oc[ic * kernel_size + k] * input[ic * in_len + in_pos];
            }
        }
        output[p] = sum * scale + bias;
    }
}

/* stride>1 SIMD path: SIMD within output channel (EncPreNet k=8, s=4, etc.) */
static void conv1d_bn_strided(const float* input, const float* w_oc,
                               float scale, float bias,
                               float* output, int in_len, int in_channels,
                               int out_len, int kernel_size, int stride,
                               int padding) {
    for (int p = 0; p < out_len; p++) {
        int base = p * stride - padding;
        float sum = 0.0f;

        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_ic = input + ic * in_len;
            const float* w_ic  = w_oc  + ic * kernel_size;

            int k_start = (base < 0) ? -base : 0;
            int k_end   = kernel_size;
            if (base + k_end > in_len) k_end = in_len - base;

            int k = k_start;
            int k4 = k_start + ((k_end - k_start) & ~3);

            f32x4 acc4 = f32x4_splat(0.0f);
            for (; k < k4; k += 4) {
                f32x4 vin = f32x4_load(in_ic + base + k);
                f32x4 vw  = f32x4_load(w_ic + k);
                acc4 = f32x4_fma(vw, vin, acc4);
            }
            float partial[4];
            f32x4_store(partial, acc4);
            sum += partial[0] + partial[1] + partial[2] + partial[3];

            for (; k < k_end; k++)
                sum += w_ic[k] * in_ic[base + k];
        }
        output[p] = sum * scale + bias;
    }
}

void fe_conv1d_bn(const float* input, const float* weight,
                  const float* bn_scale, const float* bn_bias,
                  float* output, int in_len, int in_channels, int out_channels,
                  int kernel_size, int stride, int padding) {
    if (stride <= 0) return;
    int out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    for (int oc = 0; oc < out_channels; oc++) {
        float scale = bn_scale[oc];
        float bias  = bn_bias[oc];
        float* out_oc = output + oc * out_len;

        if (kernel_size == 1 && stride == 1 && padding == 0) {
            conv1d_bn_k1(input, weight, scale, bias, out_oc,
                         out_len, in_channels, oc * in_channels);
        } else if (stride == 1) {
            const float* w_oc = weight + oc * in_channels * kernel_size;
            conv1d_bn_generic_s1(input, w_oc,
                                 scale, bias, out_oc,
                                 in_len, in_channels, out_len,
                                 kernel_size, padding);
        } else {
            const float* w_oc = weight + oc * in_channels * kernel_size;
            conv1d_bn_strided(input, w_oc,
                              scale, bias, out_oc,
                              in_len, in_channels, out_len,
                              kernel_size, stride, padding);
        }
    }
}
