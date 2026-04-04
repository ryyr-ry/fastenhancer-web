/*
 * attention.c — Implementation of Multi-Head Self-Attention
 *
 * Q = input × W_q + b_q
 * K = input × W_k + b_k
 * V = input × W_v + b_v
 * scores[h] = Q_h × K_h^T / sqrt(head_dim)
 * attn = softmax(scores)
 * out = concat(attn × V_h) × W_o + b_o
 *
 * Uses SIMD (f32x4) for linear_transform and dot products.
 */

#include "attention.h"
#include "simd.h"
#include <math.h>
#include <string.h>
#include <float.h>

void fe_softmax(const float* input, float* output, int n) {
    if (n <= 0) return;

    /* Pass 1: Find maximum only */
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    /* For NaN input: max_val becomes NaN -> fall back to uniform distribution */
    if (fe_is_nan_bits(max_val)) {
        float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; i++) output[i] = uniform;
        return;
    }

    /* For +Inf input: max_val=+Inf -> distribute equally among +Inf positions */
    if (fe_is_pos_inf(max_val)) {
        int inf_count = 0;
        for (int i = 0; i < n; i++) {
            if (fe_is_pos_inf(input[i])) inf_count++;
        }
        float share = 1.0f / (float)inf_count;
        for (int i = 0; i < n; i++) {
            output[i] = fe_is_pos_inf(input[i]) ? share : 0.0f;
        }
        return;
    }

    /* Normal path: SIMD fast exp */
    const int n4 = n & ~3;
    f32x4 vmax = f32x4_splat(max_val);
    f32x4 vsum = f32x4_splat(0.0f);
    float sum = 0.0f;

    int i = 0;
    for (; i < n4; i += 4) {
        f32x4 vx = f32x4_load(input + i);
        f32x4 ve = f32x4_fast_exp(f32x4_sub(vx, vmax));
        f32x4_store(output + i, ve);
        vsum = f32x4_add(vsum, ve);
    }
    sum = f32x4_hsum(vsum);
    for (; i < n; i++) {
        float e = fe_fast_expf(input[i] - max_val);
        output[i] = e;
        sum += e;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        f32x4 vinv = f32x4_splat(inv_sum);
        int j = 0;
        for (; j + 3 < n; j += 4) {
            f32x4_store(output + j, f32x4_mul(f32x4_load(output + j), vinv));
        }
        for (; j < n; j++) {
            output[j] *= inv_sum;
        }
    } else {
        float uniform = 1.0f / (float)n;
        for (int j = 0; j < n; j++) output[j] = uniform;
    }
}

/* Matrix x matrix: output[s][j] = sum_k input[s][k] * weight[j][k] + bias[j]
 * Weight layout: [out_dim x in_dim] (PyTorch nn.Linear compatible)
 * Uses fe_matvec_add for the inner loop. */
static void linear_transform(const float* input, const float* weight,
                              const float* bias, float* output,
                              int seq_len, int in_dim, int out_dim) {
    for (int s = 0; s < seq_len; s++) {
        for (int j = 0; j < out_dim; j++) {
            output[s * out_dim + j] = bias[j];
        }
        fe_matvec_add(weight, input + s * in_dim,
                      output + s * out_dim, out_dim, in_dim);
    }
}

void fe_mhsa(const FeMhsaWeights* w, const float* input,
             float* output, float* attn_buf, float* scratch, int seq_len) {
    int c2 = w->c2;
    int n_heads = w->n_heads;
    int head_dim = w->head_dim;

    if (seq_len <= 0 || c2 <= 0 || n_heads <= 0 || head_dim <= 0) return;
    if (c2 != n_heads * head_dim) return;

    float scale = 1.0f / sqrtf((float)head_dim);

    /* scratch layout: Q[seq*c2] | K[seq*c2] | V[seq*c2] | attn_out[seq*c2] */
    float* Q = scratch;
    float* K = scratch + seq_len * c2;
    float* V = scratch + 2 * seq_len * c2;
    float* attn_out = scratch + 3 * seq_len * c2;

    linear_transform(input, w->W_q, w->b_q, Q, seq_len, c2, c2);
    linear_transform(input, w->W_k, w->b_k, K, seq_len, c2, c2);
    linear_transform(input, w->W_v, w->b_v, V, seq_len, c2, c2);

    memset(attn_out, 0, sizeof(float) * seq_len * c2);

    for (int h = 0; h < n_heads; h++) {
        int head_offset = h * head_dim;

        /* scores[i][j] = Q_h[i] . K_h[j] / sqrt(d) — SIMD vectorized */
        for (int i = 0; i < seq_len; i++) {
            const float* qi = Q + i * c2 + head_offset;
            for (int j = 0; j < seq_len; j++) {
                const float* kj = K + j * c2 + head_offset;
                const int hd4 = head_dim & ~3;
                f32x4 acc = f32x4_splat(0.0f);
                for (int d = 0; d < hd4; d += 4) {
                    acc = f32x4_fma(f32x4_load(qi + d), f32x4_load(kj + d), acc);
                }
                float dot = f32x4_hsum(acc);
                for (int d = hd4; d < head_dim; d++) {
                    dot += qi[d] * kj[d];
                }
                attn_buf[h * seq_len * seq_len + i * seq_len + j] = dot * scale;
            }
        }

        /* softmax (row-wise, in-place) */
        for (int i = 0; i < seq_len; i++) {
            float* row = &attn_buf[h * seq_len * seq_len + i * seq_len];
            fe_softmax(row, row, seq_len);
        }

        /* attn_out_h[i][d] = sum_j attn[i][j] * V_h[j][d] — SIMD vectorized */
        for (int i = 0; i < seq_len; i++) {
            const float* attn_row = &attn_buf[h * seq_len * seq_len + i * seq_len];
            float* out_row = attn_out + i * c2 + head_offset;
            for (int d = 0; d < head_dim; d++) {
                out_row[d] = 0.0f;
            }
            for (int j = 0; j < seq_len; j++) {
                const f32x4 vw = f32x4_splat(attn_row[j]);
                const float* vj = V + j * c2 + head_offset;
                const int hd4 = head_dim & ~3;
                for (int d = 0; d < hd4; d += 4) {
                    f32x4_store(out_row + d,
                        f32x4_fma(vw, f32x4_load(vj + d), f32x4_load(out_row + d)));
                }
                for (int d = hd4; d < head_dim; d++) {
                    out_row[d] += attn_row[j] * vj[d];
                }
            }
        }
    }

    /* Output projection: output = attn_out x W_o + b_o */
    linear_transform(attn_out, w->W_o, w->b_o, output, seq_len, c2, c2);
}
