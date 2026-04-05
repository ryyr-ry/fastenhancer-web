/*
 * gru.c — Implementation of GRU (Gated Recurrent Unit)
 *
 * PyTorch-compatible separated bias scheme:
 *   z = sigmoid(W_z*x + U_z*h + b_z)
 *   r = sigmoid(W_r*x + U_r*h + b_r)
 *   n = tanh(W_n*x + b_in_n + r*(U_n*h + b_hn_n))
 *   h = (1-z)*n + z*h_prev
 *
 * b_in_n is outside the r gate, b_hn_n is inside the r gate.
 * Equivalent to PyTorch GRU (mode='linear_after_reset').
 *
 * Uses SIMD (f32x4) for matvec operations.
 */

#include "gru.h"
#include "simd.h"
#include "activations.h"
#include <math.h>
#include <string.h>

void fe_gru_step(const FeGruWeights* w, const float* input, float* hidden) {
    int hs = w->hidden_size;
    int is = w->input_size;

    if (hs <= 0 || hs > FE_GRU_MAX_HIDDEN || is <= 0) return;

    float z[FE_GRU_MAX_HIDDEN], r[FE_GRU_MAX_HIDDEN];
    float n[FE_GRU_MAX_HIDDEN], Uh[FE_GRU_MAX_HIDDEN];

    /* z = W_z·x + U_z·h + b_z */
    memcpy(z, w->b_z, sizeof(float) * hs);
    fe_matvec_add(w->W_z, input, z, hs, is);
    fe_matvec_add(w->U_z, hidden, z, hs, hs);
    {
        const int hs4 = hs & ~3;
        for (int i = 0; i < hs4; i += 4)
            f32x4_store(z + i, f32x4_fast_sigmoid(f32x4_load(z + i)));
        for (int i = hs4; i < hs; i++) z[i] = fe_sigmoid(z[i]);
    }

    /* r = W_r·x + U_r·h + b_r */
    memcpy(r, w->b_r, sizeof(float) * hs);
    fe_matvec_add(w->W_r, input, r, hs, is);
    fe_matvec_add(w->U_r, hidden, r, hs, hs);
    {
        const int hs4 = hs & ~3;
        for (int i = 0; i < hs4; i += 4)
            f32x4_store(r + i, f32x4_fast_sigmoid(f32x4_load(r + i)));
        for (int i = hs4; i < hs; i++) r[i] = fe_sigmoid(r[i]);
    }

    /* n = tanh(W_n·x + b_in_n + r * (U_n·h + b_hn_n)) */
    memcpy(Uh, w->b_hn_n, sizeof(float) * hs);
    fe_matvec_add(w->U_n, hidden, Uh, hs, hs);
    {
        const int hs4 = hs & ~3;
        for (int i = 0; i < hs4; i += 4)
            f32x4_store(Uh + i, f32x4_mul(f32x4_load(Uh + i), f32x4_load(r + i)));
        for (int i = hs4; i < hs; i++) Uh[i] *= r[i];
    }

    memcpy(n, w->b_in_n, sizeof(float) * hs);
    fe_matvec_add(w->W_n, input, n, hs, is);
    {
        const int hs4 = hs & ~3;
        for (int i = 0; i < hs4; i += 4) {
            f32x4 vn = f32x4_load(n + i);
            f32x4 vuh = f32x4_load(Uh + i);
            f32x4_store(n + i, f32x4_fast_tanh(f32x4_add(vn, vuh)));
        }
        for (int i = hs4; i < hs; i++) n[i] = tanhf(n[i] + Uh[i]);
    }

    /* h = (1-z)*n + z*h_prev — SIMD vectorized */
    {
        const int hs4 = hs & ~3;
        const f32x4 ones = f32x4_splat(1.0f);
        for (int i = 0; i < hs4; i += 4) {
            f32x4 vz = f32x4_load(z + i);
            f32x4 vn = f32x4_load(n + i);
            f32x4 vh = f32x4_load(hidden + i);
            f32x4 result = f32x4_add(
                f32x4_mul(f32x4_sub(ones, vz), vn),
                f32x4_mul(vz, vh)
            );
            f32x4_store(hidden + i, result);
        }
        for (int i = hs4; i < hs; i++) {
            hidden[i] = (1.0f - z[i]) * n[i] + z[i] * hidden[i];
        }
    }
}

void fe_gru_reset_hidden(float* hidden, int size) {
    if (size <= 0) return;
    memset(hidden, 0, sizeof(float) * (size_t)size);
}
