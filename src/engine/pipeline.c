/*
 * pipeline.c — HPF/AGC 前処理パイプライン
 */

#include "pipeline.h"
#include "simd.h"
#include <math.h>

#ifdef FE_USE_BASE_48K
#include "base_48k.h"
#elif defined(FE_USE_SMALL_48K)
#include "small_48k.h"
#else
#include "tiny_48k.h"
#endif

#define FE_PI_F               3.14159265358979323846f
#define FE_HPF_CUTOFF_HZ      80.0f
#define FE_HPF_Q              0.7071067811865476f

#define FE_AGC_TARGET_RMS     0.1f
#define FE_AGC_ENV_SMOOTH     0.9f
#define FE_AGC_GAIN_SMOOTH    0.2f
#define FE_AGC_MIN_GAIN       0.1f
#define FE_AGC_MAX_GAIN       8.0f
#define FE_AGC_EPSILON        1e-8f

static float fe_clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void fe_hpf_init(FeHpfState* s) {
    if (!s) return;

    {
        const float omega = 2.0f * FE_PI_F * FE_HPF_CUTOFF_HZ / FE_SAMPLE_RATE;
        const float cos_omega = cosf(omega);
        const float sin_omega = sinf(omega);
        const float alpha = sin_omega / (2.0f * FE_HPF_Q);
        const float a0 = 1.0f + alpha;

        s->b0 = (0.5f * (1.0f + cos_omega)) / a0;
        s->b1 = (-(1.0f + cos_omega)) / a0;
        s->b2 = s->b0;
        s->a1 = (-2.0f * cos_omega) / a0;
        s->a2 = (1.0f - alpha) / a0;
    }

    s->z1 = 0.0f;
    s->z2 = 0.0f;
}

void fe_hpf_process(FeHpfState* s, float* buf, int len) {
    if (!s || !buf || len <= 0) return;

    for (int i = 0; i < len; i++) {
        float x = buf[i];
        if (fe_is_nan_bits(x) || x > 1e15f || x < -1e15f) x = 0.0f;
        const float y = s->b0 * x + s->z1;

        s->z1 = s->b1 * x - s->a1 * y + s->z2;
        s->z2 = s->b2 * x - s->a2 * y;

        if (fe_is_nan_bits(s->z1)) s->z1 = 0.0f;
        if (fe_is_nan_bits(s->z2)) s->z2 = 0.0f;

        buf[i] = y;
    }
}

void fe_hpf_reset(FeHpfState* s) {
    if (!s) return;
    s->z1 = 0.0f;
    s->z2 = 0.0f;
}

void fe_agc_init(FeAgcState* s) {
    if (!s) return;
    s->env = FE_AGC_EPSILON;
    s->gain = 1.0f;
}

void fe_agc_process(FeAgcState* s, float* buf, int len) {
    if (!s || !buf || len <= 0) return;

    float energy = 0.0f;
    for (int i = 0; i < len; i++) {
        float v = buf[i];
        if (fe_is_nan_bits(v) || v > 1e15f || v < -1e15f) { buf[i] = 0.0f; v = 0.0f; }
        energy += v * v;
    }

    {
        const float rms = sqrtf(energy / (float)len);

        s->env = FE_AGC_ENV_SMOOTH * s->env + (1.0f - FE_AGC_ENV_SMOOTH) * rms;
        if (fe_is_nan_bits(s->env)) s->env = 0.0f;

        const float desired_gain = fe_clampf(
            FE_AGC_TARGET_RMS / (s->env > FE_AGC_EPSILON ? s->env : FE_AGC_EPSILON),
            FE_AGC_MIN_GAIN,
            FE_AGC_MAX_GAIN
        );

        s->gain = FE_AGC_GAIN_SMOOTH * s->gain + (1.0f - FE_AGC_GAIN_SMOOTH) * desired_gain;
        if (fe_is_nan_bits(s->gain)) s->gain = 1.0f;
    }

    for (int i = 0; i < len; i++) {
        buf[i] *= s->gain;
        buf[i] = fe_clampf(buf[i], -1.0f, 1.0f);
    }
}

void fe_agc_reset(FeAgcState* s) {
    if (!s) return;
    s->env = FE_AGC_EPSILON;
    s->gain = 1.0f;
}
