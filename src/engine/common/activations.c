/*
 * activations.c — 活性化関数の実装
 *
 * sigmoid: expf()ベースの正確な実装 (NaN/Inf安全)
 * SiLU: x * sigmoid(x)
 *
 * SIMD最適化: 多項式exp近似を使用したバッチ版
 */

#include "activations.h"
#include "simd.h"
#include <math.h>
#include <stdint.h>

/* -ffast-math でも正しく動作するビット操作ベースの NaN/Inf チェック */
static inline int fe_act_is_finite(float x) {
    union { float f; uint32_t u; } conv;
    conv.f = x;
    return (conv.u & 0x7F800000u) != 0x7F800000u;
}

static inline int fe_act_is_nan(float x) {
    union { float f; uint32_t u; } conv;
    conv.f = x;
    uint32_t exp_bits = conv.u & 0x7F800000u;
    uint32_t frac_bits = conv.u & 0x007FFFFFu;
    return (exp_bits == 0x7F800000u) && (frac_bits != 0);
}

/* ビット操作で符号判定（-ffast-math安全） */
static inline int fe_act_is_negative(float x) {
    union { float f; uint32_t u; } conv;
    conv.f = x;
    return (conv.u >> 31) != 0;
}

/*
 * sigmoid(x) = 1 / (1 + exp(-x))
 * 数値安全: x >= 0 のとき 1/(1+exp(-x))、x < 0 のとき exp(x)/(1+exp(x))
 */
float fe_sigmoid(float x) {
    if (fe_act_is_nan(x)) return 0.5f;
    if (!fe_act_is_finite(x)) return fe_act_is_negative(x) ? 0.0f : 1.0f;
    if (x >= FE_SIGMOID_CLAMP) return 1.0f;
    if (x <= -FE_SIGMOID_CLAMP) return 0.0f;

    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

float fe_silu(float x) {
    if (fe_act_is_nan(x)) return 0.0f;
    /* +Inf → silu(+Inf) = +Inf * 1 = FLT_MAX, -Inf → silu(-Inf) = -Inf * 0 = 0 */
    if (!fe_act_is_finite(x)) return fe_act_is_negative(x) ? 0.0f : 3.4028235e+38f;
    if (x <= -FE_EXP_OVERFLOW) return 0.0f;
    return x * fe_sigmoid(x);
}

void fe_sigmoid_batch(const float* input, float* output, int n) {
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        f32x4 vx = f32x4_load(input + i);
        f32x4_store(output + i, f32x4_fast_sigmoid(vx));
    }
    for (int i = n4; i < n; i++) {
        output[i] = fe_sigmoid(input[i]);
    }
}

void fe_silu_batch(const float* input, float* output, int n) {
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        f32x4 vx = f32x4_load(input + i);
        f32x4 vsig = f32x4_fast_sigmoid(vx);
        f32x4_store(output + i, f32x4_mul(vx, vsig));
    }
    for (int i = n4; i < n; i++) {
        output[i] = fe_silu(input[i]);
    }
    /* SIMD fast pathはNaN/Inf入力を扱えないため、
       該当要素のみスカラーfe_silu()で再計算する */
    for (int i = 0; i < n4; i++) {
        if (fe_act_is_nan(input[i]) || !fe_act_is_finite(input[i])) {
            output[i] = fe_silu(input[i]);
        }
    }
}
