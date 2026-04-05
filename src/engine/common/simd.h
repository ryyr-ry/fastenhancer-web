/*
 * simd.h — SIMD primitive abstraction layer
 *
 * Three backends:
 *   1. WASM SIMD128 (__wasm_simd128__)
 *   2. x86 SSE2 (__SSE2__) — for native benchmarks
 *   3. Scalar fallback
 * All functions are defined as static inline (header-only).
 */

#ifndef FE_SIMD_H
#define FE_SIMD_H

#include <math.h>
#include <stdint.h>

/* -ffast-math safe Infinity constant (INFINITY macro is UB with -ffinite-math-only) */
static inline float fe_inf_val(void) {
    union { uint32_t u; float f; } c = {0x7F800000u};
    return c.f;
}
static inline int fe_is_inf(float x) {
    union { float f; uint32_t u; } c;
    c.f = x;
    return (c.u & 0x7FFFFFFFu) == 0x7F800000u;
}
static inline int fe_is_nan_bits(float x) {
    union { float f; uint32_t u; } c;
    c.f = x;
    return (c.u & 0x7F800000u) == 0x7F800000u && (c.u & 0x007FFFFFu) != 0;
}
static inline int fe_is_pos_inf(float x) {
    union { float f; uint32_t u; } c;
    c.f = x;
    return c.u == 0x7F800000u;
}

#ifdef __wasm_simd128__
#include <wasm_simd128.h>

typedef v128_t f32x4;

static inline f32x4 f32x4_splat(float v)       { return wasm_f32x4_splat(v); }
static inline f32x4 f32x4_load(const float* p)  { return wasm_v128_load(p); }
static inline void  f32x4_store(float* p, f32x4 v) { wasm_v128_store(p, v); }
static inline f32x4 f32x4_add(f32x4 a, f32x4 b) { return wasm_f32x4_add(a, b); }
static inline f32x4 f32x4_sub(f32x4 a, f32x4 b) { return wasm_f32x4_sub(a, b); }
static inline f32x4 f32x4_mul(f32x4 a, f32x4 b) { return wasm_f32x4_mul(a, b); }
static inline f32x4 f32x4_neg(f32x4 a)          { return wasm_f32x4_neg(a); }
static inline f32x4 f32x4_abs(f32x4 a)          { return wasm_f32x4_abs(a); }
static inline f32x4 f32x4_max(f32x4 a, f32x4 b) { return wasm_f32x4_max(a, b); }
static inline f32x4 f32x4_min(f32x4 a, f32x4 b) { return wasm_f32x4_min(a, b); }
static inline f32x4 f32x4_div(f32x4 a, f32x4 b) { return wasm_f32x4_div(a, b); }

#ifdef __wasm_relaxed_simd__
static inline f32x4 f32x4_fma(f32x4 a, f32x4 b, f32x4 c) {
    return wasm_f32x4_relaxed_madd(a, b, c);
}
#else
static inline f32x4 f32x4_fma(f32x4 a, f32x4 b, f32x4 c) {
    return wasm_f32x4_add(wasm_f32x4_mul(a, b), c);
}
#endif

static inline float f32x4_extract0(f32x4 v)     { return wasm_f32x4_extract_lane(v, 0); }
static inline float f32x4_extract1(f32x4 v)     { return wasm_f32x4_extract_lane(v, 1); }
static inline float f32x4_extract2(f32x4 v)     { return wasm_f32x4_extract_lane(v, 2); }
static inline float f32x4_extract3(f32x4 v)     { return wasm_f32x4_extract_lane(v, 3); }

#elif defined(__SSE2__)
#include <immintrin.h>

typedef __m128 f32x4;

static inline f32x4 f32x4_splat(float v)        { return _mm_set1_ps(v); }
static inline f32x4 f32x4_load(const float* p)   { return _mm_loadu_ps(p); }
static inline void  f32x4_store(float* p, f32x4 v) { _mm_storeu_ps(p, v); }
static inline f32x4 f32x4_add(f32x4 a, f32x4 b) { return _mm_add_ps(a, b); }
static inline f32x4 f32x4_sub(f32x4 a, f32x4 b) { return _mm_sub_ps(a, b); }
static inline f32x4 f32x4_mul(f32x4 a, f32x4 b) { return _mm_mul_ps(a, b); }
static inline f32x4 f32x4_neg(f32x4 a)          { return _mm_sub_ps(_mm_setzero_ps(), a); }
static inline f32x4 f32x4_abs(f32x4 a)          { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a); }
static inline f32x4 f32x4_max(f32x4 a, f32x4 b) { return _mm_max_ps(a, b); }
static inline f32x4 f32x4_min(f32x4 a, f32x4 b) { return _mm_min_ps(a, b); }
static inline f32x4 f32x4_div(f32x4 a, f32x4 b) { return _mm_div_ps(a, b); }

#ifdef __FMA__
static inline f32x4 f32x4_fma(f32x4 a, f32x4 b, f32x4 c) {
    return _mm_fmadd_ps(a, b, c);
}
#else
static inline f32x4 f32x4_fma(f32x4 a, f32x4 b, f32x4 c) {
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}
#endif

static inline float f32x4_extract0(f32x4 v)     { return _mm_cvtss_f32(v); }
static inline float f32x4_extract1(f32x4 v)     { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1))); }
static inline float f32x4_extract2(f32x4 v)     { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2))); }
static inline float f32x4_extract3(f32x4 v)     { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3))); }

#else /* Scalar fallback */

typedef struct { float v[4]; } f32x4;

static inline f32x4 f32x4_splat(float val) {
    f32x4 r = {{val, val, val, val}};
    return r;
}
static inline f32x4 f32x4_load(const float* p) {
    f32x4 r = {{p[0], p[1], p[2], p[3]}};
    return r;
}
static inline void f32x4_store(float* p, f32x4 v) {
    p[0] = v.v[0]; p[1] = v.v[1]; p[2] = v.v[2]; p[3] = v.v[3];
}
static inline f32x4 f32x4_add(f32x4 a, f32x4 b) {
    f32x4 r = {{a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2], a.v[3]+b.v[3]}};
    return r;
}
static inline f32x4 f32x4_sub(f32x4 a, f32x4 b) {
    f32x4 r = {{a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2], a.v[3]-b.v[3]}};
    return r;
}
static inline f32x4 f32x4_mul(f32x4 a, f32x4 b) {
    f32x4 r = {{a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2], a.v[3]*b.v[3]}};
    return r;
}
static inline f32x4 f32x4_neg(f32x4 a) {
    f32x4 r = {{-a.v[0], -a.v[1], -a.v[2], -a.v[3]}};
    return r;
}
static inline f32x4 f32x4_abs(f32x4 a) {
    f32x4 r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] < 0 ? -a.v[i] : a.v[i];
    return r;
}
static inline f32x4 f32x4_max(f32x4 a, f32x4 b) {
    f32x4 r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i];
    return r;
}
static inline f32x4 f32x4_min(f32x4 a, f32x4 b) {
    f32x4 r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i];
    return r;
}
static inline f32x4 f32x4_div(f32x4 a, f32x4 b) {
    f32x4 r = {{a.v[0]/b.v[0], a.v[1]/b.v[1], a.v[2]/b.v[2], a.v[3]/b.v[3]}};
    return r;
}
static inline f32x4 f32x4_fma(f32x4 a, f32x4 b, f32x4 c) {
    f32x4 r = {{a.v[0]*b.v[0]+c.v[0], a.v[1]*b.v[1]+c.v[1],
                a.v[2]*b.v[2]+c.v[2], a.v[3]*b.v[3]+c.v[3]}};
    return r;
}

static inline float f32x4_extract0(f32x4 v) { return v.v[0]; }
static inline float f32x4_extract1(f32x4 v) { return v.v[1]; }
static inline float f32x4_extract2(f32x4 v) { return v.v[2]; }
static inline float f32x4_extract3(f32x4 v) { return v.v[3]; }

#endif /* __wasm_simd128__ / __SSE2__ / scalar */

/* Horizontal sum: returns the sum of 4 elements */
static inline float f32x4_hsum(f32x4 v) {
    return f32x4_extract0(v) + f32x4_extract1(v) +
           f32x4_extract2(v) + f32x4_extract3(v);
}

/* Horizontal maximum */
static inline float f32x4_hmax(f32x4 v) {
    float a = f32x4_extract0(v), b = f32x4_extract1(v);
    float c = f32x4_extract2(v), d = f32x4_extract3(v);
    float ab = a > b ? a : b;
    float cd = c > d ? c : d;
    return ab > cd ? ab : cd;
}

/*
 * SIMD matrix-vector product: out[i] += sum_j(mat[i*cols+j] * vec[j])
 * Processes 4 columns at a time; remainder handled with scalar operations.
 */
static inline void fe_matvec_add(const float* mat, const float* vec,
                                  float* out, int rows, int cols) {
    const int cols4 = cols & ~3;
    for (int i = 0; i < rows; i++) {
        const float* row = mat + i * cols;
        f32x4 acc = f32x4_splat(0.0f);
        for (int j = 0; j < cols4; j += 4) {
            acc = f32x4_fma(f32x4_load(row + j), f32x4_load(vec + j), acc);
        }
        float sum = f32x4_hsum(acc);
        for (int j = cols4; j < cols; j++) {
            sum += row[j] * vec[j];
        }
        out[i] += sum;
    }
}

/* ---- Fast exp/sigmoid/tanh SIMD functions ---- */

/* Sigmoid saturation range: |x| >= 16 clamps to 0.0 or 1.0 (sufficient for float32 precision) */
#define FE_SIGMOID_CLAMP 16.0f

/* float32 exp overflow boundary: exp(88) ~ 1.65e38 ~ FLT_MAX */
#define FE_EXP_OVERFLOW  88.0f

/* 1/ln(2) — used for base conversion in exp approximation */
#define FE_LOG2E         1.4426950408889634f

/*
 * f32x4_fast_exp: Fast exp approximation for 4 elements simultaneously
 * 4th-order Remez minimax polynomial + IEEE 754 exponent field manipulation
 * Relative error < 2e-5 in [-10,10], clamped to [-88,88]
 */
#ifdef __wasm_simd128__

static inline f32x4 f32x4_fast_exp(f32x4 x) {
    x = wasm_f32x4_max(x, wasm_f32x4_splat(-FE_EXP_OVERFLOW));
    x = wasm_f32x4_min(x, wasm_f32x4_splat(FE_EXP_OVERFLOW));
    v128_t t = wasm_f32x4_mul(x, wasm_f32x4_splat(FE_LOG2E));
    v128_t nf = wasm_f32x4_floor(t);
    v128_t f = wasm_f32x4_sub(t, nf);
    v128_t p = wasm_f32x4_add(wasm_f32x4_splat(0.0554953f),
                               wasm_f32x4_mul(wasm_f32x4_splat(0.0096838f), f));
    p = wasm_f32x4_add(wasm_f32x4_splat(0.2402265f), wasm_f32x4_mul(p, f));
    p = wasm_f32x4_add(wasm_f32x4_splat(0.6931472f), wasm_f32x4_mul(p, f));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(p, f));
    v128_t ni = wasm_i32x4_trunc_sat_f32x4(nf);
    v128_t scale = wasm_i32x4_shl(wasm_i32x4_add(ni, wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(p, scale);
}

#elif defined(__SSE2__)

static inline f32x4 f32x4_fast_exp(f32x4 x) {
    x = _mm_max_ps(x, _mm_set1_ps(-FE_EXP_OVERFLOW));
    x = _mm_min_ps(x, _mm_set1_ps(FE_EXP_OVERFLOW));
    __m128 t = _mm_mul_ps(x, _mm_set1_ps(FE_LOG2E));
    /* floor: truncate + adjust for negative */
    __m128i ti = _mm_cvttps_epi32(t);
    __m128 tf = _mm_cvtepi32_ps(ti);
    __m128 adj = _mm_and_ps(_mm_cmpgt_ps(tf, t), _mm_set1_ps(1.0f));
    __m128 nf = _mm_sub_ps(tf, adj);
    __m128 f = _mm_sub_ps(t, nf);
    /* Horner: p = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4))) */
    __m128 p = _mm_add_ps(_mm_set1_ps(0.0554953f),
                           _mm_mul_ps(_mm_set1_ps(0.0096838f), f));
    p = _mm_add_ps(_mm_set1_ps(0.2402265f), _mm_mul_ps(p, f));
    p = _mm_add_ps(_mm_set1_ps(0.6931472f), _mm_mul_ps(p, f));
    p = _mm_add_ps(_mm_set1_ps(1.0f), _mm_mul_ps(p, f));
    /* 2^n via IEEE 754: (n+127)<<23 reinterpret as float */
    __m128i ni = _mm_cvttps_epi32(nf);
    __m128i scale = _mm_slli_epi32(_mm_add_epi32(ni, _mm_set1_epi32(127)), 23);
    return _mm_mul_ps(p, _mm_castsi128_ps(scale));
}

#else /* scalar */

static inline f32x4 f32x4_fast_exp(f32x4 x) {
    f32x4 r;
    for (int i = 0; i < 4; i++) {
        float xi = x.v[i];
        if (xi > FE_EXP_OVERFLOW) xi = FE_EXP_OVERFLOW;
        if (xi < -FE_EXP_OVERFLOW) xi = -FE_EXP_OVERFLOW;
        float t = xi * FE_LOG2E;
        float n_f = floorf(t);
        float f = t - n_f;
        float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0554953f + f * 0.0096838f)));
        union { float fv; int iv; } scale;
        scale.iv = ((int)n_f + 127) << 23;
        r.v[i] = p * scale.fv;
    }
    return r;
}

#endif

/*
 * f32x4_fast_sigmoid: 1 / (1 + exp(-x))
 * Pure SIMD path — uses f32x4_div instead of extracting lanes to scalar.
 * Saturation clamping: x>=16 -> 1, x<=-16 -> 0 (via input clamping before exp)
 */
static inline f32x4 f32x4_fast_sigmoid(f32x4 x) {
    f32x4 clamped = f32x4_max(f32x4_min(x, f32x4_splat(FE_SIGMOID_CLAMP)),
                               f32x4_splat(-FE_SIGMOID_CLAMP));
    f32x4 e = f32x4_fast_exp(f32x4_neg(clamped));
    f32x4 one = f32x4_splat(1.0f);
    return f32x4_div(one, f32x4_add(one, e));
}

/*
 * f32x4_fast_tanh: tanh(x) = 2*sigmoid(2x) - 1
 */
static inline f32x4 f32x4_fast_tanh(f32x4 x) {
    f32x4 two_x = f32x4_add(x, x);
    f32x4 sig = f32x4_fast_sigmoid(two_x);
    return f32x4_sub(f32x4_add(sig, sig), f32x4_splat(1.0f));
}

/* Scalar fast exp approximation (same algorithm as f32x4_fast_exp) */
static inline float fe_fast_expf(float x) {
    if (x != x) return 0.0f; /* NaN guard: NaN→int is UB */
    if (x < -FE_EXP_OVERFLOW) return 0.0f;
    if (x >  FE_EXP_OVERFLOW) return fe_inf_val();
    float t = x * FE_LOG2E;
    float n = floorf(t);
    float f = t - n;
    float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0554953f + f * 0.0096838f)));
    union { float fv; int iv; } u;
    u.iv = ((int)n + 127) << 23;
    return p * u.fv;
}

/*
 * f32x4_fast_pow_const: Fast x^e for 4 positive floats with constant exponent.
 * Uses IEEE 754 decomposition + Taylor ln(1+m) + fast_exp:
 *   x^e = exp(e * ln(x)) = exp(e * (E*ln(2) + ln(1+m)))
 * Range reduction via sqrt(2) threshold keeps m ∈ [-0.293, 0.414].
 * Max relative error ~0.02% (Taylor 5-term + fast_exp combined).
 */
#ifdef __wasm_simd128__

static inline f32x4 f32x4_fast_pow_const(f32x4 x, float e) {
    v128_t xi = x;
    v128_t e_raw = wasm_u32x4_shr(xi, 23);
    v128_t mantissa = wasm_v128_or(
        wasm_v128_and(xi, wasm_i32x4_splat(0x007FFFFF)),
        wasm_i32x4_splat(0x3F800000));
    v128_t f = mantissa;
    v128_t cmp = wasm_f32x4_ge(f, wasm_f32x4_splat(1.4142135f));
    f = wasm_v128_bitselect(wasm_f32x4_mul(f, wasm_f32x4_splat(0.5f)), f, cmp);
    e_raw = wasm_i32x4_add(e_raw, wasm_v128_and(cmp, wasm_i32x4_splat(1)));
    v128_t ef = wasm_f32x4_convert_i32x4(wasm_i32x4_sub(e_raw, wasm_i32x4_splat(127)));
    v128_t m = wasm_f32x4_sub(f, wasm_f32x4_splat(1.0f));
    v128_t p = wasm_f32x4_splat(0.2f);
    p = wasm_f32x4_add(wasm_f32x4_splat(-0.25f), wasm_f32x4_mul(p, m));
    p = wasm_f32x4_add(wasm_f32x4_splat(0.33333333f), wasm_f32x4_mul(p, m));
    p = wasm_f32x4_add(wasm_f32x4_splat(-0.5f), wasm_f32x4_mul(p, m));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(p, m));
    v128_t ln_x = wasm_f32x4_add(
        wasm_f32x4_mul(ef, wasm_f32x4_splat(0.69314718f)),
        wasm_f32x4_mul(p, m));
    return f32x4_fast_exp(wasm_f32x4_mul(wasm_f32x4_splat(e), ln_x));
}

#elif defined(__SSE2__)

static inline f32x4 f32x4_fast_pow_const(f32x4 x, float e) {
    __m128i xi = _mm_castps_si128(x);
    __m128i e_raw = _mm_srli_epi32(xi, 23);
    __m128i mantissa = _mm_or_si128(
        _mm_and_si128(xi, _mm_set1_epi32(0x007FFFFF)),
        _mm_set1_epi32(0x3F800000));
    __m128 f = _mm_castsi128_ps(mantissa);
    __m128 cmp = _mm_cmpge_ps(f, _mm_set1_ps(1.4142135f));
    f = _mm_or_ps(_mm_and_ps(cmp, _mm_mul_ps(f, _mm_set1_ps(0.5f))),
                  _mm_andnot_ps(cmp, f));
    __m128i adj = _mm_and_si128(_mm_castps_si128(cmp), _mm_set1_epi32(1));
    e_raw = _mm_add_epi32(e_raw, adj);
    __m128 ef = _mm_cvtepi32_ps(_mm_sub_epi32(e_raw, _mm_set1_epi32(127)));
    __m128 m = _mm_sub_ps(f, _mm_set1_ps(1.0f));
    __m128 p = _mm_set1_ps(0.2f);
    p = _mm_add_ps(_mm_set1_ps(-0.25f), _mm_mul_ps(p, m));
    p = _mm_add_ps(_mm_set1_ps(0.33333333f), _mm_mul_ps(p, m));
    p = _mm_add_ps(_mm_set1_ps(-0.5f), _mm_mul_ps(p, m));
    p = _mm_add_ps(_mm_set1_ps(1.0f), _mm_mul_ps(p, m));
    __m128 ln_x = _mm_add_ps(
        _mm_mul_ps(ef, _mm_set1_ps(0.69314718f)),
        _mm_mul_ps(p, m));
    return f32x4_fast_exp(_mm_mul_ps(_mm_set1_ps(e), ln_x));
}

#else /* scalar */

static inline f32x4 f32x4_fast_pow_const(f32x4 x, float e) {
    f32x4 r;
    for (int i = 0; i < 4; i++) {
        float xi = x.v[i];
        if (xi <= 0.0f) { r.v[i] = 0.0f; continue; }
        union { float f; int32_t i; } u;
        u.f = xi;
        int e_raw = (u.i >> 23) & 0xFF;
        u.i = (u.i & 0x007FFFFF) | 0x3F800000;
        float fi = u.f;
        if (fi >= 1.4142135f) { fi *= 0.5f; e_raw += 1; }
        float m = fi - 1.0f;
        float p = 0.2f;
        p = -0.25f + p * m;
        p = 0.33333333f + p * m;
        p = -0.5f + p * m;
        p = 1.0f + p * m;
        float ln_x = (float)(e_raw - 127) * 0.69314718f + p * m;
        float arg = e * ln_x;
        if (arg > FE_EXP_OVERFLOW) arg = FE_EXP_OVERFLOW;
        if (arg < -FE_EXP_OVERFLOW) arg = -FE_EXP_OVERFLOW;
        float t = arg * FE_LOG2E;
        float nf = floorf(t);
        float frac = t - nf;
        float pp = 1.0f + frac * (0.6931472f + frac * (0.2402265f + frac * (0.0554953f + frac * 0.0096838f)));
        union { float fv; int iv; } sc;
        sc.iv = ((int)nf + 127) << 23;
        r.v[i] = pp * sc.fv;
    }
    return r;
}

#endif

/* SIMD vector addition: dst[i] += src[i] */
static inline void fe_vec_add(float* dst, const float* src, int n) {
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        f32x4_store(dst + i, f32x4_add(f32x4_load(dst + i), f32x4_load(src + i)));
    }
    for (int i = n4; i < n; i++) {
        dst[i] += src[i];
    }
}

#endif /* FE_SIMD_H */
