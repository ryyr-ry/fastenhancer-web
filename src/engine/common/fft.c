/*
 * fft.c — 1024-point Cooley-Tukey Radix-2 DIT FFT
 *
 * twiddle factor: W_N^k = cos(2*pi*k/N) - j*sin(2*pi*k/N)
 * 1024 points -> 512 twiddles x 2(cos,sin) x 4B = 4KB
 *
 * Bit-reversal permutation + butterfly operations.
 * Supports arbitrary power-of-two sizes (up to 1024 points).
 */

#include "fft.h"
#include "simd.h"
#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_FFT_SIZE 1024

static float twiddle_re_precise[MAX_FFT_SIZE / 2];
static float twiddle_im_precise[MAX_FFT_SIZE / 2];

#include <stdatomic.h>
static atomic_int twiddle_init_state = 0;

static void init_twiddles(void) {
    int expected = 0;
    if (!atomic_compare_exchange_strong(&twiddle_init_state, &expected, 1)) {
        while (atomic_load(&twiddle_init_state) != 2) { /* spin */ }
        return;
    }

    for (int k = 0; k < MAX_FFT_SIZE / 2; k++) {
        double angle = 2.0 * M_PI * (double)k / (double)MAX_FFT_SIZE;
        twiddle_re_precise[k] = (float)cos(angle);
        twiddle_im_precise[k] = (float)(-sin(angle));
    }

    atomic_store(&twiddle_init_state, 2);
}

static void bit_reverse_permutation(float* real, float* imag, int n) {
    int log2n = 0;
    for (int t = n; t > 1; t >>= 1) log2n++;

    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int bit = 0; bit < log2n; bit++) {
            if (i & (1 << bit))
                j |= (1 << (log2n - 1 - bit));
        }
        if (i < j) {
            float tmp;
            tmp = real[i]; real[i] = real[j]; real[j] = tmp;
            tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
        }
    }
}

void fe_fft(float* real, float* imag, int n) {
    if (real == NULL || imag == NULL) return;
    if (n <= 0 || n > MAX_FFT_SIZE) return;
    if ((n & (n - 1)) != 0) return;  /* Reject if not a power of two */

    init_twiddles();
    bit_reverse_permutation(real, imag, n);

    for (int size = 2; size <= n; size <<= 1) {
        int half = size >> 1;
        int tw_stride = MAX_FFT_SIZE / size;

        for (int block = 0; block < n; block += size) {
            int j = 0;
            const int half4 = half & ~3;

            if (tw_stride == 1) {
                /* Last stage: twiddle factors are consecutive — direct SIMD load */
                for (; j < half4; j += 4) {
                    int top = block + j;
                    int bot = top + half;
                    f32x4 vwr = f32x4_load(twiddle_re_precise + j);
                    f32x4 vwi = f32x4_load(twiddle_im_precise + j);
                    f32x4 vtr = f32x4_load(real + top);
                    f32x4 vti = f32x4_load(imag + top);
                    f32x4 vbr = f32x4_load(real + bot);
                    f32x4 vbi = f32x4_load(imag + bot);
                    f32x4 tr = f32x4_sub(f32x4_mul(vbr, vwr), f32x4_mul(vbi, vwi));
                    f32x4 ti = f32x4_add(f32x4_mul(vbr, vwi), f32x4_mul(vbi, vwr));
                    f32x4_store(real + top, f32x4_add(vtr, tr));
                    f32x4_store(imag + top, f32x4_add(vti, ti));
                    f32x4_store(real + bot, f32x4_sub(vtr, tr));
                    f32x4_store(imag + bot, f32x4_sub(vti, ti));
                }
            } else if (half4 > 0) {
                /* Other stages: gather twiddle factors from strided positions */
                for (; j < half4; j += 4) {
                    int top = block + j;
                    int bot = top + half;
                    int idx = j * tw_stride;
                    float tw_re_buf[4] = {
                        twiddle_re_precise[idx],
                        twiddle_re_precise[idx + tw_stride],
                        twiddle_re_precise[idx + 2 * tw_stride],
                        twiddle_re_precise[idx + 3 * tw_stride]
                    };
                    float tw_im_buf[4] = {
                        twiddle_im_precise[idx],
                        twiddle_im_precise[idx + tw_stride],
                        twiddle_im_precise[idx + 2 * tw_stride],
                        twiddle_im_precise[idx + 3 * tw_stride]
                    };
                    f32x4 vwr = f32x4_load(tw_re_buf);
                    f32x4 vwi = f32x4_load(tw_im_buf);
                    f32x4 vtr = f32x4_load(real + top);
                    f32x4 vti = f32x4_load(imag + top);
                    f32x4 vbr = f32x4_load(real + bot);
                    f32x4 vbi = f32x4_load(imag + bot);
                    f32x4 tr = f32x4_sub(f32x4_mul(vbr, vwr), f32x4_mul(vbi, vwi));
                    f32x4 ti = f32x4_add(f32x4_mul(vbr, vwi), f32x4_mul(vbi, vwr));
                    f32x4_store(real + top, f32x4_add(vtr, tr));
                    f32x4_store(imag + top, f32x4_add(vti, ti));
                    f32x4_store(real + bot, f32x4_sub(vtr, tr));
                    f32x4_store(imag + bot, f32x4_sub(vti, ti));
                }
            }

            for (; j < half; j++) {
                int top = block + j;
                int bot = top + half;
                float wr = twiddle_re_precise[j * tw_stride];
                float wi = twiddle_im_precise[j * tw_stride];

                float tr = real[bot] * wr - imag[bot] * wi;
                float ti = real[bot] * wi + imag[bot] * wr;

                real[bot] = real[top] - tr;
                imag[bot] = imag[top] - ti;
                real[top] = real[top] + tr;
                imag[top] = imag[top] + ti;
            }
        }
    }
}

void fe_ifft(float* real, float* imag, int n) {
    if (real == NULL || imag == NULL) return;
    if (n <= 0 || n > MAX_FFT_SIZE) return;
    if ((n & (n - 1)) != 0) return;

    /* Conjugate (SIMD) */
    {
        const int n4 = n & ~3;
        int i = 0;
        for (; i < n4; i += 4)
            f32x4_store(imag + i, f32x4_neg(f32x4_load(imag + i)));
        for (; i < n; i++) imag[i] = -imag[i];
    }

    fe_fft(real, imag, n);

    /* Conjugate + 1/N scale (SIMD) */
    {
        float inv_n = 1.0f / (float)n;
        f32x4 vinv = f32x4_splat(inv_n);
        const int n4 = n & ~3;
        int i = 0;
        for (; i < n4; i += 4) {
            f32x4_store(real + i, f32x4_mul(f32x4_load(real + i), vinv));
            f32x4_store(imag + i, f32x4_mul(f32x4_neg(f32x4_load(imag + i)), vinv));
        }
        for (; i < n; i++) {
            real[i] *= inv_n;
            imag[i] = -imag[i] * inv_n;
        }
    }
}

const float* fe_fft_get_twiddle_re(void) {
    init_twiddles();
    return twiddle_re_precise;
}

const float* fe_fft_get_twiddle_im(void) {
    init_twiddles();
    return twiddle_im_precise;
}
