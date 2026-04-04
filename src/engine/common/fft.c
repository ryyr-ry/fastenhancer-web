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
            for (int j = 0; j < half; j++) {
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

    /* Conjugate -> FFT -> conjugate -> 1/N scaling */
    for (int i = 0; i < n; i++) imag[i] = -imag[i];

    fe_fft(real, imag, n);

    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        real[i] *= inv_n;
        imag[i] = -imag[i] * inv_n;
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
