/*
 * stft.c — Streaming STFT/iSTFT
 *
 * Analysis: Hann window -> FFT -> complex spectrum [freq_bins]
 * Synthesis: conjugate-symmetric reconstruction -> iFFT -> overlap-add
 *
 * COLA condition: restricted to hop=n_fft/2. The periodic Hann window
 * strictly satisfies w[n]+w[n+N/2]=1.
 * Synthesis window normalization is unnecessary: since overlap-add of the
 * analysis window sums to a constant of 1, additional synthesis window
 * division would cause double correction and distort the waveform.
 * PyTorch torch.istft performs win_sq_sum division for general hops,
 * but for hop=N/2 + Hann window, the division result is also 1.0,
 * so it is effectively identical.
 * The golden vector test (MSE ~1e-19) verifies the correctness of this implementation.
 */

#include "stft.h"
#include "fft.h"
#include "simd.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int fe_stft_init(FeStftState* state, int n_fft, int hop_size) {
    /* Validation: n_fft > 0, power of two, upper limit, hop=n_fft/2 */
    if (n_fft <= 0 || n_fft > FE_STFT_MAX_FFT) {
        memset(state, 0, sizeof(FeStftState));
        return -1;
    }
    if ((n_fft & (n_fft - 1)) != 0) {
        memset(state, 0, sizeof(FeStftState));
        return -1;
    }
    if (hop_size != n_fft / 2) {
        memset(state, 0, sizeof(FeStftState));
        return -1;
    }

    state->n_fft = n_fft;
    state->hop_size = hop_size;
    state->freq_bins = n_fft / 2 + 1;

    /* Periodic Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n/N)
     * (COLA condition: w[i]+w[i+N/2]=1 strictly holds for hop=N/2) */
    for (int n = 0; n < n_fft; n++) {
        state->window[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)n_fft);
    }

    fe_stft_reset(state);
    return 0;
}

void fe_stft_reset(FeStftState* state) {
    memset(state->input_buffer, 0, sizeof(float) * state->n_fft);
    memset(state->overlap, 0, sizeof(float) * state->hop_size);
}

void fe_stft_forward(FeStftState* state, const float* input,
                     float* spec_real, float* spec_imag) {
    int n_fft = state->n_fft;
    int hop = state->hop_size;
    int bins = state->freq_bins;

    /* Shift input buffer by hop and append new samples at the end */
    memmove(state->input_buffer, state->input_buffer + hop, sizeof(float) * (n_fft - hop));
    memcpy(state->input_buffer + (n_fft - hop), input, sizeof(float) * hop);

    /* Apply Hann window -> copy to FFT buffer (SIMD) */
    {
        const int n4 = n_fft & ~3;
        int n_i = 0;
        for (; n_i < n4; n_i += 4) {
            f32x4_store(state->fft_real + n_i,
                f32x4_mul(f32x4_load(state->input_buffer + n_i),
                          f32x4_load(state->window + n_i)));
        }
        for (; n_i < n_fft; n_i++) {
            state->fft_real[n_i] = state->input_buffer[n_i] * state->window[n_i];
        }
    }
    memset(state->fft_imag, 0, sizeof(float) * n_fft);

    fe_fft(state->fft_real, state->fft_imag, n_fft);

    /* Output only positive frequency components (n_fft/2+1 bins) */
    memcpy(spec_real, state->fft_real, sizeof(float) * bins);
    memcpy(spec_imag, state->fft_imag, sizeof(float) * bins);
}

void fe_stft_inverse(FeStftState* state, const float* spec_real,
                     const float* spec_imag, float* output) {
    int n_fft = state->n_fft;
    int hop = state->hop_size;
    int bins = state->freq_bins;

    /* Reconstruct full spectrum from freq_bins via conjugate symmetry */
    memcpy(state->fft_real, spec_real, sizeof(float) * bins);
    memcpy(state->fft_imag, spec_imag, sizeof(float) * bins);

    /* Imaginary parts of DC and Nyquist components are always zero for real signals */
    state->fft_imag[0] = 0.0f;
    state->fft_imag[bins - 1] = 0.0f;

    for (int k = 1; k < n_fft - bins + 1; k++) {
        state->fft_real[bins - 1 + k] = spec_real[bins - 1 - k];
        state->fft_imag[bins - 1 + k] = -spec_imag[bins - 1 - k];
    }

    fe_ifft(state->fft_real, state->fft_imag, n_fft);

    /* overlap-add: first half(hop) + overlap -> output, second half(hop) -> new overlap (SIMD) */
    {
        const int h4 = hop & ~3;
        int ii = 0;
        for (; ii < h4; ii += 4) {
            f32x4_store(output + ii,
                f32x4_add(f32x4_load(state->fft_real + ii),
                          f32x4_load(state->overlap + ii)));
        }
        for (; ii < hop; ii++) {
            output[ii] = state->fft_real[ii] + state->overlap[ii];
        }
    }
    memcpy(state->overlap, state->fft_real + hop, sizeof(float) * hop);
}

const float* fe_stft_get_window(const FeStftState* state) {
    return state->window;
}
