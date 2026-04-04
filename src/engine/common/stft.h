/*
 * stft.h — STFT/iSTFT interface
 */

#ifndef FE_STFT_H
#define FE_STFT_H

#define FE_STFT_MAX_FFT 1024

typedef struct {
    float window[FE_STFT_MAX_FFT];
    float input_buffer[FE_STFT_MAX_FFT];
    float overlap[FE_STFT_MAX_FFT / 2];
    float fft_real[FE_STFT_MAX_FFT];
    float fft_imag[FE_STFT_MAX_FFT];
    int n_fft;
    int hop_size;
    int freq_bins;
} FeStftState;

int fe_stft_init(FeStftState* state, int n_fft, int hop_size);
void fe_stft_forward(FeStftState* state, const float* input,
                     float* spec_real, float* spec_imag);
void fe_stft_inverse(FeStftState* state, const float* spec_real,
                     const float* spec_imag, float* output);
void fe_stft_reset(FeStftState* state);
const float* fe_stft_get_window(const FeStftState* state);

#endif /* FE_STFT_H */
