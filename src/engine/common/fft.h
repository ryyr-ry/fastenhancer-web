/*
 * fft.h — Radix-2 DIT FFT/iFFT interface
 */

#ifndef FE_FFT_H
#define FE_FFT_H

/* in-place FFT (split format: real array + imaginary array) */
void fe_fft(float* real, float* imag, int n);

/* in-place iFFT (conjugate -> FFT -> conjugate -> 1/N scaling) */
void fe_ifft(float* real, float* imag, int n);

/* Get twiddle factor table (512 elements, for 1024-point FFT) */
const float* fe_fft_get_twiddle_re(void);
const float* fe_fft_get_twiddle_im(void);

#endif /* FE_FFT_H */
