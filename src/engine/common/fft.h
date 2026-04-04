/*
 * fft.h — Radix-2 DIT FFT/iFFT インターフェース
 */

#ifndef FE_FFT_H
#define FE_FFT_H

/* in-place FFT (split format: 実部配列 + 虚部配列) */
void fe_fft(float* real, float* imag, int n);

/* in-place iFFT (共役→FFT→共役→1/Nスケール) */
void fe_ifft(float* real, float* imag, int n);

/* twiddle factorテーブル取得 (512要素, 1024点FFT用) */
const float* fe_fft_get_twiddle_re(void);
const float* fe_fft_get_twiddle_im(void);

#endif /* FE_FFT_H */
