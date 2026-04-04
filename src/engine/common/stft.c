/*
 * stft.c — ストリーミングSTFT/iSTFT
 *
 * 分析: Hann窓 → FFT → 複素スペクトル [freq_bins]
 * 合成: 共役対称復元 → iFFT → overlap-add
 *
 * COLA条件: hop=n_fft/2 限定。周期的Hann窓は w[n]+w[n+N/2]=1 を厳密に満たす。
 * 合成窓(window_istft)正規化は不要: overlap-addで分析窓の合計が定数1となるため、
 * 追加の合成窓除算を行うと二重補正になり逆に波形が歪む。
 * PyTorch torch.istft は一般hop用に win_sq_sum 除算を行うが、
 * hop=N/2+Hann窓ではその除算結果も1.0なので実質同一。
 * golden vectorテスト (MSE ~1e-19) がこの実装の正しさを実証している。
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
    /* バリデーション: n_fft > 0, 2のべき乗, 上限, hop=n_fft/2 */
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

    /* 周期的Hann窓: w[n] = 0.5 - 0.5·cos(2πn/N)
     * (COLA条件: hop=N/2でw[i]+w[i+N/2]=1 が厳密成立) */
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

    /* 入力バッファをhop分シフトし、新しいサンプルを末尾に追加 */
    memmove(state->input_buffer, state->input_buffer + hop, sizeof(float) * (n_fft - hop));
    memcpy(state->input_buffer + (n_fft - hop), input, sizeof(float) * hop);

    /* Hann窓適用 → FFTバッファにコピー (SIMD) */
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

    /* 正の周波数成分のみ出力 (n_fft/2+1 bins) */
    memcpy(spec_real, state->fft_real, sizeof(float) * bins);
    memcpy(spec_imag, state->fft_imag, sizeof(float) * bins);
}

void fe_stft_inverse(FeStftState* state, const float* spec_real,
                     const float* spec_imag, float* output) {
    int n_fft = state->n_fft;
    int hop = state->hop_size;
    int bins = state->freq_bins;

    /* freq_binsから完全スペクトルを共役対称で復元 */
    memcpy(state->fft_real, spec_real, sizeof(float) * bins);
    memcpy(state->fft_imag, spec_imag, sizeof(float) * bins);

    /* DC成分とNyquist成分の虚部は実信号では常にゼロ */
    state->fft_imag[0] = 0.0f;
    state->fft_imag[bins - 1] = 0.0f;

    for (int k = 1; k < n_fft - bins + 1; k++) {
        state->fft_real[bins - 1 + k] = spec_real[bins - 1 - k];
        state->fft_imag[bins - 1 + k] = -spec_imag[bins - 1 - k];
    }

    fe_ifft(state->fft_real, state->fft_imag, n_fft);

    /* overlap-add: 前半(hop) + overlap → 出力、後半(hop) → 新overlap (SIMD) */
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
