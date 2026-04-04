/*
 * tiny_48k.h — 48kHz Tiny モデル設定
 *
 * FastEnhancer ICASSP 2026 論文のTinyモデル(48kHz版)の全定数。
 * コンパイル時にこのヘッダをインクルードし、モデルサイズを固定する。
 */

#ifndef FE_TINY_48K_H
#define FE_TINY_48K_H

/* オーディオパラメータ */
#define FE_SAMPLE_RATE      48000
#define FE_N_FFT            1024
#define FE_WIN_SIZE         1024
#define FE_HOP_SIZE         512
#define FE_FREQ_BINS        512     /* n_fft/2 (Nyquist除去後) */
#define FE_SPEC_BINS        513     /* n_fft/2+1 (Nyquist含む) */

/* アーキテクチャ定数 */
#define FE_C1               24      /* Encoder/Decoderチャネル数 */
#define FE_C2               20      /* RNNFormerチャネル数 */
#define FE_F1               128     /* ストライド後周波数 (FREQ_BINS / STRIDE) */
#define FE_F2               24      /* RNNFormer周波数次元 */
#define FE_ENC_BLOCKS       2       /* Encoderブロック数 */
#define FE_RF_BLOCKS        2       /* RNNFormerブロック数 */
#define FE_STRIDE           4
#define FE_ENC_K0           8       /* PreNetカーネルサイズ */
#define FE_ENC_K            3       /* Blockカーネルサイズ */
#define FE_ENC_PAD          1       /* Blockパディング */
#define FE_ENC_PRE_PAD      2       /* PreNetパディング */
#define FE_NUM_HEADS        4
#define FE_HEAD_DIM         5       /* C2 / NUM_HEADS */
#define FE_COMPRESS_EXP     0.3f

/* レイヤー別重み数 */
#define FE_W_ENC_PRENET     (FE_C1 * 2 * FE_ENC_K0 + 2 * FE_C1)
#define FE_W_ENC_BLOCK      (FE_C1 * FE_C1 * FE_ENC_K + 2 * FE_C1)
#define FE_W_RF_PRENET      (FE_F2 * FE_F1 + FE_C2 * FE_C1 + 2 * FE_C2)
#define FE_W_GRU            (6 * FE_C2 * FE_C2 + 4 * FE_C2)
#define FE_W_GRU_FC         (FE_C2 * FE_C2 + FE_C2)
#define FE_W_PE             (FE_F2 * FE_C2)
#define FE_W_MHSA           (4 * FE_C2 * FE_C2 + 4 * FE_C2)
#define FE_W_RF_BLOCK0      (FE_W_GRU + FE_W_GRU_FC + FE_W_PE + FE_W_MHSA)
#define FE_W_RF_BLOCK_N     (FE_W_GRU + FE_W_GRU_FC + FE_W_MHSA)
#define FE_W_RF_POSTNET     (FE_C1 * FE_C2 + 2 * FE_C1 + FE_F1 * FE_F2)
#define FE_W_DEC_BLOCK      (FE_C1 * 2 * FE_C1 + 2 * FE_C1 + \
                             FE_C1 * FE_C1 * FE_ENC_K + 2 * FE_C1)
#define FE_W_DEC_POSTNET    (FE_C1 * 2 * FE_C1 + 2 * FE_C1 + \
                             FE_C1 * 2 * FE_ENC_K0 + 2)

/* 総重み数: 28354 */
#define FE_TOTAL_WEIGHTS    (FE_W_ENC_PRENET + \
                             FE_ENC_BLOCKS * FE_W_ENC_BLOCK + \
                             FE_W_RF_PRENET + \
                             FE_W_RF_BLOCK0 + \
                             (FE_RF_BLOCKS - 1) * FE_W_RF_BLOCK_N + \
                             FE_W_RF_POSTNET + \
                             FE_ENC_BLOCKS * FE_W_DEC_BLOCK + \
                             FE_W_DEC_POSTNET)

#endif /* FE_TINY_48K_H */
