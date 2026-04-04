/*
 * base_48k.h — 48kHz Base model configuration
 *
 * All constants for the FastEnhancer ICASSP 2026 paper Base model (48kHz version).
 * Include this header at compile time to fix the model size.
 *
 * b.yaml: channels=48, rnnformer(num_blocks=3, channels=36, freq=36)
 */

#ifndef FE_BASE_48K_H
#define FE_BASE_48K_H

/* Audio parameters */
#define FE_SAMPLE_RATE      48000
#define FE_N_FFT            1024
#define FE_WIN_SIZE         1024
#define FE_HOP_SIZE         512
#define FE_FREQ_BINS        512     /* n_fft/2 (after Nyquist removal) */
#define FE_SPEC_BINS        513     /* n_fft/2+1 (including Nyquist) */

/* Architecture constants */
#define FE_C1               48      /* Encoder/Decoder channels */
#define FE_C2               36      /* RNNFormer channels */
#define FE_F1               128     /* Post-stride frequency (FREQ_BINS / STRIDE) */
#define FE_F2               36      /* RNNFormer frequency dimension */
#define FE_ENC_BLOCKS       2       /* Number of Encoder blocks */
#define FE_RF_BLOCKS        3       /* Number of RNNFormer blocks */
#define FE_STRIDE           4
#define FE_ENC_K0           8       /* PreNet kernel size */
#define FE_ENC_K            3       /* Block kernel size */
#define FE_ENC_PAD          1       /* Block padding */
#define FE_ENC_PRE_PAD      2       /* PreNet padding */
#define FE_NUM_HEADS        4
#define FE_HEAD_DIM         9       /* C2 / NUM_HEADS */
#define FE_COMPRESS_EXP     0.3f

/* Per-layer weight counts */
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

/* Total weight count */
#define FE_TOTAL_WEIGHTS    (FE_W_ENC_PRENET + \
                             FE_ENC_BLOCKS * FE_W_ENC_BLOCK + \
                             FE_W_RF_PRENET + \
                             FE_W_RF_BLOCK0 + \
                             (FE_RF_BLOCKS - 1) * FE_W_RF_BLOCK_N + \
                             FE_W_RF_POSTNET + \
                             FE_ENC_BLOCKS * FE_W_DEC_BLOCK + \
                             FE_W_DEC_POSTNET)

#endif /* FE_BASE_48K_H */
