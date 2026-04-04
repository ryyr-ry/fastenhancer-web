/*
 * conv.h — 1次元畳み込みインターフェース
 */

#ifndef FE_CONV_H
#define FE_CONV_H

/* Conv1d: 通常の1次元畳み込み
 * input:  [in_channels × in_len]
 * weight: [out_channels × in_channels × kernel_size]
 * bias:   [out_channels]
 * output: [out_channels × out_len]
 * out_len = (in_len + 2*padding - kernel_size) / stride + 1
 */
void fe_conv1d(const float* input, const float* weight, const float* bias,
               float* output, int in_len, int in_channels, int out_channels,
               int kernel_size, int stride, int padding);

/* StridedConv1d: padding=0のストライド畳み込み (Encoder用)
 * 戻り値: 出力長 */
int fe_strided_conv1d(const float* input, const float* weight, const float* bias,
                      float* output, int in_len, int in_channels, int out_channels,
                      int kernel_size, int stride);

/* ConvTranspose1d: 転置畳み込み (Decoder用)
 * output_len = (in_len - 1) * stride + kernel_size - 2 * padding
 * padding: PyTorch互換。フル出力から先頭/末尾のpadding分を除去。
 * 戻り値: 出力長 */
int fe_conv_transpose1d(const float* input, const float* weight, const float* bias,
                        float* output, int in_len, int in_channels, int out_channels,
                        int kernel_size, int stride, int padding);

/* Conv1d + BatchNorm融合: y = conv(x) * scale + bias
 * 前提: conv自体のbiasは0（BatchNormに吸収済み）。
 * export_weights.pyがconv.bias + BN変換を事前にbn_biasへ融合する。 */
void fe_conv1d_bn(const float* input, const float* weight,
                  const float* bn_scale, const float* bn_bias,
                  float* output, int in_len, int in_channels, int out_channels,
                  int kernel_size, int stride, int padding);

#endif /* FE_CONV_H */
