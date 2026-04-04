/*
 * conv.h — 1D convolution interface
 */

#ifndef FE_CONV_H
#define FE_CONV_H

/* Conv1d: Standard 1D convolution
 * input:  [in_channels x in_len]
 * weight: [out_channels x in_channels x kernel_size]
 * bias:   [out_channels]
 * output: [out_channels x out_len]
 * out_len = (in_len + 2*padding - kernel_size) / stride + 1
 */
void fe_conv1d(const float* input, const float* weight, const float* bias,
               float* output, int in_len, int in_channels, int out_channels,
               int kernel_size, int stride, int padding);

/* StridedConv1d: Strided convolution with padding=0 (for Encoder)
 * Returns: output length */
int fe_strided_conv1d(const float* input, const float* weight, const float* bias,
                      float* output, int in_len, int in_channels, int out_channels,
                      int kernel_size, int stride);

/* ConvTranspose1d: Transposed convolution (for Decoder)
 * output_len = (in_len - 1) * stride + kernel_size - 2 * padding
 * padding: PyTorch compatible. Removes padding from the beginning/end of full output.
 * Returns: output length */
int fe_conv_transpose1d(const float* input, const float* weight, const float* bias,
                        float* output, int in_len, int in_channels, int out_channels,
                        int kernel_size, int stride, int padding);

/* Conv1d + BatchNorm fusion: y = conv(x) * scale + bias
 * Assumption: conv's own bias is 0 (absorbed into BatchNorm).
 * export_weights.py pre-fuses conv.bias + BN transform into bn_bias. */
void fe_conv1d_bn(const float* input, const float* weight,
                  const float* bn_scale, const float* bn_bias,
                  float* output, int in_len, int in_channels, int out_channels,
                  int kernel_size, int stride, int padding);

#endif /* FE_CONV_H */
