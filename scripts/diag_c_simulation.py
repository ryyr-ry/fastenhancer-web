#!/usr/bin/env python3
"""
diag_c_simulation.py — Reproduce the C pipeline in Python using exported weights

Load the exported binary weights and run inference in the same operation order as
the C engine. Compare against PyTorch model output to verify the accuracy of the
weight export.
"""

import struct
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Constants from tiny_48k.h
C1 = 24
C2 = 20
F1 = 128
F2 = 24
FREQ_BINS = 512
SPEC_BINS = 513
N_FFT = 1024
HOP_SIZE = 512
ENC_K0 = 8
ENC_K = 3
ENC_PAD = 1
ENC_PRE_PAD = 2
STRIDE = 4
ENC_BLOCKS = 2
RF_BLOCKS = 2
NUM_HEADS = 4
HEAD_DIM = 5
COMPRESS_EXP = 0.3


def load_weights(path):
    """Load weights from a FEW1 binary."""
    with open(path, "rb") as f:
        hdr = f.read(20)
        magic = hdr[:4]
        assert magic == b"FEW1", f"Bad magic: {magic}"
        version = struct.unpack_from("<I", hdr, 4)[0]
        model_id = struct.unpack_from("<I", hdr, 8)[0]
        wcount = struct.unpack_from("<I", hdr, 12)[0]
        crc = struct.unpack_from("<I", hdr, 16)[0]
        payload = f.read(wcount * 4)
        weights = np.frombuffer(payload, dtype=np.float32).copy()
    print(f"Weights: {wcount} floats, version={version}, model_id={model_id}, crc=0x{crc:08X}")
    return weights


def parse_weights(w):
    """Assign weight pointers in the same order as setup_weights()."""
    p = 0
    result = {}

    def take(name, count):
        nonlocal p
        result[name] = w[p:p+count].copy()
        p += count

    # Encoder PreNet
    take("enc_pre_conv_w", C1 * 2 * ENC_K0)
    take("enc_pre_bn_s", C1)
    take("enc_pre_bn_b", C1)

    # Encoder Blocks
    for b in range(ENC_BLOCKS):
        take(f"enc_block_{b}_conv_w", C1 * C1 * ENC_K)
        take(f"enc_block_{b}_bn_s", C1)
        take(f"enc_block_{b}_bn_b", C1)

    # RNNFormer PreNet
    take("rf_pre_freq_w", F2 * F1)
    take("rf_pre_conv_w", C2 * C1)
    take("rf_pre_bn_s", C2)
    take("rf_pre_bn_b", C2)

    # RNNFormer Blocks
    for b in range(RF_BLOCKS):
        take(f"rf_{b}_gru_W_z", C2 * C2)
        take(f"rf_{b}_gru_U_z", C2 * C2)
        take(f"rf_{b}_gru_b_z", C2)
        take(f"rf_{b}_gru_W_r", C2 * C2)
        take(f"rf_{b}_gru_U_r", C2 * C2)
        take(f"rf_{b}_gru_b_r", C2)
        take(f"rf_{b}_gru_W_n", C2 * C2)
        take(f"rf_{b}_gru_U_n", C2 * C2)
        take(f"rf_{b}_gru_b_in_n", C2)
        take(f"rf_{b}_gru_b_hn_n", C2)

        take(f"rf_{b}_gru_fc_w", C2 * C2)
        take(f"rf_{b}_gru_fc_b", C2)

        if b == 0:
            take("rf_pe", F2 * C2)

        take(f"rf_{b}_mhsa_W_q", C2 * C2)
        take(f"rf_{b}_mhsa_b_q", C2)
        take(f"rf_{b}_mhsa_W_k", C2 * C2)
        take(f"rf_{b}_mhsa_b_k", C2)
        take(f"rf_{b}_mhsa_W_v", C2 * C2)
        take(f"rf_{b}_mhsa_b_v", C2)
        take(f"rf_{b}_mhsa_W_o", C2 * C2)
        take(f"rf_{b}_mhsa_b_o", C2)

    # RNNFormer PostNet
    take("rf_post_conv_w", C1 * C2)
    take("rf_post_bn_s", C1)
    take("rf_post_bn_b", C1)
    take("rf_post_freq_w", F1 * F2)

    # Decoder Blocks
    for b in range(ENC_BLOCKS):
        take(f"dec_{b}_skip_conv_w", C1 * 2 * C1)
        take(f"dec_{b}_skip_bn_s", C1)
        take(f"dec_{b}_skip_bn_b", C1)
        take(f"dec_{b}_conv_w", C1 * C1 * ENC_K)
        take(f"dec_{b}_bn_s", C1)
        take(f"dec_{b}_bn_b", C1)

    # Decoder PostNet
    take("dec_post_skip_conv_w", C1 * 2 * C1)
    take("dec_post_skip_bn_s", C1)
    take("dec_post_skip_bn_b", C1)
    take("dec_post_deconv_w", C1 * 2 * ENC_K0)
    take("dec_post_deconv_b", 2)

    print(f"Weight consumption: {p} / {len(w)} (expected: {len(w)})")
    assert p == len(w), f"Weight consumption mismatch: {p} != {len(w)}"
    return result


def conv1d_bn(x, weight, bn_s, bn_b, in_len, in_ch, out_ch, kernel, stride, pad):
    """Python implementation of fe_conv1d_bn."""
    w = weight.reshape(out_ch, in_ch, kernel)
    x_2d = x.reshape(in_ch, in_len)
    out_len = (in_len + 2 * pad - kernel) // stride + 1
    output = np.zeros((out_ch, out_len), dtype=np.float32)
    for oc in range(out_ch):
        for p in range(out_len):
            s = 0.0
            for ic in range(in_ch):
                for k in range(kernel):
                    pos = p * stride + k - pad
                    if 0 <= pos < in_len:
                        s += w[oc, ic, k] * x_2d[ic, pos]
            output[oc, p] = s * bn_s[oc] + bn_b[oc]
    return output


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def linear_last_dim(W, x, batch, out_dim, in_dim):
    """[batch, in_dim] → [batch, out_dim]"""
    W_2d = W.reshape(out_dim, in_dim)
    x_2d = x.reshape(batch, in_dim)
    output = np.zeros((batch, out_dim), dtype=np.float32)
    for b in range(batch):
        output[b] = W_2d @ x_2d[b]
    return output


def transpose_2d(x, rows, cols):
    return x.reshape(rows, cols).T.copy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def gru_step(x, h, W_z, U_z, b_z, W_r, U_r, b_r, W_n, U_n, b_in_n, b_hn_n, D):
    """Python implementation of fe_gru_step."""
    W_z = W_z.reshape(D, D)
    U_z = U_z.reshape(D, D)
    W_r = W_r.reshape(D, D)
    U_r = U_r.reshape(D, D)
    W_n = W_n.reshape(D, D)
    U_n = U_n.reshape(D, D)

    z = sigmoid(W_z @ x + U_z @ h + b_z)
    r = sigmoid(W_r @ x + U_r @ h + b_r)
    n = np.tanh(W_n @ x + b_in_n + r * (U_n @ h + b_hn_n))
    h_new = (1 - z) * n + z * h
    return h_new


def mhsa(x, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, n_heads, head_dim, seq_len, c2):
    """Python implementation of fe_mhsa."""
    W_q = W_q.reshape(c2, c2)
    W_k = W_k.reshape(c2, c2)
    W_v = W_v.reshape(c2, c2)
    W_o = W_o.reshape(c2, c2)
    x_2d = x.reshape(seq_len, c2)

    Q = x_2d @ W_q.T + b_q  # [seq, c2]
    K = x_2d @ W_k.T + b_k
    V = x_2d @ W_v.T + b_v

    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)
    V = V.reshape(seq_len, n_heads, head_dim)

    scale = 1.0 / np.sqrt(head_dim)
    output = np.zeros((seq_len, c2), dtype=np.float32)

    for h in range(n_heads):
        q_h = Q[:, h, :]  # [seq, hd]
        k_h = K[:, h, :]
        v_h = V[:, h, :]
        scores = q_h @ k_h.T * scale  # [seq, seq]
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        out_h = attn @ v_h  # [seq, hd]
        output[:, h * head_dim:(h+1) * head_dim] = out_h

    result = output @ W_o.T + b_o  # [seq, c2]
    return result.flatten()


def conv_transpose1d(x, weight, bias, in_len, in_ch, out_ch, kernel, stride, pad):
    """Python implementation of fe_conv_transpose1d."""
    w = weight.reshape(in_ch, out_ch, kernel)
    x_2d = x.reshape(in_ch, in_len)
    full_len = (in_len - 1) * stride + kernel
    out_len = full_len - 2 * pad
    output = np.full((out_ch, out_len), 0.0, dtype=np.float32)
    for oc in range(out_ch):
        output[oc, :] = bias[oc]

    for oc in range(out_ch):
        for ic in range(in_ch):
            for i in range(in_len):
                in_val = x_2d[ic, i]
                full_start = i * stride
                for k in range(kernel):
                    full_pos = full_start + k
                    out_pos = full_pos - pad
                    if 0 <= out_pos < out_len:
                        output[oc, out_pos] += w[ic, oc, k] * in_val
    return output


def simulate_c_pipeline(ws, input_np, target_frame=1):
    """Simulate the C pipeline frame by frame."""
    print(f"\n=== C Pipeline Simulation (target: frame {target_frame}) ===\n")

    # STFT state
    stft_buffer = np.zeros(N_FFT, dtype=np.float32)
    overlap = np.zeros(HOP_SIZE, dtype=np.float32)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)

    # GRU hidden states
    gru_h = [np.zeros((F2 * C2,), dtype=np.float32) for _ in range(RF_BLOCKS)]

    for frame_idx in range(target_frame + 1):
        frame_data = input_np[frame_idx * HOP_SIZE:(frame_idx + 1) * HOP_SIZE]
        is_target = (frame_idx == target_frame)

        # STFT forward
        stft_buffer[:HOP_SIZE] = stft_buffer[HOP_SIZE:]
        stft_buffer[HOP_SIZE:] = frame_data
        windowed = stft_buffer * hann
        spec = np.fft.rfft(windowed)
        spec_re = spec.real.astype(np.float32)
        spec_im = spec.imag.astype(np.float32)

        if is_target:
            print(f"STFT frame {frame_idx}:")
            print(f"  spec_re: min={spec_re.min():.6e}, max={spec_re.max():.6e}, rms={np.sqrt(np.mean(spec_re**2)):.6e}")
            print(f"  spec_re first5: {spec_re[:5]}")
            print(f"  spec_im first5: {spec_im[:5]}")

        # Power compression (on first 512 bins)
        re_512 = spec_re[:FREQ_BINS].copy()
        im_512 = spec_im[:FREQ_BINS].copy()
        mag = np.sqrt(re_512**2 + im_512**2)
        scale = np.where(mag > 1e-5, np.power(np.maximum(mag, 1e-5), COMPRESS_EXP - 1.0), 0.0).astype(np.float32)
        comp_re = (re_512 * scale).astype(np.float32)
        comp_im = (im_512 * scale).astype(np.float32)

        if is_target:
            print(f"\nCompressed:")
            print(f"  re: min={comp_re.min():.6e}, max={comp_re.max():.6e}, rms={np.sqrt(np.mean(comp_re**2)):.6e}")
            print(f"  re first5: {comp_re[:5]}")
            print(f"  im first5: {comp_im[:5]}")

        # Encoder input: [real(512), imag(512)] = [2, 512]
        enc_input = np.concatenate([comp_re, comp_im])

        if is_target:
            print(f"\nEncoder input [2, 512]:")
            print(f"  ch0 first5: {enc_input[:5]}")
            print(f"  ch1 first5: {enc_input[FREQ_BINS:FREQ_BINS+5]}")

        # Encoder PreNet: Conv1d_BN(2→C1, k=8, s=4, pad=2) + SiLU
        enc_pre = conv1d_bn(enc_input, ws["enc_pre_conv_w"], ws["enc_pre_bn_s"], ws["enc_pre_bn_b"],
                            FREQ_BINS, 2, C1, ENC_K0, STRIDE, ENC_PRE_PAD)
        enc_pre = silu(enc_pre)
        enc_skip = [enc_pre.copy()]

        if is_target:
            print(f"\nEncoder PreNet output [C1={C1}, F1={F1}]:")
            print(f"  rms={np.sqrt(np.mean(enc_pre**2)):.6e}")
            print(f"  first5: {enc_pre.flatten()[:5]}")

        # Encoder Blocks
        x = enc_pre
        for b in range(ENC_BLOCKS):
            x = conv1d_bn(x, ws[f"enc_block_{b}_conv_w"], ws[f"enc_block_{b}_bn_s"],
                          ws[f"enc_block_{b}_bn_b"], F1, C1, C1, ENC_K, 1, ENC_PAD)
            x = silu(x)
            enc_skip.append(x.copy())
            if is_target:
                print(f"  Encoder Block {b}: rms={np.sqrt(np.mean(x**2)):.6e}")

        # RNNFormer PreNet: Linear(F1→F2) + Conv1d_BN(C1→C2, k=1)
        rf_a = linear_last_dim(ws["rf_pre_freq_w"], x, C1, F2, F1)
        rf_b = conv1d_bn(rf_a, ws["rf_pre_conv_w"], ws["rf_pre_bn_s"], ws["rf_pre_bn_b"],
                         F2, C1, C2, 1, 1, 0)
        rf_c = transpose_2d(rf_b, C2, F2)  # [C2, F2] → [F2, C2]

        if is_target:
            print(f"\nRNNFormer PreNet output [F2={F2}, C2={C2}]:")
            print(f"  rms={np.sqrt(np.mean(rf_c**2)):.6e}")

        # RNNFormer Blocks
        for blk in range(RF_BLOCKS):
            rf_c_flat = rf_c.flatten()

            # GRU: process each frequency bin independently
            for f in range(F2):
                x_f = rf_c_flat[f * C2:(f + 1) * C2].copy()
                h_f = gru_h[blk][f * C2:(f + 1) * C2].copy()

                h_new = gru_step(
                    x_f, h_f,
                    ws[f"rf_{blk}_gru_W_z"], ws[f"rf_{blk}_gru_U_z"], ws[f"rf_{blk}_gru_b_z"],
                    ws[f"rf_{blk}_gru_W_r"], ws[f"rf_{blk}_gru_U_r"], ws[f"rf_{blk}_gru_b_r"],
                    ws[f"rf_{blk}_gru_W_n"], ws[f"rf_{blk}_gru_U_n"],
                    ws[f"rf_{blk}_gru_b_in_n"], ws[f"rf_{blk}_gru_b_hn_n"], C2
                )
                gru_h[blk][f * C2:(f + 1) * C2] = h_new

                fc_w = ws[f"rf_{blk}_gru_fc_w"].reshape(C2, C2)
                fc_b = ws[f"rf_{blk}_gru_fc_b"]
                fc_out = fc_w @ h_new + fc_b
                rf_c_flat[f * C2:(f + 1) * C2] = x_f + fc_out

            # PE (block 0 only)
            if blk == 0:
                rf_c_flat += ws["rf_pe"]

            # MHSA + residual
            mhsa_out = mhsa(
                rf_c_flat,
                ws[f"rf_{blk}_mhsa_W_q"], ws[f"rf_{blk}_mhsa_b_q"],
                ws[f"rf_{blk}_mhsa_W_k"], ws[f"rf_{blk}_mhsa_b_k"],
                ws[f"rf_{blk}_mhsa_W_v"], ws[f"rf_{blk}_mhsa_b_v"],
                ws[f"rf_{blk}_mhsa_W_o"], ws[f"rf_{blk}_mhsa_b_o"],
                NUM_HEADS, HEAD_DIM, F2, C2
            )
            rf_c_flat += mhsa_out
            rf_c = rf_c_flat.reshape(F2, C2)

            if is_target:
                print(f"  RNNFormer Block {blk}: rms={np.sqrt(np.mean(rf_c**2)):.6e}")

        # RNNFormer PostNet: Transpose + Linear(F2→F1) + Conv1d_BN(C2→C1, k=1)
        rf_b2 = transpose_2d(rf_c, F2, C2)  # [F2, C2] → [C2, F2]
        rf_b2_f1 = linear_last_dim(ws["rf_post_freq_w"], rf_b2, C2, F1, F2)
        buf_a = conv1d_bn(rf_b2_f1, ws["rf_post_conv_w"], ws["rf_post_bn_s"], ws["rf_post_bn_b"],
                          F1, C2, C1, 1, 1, 0)

        if is_target:
            print(f"\nRNNFormer PostNet output [C1={C1}, F1={F1}]:")
            print(f"  rms={np.sqrt(np.mean(buf_a**2)):.6e}")

        # Decoder Blocks
        for b in range(ENC_BLOCKS):
            skip_idx = ENC_BLOCKS - b
            cat = np.concatenate([buf_a.flatten(), enc_skip[skip_idx].flatten()])
            cat = cat.reshape(2 * C1, F1)

            buf_b = conv1d_bn(cat, ws[f"dec_{b}_skip_conv_w"], ws[f"dec_{b}_skip_bn_s"],
                              ws[f"dec_{b}_skip_bn_b"], F1, 2 * C1, C1, 1, 1, 0)
            buf_b = silu(buf_b)

            buf_a = conv1d_bn(buf_b, ws[f"dec_{b}_conv_w"], ws[f"dec_{b}_bn_s"],
                              ws[f"dec_{b}_bn_b"], F1, C1, C1, ENC_K, 1, ENC_PAD)
            buf_a = silu(buf_a)

            if is_target:
                print(f"  Decoder Block {b}: rms={np.sqrt(np.mean(buf_a**2)):.6e}")

        # Decoder PostNet
        cat = np.concatenate([buf_a.flatten(), enc_skip[0].flatten()])
        cat = cat.reshape(2 * C1, F1)
        buf_b = conv1d_bn(cat, ws["dec_post_skip_conv_w"], ws["dec_post_skip_bn_s"],
                          ws["dec_post_skip_bn_b"], F1, 2 * C1, C1, 1, 1, 0)
        buf_b = silu(buf_b)

        mask = conv_transpose1d(buf_b, ws["dec_post_deconv_w"], ws["dec_post_deconv_b"],
                                F1, C1, 2, ENC_K0, STRIDE, ENC_PRE_PAD)

        if is_target:
            mask_re = mask[0]
            mask_im = mask[1]
            print(f"\nMask [2, 512]:")
            print(f"  mask_re: rms={np.sqrt(np.mean(mask_re**2)):.6e}")
            print(f"  mask_re first5: {mask_re[:5]}")
            print(f"  mask_im first5: {mask_im[:5]}")

        # Complex multiply: out = compressed * mask
        out_re = comp_re * mask[0] - comp_im * mask[1]
        out_im = comp_re * mask[1] + comp_im * mask[0]

        # Power decompress
        out_mag = np.sqrt(out_re**2 + out_im**2)
        dec_scale = np.where(out_mag > 1e-30,
                             np.power(np.maximum(out_mag, 1e-30), 1.0/COMPRESS_EXP - 1.0),
                             0.0).astype(np.float32)
        dec_re = (out_re * dec_scale).astype(np.float32)
        dec_im = (out_im * dec_scale).astype(np.float32)

        # Nyquist restore
        full_re = np.zeros(SPEC_BINS, dtype=np.float32)
        full_im = np.zeros(SPEC_BINS, dtype=np.float32)
        full_re[:FREQ_BINS] = dec_re
        full_im[:FREQ_BINS] = dec_im

        # iSTFT
        full_spec = full_re + 1j * full_im
        full_fft = np.zeros(N_FFT, dtype=np.complex64)
        full_fft[:SPEC_BINS] = full_spec
        full_fft[0] = full_fft[0].real
        full_fft[SPEC_BINS - 1] = full_fft[SPEC_BINS - 1].real
        for k in range(1, N_FFT - SPEC_BINS + 1):
            full_fft[SPEC_BINS - 1 + k] = np.conj(full_spec[SPEC_BINS - 1 - k])

        time_domain = np.fft.ifft(full_fft).real.astype(np.float32)
        output_frame = time_domain[:HOP_SIZE] + overlap
        overlap = time_domain[HOP_SIZE:].copy()

        if is_target:
            print(f"\nOutput frame {frame_idx}:")
            print(f"  rms={np.sqrt(np.mean(output_frame**2)):.6e}")
            print(f"  first5: {output_frame[:5]}")
            return output_frame

    return None


def compare_with_pytorch():
    """Compare against PyTorch intermediate files."""
    diag_dir = os.path.join(PROJECT_ROOT, "tests", "golden", "diag")
    if not os.path.exists(diag_dir):
        print("PyTorch intermediate values were not found. Run diag_intermediates.py first.")
        return

    print("\n=== Comparison with PyTorch Intermediates ===\n")

    files = {
        "pt_enc_pre_f0": "Encoder PreNet (frame 0)",
        "pt_enc_block0_f0": "Encoder Block 0 (frame 0)",
        "pt_enc_block1_f0": "Encoder Block 1 (frame 0)",
        "pt_rf_pre_f0": "RNNFormer PreNet (frame 0)",
        "pt_rf_block0_f0": "RNNFormer Block 0 (frame 0)",
        "pt_rf_block1_f0": "RNNFormer Block 1 (frame 0)",
        "pt_rf_post_f0": "RNNFormer PostNet (frame 0)",
        "pt_mask_f0": "Mask (frame 0)",
    }

    for fname, desc in files.items():
        path = os.path.join(diag_dir, f"{fname}.bin")
        if os.path.exists(path):
            data = np.fromfile(path, dtype=np.float32)
            print(f"{desc}: {data.shape}, rms={np.sqrt(np.mean(data**2)):.6e}")


def main():
    weight_path = os.path.join(PROJECT_ROOT, "weights", "fe_tiny_48k.bin")
    input_path = os.path.join(PROJECT_ROOT, "tests", "golden", "golden_input.bin")

    weights = load_weights(weight_path)
    ws = parse_weights(weights)

    input_np = np.fromfile(input_path, dtype=np.float32)
    print(f"Input: {len(input_np)} samples, RMS={np.sqrt(np.mean(input_np**2)):.6e}")

    # Simulation for frame 1
    output = simulate_c_pipeline(ws, input_np, target_frame=1)

    # Comparison with PyTorch intermediates
    compare_with_pytorch()


if __name__ == "__main__":
    main()
