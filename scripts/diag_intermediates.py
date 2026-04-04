#!/usr/bin/env python3
"""
diag_intermediates.py — Export intermediate values from each PyTorch inference stage

Used for comparison against the C engine. Saves the output of each pipeline stage.
"""

import sys
import os
import struct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REF_PYTORCH_DIR = os.path.join(PROJECT_ROOT, "_ref_pytorch")

sys.path.insert(0, REF_PYTORCH_DIR)
_original_cwd = os.getcwd()
os.chdir(REF_PYTORCH_DIR)

import torch
import torch.nn.functional as F
import numpy as np

HOP_SIZE = 512
N_FFT = 1024
COMPRESS_EXP = 0.3
FREQ_BINS = 512

CKPT_PATH = os.path.join(PROJECT_ROOT, "ckpt_tiny_48k", "00500.pth")
CONFIG_PATH = os.path.join(REF_PYTORCH_DIR, "configs", "fastenhancer_48khz", "t.yaml")
GOLDEN_INPUT = os.path.join(PROJECT_ROOT, "tests", "golden", "golden_input.bin")
DIAG_DIR = os.path.join(PROJECT_ROOT, "tests", "golden", "diag")


def load_model():
    from utils.hparams import get_hparams
    from models.fastenhancer.default.model import Model

    def hparams_to_dict(obj):
        if hasattr(obj, "items") and not isinstance(obj, dict):
            return {k: hparams_to_dict(v) for k, v in obj.items()}
        return obj

    hps = get_hparams(config_dir=CONFIG_PATH)
    model_kwargs = hparams_to_dict(hps.model_kwargs)
    model = Model(**model_kwargs)

    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_bin(name, arr):
    path = os.path.join(DIAG_DIR, f"{name}.bin")
    arr.astype(np.float32).tofile(path)
    print(f"  {name}: shape={arr.shape}, min={arr.min():.6e}, max={arr.max():.6e}, "
          f"rms={np.sqrt(np.mean(arr**2)):.6e}")


def print_first(name, arr, n=5):
    flat = arr.flatten()
    vals = ", ".join(f"{v:.6e}" for v in flat[:n])
    print(f"  {name} first {n}: [{vals}]")


def diag_stft_frame0(model, input_np):
    """Compute frame 0 STFT output using both PyTorch and C-style streaming."""
    stft_module = model.stft

    print("\n=== STFT Frame 0 Diagnostics ===")

    # PyTorch center=True STFT (the method used by the full model)
    x = torch.from_numpy(input_np).unsqueeze(0)  # [1, N_SAMPLES]
    with torch.no_grad():
        spec_full = torch.stft(
            x, N_FFT, HOP_SIZE, N_FFT,
            window=torch.hann_window(N_FFT),
            center=True, pad_mode="reflect",
            return_complex=True
        )
    # spec_full: [1, 513, T]
    frame0_spec = spec_full[0, :, 0].numpy()  # [513] complex
    frame0_re = frame0_spec.real.astype(np.float32)
    frame0_im = frame0_spec.imag.astype(np.float32)

    print(f"PyTorch STFT frame 0 (center=True):")
    save_bin("pt_stft_frame0_re", frame0_re)
    save_bin("pt_stft_frame0_im", frame0_im)
    print_first("re", frame0_re)
    print_first("im", frame0_im)

    # C-style streaming STFT: apply a Hann window to [zeros(512) | input[0:512]]
    c_buffer = np.zeros(N_FFT, dtype=np.float32)
    c_buffer[HOP_SIZE:] = input_np[:HOP_SIZE]  # Input in the second half

    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    hann = hann.astype(np.float32)
    windowed = c_buffer * hann

    c_spec = np.fft.rfft(windowed)  # [513] complex
    c_re = c_spec.real.astype(np.float32)
    c_im = c_spec.imag.astype(np.float32)

    print(f"\nC-style streaming STFT frame 0:")
    save_bin("c_sim_stft_frame0_re", c_re)
    save_bin("c_sim_stft_frame0_im", c_im)
    print_first("re", c_re)
    print_first("im", c_im)

    # Comparison
    diff_re = np.abs(frame0_re - c_re)
    diff_im = np.abs(frame0_im - c_im)
    print(f"\nSTFT difference: max_diff_re={diff_re.max():.6e}, max_diff_im={diff_im.max():.6e}")

    return frame0_re, frame0_im, c_re, c_im


def diag_compression(pt_re, pt_im):
    """Intermediate values for power compression."""
    print("\n=== Power Compression Frame 0 Diagnostics ===")

    # Remove Nyquist: [513] → [512]
    re_512 = pt_re[:FREQ_BINS]
    im_512 = pt_im[:FREQ_BINS]

    mag = np.sqrt(re_512**2 + im_512**2)
    scale = np.where(mag > 1e-30, np.power(np.maximum(mag, 1e-30), COMPRESS_EXP - 1.0), 0.0)
    comp_re = re_512 * scale
    comp_im = im_512 * scale

    save_bin("pt_compressed_re", comp_re)
    save_bin("pt_compressed_im", comp_im)
    print_first("compressed re", comp_re)
    print_first("compressed im", comp_im)

    # PyTorch method (CompressedSTFT.forward)
    mag_pt = np.sqrt(comp_re**2 + comp_im**2)
    print(f"  compressed mag: min={mag_pt.min():.6e}, max={mag_pt.max():.6e}")

    return comp_re, comp_im


def diag_model_intermediates(model, input_np):
    """Export Encoder/RNNFormer/Decoder intermediates from model inference."""
    print("\n=== Model Internal Intermediate Diagnostics ===")

    x = torch.from_numpy(input_np).unsqueeze(0)  # [1, N_SAMPLES]

    with torch.no_grad():
        # CompressedSTFT
        spec = model.stft(x)  # [B, F, T, 2]
        print(f"CompressedSTFT output: shape={spec.shape}")
        # Frame 0
        spec_f0 = spec[0, :, 0, :].numpy()  # [F, 2]
        save_bin("pt_model_spec_f0", spec_f0)
        print_first("spec_f0 re (first 5 freq bins)", spec_f0[:5, 0])
        print_first("spec_f0 im (first 5 freq bins)", spec_f0[:5, 1])

        # Manually step through model_forward()
        B, F_dim, T, C2dim = spec.shape
        print(f"  B={B}, F={F_dim}, T={T}, C2dim={C2dim}")

        # Reshape: [B, F, T, 2] → [B*T, 2, F]
        x_enc = spec.permute(0, 2, 3, 1).reshape(B * T, C2dim, F_dim)
        print(f"\nEncoder input (B*T, 2, F): shape={x_enc.shape}")
        enc_input_f0 = x_enc[0].numpy()  # [2, F] for frame 0
        save_bin("pt_enc_input_f0", enc_input_f0)
        print_first("enc_input_f0 ch0 (real)", enc_input_f0[0, :5])
        print_first("enc_input_f0 ch1 (imag)", enc_input_f0[1, :5])

        # Encoder PreNet
        enc_pre_out = model.enc_pre(x_enc)
        print(f"\nEncoder PreNet output: shape={enc_pre_out.shape}")
        enc_pre_f0 = enc_pre_out[0].numpy()  # [C1, F1]
        save_bin("pt_enc_pre_f0", enc_pre_f0)
        print_first("enc_pre_f0", enc_pre_f0.flatten())
        print(f"  enc_pre_f0 rms={np.sqrt(np.mean(enc_pre_f0**2)):.6e}")

        # Encoder Blocks
        encoder_outs = [enc_pre_out]
        x_blk = enc_pre_out
        resnet = model.resnet
        for idx, module in enumerate(model.encoder):
            x_in = x_blk
            x_blk = module(x_blk)
            if resnet:
                x_blk = x_blk + x_in
            encoder_outs.append(x_blk)
            blk_f0 = x_blk[0].numpy()
            print(f"\nEncoder Block {idx} output: rms={np.sqrt(np.mean(blk_f0**2)):.6e}")
            save_bin(f"pt_enc_block{idx}_f0", blk_f0)

        # RNNFormer PreNet
        x_in_rf = x_blk  # save for residual
        x_rf = model.rf_pre(x_blk)
        rf_pre_f0 = x_rf[0].numpy()
        print(f"\nRNNFormer PreNet output: shape={x_rf.shape}, rms={np.sqrt(np.mean(rf_pre_f0**2)):.6e}")
        save_bin("pt_rf_pre_f0", rf_pre_f0)

        # Reshape for RNNFormer: [B*T, C2, F2] → [T, B, F2, C2]
        x_rf_reshaped = x_rf.reshape(B, T, -1, x_rf.shape[-1])
        x_rf_reshaped = x_rf_reshaped.permute(1, 0, 3, 2)
        print(f"  RNNFormer input reshaped: shape={x_rf_reshaped.shape}")

        # RNNFormer Blocks
        cache_in = [None] * len(model.rf_block)
        for bidx, block in enumerate(model.rf_block):
            x_rf_reshaped, _ = block(x_rf_reshaped, cache_in[bidx])
            rf_blk_f0 = x_rf_reshaped[0, 0].numpy()  # [F2, C2] for frame 0
            print(f"\nRNNFormer Block {bidx} output (frame 0): shape={rf_blk_f0.shape}, "
                  f"rms={np.sqrt(np.mean(rf_blk_f0**2)):.6e}")
            save_bin(f"pt_rf_block{bidx}_f0", rf_blk_f0)

        # Reshape back: [T, B, F2, C2] → [B*T, C2, F2]
        x_rf_back = x_rf_reshaped.permute(1, 0, 3, 2).reshape(B * T, -1, x_rf_reshaped.shape[2])
        x_rf_back = model.rf_post(x_rf_back)
        if resnet:
            x_rf_back = x_rf_back + x_in_rf
        rf_post_f0 = x_rf_back[0].numpy()
        print(f"\nRNNFormer PostNet output: shape={x_rf_back.shape}, rms={np.sqrt(np.mean(rf_post_f0**2)):.6e}")
        save_bin("pt_rf_post_f0", rf_post_f0)

        # Decoder
        x_dec = x_rf_back
        for didx, module in enumerate(model.decoder):
            x_in_dec = x_dec
            enc_out = encoder_outs.pop(-1)
            x_dec = torch.cat([x_dec, enc_out], dim=1)
            x_dec = module(x_dec)
            if resnet:
                x_dec = x_dec + x_in_dec
            dec_f0 = x_dec[0].numpy()
            print(f"\nDecoder Block {didx} output: rms={np.sqrt(np.mean(dec_f0**2)):.6e}")
            save_bin(f"pt_dec_block{didx}_f0", dec_f0)

        # Decoder PostNet
        enc_out_first = encoder_outs.pop(-1)
        x_dec = torch.cat([x_dec, enc_out_first], dim=1)
        x_dec = model.dec_post(x_dec)
        dec_post_f0 = x_dec[0].numpy()  # [2, F] mask
        print(f"\nDecoder PostNet output (mask): shape={x_dec.shape}, rms={np.sqrt(np.mean(dec_post_f0**2)):.6e}")
        save_bin("pt_mask_f0", dec_post_f0)
        print_first("mask_re", dec_post_f0[0, :5])
        print_first("mask_im", dec_post_f0[1, :5])

        # mask activation (if any)
        mask_activated = model.mask(x_dec)
        mask_act_f0 = mask_activated[0].numpy()
        print(f"\nAfter mask activation: rms={np.sqrt(np.mean(mask_act_f0**2)):.6e}")
        save_bin("pt_mask_activated_f0", mask_act_f0)

        # Comparison with the full inference result
        wav_out, _ = model(torch.from_numpy(input_np).unsqueeze(0))
        wav_f0 = wav_out[0, :HOP_SIZE].numpy()
        print(f"\nFull inference frame 0: rms={np.sqrt(np.mean(wav_f0**2)):.6e}")
        save_bin("pt_output_f0", wav_f0)


def main():
    os.makedirs(DIAG_DIR, exist_ok=True)

    print("=== PyTorch Intermediate Diagnostics ===\n")
    print(f"Input: {GOLDEN_INPUT}")

    input_np = np.fromfile(GOLDEN_INPUT, dtype=np.float32)
    print(f"Sample count: {len(input_np)}, RMS: {np.sqrt(np.mean(input_np**2)):.6e}")

    model = load_model()
    print("Model load complete\n")

    # STFT comparison
    pt_re, pt_im, c_re, c_im = diag_stft_frame0(model, input_np)

    # Power compression
    diag_compression(pt_re, pt_im)

    # Model intermediates
    diag_model_intermediates(model, input_np)

    os.chdir(_original_cwd)
    print("\n=== Diagnostics Complete ===")
    print(f"Intermediate files: {DIAG_DIR}")


if __name__ == "__main__":
    main()
