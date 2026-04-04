#!/usr/bin/env python3
"""
golden_vectors_streaming.py — streaming STFT版 golden vector生成

Cエンジンと同じstreaming STFTを使い、
PyTorch model_forward() でフレーム単位推論を行い、
golden_input.bin / golden_output.bin を生成する。

これにより STFT framing 差異（center=True vs streaming）が排除され、
NNパイプラインの数値一致のみをテストできる。

使い方:
  python scripts/golden_vectors_streaming.py

前提条件:
  - _ref_pytorch/ に https://github.com/aask1357/fastenhancer がクローン済み
  - ckpt_tiny_48k/00500.pth が存在
  - pip install torch numpy
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REF_PYTORCH_DIR = os.path.join(PROJECT_ROOT, "_ref_pytorch")

sys.path.insert(0, REF_PYTORCH_DIR)
_original_cwd = os.getcwd()
os.chdir(REF_PYTORCH_DIR)

import torch
import numpy as np

N_FRAMES = 40
HOP_SIZE = 512
N_FFT = 1024
FREQ_BINS = N_FFT // 2 + 1  # 513
NN_FREQ = FREQ_BINS - 1     # 512 (Nyquist除去後)
N_SAMPLES = N_FRAMES * HOP_SIZE
RANDOM_SEED = 42
INPUT_AMPLITUDE = 0.1
COMPRESS_EXP = 0.3
MAG_FLOOR = 1e-5

MODEL_CONFIGS = {
    "tiny": {
        "ckpt": os.path.join(PROJECT_ROOT, "ckpt_tiny_48k", "00500.pth"),
        "config": os.path.join(REF_PYTORCH_DIR, "configs", "fastenhancer_48khz", "t.yaml"),
        "output_dir": os.path.join(PROJECT_ROOT, "tests", "golden"),
    },
    "base": {
        "ckpt": os.path.join(PROJECT_ROOT, "ckpt_base_48k", "00500.pth"),
        "config": os.path.join(REF_PYTORCH_DIR, "configs", "fastenhancer_48khz", "b.yaml"),
        "output_dir": os.path.join(PROJECT_ROOT, "tests", "golden_base"),
    },
    "small": {
        "ckpt": os.path.join(PROJECT_ROOT, "ckpt_small_48k", "00500.pth"),
        "config": os.path.join(REF_PYTORCH_DIR, "configs", "fastenhancer_48khz", "s.yaml"),
        "output_dir": os.path.join(PROJECT_ROOT, "tests", "golden_small"),
    },
}

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Golden vector生成（streaming STFT版）")
    parser.add_argument("--model", "-m", choices=["tiny", "base", "small"], default="tiny",
                        help="モデルサイズ (default: tiny)")
    return parser.parse_args()


CKPT_PATH = None
CONFIG_PATH = None
OUTPUT_DIR = None


def hparams_to_dict(obj):
    """HParamsオブジェクトを再帰的にdictに変換"""
    if hasattr(obj, "items") and not isinstance(obj, dict):
        return {k: hparams_to_dict(v) for k, v in obj.items()}
    return obj


def load_model():
    """PyTorchモデルをロードして推論モードに設定"""
    from utils.hparams import get_hparams
    from models.fastenhancer.default.model import Model

    hps = get_hparams(config_dir=CONFIG_PATH)
    model_kwargs = hparams_to_dict(hps.model_kwargs)

    model = Model(**model_kwargs)

    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    prefix = ""
    sample_key = next(iter(state_dict.keys()), "")
    if sample_key.startswith("model."):
        prefix = "model."
    if prefix:
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    model.load_state_dict(state_dict)

    model.eval()
    return model


def generate_test_input():
    """決定論的テスト入力を生成（seed=42, 振幅0.1の白色雑音）"""
    rng = np.random.RandomState(RANDOM_SEED)
    return (rng.randn(N_SAMPLES).astype(np.float32) * INPUT_AMPLITUDE)


class StreamingSTFT:
    """Cエンジンと同一のstreaming STFT/iSTFT（float32精度）

    fe_stft_forward / fe_stft_inverse を完全再現:
    - 周期的Hann窓: 0.5 - 0.5*cos(2πn/N), N=n_fft
    - 分析: shift buffer → 窓適用 → FFT → 正の周波数ビン
    - 合成: 共役対称復元 → iFFT → overlap-add（合成窓なし）

    torch.fft を使用して float32 精度の FFT/iFFT を行い、
    Cエンジンの radix-2 float32 FFT との精度差を最小化する。
    """

    def __init__(self, n_fft=1024, hop_size=512):
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.freq_bins = n_fft // 2 + 1

        self.window = torch.tensor([
            0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)
            for n in range(n_fft)
        ], dtype=torch.float32)

        self.input_buffer = torch.zeros(n_fft, dtype=torch.float32)
        self.overlap = torch.zeros(hop_size, dtype=torch.float32)

    def reset(self):
        self.input_buffer.zero_()
        self.overlap.zero_()

    def forward(self, input_hop):
        """Cエンジンのfe_stft_forwardと同一のSTFT分析

        Returns: (spec_real[freq_bins], spec_imag[freq_bins]) as numpy float32
        """
        n = self.n_fft
        hop = self.hop_size

        self.input_buffer[:n - hop] = self.input_buffer[hop:].clone()
        self.input_buffer[n - hop:] = torch.from_numpy(input_hop.astype(np.float32))

        windowed = self.input_buffer * self.window

        spectrum = torch.fft.rfft(windowed)
        return spectrum.real.numpy().copy(), spectrum.imag.numpy().copy()

    def inverse(self, spec_real, spec_imag):
        """Cエンジンのfe_stft_inverseと同一のiSTFT合成

        DC成分とNyquist成分の虚部を0にし、irfftで合成。
        合成窓なし、overlap-addのみ。float32精度。
        """
        n = self.n_fft
        hop = self.hop_size
        bins = self.freq_bins

        spec_real = np.asarray(spec_real, dtype=np.float32).copy()
        spec_imag = np.asarray(spec_imag, dtype=np.float32).copy()
        spec_imag[0] = 0.0
        spec_imag[bins - 1] = 0.0

        half_spec = torch.complex(
            torch.from_numpy(spec_real),
            torch.from_numpy(spec_imag),
        )

        time_domain = torch.fft.irfft(half_spec, n=n)

        output = time_domain[:hop] + self.overlap
        self.overlap = time_domain[hop:].clone()

        return output.numpy().copy()


def power_compress_complex(re, im, exponent=0.3, floor=MAG_FLOOR):
    """Cエンジン fe_power_compress_complex と同一（float32精度）

    scale = mag^(exponent - 1) = mag^(-0.7)
    mag < floor の場合は出力0
    """
    re = re.astype(np.float32)
    im = im.astype(np.float32)
    mag = np.sqrt(re * re + im * im).astype(np.float32)
    scale_exp = np.float32(exponent - 1.0)

    valid = mag >= np.float32(floor)
    safe_mag = np.where(valid, mag, np.float32(1.0))
    scale = np.where(valid, np.power(safe_mag, scale_exp), np.float32(0.0)).astype(np.float32)
    out_re = np.where(valid, re * scale, np.float32(0.0)).astype(np.float32)
    out_im = np.where(valid, im * scale, np.float32(0.0)).astype(np.float32)
    return out_re, out_im


def power_decompress_complex(re, im, exponent=0.3, floor=MAG_FLOOR):
    """Cエンジン fe_power_decompress_complex と同一（float32精度）

    scale = mag^(1/exponent - 1) = mag^(2.333...)
    mag < floor の場合は出力0
    """
    re = re.astype(np.float32)
    im = im.astype(np.float32)
    mag = np.sqrt(re * re + im * im).astype(np.float32)
    scale_exp = np.float32(1.0 / exponent - 1.0)

    valid = mag >= np.float32(floor)
    safe_mag = np.where(valid, mag, np.float32(1.0))
    scale = np.where(valid, np.power(safe_mag, scale_exp), np.float32(0.0)).astype(np.float32)
    out_re = np.where(valid, re * scale, np.float32(0.0)).astype(np.float32)
    out_im = np.where(valid, im * scale, np.float32(0.0)).astype(np.float32)
    return out_re, out_im


def run_streaming_inference(model, test_input):
    """Streaming STFT + PyTorch model_forward + Streaming iSTFT

    1. streaming STFT（Cエンジンと同一フレーミング）
    2. Nyquist除去 + パワー圧縮
    3. PyTorch model_forward() per-frame with GRU cache
    4. 複素マスク適用（compressed × mask）
    5. パワー復号
    6. Nyquist復元 + streaming iSTFT
    """
    stft = StreamingSTFT(N_FFT, HOP_SIZE)
    output = np.zeros(N_SAMPLES, dtype=np.float32)

    cache_list = None

    for f in range(N_FRAMES):
        frame_input = test_input[f * HOP_SIZE : (f + 1) * HOP_SIZE]

        # Step 1: Streaming STFT → [513] real + [513] imag
        spec_re, spec_im = stft.forward(frame_input)

        # Step 2: Nyquist除去 → [512]
        nn_re = spec_re[:NN_FREQ].astype(np.float32)
        nn_im = spec_im[:NN_FREQ].astype(np.float32)

        # Step 3: パワー圧縮
        comp_re, comp_im = power_compress_complex(nn_re, nn_im, COMPRESS_EXP, MAG_FLOOR)

        # Step 4: [1, 512, 1, 2] テンソルに変換
        spec_tensor = torch.zeros(1, NN_FREQ, 1, 2, dtype=torch.float32)
        spec_tensor[0, :, 0, 0] = torch.from_numpy(comp_re)
        spec_tensor[0, :, 0, 1] = torch.from_numpy(comp_im)

        # Step 5: model_forward (GRU cache受け渡し)
        with torch.no_grad():
            if cache_list is None:
                mask, cache_list = model.model_forward(spec_tensor)
            else:
                mask, cache_list = model.model_forward(spec_tensor, *cache_list)

        mask_re = mask[0, :, 0, 0].numpy()
        mask_im = mask[0, :, 0, 1].numpy()

        # Step 6: 複素マスク適用 (compressed × mask)
        out_re = comp_re * mask_re - comp_im * mask_im
        out_im = comp_re * mask_im + comp_im * mask_re

        # Step 7: パワー復号
        dec_re, dec_im = power_decompress_complex(out_re, out_im, COMPRESS_EXP, MAG_FLOOR)

        # Step 8: Nyquist復元 → [513] (float32)
        full_re = np.zeros(FREQ_BINS, dtype=np.float32)
        full_im = np.zeros(FREQ_BINS, dtype=np.float32)
        full_re[:NN_FREQ] = dec_re.astype(np.float32)
        full_im[:NN_FREQ] = dec_im.astype(np.float32)

        # Step 9: Streaming iSTFT
        frame_output = stft.inverse(full_re, full_im)
        output[f * HOP_SIZE : (f + 1) * HOP_SIZE] = frame_output

        if f < 5 or f == N_FRAMES - 1:
            rms = np.sqrt(np.mean(frame_output**2))
            print(f"  Frame {f:2d}: output RMS={rms:.6e}")

    return output


def save_golden_vectors(test_input, test_output):
    """golden vectorをバイナリファイルとして保存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(OUTPUT_DIR, "golden_input.bin")
    output_path = os.path.join(OUTPUT_DIR, "golden_output.bin")

    test_input.tofile(input_path)
    test_output.tofile(output_path)

    print(f"\n入力: {input_path} ({os.path.getsize(input_path)} bytes)")
    print(f"出力: {output_path} ({os.path.getsize(output_path)} bytes)")


def print_diagnostics(test_input, test_output):
    """診断情報を出力"""
    print(f"\n=== Streaming Golden Vector 生成完了 ===")
    print(f"STFT方式:       streaming（Cエンジン互換、zero-init overlap）")
    print(f"フレーム数:     {N_FRAMES}")
    print(f"ホップサイズ:   {HOP_SIZE}")
    print(f"FFTサイズ:      {N_FFT}")
    print(f"サンプル数:     {N_SAMPLES}")
    print(f"乱数シード:     {RANDOM_SEED}")
    print(f"入力振幅:       {INPUT_AMPLITUDE}")
    print(f"圧縮指数:       {COMPRESS_EXP}")
    print(f"入力範囲:       [{test_input.min():.6f}, {test_input.max():.6f}]")
    print(f"出力範囲:       [{test_output.min():.6f}, {test_output.max():.6f}]")
    print(f"出力RMS:        {np.sqrt(np.mean(test_output**2)):.6e}")
    print(f"出力最大絶対値: {np.max(np.abs(test_output)):.6e}")

    has_nan = np.any(np.isnan(test_output))
    has_inf = np.any(np.isinf(test_output))
    print(f"NaN検出:        {'あり ⚠' if has_nan else 'なし ✓'}")
    print(f"Inf検出:        {'あり ⚠' if has_inf else 'なし ✓'}")

    print(f"\nフレーム別RMS (先頭10フレーム):")
    for f in range(min(10, N_FRAMES)):
        frame = test_output[f * HOP_SIZE : (f + 1) * HOP_SIZE]
        rms = np.sqrt(np.mean(frame**2))
        print(f"  Frame {f:2d}: RMS={rms:.6e}")


def main():
    global CKPT_PATH, CONFIG_PATH, OUTPUT_DIR

    args = parse_args()
    cfg = MODEL_CONFIGS[args.model]
    CKPT_PATH = cfg["ckpt"]
    CONFIG_PATH = cfg["config"]
    OUTPUT_DIR = cfg["output_dir"]

    print(f"Streaming STFT + PyTorch model_forward() で golden vector生成中... (model={args.model})")
    print(f"  STFT: streaming（zero-init, hop={HOP_SIZE}, n_fft={N_FFT}）")
    print(f"  NN:   PyTorch model_forward() frame-by-frame with GRU cache")
    print(f"  CKPT: {CKPT_PATH}")
    print()

    model = load_model()
    print("モデルロード完了")

    test_input = generate_test_input()
    print(f"テスト入力生成完了: {N_SAMPLES} samples")
    print()

    print("フレーム単位推論:")
    test_output = run_streaming_inference(model, test_input)
    print(f"\n推論完了: {len(test_output)} samples")

    save_golden_vectors(test_input, test_output)
    print_diagnostics(test_input, test_output)

    os.chdir(_original_cwd)


if __name__ == "__main__":
    main()
