#!/usr/bin/env python3
"""
golden_vectors.py — PyTorch推論によるgolden vector生成

48kHz Tinyモデルで決定論的テスト入力を推論し、
C エンジンとの比較用 golden_input.bin / golden_output.bin を出力する。

使い方:
  python scripts/golden_vectors.py

前提条件:
  - _ref_pytorch/ に https://github.com/aask1357/fastenhancer がクローン済み
  - ckpt_tiny_48k/00500.pth が存在
  - pip install torch torchaudio numpy tensorboard pesq pystoi librosa
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
N_SAMPLES = N_FRAMES * HOP_SIZE
RANDOM_SEED = 42
INPUT_AMPLITUDE = 0.1

CKPT_PATH = os.path.join(PROJECT_ROOT, "ckpt_tiny_48k", "00500.pth")
CONFIG_PATH = os.path.join(REF_PYTORCH_DIR, "configs", "fastenhancer_48khz", "t.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "golden")


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
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.eval()
    return model


def generate_test_input():
    """決定論的テスト入力を生成（seed=42, 振幅0.1の白色雑音）"""
    rng = np.random.RandomState(RANDOM_SEED)
    return (rng.randn(N_SAMPLES).astype(np.float32) * INPUT_AMPLITUDE)


def run_inference(model, test_input):
    """モデル推論を実行し出力波形を返す"""
    x = torch.from_numpy(test_input).unsqueeze(0)
    with torch.no_grad():
        wav_out, _ = model(x)
    output = wav_out.squeeze(0).numpy()

    if len(output) != N_SAMPLES:
        print(f"警告: 出力長 {len(output)} != 期待値 {N_SAMPLES}")
        if len(output) > N_SAMPLES:
            output = output[:N_SAMPLES]
        else:
            output = np.pad(output, (0, N_SAMPLES - len(output)))

    return output


def save_golden_vectors(test_input, test_output):
    """golden vectorをバイナリファイルとして保存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(OUTPUT_DIR, "golden_input.bin")
    output_path = os.path.join(OUTPUT_DIR, "golden_output.bin")

    test_input.tofile(input_path)
    test_output.tofile(output_path)

    print(f"入力: {input_path} ({os.path.getsize(input_path)} bytes)")
    print(f"出力: {output_path} ({os.path.getsize(output_path)} bytes)")


def print_diagnostics(test_input, test_output):
    """診断情報を出力"""
    print(f"\n=== Golden Vector 生成完了 ===")
    print(f"フレーム数:     {N_FRAMES}")
    print(f"ホップサイズ:   {HOP_SIZE}")
    print(f"サンプル数:     {N_SAMPLES}")
    print(f"乱数シード:     {RANDOM_SEED}")
    print(f"入力振幅:       {INPUT_AMPLITUDE}")
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
    print("PyTorch 48kHz Tiny モデルでgolden vector生成中...")

    model = load_model()
    print("モデルロード完了")

    test_input = generate_test_input()
    print(f"テスト入力生成完了: {N_SAMPLES} samples")

    test_output = run_inference(model, test_input)
    print(f"推論完了: {len(test_output)} samples")

    save_golden_vectors(test_input, test_output)
    print_diagnostics(test_input, test_output)

    os.chdir(_original_cwd)


if __name__ == "__main__":
    main()
