#!/usr/bin/env python3
"""
golden_vectors.py — Generate golden vectors using PyTorch inference

Run deterministic test input through the 48 kHz Tiny model and output
golden_input.bin / golden_output.bin for comparison with the C engine.

Usage:
  python scripts/golden_vectors.py

Prerequisites:
  - https://github.com/aask1357/fastenhancer has been cloned into _ref_pytorch/
  - ckpt_tiny_48k/00500.pth exists
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
    """Recursively convert an HParams object into a dict."""
    if hasattr(obj, "items") and not isinstance(obj, dict):
        return {k: hparams_to_dict(v) for k, v in obj.items()}
    return obj


def load_model():
    """Load the PyTorch model and switch it to inference mode."""
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
    """Generate deterministic test input (seed=42, white noise with amplitude 0.1)."""
    rng = np.random.RandomState(RANDOM_SEED)
    return (rng.randn(N_SAMPLES).astype(np.float32) * INPUT_AMPLITUDE)


def run_inference(model, test_input):
    """Run model inference and return the output waveform."""
    x = torch.from_numpy(test_input).unsqueeze(0)
    with torch.no_grad():
        wav_out, _ = model(x)
    output = wav_out.squeeze(0).numpy()

    if len(output) != N_SAMPLES:
        print(f"Warning: output length {len(output)} != expected value {N_SAMPLES}")
        if len(output) > N_SAMPLES:
            output = output[:N_SAMPLES]
        else:
            output = np.pad(output, (0, N_SAMPLES - len(output)))

    return output


def save_golden_vectors(test_input, test_output):
    """Save golden vectors as binary files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(OUTPUT_DIR, "golden_input.bin")
    output_path = os.path.join(OUTPUT_DIR, "golden_output.bin")

    test_input.tofile(input_path)
    test_output.tofile(output_path)

    print(f"Input: {input_path} ({os.path.getsize(input_path)} bytes)")
    print(f"Output: {output_path} ({os.path.getsize(output_path)} bytes)")


def print_diagnostics(test_input, test_output):
    """Print diagnostic information."""
    print(f"\n=== Golden Vector Generation Complete ===")
    print(f"Frame count:        {N_FRAMES}")
    print(f"Hop size:           {HOP_SIZE}")
    print(f"Sample count:       {N_SAMPLES}")
    print(f"Random seed:        {RANDOM_SEED}")
    print(f"Input amplitude:    {INPUT_AMPLITUDE}")
    print(f"Input range:        [{test_input.min():.6f}, {test_input.max():.6f}]")
    print(f"Output range:       [{test_output.min():.6f}, {test_output.max():.6f}]")
    print(f"Output RMS:         {np.sqrt(np.mean(test_output**2)):.6e}")
    print(f"Output max abs:     {np.max(np.abs(test_output)):.6e}")

    has_nan = np.any(np.isnan(test_output))
    has_inf = np.any(np.isinf(test_output))
    print(f"NaN detected:       {'yes ⚠' if has_nan else 'no ✓'}")
    print(f"Inf detected:       {'yes ⚠' if has_inf else 'no ✓'}")

    print(f"\nPer-frame RMS (first 10 frames):")
    for f in range(min(10, N_FRAMES)):
        frame = test_output[f * HOP_SIZE : (f + 1) * HOP_SIZE]
        rms = np.sqrt(np.mean(frame**2))
        print(f"  Frame {f:2d}: RMS={rms:.6e}")


def main():
    print("Generating golden vectors with the PyTorch 48 kHz Tiny model...")

    model = load_model()
    print("Model load complete")

    test_input = generate_test_input()
    print(f"Test input generation complete: {N_SAMPLES} samples")

    test_output = run_inference(model, test_input)
    print(f"Inference complete: {len(test_output)} samples")

    save_golden_vectors(test_input, test_output)
    print_diagnostics(test_input, test_output)

    os.chdir(_original_cwd)


if __name__ == "__main__":
    main()
