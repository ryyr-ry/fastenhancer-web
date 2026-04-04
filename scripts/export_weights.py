#!/usr/bin/env python3
"""
export_weights.py — Convert a PyTorch FastEnhancer checkpoint into a binary for the C engine

Usage:
  python scripts/export_weights.py --checkpoint path/to/model.ckpt --model tiny --output weights/fe_tiny_48k.bin
  python scripts/export_weights.py --checkpoint path/to/model.ckpt --model tiny --dump-keys

Binary format (FEW1):
  [0:4]   Magic "FEW1" (ASCII)
  [4:8]   Version (uint32 LE, current=1)
  [8:12]  Model size (uint32 LE, 0=Tiny, 1=Base, 2=Small)
  [12:16] Weight count (uint32 LE, number of float32 elements)
  [16:20] CRC32 (uint32 LE, over the payload)
  [20:]   float32 LE payload
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MODEL_CONFIGS = {
    "tiny": {
        "model_id": 0,
        "C1": 24,
        "C2": 20,
        "F1": 128,
        "F2": 24,
        "enc_k0": 8,
        "enc_k": 3,
        "enc_blocks": 2,
        "rf_blocks": 2,
        "n_heads": 4,
        "head_dim": 5,
        "channels": 24,
        "freq": 24,
    },
    "base": {
        "model_id": 1,
        "C1": 48,
        "C2": 36,
        "F1": 128,
        "F2": 36,
        "enc_k0": 8,
        "enc_k": 3,
        "enc_blocks": 2,
        "rf_blocks": 3,
        "n_heads": 4,
        "head_dim": 9,
        "channels": 48,
        "freq": 36,
    },
    "small": {
        "model_id": 2,
        "C1": 64,
        "C2": 48,
        "F1": 128,
        "F2": 48,
        "enc_k0": 8,
        "enc_k": 3,
        "enc_blocks": 3,
        "rf_blocks": 3,
        "n_heads": 4,
        "head_dim": 12,
        "channels": 64,
        "freq": 48,
    },
}

BN_EPS = 1e-5


def compute_total_weights(cfg):
    """Compute the same value as FE_TOTAL_WEIGHTS on the C side."""
    C1, C2, F1, F2 = cfg["C1"], cfg["C2"], cfg["F1"], cfg["F2"]
    K0, K = cfg["enc_k0"], cfg["enc_k"]
    enc_b, rf_b = cfg["enc_blocks"], cfg["rf_blocks"]

    w_enc_prenet = C1 * 2 * K0 + 2 * C1
    w_enc_block = C1 * C1 * K + 2 * C1
    w_rf_prenet = F2 * F1 + C2 * C1 + 2 * C2
    w_gru = 6 * C2 * C2 + 4 * C2
    w_gru_fc = C2 * C2 + C2
    w_pe = F2 * C2
    w_mhsa = 4 * C2 * C2 + 4 * C2
    w_rf_block0 = w_gru + w_gru_fc + w_pe + w_mhsa
    w_rf_block_n = w_gru + w_gru_fc + w_mhsa
    w_rf_postnet = C1 * C2 + 2 * C1 + F1 * F2
    w_dec_block = C1 * 2 * C1 + 2 * C1 + C1 * C1 * K + 2 * C1
    w_dec_postnet = C1 * 2 * C1 + 2 * C1 + C1 * 2 * K0 + 2

    total = (
        w_enc_prenet
        + enc_b * w_enc_block
        + w_rf_prenet
        + w_rf_block0
        + (rf_b - 1) * w_rf_block_n
        + w_rf_postnet
        + enc_b * w_dec_block
        + w_dec_postnet
    )
    return total


def crc32_compute(data: bytes) -> int:
    """Use the same algorithm as crc32_compute on the C side (standard CRC-32)."""
    import zlib

    return zlib.crc32(data) & 0xFFFFFFFF


def resolve_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove weight_norm: w = g * (v / ||v||)

    In PyTorch weight_norm parametrizations:
      original0 = g (scaling factor)   [out_features, 1]
      original1 = v (direction vector) [out_features, in_features]
    Normalization is along dim=1 (the L2 norm of each row).
    """
    g = g.squeeze()
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_norm = np.maximum(v_norm, 1e-12)
    w = g[:, np.newaxis] * (v / v_norm)
    return w


def fuse_batchnorm(
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    conv_bias: np.ndarray | None = None,
    eps: float = BN_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BatchNorm into scale/offset.

    Returns: (scale, offset)
      y = x * scale + offset

    If conv_bias is provided, absorb it into the offset.
    """
    scale = bn_weight / np.sqrt(running_var + eps)
    if conv_bias is not None:
        offset = (conv_bias - running_mean) * scale + bn_bias
    else:
        offset = -running_mean * scale + bn_bias
    return scale.astype(np.float32), offset.astype(np.float32)


def split_gru_weights(
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
    hidden_size: int,
) -> list[np.ndarray]:
    """Convert fused PyTorch GRU weights into the separated C-side format.

    PyTorch gate order: [r(reset), z(update), n(new)]
    C-side storage order: W_z, U_z, b_z, W_r, U_r, b_r, W_n, U_n, b_in_n, b_hn_n
    """
    D = hidden_size
    r_ih, z_ih, n_ih = weight_ih[:D], weight_ih[D : 2 * D], weight_ih[2 * D : 3 * D]
    r_hh, z_hh, n_hh = weight_hh[:D], weight_hh[D : 2 * D], weight_hh[2 * D : 3 * D]
    br_ih, bz_ih, bn_ih = bias_ih[:D], bias_ih[D : 2 * D], bias_ih[2 * D : 3 * D]
    br_hh, bz_hh, bn_hh = bias_hh[:D], bias_hh[D : 2 * D], bias_hh[2 * D : 3 * D]

    W_z = z_ih
    U_z = z_hh
    b_z = bz_ih + bz_hh

    W_r = r_ih
    U_r = r_hh
    b_r = br_ih + br_hh

    W_n = n_ih
    U_n = n_hh
    b_in_n = bn_ih
    b_hn_n = bn_hh

    return [W_z, U_z, b_z, W_r, U_r, b_r, W_n, U_n, b_in_n, b_hn_n]


def split_qkv_weights(
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray | None,
    c2: int,
    n_heads: int,
    head_dim: int,
) -> list[np.ndarray]:
    """Split fused QKV weights from interleaved layout into Q/K/V.

    PyTorch fused QKV linear weight layout (row-wise):
      [Q_h0(head_dim rows), K_h0(head_dim rows), V_h0(head_dim rows),
       Q_h1(head_dim rows), K_h1(head_dim rows), V_h1(head_dim rows), ...]
    Full shape: [3*C2, C2] = [n_heads * 3 * head_dim, C2]

    The C engine expects separated W_q/W_k/W_v (each [C2, C2]), so extract
    the Q/K/V rows for each head and concatenate them.

    qkv_weight: [3*C2, C2]
    qkv_bias: [3*C2] or None

    Returns: [W_q, b_q, W_k, b_k, W_v, b_v]
    """
    assert qkv_weight.shape[0] == 3 * c2, (
        f"QKV weight row count mismatch: {qkv_weight.shape[0]} != 3*{c2}"
    )
    assert c2 == n_heads * head_dim, (
        f"C2 != n_heads*head_dim: {c2} != {n_heads}*{head_dim}"
    )

    in_dim = qkv_weight.shape[1]
    W_q = np.empty((c2, in_dim), dtype=np.float32)
    W_k = np.empty((c2, in_dim), dtype=np.float32)
    W_v = np.empty((c2, in_dim), dtype=np.float32)

    for h in range(n_heads):
        src = h * 3 * head_dim
        dst = h * head_dim
        W_q[dst : dst + head_dim] = qkv_weight[src : src + head_dim]
        W_k[dst : dst + head_dim] = qkv_weight[src + head_dim : src + 2 * head_dim]
        W_v[dst : dst + head_dim] = qkv_weight[src + 2 * head_dim : src + 3 * head_dim]

    if qkv_bias is not None:
        assert qkv_bias.shape[0] == 3 * c2, (
            f"QKV bias size mismatch: {qkv_bias.shape[0]} != 3*{c2}"
        )
        b_q = np.empty(c2, dtype=np.float32)
        b_k = np.empty(c2, dtype=np.float32)
        b_v = np.empty(c2, dtype=np.float32)
        for h in range(n_heads):
            src = h * 3 * head_dim
            dst = h * head_dim
            b_q[dst : dst + head_dim] = qkv_bias[src : src + head_dim]
            b_k[dst : dst + head_dim] = qkv_bias[src + head_dim : src + 2 * head_dim]
            b_v[dst : dst + head_dim] = qkv_bias[src + 2 * head_dim : src + 3 * head_dim]
    else:
        b_q = np.zeros(c2, dtype=np.float32)
        b_k = np.zeros(c2, dtype=np.float32)
        b_v = np.zeros(c2, dtype=np.float32)

    return [W_q, b_q, W_k, b_k, W_v, b_v]


def get_param(sd: dict, key: str) -> np.ndarray:
    """Fetch a parameter from state_dict and convert it to a NumPy array."""
    if key not in sd:
        raise KeyError(f"Key '{key}' does not exist in state_dict")
    val = sd[key]
    if hasattr(val, "cpu"):
        return val.cpu().numpy().astype(np.float32)
    return np.array(val, dtype=np.float32)


def try_get_param(sd: dict, key: str) -> np.ndarray | None:
    """Fetch a parameter from state_dict, or return None if missing."""
    if key not in sd:
        return None
    val = sd[key]
    if hasattr(val, "cpu"):
        return val.cpu().numpy().astype(np.float32)
    return np.array(val, dtype=np.float32)


def resolve_gru_weight(sd: dict, prefix: str, name: str) -> np.ndarray:
    """Resolve a GRU weight (supports both with and without weight_norm).

    When weight_norm is enabled: parametrizations.{name}.original0 (g), original1 (v)
    When weight_norm is disabled: {name} (direct)
    """
    param_g_key = f"{prefix}.parametrizations.{name}.original0"
    param_v_key = f"{prefix}.parametrizations.{name}.original1"
    direct_key = f"{prefix}.{name}"

    if param_g_key in sd and param_v_key in sd:
        g = get_param(sd, param_g_key)
        v = get_param(sd, param_v_key)
        return resolve_weight_norm(g, v)
    elif direct_key in sd:
        return get_param(sd, direct_key)
    else:
        raise KeyError(
            f"GRU weight '{name}' was not found (prefix='{prefix}')"
        )


def resolve_attn_qkv_weight(sd: dict, prefix: str) -> np.ndarray:
    """Resolve attention QKV weights (supports both with and without weight_norm)."""
    param_g_key = f"{prefix}.parametrizations.weight.original0"
    param_v_key = f"{prefix}.parametrizations.weight.original1"
    direct_key = f"{prefix}.weight"

    if param_g_key in sd and param_v_key in sd:
        g = get_param(sd, param_g_key)
        v = get_param(sd, param_v_key)
        return resolve_weight_norm(g, v)
    elif direct_key in sd:
        return get_param(sd, direct_key)
    else:
        raise KeyError(f"QKV weights were not found (prefix='{prefix}')")


def extract_bn_params(
    sd: dict, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch BatchNorm parameters."""
    w = get_param(sd, f"{prefix}.weight")
    b = get_param(sd, f"{prefix}.bias")
    mean = get_param(sd, f"{prefix}.running_mean")
    var = get_param(sd, f"{prefix}.running_var")
    return w, b, mean, var


def convert_strided_conv_weight(
    w_pt: np.ndarray, in_ch: int, kernel_size: int, stride: int
) -> np.ndarray:
    """Convert StridedConv1d-format weights to standard Conv1d format.

    PyTorch StridedConv1d: [out_ch, in_ch*stride, kernel/stride]
    Standard Conv1d:       [out_ch, in_ch, kernel_size]

    Conversion: w_std[oc, ic, k] = w_strided[oc, ic*stride + k%stride, k//stride]
    """
    out_ch = w_pt.shape[0]
    w_std = np.zeros((out_ch, in_ch, kernel_size), dtype=np.float32)
    for oc in range(out_ch):
        for ic in range(in_ch):
            for k in range(kernel_size):
                vic = (k % stride) * in_ch + ic
                vk = k // stride
                w_std[oc, ic, k] = w_pt[oc, vic, vk]
    return w_std


def fuse_linear_bn(
    fc_w: np.ndarray,
    fc_bias: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = BN_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse Linear + BatchNorm.

    Linear: y = x @ W^T (+ b)
    BN: z = (y - mean) / sqrt(var+eps) * w + b

    After fusion: z = x @ fused_W^T + fused_b
      fused_W[i,:] = W[i,:] * scale[i]
      fused_b[i] = (fc_b[i] - mean[i]) * scale[i] + bn_b[i]   (0 if fc_b is absent)
    """
    scale = bn_weight / np.sqrt(running_var + eps)
    fused_w = fc_w * scale[:, np.newaxis]
    if fc_bias is not None:
        fused_b = (fc_bias - running_mean) * scale + bn_bias
    else:
        fused_b = -running_mean * scale + bn_bias
    return fused_w.astype(np.float32), fused_b.astype(np.float32)


def export_weights(sd: dict, cfg: dict) -> np.ndarray:
    """Generate a float32 array for the C engine from a PyTorch state_dict.

    Returns: np.ndarray (dtype=float32), shape=(FE_TOTAL_WEIGHTS,)

    PyTorch layer names → corresponding C weight order:
      enc_pre: [0]=StridedConv1d(2→C1,k=8,s=4), [1]=BN(C1)
      encoder.{b}: [0]=Conv1d(C1→C1,k=3), [1]=BN(C1)
      rf_pre: [0]=Linear(F1→F2), [1]=Conv1d(C1→C2,k=1), [2]=BN(C2)
      rf_block.{b}: .rnn=GRU(weight_norm), .rnn_fc=Linear, .rnn_post_norm=BN,
                     .pe=positional embedding, .attn.qkv=QKV(weight_norm),
                     .attn_fc=Linear, .attn_post_norm=BN
      rf_post: [0]=Linear(F2→F1), [1]=Conv1d(C2→C1,k=1), [2]=BN(C1)
      decoder.{b}: [0]=Conv1d(2C1→C1,k=1), [1]=BN(C1), [3]=Conv1d(C1→C1,k=3), [4]=BN(C1)
      dec_post: [0]=Conv1d(2C1→C1,k=1), [1]=BN(C1), [3]=ScaledConvTranspose1d(C1→2,k=8,s=4)
    """
    C1 = cfg["C1"]
    C2 = cfg["C2"]
    F1 = cfg["F1"]
    F2 = cfg["F2"]
    stride = 4

    parts: list[np.ndarray] = []

    def emit(arr: np.ndarray, label: str = "") -> None:
        flat = arr.flatten().astype(np.float32)
        parts.append(flat)

    # === 1. Encoder PreNet ===
    # StridedConv1d(2→C1, k=8, s=4): PyTorch shape (C1, 2*4, 8/4)=(24,8,2)
    enc_pre_w_pt = get_param(sd, "enc_pre.0.weight")
    enc_pre_w = convert_strided_conv_weight(enc_pre_w_pt, in_ch=2, kernel_size=8, stride=stride)
    enc_pre_bias = try_get_param(sd, "enc_pre.0.bias")
    bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, "enc_pre.1")
    bn_scale, bn_offset = fuse_batchnorm(bn_w, bn_b, bn_mean, bn_var, enc_pre_bias)
    emit(enc_pre_w, "enc_pre_conv_w")
    emit(bn_scale, "enc_pre_bn_s")
    emit(bn_offset, "enc_pre_bn_b")

    # === 2. Encoder Blocks ===
    # Conv1d(C1→C1, k=3, s=1): standard format, no conversion needed
    for b in range(cfg["enc_blocks"]):
        conv_w = get_param(sd, f"encoder.{b}.0.weight")
        conv_bias = try_get_param(sd, f"encoder.{b}.0.bias")
        bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, f"encoder.{b}.1")
        bn_scale, bn_offset = fuse_batchnorm(bn_w, bn_b, bn_mean, bn_var, conv_bias)
        emit(conv_w, f"enc_block_{b}_conv_w")
        emit(bn_scale, f"enc_block_{b}_bn_s")
        emit(bn_offset, f"enc_block_{b}_bn_b")

    # === 3. RNNFormer PreNet ===
    # rf_pre.0: Linear(F1→F2), shape=(F2, F1)
    # rf_pre.1: Conv1d(C1→C2, k=1), shape=(C2, C1, 1)
    # rf_pre.2: BN(C2)
    rf_pre_freq_w = get_param(sd, "rf_pre.0.weight")
    rf_pre_conv_w = get_param(sd, "rf_pre.1.weight")
    rf_pre_conv_bias = try_get_param(sd, "rf_pre.1.bias")
    bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, "rf_pre.2")
    bn_scale, bn_offset = fuse_batchnorm(bn_w, bn_b, bn_mean, bn_var, rf_pre_conv_bias)
    emit(rf_pre_freq_w, "rf_pre_freq_w")
    emit(rf_pre_conv_w, "rf_pre_conv_w")
    emit(bn_scale, "rf_pre_bn_s")
    emit(bn_offset, "rf_pre_bn_b")

    # === 4. RNNFormer Blocks ===
    for b in range(cfg["rf_blocks"]):
        prefix = f"rf_block.{b}"

        # --- GRU weights (resolve weight_norm + split gates) ---
        w_ih = resolve_gru_weight(sd, f"{prefix}.rnn", "weight_ih_l0")
        w_hh = resolve_gru_weight(sd, f"{prefix}.rnn", "weight_hh_l0")
        b_ih = get_param(sd, f"{prefix}.rnn.bias_ih_l0")
        b_hh = get_param(sd, f"{prefix}.rnn.bias_hh_l0")
        gru_parts = split_gru_weights(w_ih, w_hh, b_ih, b_hh, C2)
        for gp in gru_parts:
            emit(gp)

        # --- GRU FC + rnn_post_norm fusion ---
        gru_fc_w = get_param(sd, f"{prefix}.rnn_fc.weight")
        gru_fc_bias = try_get_param(sd, f"{prefix}.rnn_fc.bias")
        bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, f"{prefix}.rnn_post_norm")
        fused_fc_w, fused_fc_b = fuse_linear_bn(
            gru_fc_w, gru_fc_bias, bn_w, bn_b, bn_mean, bn_var
        )
        emit(fused_fc_w, f"rf_{b}_gru_fc_w")
        emit(fused_fc_b, f"rf_{b}_gru_fc_b")

        # --- Positional embedding (block 0 only) ---
        if b == 0:
            pe = get_param(sd, f"{prefix}.pe")
            emit(pe, "rf_pe")

        # --- MHSA QKV (resolve weight_norm + split) ---
        qkv_w = resolve_attn_qkv_weight(sd, f"{prefix}.attn.qkv")
        qkv_b = try_get_param(sd, f"{prefix}.attn.qkv.bias")
        qkv_parts = split_qkv_weights(
            qkv_w, qkv_b, C2, cfg["n_heads"], cfg["head_dim"]
        )
        for qp in qkv_parts:
            emit(qp)

        # --- MHSA output (attn_fc + attn_post_norm fusion) → W_o, b_o ---
        attn_fc_w = get_param(sd, f"{prefix}.attn_fc.weight")
        attn_fc_bias = try_get_param(sd, f"{prefix}.attn_fc.bias")
        bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, f"{prefix}.attn_post_norm")
        fused_o_w, fused_o_b = fuse_linear_bn(
            attn_fc_w, attn_fc_bias, bn_w, bn_b, bn_mean, bn_var
        )
        emit(fused_o_w, f"rf_{b}_attn_fc_w")
        emit(fused_o_b, f"rf_{b}_attn_fc_b")

    # === 5. RNNFormer PostNet ===
    # rf_post.0: Linear(F2→F1), shape=(F1, F2)=(128, 24)
    # rf_post.1: Conv1d(C2→C1, k=1), shape=(C1, C2, 1)=(24, 20, 1)
    # rf_post.2: BN(C1)
    rf_post_freq_w = get_param(sd, "rf_post.0.weight")
    rf_post_conv_w = get_param(sd, "rf_post.1.weight")
    rf_post_conv_bias = try_get_param(sd, "rf_post.1.bias")
    bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, "rf_post.2")
    bn_scale, bn_offset = fuse_batchnorm(bn_w, bn_b, bn_mean, bn_var, rf_post_conv_bias)
    emit(rf_post_conv_w, "rf_post_conv_w")
    emit(bn_scale, "rf_post_bn_s")
    emit(bn_offset, "rf_post_bn_b")
    emit(rf_post_freq_w, "rf_post_freq_w")

    # === 6. Decoder Blocks ===
    for b in range(cfg["enc_blocks"]):
        # skip: Conv1d(2C1→C1, k=1) + BN(C1)
        skip_w = get_param(sd, f"decoder.{b}.0.weight")
        skip_bias = try_get_param(sd, f"decoder.{b}.0.bias")
        bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, f"decoder.{b}.1")
        skip_bn_s, skip_bn_b = fuse_batchnorm(bn_w, bn_b, bn_mean, bn_var, skip_bias)
        emit(skip_w, f"dec_{b}_skip_conv_w")
        emit(skip_bn_s, f"dec_{b}_skip_bn_s")
        emit(skip_bn_b, f"dec_{b}_skip_bn_b")

        # main: Conv1d(C1→C1, k=3) + BN(C1)
        conv_w = get_param(sd, f"decoder.{b}.3.weight")
        conv_bias = try_get_param(sd, f"decoder.{b}.3.bias")
        bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, f"decoder.{b}.4")
        conv_bn_s, conv_bn_b = fuse_batchnorm(bn_w, bn_b, bn_mean, bn_var, conv_bias)
        emit(conv_w, f"dec_{b}_conv_w")
        emit(conv_bn_s, f"dec_{b}_bn_s")
        emit(conv_bn_b, f"dec_{b}_bn_b")

    # === 7. Decoder PostNet ===
    # dec_post.0: Conv1d(2C1→C1, k=1) + dec_post.1: BN(C1)
    dec_post_skip_w = get_param(sd, "dec_post.0.weight")
    dec_post_skip_bias = try_get_param(sd, "dec_post.0.bias")
    bn_w, bn_b, bn_mean, bn_var = extract_bn_params(sd, "dec_post.1")
    dec_post_skip_bn_s, dec_post_skip_bn_b = fuse_batchnorm(
        bn_w, bn_b, bn_mean, bn_var, dec_post_skip_bias
    )
    emit(dec_post_skip_w, "dec_post_skip_conv_w")
    emit(dec_post_skip_bn_s, "dec_post_skip_bn_s")
    emit(dec_post_skip_bn_b, "dec_post_skip_bn_b")

    # dec_post.3: ScaledConvTranspose1d(C1→2, k=8, s=4) + scale weight_norm
    # normalize=True: F.normalize(weight, dim=(0,1,2)) * scale
    # → Normalize by the L2 norm of the entire tensor (not per-row)
    deconv_w = get_param(sd, "dec_post.3.weight")
    deconv_scale = try_get_param(sd, "dec_post.3.scale")
    if deconv_scale is not None:
        total_norm = np.linalg.norm(deconv_w.flatten())
        total_norm = max(total_norm, 1e-12)
        w_normalized = deconv_w / total_norm
        scale_scalar = deconv_scale.item() if deconv_scale.ndim <= 1 else deconv_scale
        deconv_w = w_normalized * scale_scalar

    deconv_b = try_get_param(sd, "dec_post.3.bias")
    if deconv_b is None:
        deconv_b = np.zeros(2, dtype=np.float32)
    emit(deconv_w, "dec_post_deconv_w")
    emit(deconv_b, "dec_post_deconv_b")

    result = np.concatenate(parts)
    return result


def write_binary(weights: np.ndarray, model_id: int, output_path: Path) -> None:
    """Write a FEW1 binary file."""
    weight_count = len(weights)
    payload = weights.astype('<f4').tobytes()

    crc = crc32_compute(payload)

    header = struct.pack(
        "<4sIIII",
        b"FEW1",
        1,
        model_id,
        weight_count,
        crc,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(payload)

    total_size = len(header) + len(payload)
    print(f"Output: {output_path}")
    print(f"  Header: {len(header)} bytes")
    print(f"  Payload: {len(payload)} bytes ({weight_count} floats)")
    print(f"  Total: {total_size} bytes")
    print(f"  CRC32: 0x{crc:08X}")


def dump_state_dict_keys(sd: dict) -> None:
    """Print all state_dict keys and shapes."""
    print(f"state_dict key count: {len(sd)}")
    print("-" * 80)
    for key in sorted(sd.keys()):
        val = sd[key]
        if hasattr(val, "shape"):
            shape = tuple(val.shape)
            dtype = val.dtype
            print(f"  {key:60s}  shape={str(shape):20s}  dtype={dtype}")
        else:
            print(f"  {key:60s}  value={val}")


EXPECTED_WEIGHTS = {
    "tiny": 28354,
    "base": 101654,
    "small": 208242,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch FastEnhancer checkpoint to FEW1 binary"
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        help="Path to the PyTorch checkpoint (example: ckpt_tiny_48k/00500.pth)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "base", "small"],
        help="Model size",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path (default: weights/fe_{model}_48k.bin)",
    )
    parser.add_argument(
        "--dump-keys",
        action="store_true",
        help="Print all state_dict keys and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all three models (tiny/base/small) in one batch",
    )
    args = parser.parse_args()

    if args.all:
        ckpt_map = {
            "tiny": "ckpt_tiny_48k/00500.pth",
            "base": "ckpt_base_48k/00500.pth",
            "small": "ckpt_small_48k/00500.pth",
        }
        for model_name, ckpt_path in ckpt_map.items():
            ckpt_file = Path(ckpt_path)
            if not ckpt_file.exists():
                print(f"Warning: {ckpt_path} does not exist. Skipping.")
                continue
            print(f"\n{'='*60}")
            print(f"  Exporting the {model_name.upper()} model")
            print(f"{'='*60}")
            export_single(ckpt_file, model_name, None)
        return

    if not args.checkpoint or not args.model:
        parser.error("Please specify --checkpoint and --model (or use --all)")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint '{ckpt_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    export_single(ckpt_path, args.model, args.output, args.dump_keys)


def export_single(
    ckpt_path: Path,
    model_name: str,
    output_path_str: str | None,
    dump_keys: bool = False,
) -> None:
    """Export a single model."""
    import torch

    cfg = MODEL_CONFIGS[model_name]

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "model" in ckpt:
        sd = ckpt["model"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    print(f"state_dict key count: {len(sd)}")

    if dump_keys:
        dump_state_dict_keys(sd)
        return

    expected = compute_total_weights(cfg)
    expected_hardcoded = EXPECTED_WEIGHTS.get(model_name)
    print(f"Computed expected weight count: {expected}")
    if expected_hardcoded is not None:
        assert expected == expected_hardcoded, (
            f"Weight count computation mismatch: compute={expected} vs hardcoded={expected_hardcoded}"
        )
        print(f"Matches hardcoded value: {expected_hardcoded} ✓")

    print(f"Exporting weights...")
    weights = export_weights(sd, cfg)
    actual_count = len(weights)

    print(f"Exported weight count: {actual_count}")
    if actual_count != expected:
        print(
            f"Error: weight count mismatch! expected={expected}, actual={actual_count}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Weight count matches: {actual_count} == {expected} ✓")

    if output_path_str is None:
        output_path_str = f"weights/fe_{model_name}_48k.bin"
    out_path = Path(output_path_str)

    write_binary(weights, cfg["model_id"], out_path)
    print(f"\nFinished exporting the {model_name.upper()} model ✓")


if __name__ == "__main__":
    main()
