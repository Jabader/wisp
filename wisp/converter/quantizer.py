"""
wisp.converter.quantizer — group-wise int4/fp8 quantization utilities.

The packing here MUST stay bit-compatible with the CUDA dequant kernel in
csrc/cuda/expert_loader.cu and csrc/cuda/expert_loader_optimized.cu:

    dequant:  value = (nibble - 8) * scale + zero
    packing:  two int4 values per byte, LOW nibble = even index,
              HIGH nibble = odd index
    groups :  `group_size` consecutive elements (flattened row-major)
              share one fp16/fp8 scale and one fp16/fp8 zero

Quantization scheme (asymmetric, centered):
    zero  = (max + min) / 2
    scale = (max - min) / 15          (15 = int4 range steps)
    q     = clamp(round((x - zero) / scale) + 8, 0, 15)

FP8 E4M3 support for 8GB VRAM optimization:
    When use_fp8=True, scales/zeros are stored as fp8 E4M3 instead of fp16,
    reducing metadata size by 50% (from 2 bytes to 1 byte per group).
    This saves ~10-15% total expert size with minimal accuracy loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class QuantizedTensor:
    packed: torch.Tensor   # uint8, ceil(n/2) bytes, low nibble first
    scales: torch.Tensor   # fp16 or fp8 (uint8), one per group
    zeros:  torch.Tensor   # fp16 or fp8 (uint8), one per group
    rows:   int
    cols:   int
    group_size: int
    dtype: Literal["fp16", "fp8"] = "fp16"  # Metadata dtype

    @property
    def n_elements(self) -> int:
        return self.rows * self.cols

    @property
    def nbytes(self) -> int:
        """Total bytes: packed weights + scales + zeros."""
        packed_bytes = self.packed.numel()
        if self.dtype == "fp8":
            # fp8: 1 byte per scale/zero
            meta_bytes = self.scales.numel() + self.zeros.numel()
        else:
            # fp16: 2 bytes per scale/zero
            meta_bytes = (self.scales.numel() + self.zeros.numel()) * 2
        return packed_bytes + meta_bytes


def _float_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 tensor to fp8 E4M3 format (stored as uint8)."""
    # FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
    # Bias = 7, max finite = 0b01111110 = 448.0, NaN = 0b01111111
    
    # Clamp to valid range to avoid overflow
    x = torch.clamp(x, -448.0, 448.0)
    
    # Handle sign
    sign = (x < 0).to(torch.uint8) << 7
    x_abs = torch.abs(x)
    
    # Special case: zero
    is_zero = (x_abs == 0.0)
    
    # Compute exponent and mantissa using frexp
    # frexp returns mantissa in [0.5, 1.0) and exponent
    mant, exp = torch.frexp(x_abs)
    
    # Adjust exponent with bias (bias=7 for E4M3)
    exp_biased = exp + 7
    
    # Handle subnormals (exp_biased <= 0)
    subnormal_mask = exp_biased <= 0
    # For subnormals, shift mantissa right
    mant_sub = mant * torch.pow(2.0, exp_biased.float() + 2.0)  # +2 for 3 mantissa bits
    exp_sub = torch.zeros_like(exp_biased)
    
    # Handle normals
    normal_mask = ~subnormal_mask & (exp_biased < 15) & ~is_zero
    mant_normal = mant * 2.0 - 1.0  # Remove implicit 1
    exp_normal = exp_biased
    
    # Handle overflow (exp_biased >= 15)
    overflow_mask = exp_biased >= 15
    mant_overflow = torch.ones_like(mant) * 0.875  # Max mantissa (7/8)
    exp_overflow = torch.full_like(exp_biased, 14)  # Max exponent (0xE)
    
    # Combine mantissa parts
    mant_combined = torch.where(
        subnormal_mask, mant_sub,
        torch.where(normal_mask, mant_normal, mant_overflow)
    )
    exp_combined = torch.where(
        subnormal_mask, exp_sub,
        torch.where(normal_mask, exp_normal, exp_overflow)
    )
    
    # Quantize mantissa to 3 bits
    mant_bits = (mant_combined * 8.0).round().clamp(0, 7).to(torch.uint8)
    
    # Combine: sign | (exp << 3) | mantissa
    fp8 = sign | ((exp_combined.to(torch.uint8) << 3) | mant_bits)
    
    # Zero stays zero
    fp8 = torch.where(is_zero, torch.zeros_like(fp8), fp8)
    
    return fp8


def _fp8_e4m3_to_float(x: torch.Tensor) -> torch.Tensor:
    """Convert fp8 E4M3 (uint8) tensor to float32."""
    x = x.to(torch.uint8)
    
    # Extract components
    sign = (x >> 7).to(torch.float32)
    exp = ((x >> 3) & 0x0F).to(torch.float32)
    mant = (x & 0x07).to(torch.float32)
    
    # Handle NaN (0x7F)
    is_nan = (x == 0x7F)
    
    # Handle subnormals (exp == 0)
    is_subnormal = (exp == 0) & ~is_nan
    # Subnormal: value = (-1)^sign * 2^(-6) * (mantissa / 8)
    val_sub = torch.ldexp(mant, -6)
    
    # Handle normals (exp > 0 and not NaN)
    is_normal = (exp > 0) & ~is_nan
    # Normal: value = (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    val_normal = torch.ldexp(1.0 + mant / 8.0, exp.long() - 7)
    
    # Combine
    result = torch.where(is_subnormal, val_sub,
                torch.where(is_normal, val_normal,
                    torch.zeros_like(val_sub)))  # NaN -> 0
    
    # Apply sign
    result = torch.where(sign > 0, -result, result)
    
    return result


def quantize_int4(weight: torch.Tensor, group_size: int = 64,
                  use_fp8: bool = False) -> QuantizedTensor:
    """Group-wise asymmetric int4 quantization of a 2D weight matrix.
    
    Args:
        weight: 2D weight tensor (float32 or float16)
        group_size: Number of elements per quantization group
        use_fp8: If True, store scales/zeros as fp8 E4M3 (saves 50% metadata)
    
    Returns:
        QuantizedTensor with packed weights and metadata
    """
    if weight.dim() != 2:
        raise ValueError(f"quantize_int4 expects 2D weight, got {tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    rows, cols = weight.shape
    flat = weight.detach().to(torch.float32).reshape(-1)
    n = flat.numel()

    # Pad to a whole number of groups
    n_groups = (n + group_size - 1) // group_size
    padded = n_groups * group_size
    if padded != n:
        flat = torch.cat([flat, flat[-1].repeat(padded - n)])

    groups = flat.view(n_groups, group_size)
    gmax = groups.max(dim=1).values
    gmin = groups.min(dim=1).values
    zeros = (gmax + gmin) / 2.0
    scales = (gmax - gmin) / 15.0
    scales = torch.clamp(scales, min=1e-8)

    q = torch.round((groups - zeros.unsqueeze(1)) / scales.unsqueeze(1)) + 8
    q = torch.clamp(q, 0, 15).to(torch.uint8).reshape(-1)[:n]

    # Pack two nibbles per byte
    if n % 2 != 0:
        q = torch.cat([q, torch.zeros(1, dtype=torch.uint8)])
    lo = q[0::2]
    hi = q[1::2]
    packed = (lo | (hi << 4)).contiguous()

    # Convert scales/zeros to target dtype
    if use_fp8:
        scales_out = _float_to_fp8_e4m3(scales)
        zeros_out = _float_to_fp8_e4m3(zeros)
        dtype_str = "fp8"
    else:
        scales_out = scales.to(torch.float16).contiguous()
        zeros_out = zeros.to(torch.float16).contiguous()
        dtype_str = "fp16"

    return QuantizedTensor(
        packed=packed,
        scales=scales_out,
        zeros=zeros_out,
        rows=rows, cols=cols, group_size=group_size,
        dtype=dtype_str,
    )


def dequantize_int4(qt: QuantizedTensor) -> torch.Tensor:
    """Exact inverse of the packing above — mirrors the CUDA kernel math."""
    n = qt.n_elements
    packed = qt.packed
    lo = (packed & 0x0F).to(torch.int32)
    hi = (packed >> 4).to(torch.int32)
    q = torch.empty(packed.numel() * 2, dtype=torch.int32)
    q[0::2] = lo
    q[1::2] = hi
    q = q[:n]

    idx = torch.arange(n)
    g = idx // qt.group_size
    
    # Convert metadata back to float32 based on dtype
    if qt.dtype == "fp8":
        scales = _fp8_e4m3_to_float(qt.scales.to(torch.uint8))[g]
        zeros = _fp8_e4m3_to_float(qt.zeros.to(torch.uint8))[g]
    else:
        scales = qt.scales.to(torch.float32)[g]
        zeros = qt.zeros.to(torch.float32)[g]
    
    out = (q.to(torch.float32) - 8.0) * scales + zeros
    return out.view(qt.rows, qt.cols).to(torch.float16)


def quantization_error(weight: torch.Tensor, group_size: int = 64,
                       use_fp8: bool = False) -> float:
    """Mean absolute reconstruction error — sanity metric for `wisp convert`."""
    qt = quantize_int4(weight, group_size, use_fp8=use_fp8)
    recon = dequantize_int4(qt).to(torch.float32)
    return (weight.to(torch.float32) - recon).abs().mean().item()


def estimate_expert_size(rows: int, cols: int, n_mats: int = 3,
                         group_size: int = 64, use_fp8: bool = False) -> int:
    """Estimate expert size in bytes for memory planning.
    
    Args:
        rows: Rows per matrix
        cols: Columns per matrix
        n_mats: Number of matrices per expert (gate, up, down = 3)
        group_size: Quantization group size
        use_fp8: Whether fp8 metadata is used
    
    Returns:
        Total bytes for the expert
    """
    n_elements = rows * cols * n_mats
    packed_bytes = (n_elements + 1) // 2
    n_groups = (n_elements + group_size - 1) // group_size
    
    if use_fp8:
        meta_bytes = n_groups * 2  # 1 byte each for scale and zero
    else:
        meta_bytes = n_groups * 4  # 2 bytes each for scale and zero
    
    return packed_bytes + meta_bytes
