"""Closed registry of persistent NInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class TernaryFormat:
    """Prism ternary codes: grouped {-1,0,+1} codes with one binary16 scale per group.

    ⚠️ NOT part of upstream NInfer — reconstructed for the ternary Bonsai port.
    The released bundle ships pack.py (which names these formats) and the C++ side
    of the artifact layer, but NOT the matching Python-side registration, so
    `get_format("PQ2_0_G128")` raises "unknown numeric format" on a clean tree.

    Why the existing QuantFormat cannot express them: `row_split_geometry` derives
    `base_bytes_per_group` as `group_size // 2` (for bits != 8), giving 64 for a
    128-wide group.  PQ2_0 actually stores **32** base bytes per group and PTQ1_0
    stores **24 base + 2 high**.  So the per-group byte counts are stated explicitly.
    """

    name: str
    group_size: int
    base_bytes_per_group: int
    high_bytes_per_group: int


NumericFormat: TypeAlias = (
    DirectFormat | QuantFormat | Nvfp4Format | Fp8RowFormat | TernaryFormat
)


BF16 = DirectFormat("BF16", 2)
FP32 = DirectFormat("FP32", 4)
I32 = DirectFormat("I32", 4)

Q4G64_F16S = QuantFormat("Q4G64_F16S", 4, 64, -8, 7)
Q5G64_F16S = QuantFormat("Q5G64_F16S", 5, 64, -16, 15)
Q6G64_F16S = QuantFormat("Q6G64_F16S", 6, 64, -32, 31)
W8G32_F16S = QuantFormat("W8G32_F16S", 8, 32, -127, 127)
NVFP4 = Nvfp4Format("NVFP4", 16)
FP8_E4M3FN_ROW_BF16S = Fp8RowFormat("FP8_E4M3FN_ROW_BF16S")

# Ternary Bonsai formats (see TernaryFormat docstring).  Byte counts per 128-weight
# group, verified against the sizes recorded in docs/03-三元模型转-NInfer.md §1.1:
#   [248320, 5120] -> PTQ1_0 278,118,400 B / PQ2_0 337,715,200 B
PQ2_0_G128 = TernaryFormat("PQ2_0_G128", 128, 32, 0)
PTQ1_0_G128 = TernaryFormat("PTQ1_0_G128", 128, 24, 2)


DIRECT_FORMATS = MappingProxyType(
    {item.name: item for item in (BF16, FP32, I32)}
)
QUANT_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, W8G32_F16S)
    }
)
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType(
    {FP8_E4M3FN_ROW_BF16S.name: FP8_E4M3FN_ROW_BF16S}
)
TERNARY_FORMATS = MappingProxyType(
    {item.name: item for item in (PQ2_0_G128, PTQ1_0_G128)}
)
NUMERIC_FORMATS = MappingProxyType(
    {
        **DIRECT_FORMATS,
        **QUANT_FORMATS,
        **NVFP4_FORMATS,
        **FP8_ROW_FORMATS,
        **TERNARY_FORMATS,
    }
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return (
        type(word) is int
        and 0 <= word <= 0xFF
        and word & 0x80 == 0
        and word != 0x7F
    )


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "DIRECT_FORMATS",
    "DirectFormat",
    "FP8_E4M3FN_ROW_BF16S",
    "FP8_ROW_FORMATS",
    "FP32",
    "Fp8RowFormat",
    "I32",
    "NUMERIC_FORMATS",
    "NVFP4",
    "NVFP4_FORMATS",
    "Nvfp4Format",
    "NumericFormat",
    "Q4G64_F16S",
    "Q5G64_F16S",
    "Q6G64_F16S",
    "QUANT_FORMATS",
    "QuantFormat",
    "W8G32_F16S",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "get_format",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_positive_fp32_word",
]
