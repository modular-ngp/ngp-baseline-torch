"""SISO type definitions and shape validation for NGP baseline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class RayBatch:
    """Structure-of-Arrays ray batch on device.

    All tensors must share device and dtype (float32).
    Lengths must match across all arrays.
    """
    orig_x: torch.Tensor  # [N]
    orig_y: torch.Tensor  # [N]
    orig_z: torch.Tensor  # [N]
    dir_x: torch.Tensor   # [N]
    dir_y: torch.Tensor   # [N]
    dir_z: torch.Tensor   # [N]
    tmin: torch.Tensor    # [N]
    tmax: torch.Tensor    # [N]
    mask: torch.Tensor | None = None  # [N] bool, optional

    def __post_init__(self):
        assert_ray_batch_valid(self)

    @property
    def device(self) -> torch.device:
        return self.orig_x.device

    @property
    def dtype(self) -> torch.dtype:
        return self.orig_x.dtype

    @property
    def N(self) -> int:
        return self.orig_x.shape[0]


@dataclass
class EncodedFeat:
    """Encoded feature tensor with padding to multiple of 16."""
    feat: torch.Tensor  # [B, F], F % 16 == 0

    def __post_init__(self):
        assert self.feat.ndim == 2, f"EncodedFeat must be 2D, got {self.feat.ndim}D"
        assert self.feat.shape[1] % 16 == 0, f"Feature dim must be multiple of 16, got {self.feat.shape[1]}"


@dataclass
class FieldIO:
    """Field input/output structure."""
    input_feat: torch.Tensor  # [B, F]
    sigma: torch.Tensor       # [B]
    rgb_feat: torch.Tensor    # [B, R]
    rgb: torch.Tensor | None = None  # [B, 3], optional


@dataclass
class GridState:
    """Occupancy grid state with bitset and EMA parameters."""
    bitset: torch.Tensor      # [L, Gpack] uint32 or uint64
    ema_tau: float
    threshold: float
    levels: int
    resolution: int

    def __post_init__(self):
        assert self.bitset.dtype in (torch.uint8, torch.int32, torch.int64), \
            f"Bitset must be integer type, got {self.bitset.dtype}"
        assert self.bitset.ndim == 2, f"Bitset must be 2D, got {self.bitset.ndim}D"


@dataclass
class ArtifactMeta:
    """Metadata for exported artifacts."""
    version: str
    endianness: str
    alignment_bytes: int
    arch_req: str
    precision: dict[str, str]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'endianness': self.endianness,
            'alignment_bytes': self.alignment_bytes,
            'arch_req': self.arch_req,
            'precision': self.precision,
            'seed': self.seed
        }


def assert_ray_batch_valid(rays: RayBatch) -> None:
    """Validate RayBatch SISO invariants."""
    tensors = [rays.orig_x, rays.orig_y, rays.orig_z,
               rays.dir_x, rays.dir_y, rays.dir_z,
               rays.tmin, rays.tmax]

    if rays.mask is not None:
        tensors.append(rays.mask)

    # Check all tensors have same length
    N = tensors[0].shape[0]
    for i, t in enumerate(tensors):
        assert t.ndim == 1, f"Tensor {i} must be 1D, got {t.ndim}D"
        assert t.shape[0] == N, f"Tensor {i} has length {t.shape[0]}, expected {N}"

    # Check device and dtype consistency (except mask which is bool)
    device = tensors[0].device
    dtype = tensors[0].dtype
    for i, t in enumerate(tensors[:-1] if rays.mask is not None else tensors):
        assert t.device == device, f"Tensor {i} on device {t.device}, expected {device}"
        assert t.dtype == dtype, f"Tensor {i} has dtype {t.dtype}, expected {dtype}"

    if rays.mask is not None:
        assert rays.mask.dtype == torch.bool, f"Mask must be bool, got {rays.mask.dtype}"


def assert_shapes_encoded(feat: EncodedFeat) -> None:
    """Validate EncodedFeat shape invariants."""
    assert feat.feat.ndim == 2, f"Feature must be 2D, got {feat.feat.ndim}D"
    assert feat.feat.shape[1] % 16 == 0, f"Feature dim must be multiple of 16, got {feat.feat.shape[1]}"


def pad_to_multiple(size: int, multiple: int = 16) -> int:
    """Pad size to next multiple."""
    return ((size + multiple - 1) // multiple) * multiple

