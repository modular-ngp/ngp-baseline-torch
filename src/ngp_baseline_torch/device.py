"""Device and dtype management."""
from __future__ import annotations
import torch
from .config import PrecisionConfig


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the default device for computation."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def setup_device(cfg: PrecisionConfig, prefer_cuda: bool = True) -> tuple[torch.device, torch.dtype, torch.dtype]:
    """Setup device and dtypes from config.

    Returns:
        device: computation device
        param_dtype: parameter dtype
        compute_dtype: computation dtype
    """
    device = get_device(prefer_cuda)
    param_dtype = get_dtype(cfg.param_dtype)
    compute_dtype = get_dtype(cfg.compute_dtype)

    return device, param_dtype, compute_dtype


def optimize_cuda() -> None:
    """Enable CUDA optimizations for modern GPUs."""
    if torch.cuda.is_available():
        # Enable TF32 for matmul on Ampere+ GPUs (PyTorch 2.9+ API)
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

        # Enable cudnn benchmarking for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True

        # Use deterministic algorithms when needed (set per config)
        # torch.use_deterministic_algorithms(True)
