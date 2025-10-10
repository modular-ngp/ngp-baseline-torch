"""Random number generation and determinism control."""
from __future__ import annotations
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch 2.0+, use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded generator for a specific device.

    Args:
        seed: Random seed
        device: Target device

    Returns:
        Seeded torch Generator
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen

