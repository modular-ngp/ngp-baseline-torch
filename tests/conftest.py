"""Pytest configuration and shared fixtures."""
import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def small_batch_size():
    """Small batch size for fast tests."""
    return 256


@pytest.fixture
def data_root():
    """Path to test data."""
    return Path(__file__).parent.parent / "data" / "nerf-synthetic"


@pytest.fixture
def has_test_data(data_root):
    """Check if test data exists."""
    return (data_root / "lego").exists()

