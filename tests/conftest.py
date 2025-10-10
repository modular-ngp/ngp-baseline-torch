"""Pytest configuration and fixtures."""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def device():
    """Get default device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def seed():
    """Default random seed."""
    return 1337


@pytest.fixture
def data_root():
    """Path to test data."""
    return Path("data/nerf-synthetic")


@pytest.fixture
def scene_path(data_root):
    """Path to lego scene."""
    return data_root / "lego"


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "quick: tests that run in < 60s")
    config.addinivalue_line("markers", "slow: tests that may take several minutes")
    config.addinivalue_line("markers", "perf: performance benchmarks requiring GPU")

