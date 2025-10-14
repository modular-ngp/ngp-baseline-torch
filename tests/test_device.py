"""Test device utilities."""
import pytest
import torch
from ngp_baseline_torch.device import get_device, get_dtype


class TestDevice:
    """Test device utilities."""

    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'cpu']

    def test_get_dtype(self):
        """Test dtype conversion."""
        dtype = get_dtype('float32')
        assert dtype == torch.float32

        dtype = get_dtype('float16')
        assert dtype == torch.float16

    def test_unknown_dtype_returns_default(self):
        """Test that unknown dtype returns default float32."""
        dtype = get_dtype('invalid_dtype')
        assert dtype == torch.float32  # Returns default instead of raising error
