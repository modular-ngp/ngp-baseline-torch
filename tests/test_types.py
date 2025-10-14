"""Test type definitions and validation."""
import pytest
import torch
from ngp_baseline_torch.types import RayBatch, EncodedFeat


class TestRayBatch:
    """Test RayBatch dataclass."""

    def test_creation(self, device):
        """Test basic RayBatch creation."""
        N = 100
        rays = RayBatch(
            orig_x=torch.zeros(N, device=device),
            orig_y=torch.zeros(N, device=device),
            orig_z=torch.zeros(N, device=device),
            dir_x=torch.ones(N, device=device),
            dir_y=torch.zeros(N, device=device),
            dir_z=torch.zeros(N, device=device),
            tmin=torch.zeros(N, device=device),
            tmax=torch.ones(N, device=device)
        )
        assert rays.N == N
        assert rays.device.type == device.type  # Compare device type, not exact device

    def test_with_mask(self, device):
        """Test RayBatch with optional mask."""
        N = 50
        rays = RayBatch(
            orig_x=torch.zeros(N, device=device),
            orig_y=torch.zeros(N, device=device),
            orig_z=torch.zeros(N, device=device),
            dir_x=torch.ones(N, device=device),
            dir_y=torch.zeros(N, device=device),
            dir_z=torch.zeros(N, device=device),
            tmin=torch.zeros(N, device=device),
            tmax=torch.ones(N, device=device),
            mask=torch.ones(N, dtype=torch.bool, device=device)
        )
        assert rays.mask is not None
        assert rays.mask.shape[0] == N


class TestEncodedFeat:
    """Test EncodedFeat dataclass."""

    def test_creation(self, device):
        """Test EncodedFeat creation with valid dimensions."""
        B, F = 100, 32  # F must be multiple of 16
        feat = EncodedFeat(feat=torch.randn(B, F, device=device))
        assert feat.feat.shape == (B, F)

    def test_padding_validation(self, device):
        """Test that feature dimension must be multiple of 16."""
        with pytest.raises(AssertionError):
            EncodedFeat(feat=torch.randn(100, 30, device=device))  # 30 % 16 != 0
