"""Test hash grid encoder."""
import pytest
import torch
from ngp_baseline_torch.encoder.hashgrid_torch import HashGridEncoder


@pytest.mark.quick
def test_hashgrid_output_shape(device):
    """Test hash grid encoder output shape."""
    B = 100
    num_levels = 16
    features_per_level = 2

    encoder = HashGridEncoder(
        num_levels=num_levels,
        base_resolution=16,
        max_resolution=2048,
        features_per_level=features_per_level,
        log2_hashmap_size=19
    )
    encoder.to(device)

    xyz = torch.rand(B, 3, device=device) * 2 - 1  # [-1, 1]
    encoded = encoder(xyz)

    assert encoded.feat.shape[0] == B
    assert encoded.feat.shape[1] % 16 == 0
    assert encoded.feat.shape[1] >= num_levels * features_per_level


@pytest.mark.quick
def test_hashgrid_trilinear_interpolation(device):
    """Test trilinear interpolation in hash grid."""
    encoder = HashGridEncoder(
        num_levels=1,
        base_resolution=16,
        max_resolution=16,
        features_per_level=2
    )
    encoder.to(device)

    # Test point in middle of voxel should interpolate neighbors
    xyz = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    encoded = encoder(xyz)

    # Should produce valid output
    assert not torch.isnan(encoded.feat).any()
    assert not torch.isinf(encoded.feat).any()


@pytest.mark.quick
def test_hashgrid_gradients(device):
    """Test that hash grid produces gradients."""
    encoder = HashGridEncoder(num_levels=4, features_per_level=2)
    encoder.to(device)

    xyz = torch.rand(10, 3, device=device, requires_grad=True)
    encoded = encoder(xyz)

    loss = encoded.feat.sum()
    loss.backward()

    assert xyz.grad is not None
    assert not torch.isnan(xyz.grad).any()

