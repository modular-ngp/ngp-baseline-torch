"""Test encoder modules."""
import pytest
import torch
from ngp_baseline_torch.encoder.pe import PositionalEncoder
from ngp_baseline_torch.encoder.hashgrid_torch import HashGridEncoder


class TestPositionalEncoder:
    """Test positional encoding."""

    def test_forward(self, device, set_seed):
        """Test forward pass."""
        encoder = PositionalEncoder(num_bands=4, include_input=True).to(device)
        xyz = torch.rand(100, 3, device=device)

        encoded = encoder(xyz)

        assert encoded.feat.ndim == 2
        assert encoded.feat.shape[0] == 100
        assert encoded.feat.shape[1] % 16 == 0  # Padded to multiple of 16

    def test_output_dim(self, device):
        """Test output dimension calculation."""
        encoder = PositionalEncoder(num_bands=6, include_input=True).to(device)
        # 3 (input) + 3*2*6 (sin/cos for 6 bands) = 3 + 36 = 39
        # Padded to 48 (next multiple of 16)
        xyz = torch.rand(10, 3, device=device)
        encoded = encoder(xyz)
        assert encoded.feat.shape[1] == 48

    def test_gradient_flow(self, device, set_seed):
        """Test gradient backpropagation."""
        encoder = PositionalEncoder(num_bands=4).to(device)
        xyz = torch.rand(50, 3, device=device, requires_grad=True)

        encoded = encoder(xyz)
        loss = encoded.feat.sum()
        loss.backward()

        assert xyz.grad is not None
        assert xyz.grad.shape == xyz.shape


class TestHashGridEncoder:
    """Test hash grid encoder."""

    def test_forward(self, device, set_seed):
        """Test forward pass."""
        encoder = HashGridEncoder(
            num_levels=4,
            base_resolution=16,
            features_per_level=2
        ).to(device)
        xyz = torch.rand(100, 3, device=device)

        encoded = encoder(xyz)

        assert encoded.feat.ndim == 2
        assert encoded.feat.shape[0] == 100
        assert encoded.feat.shape[1] % 16 == 0

    def test_output_dim(self, device):
        """Test output dimension."""
        num_levels = 8
        features_per_level = 2
        encoder = HashGridEncoder(
            num_levels=num_levels,
            features_per_level=features_per_level
        ).to(device)
        xyz = torch.rand(10, 3, device=device)
        encoded = encoder(xyz)

        # num_levels * features_per_level = 16, already multiple of 16
        assert encoded.feat.shape[1] == 16

    def test_aabb_bounds(self, device, set_seed):
        """Test coordinates within AABB."""
        aabb = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
        encoder = HashGridEncoder(aabb=aabb).to(device)

        # Test with coordinates inside AABB
        xyz_inside = torch.rand(50, 3, device=device) * 2 - 1  # [-1, 1]
        encoded = encoder(xyz_inside)
        assert not torch.isnan(encoded.feat).any()

    def test_gradient_flow(self, device, set_seed):
        """Test gradient backpropagation."""
        encoder = HashGridEncoder(num_levels=4).to(device)
        xyz = torch.rand(50, 3, device=device, requires_grad=True)

        encoded = encoder(xyz)
        loss = encoded.feat.sum()
        loss.backward()

        assert xyz.grad is not None
        # Check that hash table parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None

