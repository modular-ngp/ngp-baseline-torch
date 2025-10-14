"""Test field (MLP) modules."""
import pytest
import torch
from ngp_baseline_torch.field.mlp import NGP_MLP
from ngp_baseline_torch.field.heads import RGBHead


class TestNGP_MLP:
    """Test NGP MLP."""

    def test_forward(self, device, set_seed):
        """Test forward pass."""
        mlp = NGP_MLP(
            input_dim=32,
            hidden_dim=64,
            num_layers=2,
            output_dim=16
        ).to(device)

        x = torch.randn(100, 32, device=device)
        sigma, feat = mlp(x)

        assert sigma.shape == (100,)  # Sigma is squeezed to [N]
        assert feat.shape == (100, 16)
        assert sigma.min() >= 0  # Sigma should be non-negative

    def test_activations(self, device):
        """Test different activation functions."""
        for activation in ["relu", "silu", "softplus"]:
            mlp = NGP_MLP(input_dim=16, activation=activation).to(device)
            x = torch.randn(10, 16, device=device)
            sigma, feat = mlp(x)
            assert not torch.isnan(sigma).any()
            assert not torch.isnan(feat).any()

    def test_gradient_flow(self, device, set_seed):
        """Test gradient backpropagation."""
        mlp = NGP_MLP(input_dim=32, hidden_dim=64).to(device)
        x = torch.randn(50, 32, device=device, requires_grad=True)

        sigma, feat = mlp(x)
        loss = sigma.sum() + feat.sum()
        loss.backward()

        assert x.grad is not None
        for param in mlp.parameters():
            assert param.grad is not None


class TestRGBHead:
    """Test RGB head."""

    def test_forward(self, device, set_seed):
        """Test forward pass."""
        rgb_head = RGBHead(
            rgb_feat_dim=16,
            view_dependent=True,
            viewdir_dim=12
        ).to(device)

        feat = torch.randn(100, 16, device=device)
        viewdir = torch.randn(100, 12, device=device)  # Pre-encoded view direction

        rgb = rgb_head(feat, viewdir)

        assert rgb.shape == (100, 3)
        assert rgb.min() >= 0 and rgb.max() <= 1  # RGB in [0, 1]

    def test_without_viewdir(self, device, set_seed):
        """Test RGB head without view direction."""
        rgb_head = RGBHead(rgb_feat_dim=16, view_dependent=False).to(device)
        feat = torch.randn(50, 16, device=device)

        rgb = rgb_head(feat, None)

        assert rgb.shape == (50, 3)
        assert rgb.min() >= 0 and rgb.max() <= 1

    def test_gradient_flow(self, device, set_seed):
        """Test gradient backpropagation."""
        rgb_head = RGBHead(rgb_feat_dim=16, view_dependent=True, viewdir_dim=12).to(device)
        feat = torch.randn(50, 16, device=device, requires_grad=True)
        viewdir = torch.randn(50, 12, device=device, requires_grad=True)

        rgb = rgb_head(feat, viewdir)
        loss = rgb.sum()
        loss.backward()

        assert feat.grad is not None
        assert viewdir.grad is not None
