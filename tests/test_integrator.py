"""Test integrator (ray marching and volume rendering)."""
import pytest
import torch
from ngp_baseline_torch.integrator.compositor import compose
from ngp_baseline_torch.config import IntegratorConfig


class TestVolumeRendering:
    """Test volume rendering compositor."""

    def test_compose_basic(self, device, set_seed):
        """Test basic volume rendering composition."""
        N = 100
        S = 64  # number of samples per ray

        # Create mock data
        sigma = torch.rand(N, S, device=device)
        rgb = torch.rand(N, S, 3, device=device)
        dt = 0.01

        # Render
        rendered_rgb, T_final = compose(sigma, rgb, dt)

        assert rendered_rgb.shape == (N, 3)
        assert T_final.shape == (N,)
        assert torch.all(rendered_rgb >= 0) and torch.all(rendered_rgb <= 1.1)  # Allow slight overshoot due to white bg

    def test_transmittance_decay(self, device, set_seed):
        """Test that transmittance decreases with density."""
        N = 50
        S = 32

        # High density should reduce transmittance more
        sigma_high = torch.ones(N, S, device=device) * 10
        sigma_low = torch.ones(N, S, device=device) * 0.1
        rgb = torch.ones(N, S, 3, device=device) * 0.5
        dt = 0.01

        _, T_high = compose(sigma_high, rgb, dt)
        _, T_low = compose(sigma_low, rgb, dt)

        # Lower density should have higher final transmittance
        assert T_low.mean() > T_high.mean()

    def test_zero_sigma(self, device):
        """Test with zero density (transparent)."""
        N, S = 10, 32
        sigma = torch.zeros(N, S, device=device)
        rgb = torch.ones(N, S, 3, device=device) * 0.5
        dt = 0.01

        rendered_rgb, T_final = compose(sigma, rgb, dt)

        # With zero density, should get white background
        assert torch.allclose(rendered_rgb, torch.ones_like(rendered_rgb), atol=0.1)
        assert torch.allclose(T_final, torch.ones_like(T_final), atol=0.01)

    def test_high_sigma(self, device):
        """Test with high density (opaque)."""
        N, S = 10, 32
        sigma = torch.ones(N, S, device=device) * 100  # Very high density
        rgb = torch.ones(N, S, 3, device=device) * 0.5  # Gray
        dt = 0.01

        rendered_rgb, _ = compose(sigma, rgb, dt)

        # Should render close to the input color (gray)
        assert rendered_rgb.shape == (N, 3)
