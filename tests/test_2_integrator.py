"""Test 2: Ray Marching and Volume Rendering"""
import torch
import pytest
from ngp_baseline_torch.config import IntegratorConfig
from ngp_baseline_torch.integrator.marcher import RayMarcher
from ngp_baseline_torch.integrator.compositor import compose, compose_with_early_stop
from ngp_baseline_torch.types import RayBatch


class TestRayMarcher:
    """Test ray marching with jittering."""

    def test_sample_jittering(self):
        """CRITICAL: Sample jittering must be enabled during training."""
        cfg = IntegratorConfig(n_steps_fixed=64, perturb=True)
        marcher = RayMarcher(cfg)

        # Create test rays
        rays = RayBatch(
            orig_x=torch.zeros(10),
            orig_y=torch.zeros(10),
            orig_z=torch.zeros(10),
            dir_x=torch.ones(10),
            dir_y=torch.zeros(10),
            dir_z=torch.zeros(10),
            tmin=torch.ones(10) * 2.0,
            tmax=torch.ones(10) * 6.0,
        )

        # Test training mode (should have jitter)
        marcher.train()
        t1, dt1 = marcher._sample_fixed_steps(rays)
        t2, dt2 = marcher._sample_fixed_steps(rays)

        diff = (t1 - t2).abs().mean().item()
        assert diff > 0, "Training mode should have jittering"

        # Test eval mode (no jitter)
        marcher.eval()
        t3, dt3 = marcher._sample_fixed_steps(rays)
        t4, dt4 = marcher._sample_fixed_steps(rays)

        diff_eval = (t3 - t4).abs().mean().item()
        assert diff_eval == 0, "Eval mode should have no jittering"

        print(f"✓ Jittering: train diff={diff:.6f}, eval diff={diff_eval:.6f}")

    def test_sample_count(self):
        """Test correct number of samples."""
        cfg = IntegratorConfig(n_steps_fixed=128)
        marcher = RayMarcher(cfg)

        rays = RayBatch(
            orig_x=torch.zeros(5),
            orig_y=torch.zeros(5),
            orig_z=torch.zeros(5),
            dir_x=torch.ones(5),
            dir_y=torch.zeros(5),
            dir_z=torch.zeros(5),
            tmin=torch.ones(5) * 2.0,
            tmax=torch.ones(5) * 6.0,
        )

        t_samples, dt = marcher._sample_fixed_steps(rays)

        assert t_samples.shape == (5, 128)
        assert dt > 0

        print(f"✓ Sample count correct: {t_samples.shape}, dt={dt:.4f}")


class TestVolumeCompositing:
    """Test volume rendering compositing."""

    def test_basic_compositing(self):
        """Test basic alpha compositing."""
        N, S = 10, 64

        # Create test data
        sigma = torch.rand(N, S) * 10.0
        rgb = torch.rand(N, S, 3)
        dt = 0.1

        rgb_out, T_out = compose(sigma, rgb, dt)

        assert rgb_out.shape == (N, 3)
        assert T_out.shape == (N,)
        assert (rgb_out >= 0).all() and (rgb_out <= 1.5).all()  # Allow for white background

        print(f"✓ Compositing output range: [{rgb_out.min():.3f}, {rgb_out.max():.3f}]")

    def test_early_stopping(self):
        """Test early stopping for efficiency."""
        N, S = 10, 64

        # High density in first half
        sigma = torch.zeros(N, S)
        sigma[:, :32] = 100.0  # Very high density
        rgb = torch.ones(N, S, 3) * 0.5
        dt = 0.05

        rgb_out, T_out, stop_mask = compose_with_early_stop(sigma, rgb, dt, T_threshold=1e-4)

        # Should stop early due to high density
        assert stop_mask.any(), "Should have early stopping"

        print(f"✓ Early stopping: {stop_mask.sum().item()}/{stop_mask.numel()} samples stopped")

    def test_white_background(self):
        """Test white background compositing."""
        N, S = 5, 32

        # Low density (mostly transparent)
        sigma = torch.ones(N, S) * 0.1
        rgb = torch.zeros(N, S, 3)  # Black color
        dt = 0.1

        rgb_out, T_out = compose(sigma, rgb, dt)

        # With white background, should be close to white where transparent
        assert (rgb_out > 0.5).any(), "White background not applied"

        print(f"✓ White background compositing working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

