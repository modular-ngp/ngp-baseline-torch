"""Test ray generation and camera utilities."""
import pytest
import torch
from ngp_baseline_torch.rays.rays import make_rays_single
from ngp_baseline_torch.types import RayBatch


class TestRayGeneration:
    """Test ray generation."""

    def test_make_rays_single(self, device, set_seed):
        """Test single camera ray generation."""
        H, W = 100, 100
        focal = 138.88
        pose = torch.eye(4, device=device)
        near, far = 2.0, 6.0

        rays = make_rays_single(H, W, pose, focal, near, far, device)

        assert isinstance(rays, RayBatch)
        assert rays.N == H * W
        assert rays.device.type == device.type

    def test_ray_directions_normalized(self, device, set_seed):
        """Test that ray directions are normalized."""
        H, W = 50, 50
        focal = 100.0
        pose = torch.eye(4, device=device)
        near, far = 0.1, 10.0

        rays = make_rays_single(H, W, pose, focal, near, far, device)

        # Reconstruct direction vectors
        dir_norm = torch.sqrt(rays.dir_x**2 + rays.dir_y**2 + rays.dir_z**2)

        # Check approximately normalized (within floating point tolerance)
        assert torch.allclose(dir_norm, torch.ones_like(dir_norm), atol=1e-5)

    def test_multiple_poses(self, device, set_seed):
        """Test ray generation with different poses."""
        H, W = 64, 64
        focal = 100.0
        near, far = 0.1, 10.0

        # Generate rays for two different poses
        pose1 = torch.eye(4, device=device)
        pose2 = torch.eye(4, device=device)
        pose2[0, 3] = 1.0  # Translate in x

        rays1 = make_rays_single(H, W, pose1, focal, near, far, device)
        rays2 = make_rays_single(H, W, pose2, focal, near, far, device)

        # Origins should be different
        assert not torch.allclose(rays1.orig_x, rays2.orig_x)
