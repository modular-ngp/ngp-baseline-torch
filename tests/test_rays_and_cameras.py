"""Test ray generation and camera loading."""
import pytest
import torch
from pathlib import Path
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays, make_rays_single


@pytest.mark.quick
def test_camera_loading(scene_path):
    """Test loading camera data from NeRF synthetic."""
    if not scene_path.exists():
        pytest.skip("Test data not available")

    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")

    assert cameras.N > 0
    assert cameras.focal > 0
    assert cameras.width > 0
    assert cameras.height > 0
    assert cameras.near < cameras.far
    assert cameras.poses.shape == (cameras.N, 4, 4)


@pytest.mark.quick
def test_ray_generation(scene_path, device):
    """Test ray generation from cameras."""
    if not scene_path.exists():
        pytest.skip("Test data not available")

    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")
    cameras.poses = cameras.poses.to(device)

    # Generate rays for small resolution
    H, W = 64, 64
    indices = torch.tensor([0], device=device)  # First camera only

    rays = make_rays(H, W, cameras, indices, device)

    # Check structure
    assert rays.N == H * W
    # Compare device types, not exact device instances
    assert rays.device.type == device.type
    assert rays.dtype == torch.float32

    # Check directions are normalized
    dirs = torch.stack([rays.dir_x, rays.dir_y, rays.dir_z], dim=-1)
    norms = torch.norm(dirs, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Check near < far
    assert torch.all(rays.tmin < rays.tmax)


@pytest.mark.quick
def test_ray_single_camera(device):
    """Test ray generation for single camera."""
    H, W = 32, 32
    pose = torch.eye(4, device=device)
    focal = 400.0
    near, far = 2.0, 6.0

    rays = make_rays_single(H, W, pose, focal, near, far, device)

    assert rays.N == H * W
    # Compare device types
    assert rays.device.type == device.type

    # Check all rays have same origin (camera at origin)
    assert torch.allclose(rays.orig_x, torch.zeros_like(rays.orig_x), atol=1e-5)
    assert torch.allclose(rays.orig_y, torch.zeros_like(rays.orig_y), atol=1e-5)
    assert torch.allclose(rays.orig_z, torch.zeros_like(rays.orig_z), atol=1e-5)
