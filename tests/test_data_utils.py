"""Test data loading utilities."""
import pytest
import torch
from pathlib import Path
from ngp_baseline_torch.data_utils import load_nerf_synthetic_images, load_image
from ngp_baseline_torch.rays.cameras import load_nerf_synthetic


class TestDataUtils:
    """Test data loading utilities."""

    @pytest.mark.skipif(not Path("data/nerf-synthetic/lego").exists(),
                        reason="Test data not available")
    def test_load_nerf_synthetic_cameras(self, device):
        """Test loading camera data."""
        scene_path = "data/nerf-synthetic/lego"

        camera_data = load_nerf_synthetic(scene_path, "transforms_train.json")

        assert camera_data.poses.shape[0] > 0
        assert camera_data.poses.shape[1:] == (4, 4)
        assert camera_data.focal > 0
        assert camera_data.width > 0
        assert camera_data.height > 0

    @pytest.mark.skipif(not Path("data/nerf-synthetic/lego").exists(),
                        reason="Test data not available")
    def test_load_nerf_synthetic_images(self, device):
        """Test loading images from dataset."""
        scene_path = "data/nerf-synthetic/lego"

        images, image_paths = load_nerf_synthetic_images(
            scene_path,
            "train",
            device=device
        )

        assert images.shape[0] > 0
        assert images.ndim == 4  # [N, H, W, 3]
        assert images.device.type == device.type
        assert len(image_paths) == images.shape[0]

    @pytest.mark.skipif(not Path("data/nerf-synthetic/lego").exists(),
                        reason="Test data not available")
    def test_image_range(self, device):
        """Test that loaded images are in [0, 1] range."""
        scene_path = "data/nerf-synthetic/lego"

        images, _ = load_nerf_synthetic_images(
            scene_path,
            "train",
            device=device
        )

        assert torch.all(images >= 0) and torch.all(images <= 1)
