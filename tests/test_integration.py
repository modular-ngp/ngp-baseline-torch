"""Integration tests for complete pipeline."""
import pytest
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import (
    create_encoder, create_field, create_rgb_head
)
from ngp_baseline_torch.rays.rays import make_rays_single
from ngp_baseline_torch.integrator import render_batch
from ngp_baseline_torch.opt.adam import make_optimizer
from ngp_baseline_torch.runtime.train import train_step
from ngp_baseline_torch.types import RayBatch


@pytest.mark.slow
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_forward_pass(self, device, set_seed):
        """Test complete forward pass from rays to RGB."""
        cfg = Config()
        cfg.model.hash_levels = 4
        cfg.integrator.n_steps_fixed = 32

        # Create model components
        encoder = create_encoder(cfg, device)
        field = create_field(cfg, encoder, device)
        rgb_head = create_rgb_head(cfg, device)

        # Generate rays
        H, W = 64, 64
        focal = 100.0
        pose = torch.eye(4, device=device)
        rays = make_rays_single(H, W, pose, focal, 2.0, 6.0, device)

        # Subsample for speed
        indices = torch.randperm(rays.N, device=device)[:1024]
        rays_subset = RayBatch(
            orig_x=rays.orig_x[indices],
            orig_y=rays.orig_y[indices],
            orig_z=rays.orig_z[indices],
            dir_x=rays.dir_x[indices],
            dir_y=rays.dir_y[indices],
            dir_z=rays.dir_z[indices],
            tmin=rays.tmin[indices],
            tmax=rays.tmax[indices]
        )

        # Render
        rgb, aux = render_batch(rays_subset, encoder, field, rgb_head, cfg.integrator)

        assert rgb.shape == (1024, 3)
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1.1)  # Allow slight overshoot

    def test_training_overfitting(self, device, set_seed):
        """Test that model can overfit to a single batch."""
        cfg = Config()
        cfg.model.hash_levels = 2
        cfg.integrator.n_steps_fixed = 16
        cfg.train.batch_rays = 128
        cfg.precision.use_amp = False
        cfg.train.lr = 0.01

        encoder = create_encoder(cfg, device)
        field = create_field(cfg, encoder, device)
        rgb_head = create_rgb_head(cfg, device)

        # Create combined model for optimizer
        class CombinedModel(torch.nn.Module):
            def __init__(self, encoder, field, rgb_head):
                super().__init__()
                self.encoder = encoder
                self.field = field
                self.rgb_head = rgb_head

        model = CombinedModel(encoder, field, rgb_head)
        optimizer = make_optimizer(model, cfg.train)

        # Create fixed batch
        N = 128
        rays = RayBatch(
            orig_x=torch.zeros(N, device=device),
            orig_y=torch.zeros(N, device=device),
            orig_z=torch.zeros(N, device=device),
            dir_x=torch.ones(N, device=device),
            dir_y=torch.zeros(N, device=device),
            dir_z=torch.zeros(N, device=device),
            tmin=torch.ones(N, device=device) * 2.0,
            tmax=torch.ones(N, device=device) * 6.0
        )
        target_rgb = torch.rand(N, 3, device=device)

        # Train for several iterations
        initial_loss = None
        final_loss = None

        for i in range(50):
            metrics = train_step(rays, target_rgb, encoder, field,
                               rgb_head, optimizer, cfg)
            if i == 0:
                initial_loss = metrics['loss']
            if i == 49:
                final_loss = metrics['loss']

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Expected loss to decrease, but got {initial_loss} -> {final_loss}"

    @pytest.mark.skipif(not Path("data/nerf-synthetic/lego").exists(),
                        reason="Test data not available")
    def test_load_and_train_single_image(self, device, set_seed):
        """Test loading data and training on a single image."""
        from ngp_baseline_torch.data_utils import load_nerf_synthetic_images
        from ngp_baseline_torch.rays.cameras import load_nerf_synthetic

        cfg = Config()
        cfg.dataset.scene = "lego"
        cfg.model.hash_levels = 4
        cfg.train.batch_rays = 256
        cfg.precision.use_amp = False

        # Load one image
        scene_path = f"data/nerf-synthetic/{cfg.dataset.scene}"

        # Load cameras
        camera_data = load_nerf_synthetic(scene_path, "transforms_train.json")

        # Load images
        images, _ = load_nerf_synthetic_images(scene_path, "train", device=device)

        # Create model
        encoder = create_encoder(cfg, device)
        field = create_field(cfg, encoder, device)
        rgb_head = create_rgb_head(cfg, device)

        class CombinedModel(torch.nn.Module):
            def __init__(self, encoder, field, rgb_head):
                super().__init__()
                self.encoder = encoder
                self.field = field
                self.rgb_head = rgb_head

        model = CombinedModel(encoder, field, rgb_head)
        optimizer = make_optimizer(model, cfg.train)

        # Get first image and pose
        image = images[0]  # [H, W, 3]
        pose = camera_data.poses[0].to(device)  # [4, 4]
        focal = camera_data.focal

        H, W = image.shape[:2]
        rays = make_rays_single(H, W, pose, focal, camera_data.near, camera_data.far, device)

        # Subsample rays
        indices = torch.randperm(rays.N, device=device)[:cfg.train.batch_rays]
        rays_subset = RayBatch(
            orig_x=rays.orig_x[indices],
            orig_y=rays.orig_y[indices],
            orig_z=rays.orig_z[indices],
            dir_x=rays.dir_x[indices],
            dir_y=rays.dir_y[indices],
            dir_z=rays.dir_z[indices],
            tmin=rays.tmin[indices],
            tmax=rays.tmax[indices]
        )
        target_rgb = image.view(-1, 3)[indices]

        # Run a few training steps
        for _ in range(5):
            metrics = train_step(rays_subset, target_rgb, encoder,
                               field, rgb_head, optimizer, cfg)
            assert metrics['loss'] >= 0
