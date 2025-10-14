"""Test training utilities and workflows."""
import pytest
import torch
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_encoder, create_field, create_rgb_head
from ngp_baseline_torch.runtime.train import train_step
from ngp_baseline_torch.types import RayBatch
from ngp_baseline_torch.opt.adam import make_optimizer


class TestTraining:
    """Test training functionality."""

    def test_train_step_basic(self, device, set_seed, small_batch_size):
        """Test basic training step."""
        cfg = Config()
        cfg.train.batch_rays = small_batch_size
        cfg.precision.use_amp = False  # Disable AMP for CPU testing
        cfg.model.hash_levels = 4

        # Create model
        encoder = create_encoder(cfg, device)
        field = create_field(cfg, encoder, device)
        rgb_head = create_rgb_head(cfg, device)

        # Create single model for optimizer
        class CombinedModel(torch.nn.Module):
            def __init__(self, encoder, field, rgb_head):
                super().__init__()
                self.encoder = encoder
                self.field = field
                self.rgb_head = rgb_head

        model = CombinedModel(encoder, field, rgb_head)
        optimizer = make_optimizer(model, cfg.train)

        # Create dummy data
        N = small_batch_size
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

        # Run training step
        metrics = train_step(
            rays=rays,
            target_rgb=target_rgb,
            encoder=encoder,
            field=field,
            rgb_head=rgb_head,
            optimizer=optimizer,
            cfg=cfg
        )

        assert 'loss' in metrics
        assert 'psnr' in metrics
        assert metrics['loss'] >= 0

    def test_gradient_update(self, device, set_seed, small_batch_size):
        """Test that gradients update parameters."""
        cfg = Config()
        cfg.train.batch_rays = small_batch_size
        cfg.precision.use_amp = False
        cfg.model.hash_levels = 2

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

        # Save initial parameters
        initial_params = [p.clone() for p in encoder.parameters()]

        # Create dummy data
        N = small_batch_size
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

        # Run training step
        train_step(rays, target_rgb, encoder, field, rgb_head, optimizer, cfg)

        # Check that parameters changed
        params_changed = False
        for initial, current in zip(initial_params, encoder.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break

        assert params_changed, "Parameters should change after training step"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
    def test_train_step_with_amp(self, device, set_seed, small_batch_size):
        """Test training step with automatic mixed precision."""
        if device.type != 'cuda':
            pytest.skip("AMP requires CUDA")

        cfg = Config()
        cfg.train.batch_rays = small_batch_size
        cfg.precision.use_amp = True
        cfg.model.hash_levels = 4

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
        scaler = torch.cuda.amp.GradScaler()

        N = small_batch_size
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

        metrics = train_step(
            rays, target_rgb, encoder, field, rgb_head,
            optimizer, cfg, scaler=scaler
        )

        assert 'loss' in metrics
        assert metrics['loss'] >= 0
