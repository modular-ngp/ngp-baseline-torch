"""Test training step functionality."""
import pytest
import torch
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.runtime import train_step
from ngp_baseline_torch.types import RayBatch


@pytest.mark.quick
def test_train_step_execution(device):
    """Test that a single training step executes successfully."""
    cfg = Config()
    cfg.model.hash_levels = 4
    cfg.model.mlp_width = 32
    cfg.integrator.n_steps_fixed = 64

    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)

    # Create dummy ray batch
    N = 256
    rays = RayBatch(
        orig_x=torch.zeros(N, device=device),
        orig_y=torch.zeros(N, device=device),
        orig_z=torch.zeros(N, device=device),
        dir_x=torch.zeros(N, device=device),
        dir_y=torch.zeros(N, device=device),
        dir_z=torch.ones(N, device=device),
        tmin=torch.ones(N, device=device) * 2.0,
        tmax=torch.ones(N, device=device) * 6.0,
    )
    target_rgb = torch.rand(N, 3, device=device)

    # Execute training step
    metrics = train_step(rays, target_rgb, encoder, field, rgb_head, optimizer, cfg)

    # Check metrics
    assert 'loss' in metrics
    assert 'psnr' in metrics
    assert metrics['loss'] >= 0
    assert not torch.isnan(torch.tensor(metrics['loss']))


@pytest.mark.quick
def test_train_step_updates_parameters(device):
    """Test that training step updates model parameters."""
    cfg = Config()
    cfg.model.hash_levels = 4
    cfg.model.mlp_width = 32

    encoder, field, rgb_head, _ = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)

    # Save initial parameters
    initial_params = {name: param.data.clone() for name, param in field.named_parameters()}

    # Create dummy data
    N = 128
    rays = RayBatch(
        orig_x=torch.zeros(N, device=device),
        orig_y=torch.zeros(N, device=device),
        orig_z=torch.zeros(N, device=device),
        dir_x=torch.zeros(N, device=device),
        dir_y=torch.zeros(N, device=device),
        dir_z=torch.ones(N, device=device),
        tmin=torch.ones(N, device=device) * 2.0,
        tmax=torch.ones(N, device=device) * 6.0,
    )
    target_rgb = torch.rand(N, 3, device=device)

    # Training step
    train_step(rays, target_rgb, encoder, field, rgb_head, optimizer, cfg)

    # Check that at least some parameters changed
    changed = False
    for name, param in field.named_parameters():
        if not torch.allclose(param.data, initial_params[name]):
            changed = True
            break

    assert changed, "No parameters were updated"

