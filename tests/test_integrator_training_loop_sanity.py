"""Test training loop sanity checks."""
import pytest
import torch
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.runtime import Trainer


@pytest.mark.quick
def test_training_step_improves_loss(scene_path, device, seed):
    """Test that training improves PSNR over initial random state."""
    if not scene_path.exists():
        pytest.skip("Test data not available")

    from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays

    seed_everything(seed)

    # Create minimal config for quick test
    cfg = Config()
    cfg.model.hash_levels = 8
    cfg.model.mlp_width = 32
    cfg.model.mlp_depth = 1
    cfg.integrator.n_steps_fixed = 64
    cfg.train.batch_rays = 512

    # Load data
    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")
    cameras.poses = cameras.poses.to(device)

    # Create model
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)

    # Create trainer
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Generate small batch of rays
    H, W = 32, 32
    indices = torch.tensor([0], device=device)
    rays = make_rays(H, W, cameras, indices, device)

    # Sample random rays
    num_rays = min(cfg.train.batch_rays, rays.N)
    ray_indices = torch.randperm(rays.N, device=device)[:num_rays]

    from ngp_baseline_torch.types import RayBatch
    ray_batch = RayBatch(
        orig_x=rays.orig_x[ray_indices],
        orig_y=rays.orig_y[ray_indices],
        orig_z=rays.orig_z[ray_indices],
        dir_x=rays.dir_x[ray_indices],
        dir_y=rays.dir_y[ray_indices],
        dir_z=rays.dir_z[ray_indices],
        tmin=rays.tmin[ray_indices],
        tmax=rays.tmax[ray_indices],
    )

    # Create dummy target (all white for now)
    target_rgb = torch.ones(num_rays, 3, device=device)

    # Get initial loss
    metrics_0 = trainer.step(ray_batch, target_rgb)
    initial_psnr = metrics_0['psnr']

    # Train for a few steps
    for _ in range(50):
        trainer.step(ray_batch, target_rgb)

    # Final evaluation
    metrics_final = trainer.step(ray_batch, target_rgb)
    final_psnr = metrics_final['psnr']

    # PSNR should improve
    assert final_psnr > initial_psnr + 1.0, f"PSNR did not improve: {initial_psnr:.2f} -> {final_psnr:.2f}"

