"""End-to-end rendering and PSNR test."""
import pytest
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays_single
from ngp_baseline_torch.runtime import Trainer, render_image
from ngp_baseline_torch.rng import seed_everything


@pytest.mark.quick
def test_end_to_end_render_small(device, seed):
    """Test end-to-end rendering pipeline on small resolution."""
    seed_everything(seed, deterministic=True)

    # Minimal config for quick test
    cfg = Config()
    cfg.model.hash_levels = 8
    cfg.model.mlp_width = 32
    cfg.model.mlp_depth = 1
    cfg.integrator.n_steps_fixed = 32
    cfg.train.batch_rays = 256

    # Create model
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)

    # Test rendering without training (should produce some output)
    H, W = 32, 32
    pose = torch.eye(4, device=device)

    image = render_image(
        H=H, W=W,
        pose=pose,
        focal=400.0,
        near=2.0,
        far=6.0,
        encoder=encoder,
        field=field,
        rgb_head=rgb_head,
        cfg=cfg,
        device=device,
        chunk_size=256
    )

    # Check output shape and range
    assert image.shape == (H, W, 3)
    assert torch.all(image >= 0)
    assert torch.all(image <= 1.1)  # Allow slight overflow from background
    assert not torch.isnan(image).any()
    assert not torch.isinf(image).any()


@pytest.mark.quick
def test_psnr_improves_with_training(scene_path, device, seed):
    """Test that PSNR improves with training iterations."""
    if not scene_path.exists():
        pytest.skip("Test data not available")

    seed_everything(seed, deterministic=True)

    # Quick training config
    cfg = Config()
    cfg.model.hash_levels = 8
    cfg.model.mlp_width = 32
    cfg.model.mlp_depth = 1
    cfg.integrator.n_steps_fixed = 64
    cfg.train.batch_rays = 512
    cfg.train.lr = 1e-2

    # Load single camera
    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")
    cameras.poses = cameras.poses.to(device)

    # Create model
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Generate rays for small resolution
    H, W = 32, 32
    rays = make_rays_single(H, W, cameras.poses[0], cameras.focal,
                           cameras.near, cameras.far, device)

    # Create dummy target (white background)
    target_rgb = torch.ones(rays.N, 3, device=device)

    # Get initial PSNR
    from ngp_baseline_torch.types import RayBatch
    ray_batch = RayBatch(
        orig_x=rays.orig_x,
        orig_y=rays.orig_y,
        orig_z=rays.orig_z,
        dir_x=rays.dir_x,
        dir_y=rays.dir_y,
        dir_z=rays.dir_z,
        tmin=rays.tmin,
        tmax=rays.tmax,
    )

    metrics_initial = trainer.step(ray_batch, target_rgb)
    initial_psnr = metrics_initial['psnr']

    # Train for several iterations
    for _ in range(20):
        trainer.step(ray_batch, target_rgb)

    # Get final PSNR
    metrics_final = trainer.step(ray_batch, target_rgb)
    final_psnr = metrics_final['psnr']

    # PSNR should improve significantly
    print(f"\nInitial PSNR: {initial_psnr:.2f} dB")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Improvement: {final_psnr - initial_psnr:.2f} dB")

    assert final_psnr > initial_psnr + 2.0, \
        f"PSNR did not improve enough: {initial_psnr:.2f} -> {final_psnr:.2f} dB"


@pytest.mark.slow
def test_convergence_to_target(scene_path, device, seed):
    """Test that model can converge to a simple target pattern."""
    if not scene_path.exists():
        pytest.skip("Test data not available")

    seed_everything(seed, deterministic=True)

    cfg = Config()
    cfg.model.hash_levels = 12
    cfg.model.mlp_width = 64
    cfg.model.mlp_depth = 2
    cfg.integrator.n_steps_fixed = 128
    cfg.train.batch_rays = 1024
    cfg.train.lr = 1e-2

    # Load data
    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")
    cameras.poses = cameras.poses.to(device)

    # Create model
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Generate rays
    H, W = 64, 64
    rays = make_rays_single(H, W, cameras.poses[0], cameras.focal,
                           cameras.near, cameras.far, device)

    # Simple target: white background
    target_rgb = torch.ones(rays.N, 3, device=device)

    from ngp_baseline_torch.types import RayBatch
    ray_batch = RayBatch(
        orig_x=rays.orig_x,
        orig_y=rays.orig_y,
        orig_z=rays.orig_z,
        dir_x=rays.dir_x,
        dir_y=rays.dir_y,
        dir_z=rays.dir_z,
        tmin=rays.tmin,
        tmax=rays.tmax,
    )

    # Train until convergence
    for i in range(200):
        metrics = trainer.step(ray_batch, target_rgb)
        if (i + 1) % 50 == 0:
            print(f"Iter {i+1}: PSNR = {metrics['psnr']:.2f} dB")

    # Should achieve high PSNR on simple target
    final_metrics = trainer.step(ray_batch, target_rgb)
    final_psnr = final_metrics['psnr']

    print(f"\nFinal PSNR: {final_psnr:.2f} dB")
    assert final_psnr > 30.0, f"Failed to converge: PSNR = {final_psnr:.2f} dB"


@pytest.mark.quick
def test_render_consistency(device, seed):
    """Test that rendering is deterministic with same seed."""
    cfg = Config()
    cfg.model.hash_levels = 8
    cfg.model.mlp_width = 32

    H, W = 16, 16
    pose = torch.eye(4, device=device)

    # Render 1
    seed_everything(seed, deterministic=True)
    encoder1, field1, rgb_head1, _ = create_all(cfg, device)
    image1 = render_image(H, W, pose, 400.0, 2.0, 6.0,
                         encoder1, field1, rgb_head1, cfg, device, chunk_size=128)

    # Render 2 with same seed
    seed_everything(seed, deterministic=True)
    encoder2, field2, rgb_head2, _ = create_all(cfg, device)
    image2 = render_image(H, W, pose, 400.0, 2.0, 6.0,
                         encoder2, field2, rgb_head2, cfg, device, chunk_size=128)

    # Should be identical
    assert torch.allclose(image1, image2, atol=1e-6)

