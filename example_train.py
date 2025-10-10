"""Simple training example for NGP baseline."""
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays
from ngp_baseline_torch.runtime import Trainer
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.device import optimize_cuda


def train_nerf(scene: str = "lego", num_iters: int = 10000):
    """Train a simple NGP NeRF model.

    Args:
        scene: Scene name from nerf-synthetic dataset
        num_iters: Number of training iterations
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create configuration
    cfg = Config()
    cfg.dataset.scene = scene
    cfg.model.hash_levels = 16
    cfg.model.mlp_width = 64
    cfg.model.mlp_depth = 2
    cfg.integrator.n_steps_fixed = 128
    cfg.train.batch_rays = 4096
    cfg.train.lr = 1e-2
    cfg.train.iters = num_iters

    # Seed for reproducibility
    seed_everything(cfg.train.seed, deterministic=False)  # Allow performance optimizations
    if device.type == 'cuda':
        optimize_cuda()

    # Load dataset
    scene_path = Path(cfg.dataset.root) / cfg.dataset.scene
    print(f"Loading scene from {scene_path}")
    cameras = load_nerf_synthetic(scene_path, cfg.dataset.train_split)
    cameras.poses = cameras.poses.to(device)
    print(f"Loaded {cameras.N} training cameras")

    # Create model
    print("Creating model...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in field.parameters()) + \
                   sum(p.numel() for p in rgb_head.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Generate all training rays once
    print("Generating training rays...")
    H, W = 800, 800  # Full resolution
    rays = make_rays(H, W, cameras, None, device)
    print(f"Generated {rays.N:,} rays")

    # Load ground truth images
    print("Loading ground truth images...")
    from ngp_baseline_torch.data_utils import load_nerf_synthetic_images
    try:
        images, _ = load_nerf_synthetic_images(scene_path, "train", device)
        print(f"Loaded {images.shape[0]} images with shape {images.shape[1:3]}")

        # Flatten images to match rays: [N_cameras, H, W, 3] -> [N_cameras * H * W, 3]
        target_rgb = images.reshape(-1, 3)

        # Verify dimensions match
        assert target_rgb.shape[0] == rays.N, \
            f"Ray count {rays.N} doesn't match image pixels {target_rgb.shape[0]}"

        print(f"Target RGB range: [{target_rgb.min():.3f}, {target_rgb.max():.3f}]")

    except Exception as e:
        print(f"Warning: Could not load images ({e}), using synthetic target")
        # Fallback: create a simple pattern for testing
        # Create a gradient pattern instead of all white
        coords = torch.arange(rays.N, device=device).float() / rays.N
        target_rgb = torch.stack([
            coords,
            1.0 - coords,
            torch.ones_like(coords) * 0.5
        ], dim=-1)
        print("Using synthetic gradient pattern for testing")

    # Training loop
    print(f"\nTraining for {num_iters} iterations...")
    for iteration in range(num_iters):
        # Sample random rays
        indices = torch.randperm(rays.N, device=device)[:cfg.train.batch_rays]

        from ngp_baseline_torch.types import RayBatch
        ray_batch = RayBatch(
            orig_x=rays.orig_x[indices],
            orig_y=rays.orig_y[indices],
            orig_z=rays.orig_z[indices],
            dir_x=rays.dir_x[indices],
            dir_y=rays.dir_y[indices],
            dir_z=rays.dir_z[indices],
            tmin=rays.tmin[indices],
            tmax=rays.tmax[indices],
        )
        target_batch = target_rgb[indices]

        # Training step
        metrics = trainer.step(ray_batch, target_batch)

        # Log
        if (iteration + 1) % cfg.train.log_every == 0:
            print(f"Iter {iteration + 1}/{num_iters}: "
                  f"Loss={metrics['loss']:.6f}, "
                  f"PSNR={metrics['psnr']:.2f} dB")

    print("\nTraining complete!")

    # Save checkpoint
    from ngp_baseline_torch.artifact import export
    output_dir = Path("outputs") / scene
    export(encoder, field, rgb_head, cfg, output_dir, occ_grid)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="lego", help="Scene name")
    parser.add_argument("--iters", type=int, default=10000, help="Training iterations")
    args = parser.parse_args()

    train_nerf(args.scene, args.iters)
