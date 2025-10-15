"""Training example with real-time visualization via shared memory."""
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays
from ngp_baseline_torch.runtime import Trainer
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.device import optimize_cuda
from ngp_baseline_torch.logger import NGPLogger
from ngp_baseline_torch.visualization import (
    NGPDebugServer,
    sample_density_grid,
    extract_training_metrics,
    filter_by_density_threshold,
)
import numpy as np


def train_nerf_with_visualization(
    scene: str = "lego",
    num_iters: int = 10000,
    enable_visualization: bool = True,
    vis_update_interval: int = 10,
    vis_max_points: int = 100_000,
    vis_density_threshold: float = 0.01,
):
    """
    Train NGP NeRF with real-time visualization via shared memory.

    Args:
        scene: Scene name from nerf-synthetic dataset
        num_iters: Number of training iterations
        enable_visualization: Enable shared memory visualization server
        vis_update_interval: Update visualization every N iterations
        vis_max_points: Maximum points to send for visualization
        vis_density_threshold: Minimum density threshold for visualization
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
    cfg.train.lr = 5e-3
    cfg.train.iters = num_iters

    # Initialize logger
    log_dir = Path("outputs") / scene / "logs"
    logger = NGPLogger(log_dir, cfg, run_name=f"{scene}_vis_{num_iters}iters")
    logger.log_system_info()
    logger.log_complete_config()

    # Initialize visualization server
    debug_server = None
    if enable_visualization:
        try:
            debug_server = NGPDebugServer(
                name=f"ngp_{scene}",
                max_points=vis_max_points,
                slots=4,
                reader_slots=8,
            )
            if debug_server.initialize():
                logger.logger.info("✓ Visualization server initialized")
            else:
                logger.logger.warning("✗ Failed to initialize visualization server")
                debug_server = None
        except Exception as e:
            logger.logger.warning(f"✗ Visualization not available: {e}")
            debug_server = None

    # Seed for reproducibility
    seed_everything(cfg.train.seed, deterministic=False)
    if device.type == 'cuda':
        optimize_cuda()

    # Load dataset
    scene_path = Path(cfg.dataset.root) / cfg.dataset.scene
    logger.logger.info(f"Loading scene from {scene_path}")
    cameras = load_nerf_synthetic(scene_path, cfg.dataset.train_split)
    cameras.poses = cameras.poses.to(device)
    logger.logger.info(f"Loaded {cameras.N} training cameras")

    # Create model
    logger.logger.info("Creating model...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)
    logger.log_model_info(encoder, field, rgb_head, occ_grid)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-5
    )

    # Create trainer
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Generate training rays
    logger.logger.info("Generating training rays...")
    H, W = 800, 800
    rays = make_rays(H, W, cameras, None, device)
    logger.logger.info(f"Generated {rays.N:,} rays")

    # Load ground truth images
    logger.logger.info("Loading ground truth images...")
    from ngp_baseline_torch.data_utils import load_nerf_synthetic_images
    try:
        images, _ = load_nerf_synthetic_images(scene_path, "train", device)
        logger.logger.info(f"Loaded {images.shape[0]} images")
        target_rgb = images.reshape(-1, 3)
        logger.log_dataset_info(images.shape[0], (H, W), rays.N)
    except Exception as e:
        logger.logger.warning(f"Could not load images ({e}), using synthetic target")
        coords = torch.arange(rays.N, device=device).float() / rays.N
        target_rgb = torch.stack([coords, 1.0 - coords, torch.ones_like(coords) * 0.5], dim=-1)

    # Training loop
    logger.log_training_start()
    best_psnr = 0.0
    losses = []
    psnrs = []

    for iteration in range(num_iters):
        # Sample ray batch
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
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        losses.append(metrics['loss'])
        psnrs.append(metrics['psnr'])

        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']

        # Logging
        if (iteration + 1) % cfg.train.log_every == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            avg_psnr = sum(psnrs[-100:]) / min(100, len(psnrs))

            log_metrics = {
                'loss': metrics['loss'],
                'psnr': metrics['psnr'],
                'avg_loss_100': avg_loss,
                'avg_psnr_100': avg_psnr,
                'best_psnr': best_psnr,
            }
            logger.log_step(iteration + 1, log_metrics, current_lr)

        # === Visualization update ===
        if debug_server is not None and (iteration % vis_update_interval == 0 or iteration == 0):
            try:
                with torch.no_grad():
                    # Sample density field
                    positions, colors, densities = sample_density_grid(
                        encoder=encoder,
                        field=field,
                        rgb_head=rgb_head,
                        bbox_min=np.array([-1.5, -1.5, -1.5], dtype=np.float32),
                        bbox_max=np.array([1.5, 1.5, 1.5], dtype=np.float32),
                        num_samples=vis_max_points,
                        device=device,
                        batch_size=8192,
                    )

                    # Filter by density threshold
                    positions, colors, densities = filter_by_density_threshold(
                        positions, colors, densities,
                        threshold=vis_density_threshold,
                        max_points=vis_max_points,
                    )

                    # Extract camera info (use first training camera)
                    camera_pose = cameras.poses[0].cpu().numpy()
                    camera_pos = camera_pose[:3, 3]
                    camera_target = camera_pos + camera_pose[:3, 2]

                    # Publish frame
                    success = debug_server.publish_frame(
                        iteration=iteration,
                        positions=positions,
                        colors=colors,
                        densities=densities,
                        loss=metrics['loss'],
                        psnr=metrics['psnr'],
                        learning_rate=current_lr,
                        camera_pos=camera_pos,
                        camera_target=camera_target,
                        camera_matrix=camera_pose,
                    )

                    if success and iteration % 100 == 0:
                        logger.logger.info(
                            f"  → Published {positions.shape[0]:,} points to visualization "
                            f"(frame #{debug_server.frame_count})"
                        )

            except Exception as e:
                logger.logger.warning(f"Visualization update failed: {e}")

    # Training complete
    final_metrics = {
        'final_loss': losses[-1],
        'final_psnr': psnrs[-1],
        'best_psnr': best_psnr,
        'avg_loss_final_100': sum(losses[-100:]) / len(losses[-100:]),
        'avg_psnr_final_100': sum(psnrs[-100:]) / len(psnrs[-100:]),
    }
    logger.log_training_complete(num_iters, final_metrics)

    logger.logger.info(f"\nTraining complete!")
    logger.logger.info(f"Best PSNR: {best_psnr:.2f} dB")

    # Cleanup
    if debug_server is not None:
        debug_server.shutdown()

    return encoder, field, rgb_head


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NGP with visualization")
    parser.add_argument("--scene", type=str, default="lego", help="Scene name")
    parser.add_argument("--iters", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--vis-interval", type=int, default=10, help="Visualization update interval")
    parser.add_argument("--vis-max-points", type=int, default=100_000, help="Max visualization points")
    parser.add_argument("--vis-threshold", type=float, default=0.01, help="Density threshold")

    args = parser.parse_args()

    train_nerf_with_visualization(
        scene=args.scene,
        num_iters=args.iters,
        enable_visualization=not args.no_vis,
        vis_update_interval=args.vis_interval,
        vis_max_points=args.vis_max_points,
        vis_density_threshold=args.vis_threshold,
    )

