"""Optimized training example for NGP baseline - Fixed critical performance issues."""
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic
from ngp_baseline_torch.runtime import Trainer, render_image
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.device import optimize_cuda
from ngp_baseline_torch.logger import NGPLogger
import numpy as np
from PIL import Image
import random


def train_nerf(scene: str = "lego", num_iters: int = 30000):
    """Train NGP NeRF model with proper Instant-NGP configuration.

    Args:
        scene: Scene name from nerf-synthetic dataset
        num_iters: Number of training iterations (30k for baseline results)
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create configuration with optimized defaults
    cfg = Config()
    cfg.dataset.scene = scene
    cfg.dataset.white_background = True  # Critical for NeRF-Synthetic

    # Model config - view-dependent colors enabled
    cfg.model.view_dependent = True  # CRITICAL FIX
    cfg.model.hash_levels = 16
    cfg.model.mlp_width = 64
    cfg.model.mlp_depth = 2

    # Integrator - reduced steps, enable early stopping
    cfg.integrator.n_steps_fixed = 64  # Down from 128 for speed
    cfg.integrator.early_stop_T = 1e-4  # Enable early stopping
    cfg.integrator.sigma_thresh = 0.01

    # Grid - multi-level occupancy
    cfg.grid.levels = 4  # Multi-level (was 1)
    cfg.grid.update_every = 16

    # Training - increased batch, separate LRs, enable AMP
    cfg.train.batch_rays = 8192  # Increased from 4096
    cfg.train.lr_encoder = 1e-2
    cfg.train.lr_mlp = 1e-3  # Lower LR for MLP
    cfg.train.iters = num_iters
    cfg.train.deterministic = False  # Disable for speed

    # Precision - enable AMP for speed
    cfg.precision.use_amp = True

    # Initialize logger
    log_dir = Path("outputs") / scene / "logs"
    logger = NGPLogger(log_dir, cfg, run_name=f"{scene}_{num_iters}iters_optimized")

    logger.log_system_info()
    logger.log_complete_config()

    # Seed for reproducibility (but not deterministic mode)
    seed_everything(cfg.train.seed, deterministic=False)
    if device.type == 'cuda':
        optimize_cuda()

    # Load dataset
    scene_path = Path(cfg.dataset.root) / cfg.dataset.scene
    logger.logger.info(f"Loading scene from {scene_path}")
    cameras = load_nerf_synthetic(scene_path, cfg.dataset.train_split)
    cameras.poses = cameras.poses.to(device)
    n_cameras = cameras.N
    logger.logger.info(f"Loaded {n_cameras} training cameras")

    # Create model
    logger.logger.info("Creating model with view-dependent colors...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)
    logger.log_model_info(encoder, field, rgb_head, occ_grid)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-5
    )

    # Create trainer
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Load ground truth images for validation
    logger.logger.info("Loading ground truth images...")
    from ngp_baseline_torch.data_utils import load_nerf_synthetic_images
    H, W = 800, 800

    try:
        images, _ = load_nerf_synthetic_images(scene_path, "train", device)
        logger.logger.info(f"Loaded {images.shape[0]} images with shape {images.shape[1:3]}")

        # Handle background
        if cfg.dataset.white_background:
            # Images are already on white background in NeRF-Synthetic
            pass

        logger.log_dataset_info(images.shape[0], (H, W), images.shape[0] * H * W)
        gt_image = images[0].cpu().numpy()
        has_gt = True
    except Exception as e:
        logger.logger.warning(f"Could not load images: {e}")
        images = None
        gt_image = None
        has_gt = False

    # Setup output directories
    render_dir = Path("outputs") / scene / "training_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    if gt_image is not None:
        gt_img_pil = Image.fromarray((np.clip(gt_image, 0, 1) * 255).astype(np.uint8))
        gt_img_pil.save(render_dir / "ground_truth.png")
        logger.logger.info(f"Saved ground truth to {render_dir / 'ground_truth.png'}")

    # Test camera for intermediate renders
    test_pose = cameras.poses[0]
    test_focal = cameras.focal
    test_near = cameras.near
    test_far = cameras.far

    # Render interval
    render_interval = max(500, num_iters // 40)
    logger.logger.info(f"Will render every {render_interval} iterations")

    # Training statistics
    best_psnr = 0.0
    best_render_psnr = 0.0
    losses = []
    psnrs = []

    logger.log_training_start()
    logger.logger.info(f"Starting training with optimized configuration")
    logger.logger.info(f"Encoder LR: {cfg.train.lr_encoder}, MLP LR: {cfg.train.lr_mlp}")
    logger.logger.info(f"AMP enabled: {cfg.precision.use_amp}")
    logger.logger.info(f"View-dependent: {cfg.model.view_dependent}")

    # CRITICAL FIX: On-the-fly ray generation instead of pre-generating all rays
    for iteration in range(num_iters):
        # Sample random camera
        cam_idx = random.randint(0, n_cameras - 1)

        # Sample random pixels from this camera
        pixel_indices = torch.randint(0, H * W, (cfg.train.batch_rays,), device=device)

        # Generate rays on-the-fly for this batch only
        from ngp_baseline_torch.rays import make_rays_single
        from ngp_baseline_torch.types import RayBatch

        # Get camera
        pose = cameras.poses[cam_idx]

        # Generate full rays for one camera (much smaller than all cameras)
        rays_full = make_rays_single(H, W, pose, cameras.focal, cameras.near, cameras.far, device)

        # Sample subset
        ray_batch = RayBatch(
            orig_x=rays_full.orig_x[pixel_indices],
            orig_y=rays_full.orig_y[pixel_indices],
            orig_z=rays_full.orig_z[pixel_indices],
            dir_x=rays_full.dir_x[pixel_indices],
            dir_y=rays_full.dir_y[pixel_indices],
            dir_z=rays_full.dir_z[pixel_indices],
            tmin=rays_full.tmin[pixel_indices],
            tmax=rays_full.tmax[pixel_indices],
        )

        # Get target colors if available
        if has_gt:
            target_batch = images[cam_idx].reshape(-1, 3)[pixel_indices]
        else:
            # Fallback synthetic target
            target_batch = torch.rand(cfg.train.batch_rays, 3, device=device)

        # Training step
        metrics = trainer.step(ray_batch, target_batch)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Record statistics
        losses.append(metrics['loss'])
        psnrs.append(metrics['psnr'])

        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']

        # Log training step
        if (iteration + 1) % cfg.train.log_every == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            avg_psnr = sum(psnrs[-100:]) / min(100, len(psnrs))

            log_metrics = {
                'loss': metrics['loss'],
                'psnr': metrics['psnr'],
                'avg_loss_100': avg_loss,
                'avg_psnr_100': avg_psnr,
                'best_psnr': best_psnr,
                'num_steps': metrics.get('num_steps', cfg.integrator.n_steps_fixed)
            }

            logger.log_step(iteration + 1, log_metrics, current_lr)

            # Gradient and weight stats (less frequent to avoid NaN issues)
            if (iteration + 1) % 1000 == 0:
                try:
                    logger.log_gradient_stats(encoder, field, rgb_head)
                    logger.log_weight_stats(encoder, field, rgb_head)
                except Exception as e:
                    logger.logger.warning(f"Stats error: {e}")

        # Epoch summary
        if (iteration + 1) % 1000 == 0 and iteration > 0:
            start_idx = max(0, len(losses) - 1000)
            epoch_metrics = {
                'loss': sum(losses[start_idx:]) / len(losses[start_idx:]),
                'psnr': sum(psnrs[start_idx:]) / len(psnrs[start_idx:]),
            }
            logger.log_epoch_summary(iteration - 999, iteration + 1, epoch_metrics)

        # Render intermediate results
        if (iteration + 1) % render_interval == 0 or iteration == 0:
            logger.logger.info(f"Rendering at iteration {iteration + 1}...")

            render_H, render_W = 400, 400
            scale_factor = render_H / H
            render_focal = test_focal * scale_factor

            with torch.no_grad():
                rendered_image, aux = render_image(
                    H=render_H,
                    W=render_W,
                    pose=test_pose,
                    focal=render_focal,
                    near=test_near,
                    far=test_far,
                    encoder=encoder,
                    field=field,
                    rgb_head=rgb_head,
                    cfg=cfg,
                    device=device,
                    chunk_size=4096,
                    occupancy_grid=occ_grid
                )

                try:
                    logger.log_rendering_stats(aux)
                except:
                    pass

            img_np = rendered_image.cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(render_dir / f"iter_{iteration + 1:06d}.png")

            if gt_image is not None:
                from PIL import Image as PILImage
                gt_resized = np.array(PILImage.fromarray((gt_image * 255).astype(np.uint8)).resize(
                    (render_W, render_H), PILImage.LANCZOS
                )) / 255.0

                mse = np.mean((img_np - gt_resized) ** 2)
                render_psnr = -10 * np.log10(mse + 1e-8)

                if render_psnr > best_render_psnr:
                    best_render_psnr = render_psnr

                val_metrics = {
                    'render_psnr': render_psnr,
                    'render_mse': mse,
                    'best_render_psnr': best_render_psnr
                }
                logger.log_validation(iteration + 1, val_metrics)

    # Training complete
    final_metrics = {
        'final_loss': losses[-1],
        'final_psnr': psnrs[-1],
        'best_batch_psnr': best_psnr,
        'best_render_psnr': best_render_psnr,
        'avg_loss_final_100': sum(losses[-100:]) / len(losses[-100:]),
        'avg_psnr_final_100': sum(psnrs[-100:]) / len(psnrs[-100:]),
    }
    logger.log_training_complete(num_iters, final_metrics)

    logger.logger.info(f"\n{'='*80}")
    logger.logger.info("TRAINING COMPLETE")
    logger.logger.info(f"{'='*80}")
    logger.logger.info(f"Best batch PSNR: {best_psnr:.2f} dB")
    if gt_image is not None:
        logger.logger.info(f"Best render PSNR: {best_render_psnr:.2f} dB")
    logger.logger.info(f"Expected PSNR ranges (Lego, 800x800, white background):")
    logger.logger.info(f"  5k steps: 30-32 dB")
    logger.logger.info(f"  10k steps: 33-34.5 dB")
    logger.logger.info(f"  30-35k steps: 35.5-36.0 dB")
    logger.logger.info(f"Logs: {log_dir}")
    logger.logger.info(f"Renders: {render_dir}")

    return encoder, field, rgb_head, occ_grid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="lego", help="Scene name")
    parser.add_argument("--iters", type=int, default=30000, help="Training iterations (30k for baseline)")
    args = parser.parse_args()

    train_nerf(args.scene, args.iters)

