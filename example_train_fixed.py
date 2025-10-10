"""Improved training example with all convergence fixes."""
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays
from ngp_baseline_torch.runtime import Trainer, render_image
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.device import optimize_cuda
from ngp_baseline_torch.logger import NGPLogger
import numpy as np
from PIL import Image


def exponential_lr_scheduler(optimizer, iteration, warmup_iters=256, decay_start=1000, decay_end=20000,
                             initial_lrs=None, min_factor=0.1):
    """Custom exponential learning rate scheduler with warmup."""
    if initial_lrs is None:
        initial_lrs = [group['lr'] for group in optimizer.param_groups]

    for i, param_group in enumerate(optimizer.param_groups):
        if iteration < warmup_iters:
            # Linear warmup
            lr = initial_lrs[i] * (iteration / warmup_iters)
        elif iteration < decay_start:
            # Constant LR
            lr = initial_lrs[i]
        else:
            # Exponential decay
            progress = (iteration - decay_start) / (decay_end - decay_start)
            progress = min(progress, 1.0)
            lr = initial_lrs[i] * (min_factor ** progress)

        param_group['lr'] = lr


def train_nerf(scene: str = "lego", num_iters: int = 20000):
    """Train an improved NGP NeRF model with convergence fixes.

    Args:
        scene: Scene name from nerf-synthetic dataset
        num_iters: Number of training iterations
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create configuration with improved defaults
    cfg = Config()
    cfg.dataset.scene = scene

    # Model config
    cfg.model.hash_levels = 16
    cfg.model.mlp_width = 64
    cfg.model.mlp_depth = 2
    cfg.model.density_activation = "trunc_exp"
    cfg.model.view_dependent = True

    # Integrator config
    cfg.integrator.n_steps_fixed = 128
    cfg.integrator.perturb = True  # CRITICAL: Enable jittering
    cfg.integrator.early_stop_T = 1e-4

    # Training config
    cfg.train.batch_rays = 4096
    cfg.train.lr_encoder = 1e-2  # High LR for hash grid
    cfg.train.lr_mlp = 1e-3  # Lower LR for MLP
    cfg.train.iters = num_iters
    cfg.train.use_lr_scheduler = True

    # Precision
    cfg.precision.use_amp = False  # Disable for stability

    # Initialize logger
    log_dir = Path("outputs") / scene / "logs_fixed"
    logger = NGPLogger(log_dir, cfg, run_name=f"{scene}_fixed_{num_iters}iters")
    logger.log_system_info()
    logger.log_complete_config()

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
    logger.logger.info("Creating model with improved initialization...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)

    # Log model info
    logger.log_model_info(encoder, field, rgb_head, occ_grid)

    # Store initial LRs for scheduler
    initial_lrs = [group['lr'] for group in optimizer.param_groups]

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
    images, _ = load_nerf_synthetic_images(scene_path, "train", device)
    logger.logger.info(f"Loaded {images.shape[0]} images with shape {images.shape[1:3]}")

    target_rgb = images.reshape(-1, 3)
    assert target_rgb.shape[0] == rays.N
    logger.log_dataset_info(images.shape[0], (H, W), rays.N)

    gt_image = images[0].cpu().numpy()

    # Create output directory
    render_dir = Path("outputs") / scene / "training_renders_fixed"
    render_dir.mkdir(parents=True, exist_ok=True)

    # Save ground truth
    gt_img_pil = Image.fromarray((gt_image * 255).astype(np.uint8))
    gt_img_pil.save(render_dir / "ground_truth.png")

    # Test view for rendering
    test_pose = cameras.poses[0]
    test_focal = cameras.focal
    test_near = cameras.near
    test_far = cameras.far

    render_interval = 500
    logger.logger.info(f"Will render results every {render_interval} iterations")

    best_psnr = 0.0
    best_render_psnr = 0.0
    losses = []
    psnrs = []

    # Training loop
    logger.log_training_start()

    for iteration in range(num_iters):
        # Update learning rate with custom scheduler
        if cfg.train.use_lr_scheduler:
            exponential_lr_scheduler(
                optimizer, iteration,
                warmup_iters=256,
                decay_start=cfg.train.lr_decay_start,
                decay_end=cfg.train.lr_decay_end,
                initial_lrs=initial_lrs,
                min_factor=cfg.train.lr_min_factor
            )

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
        current_lr = optimizer.param_groups[0]['lr']

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
                'num_steps': metrics.get('num_steps', cfg.integrator.n_steps_fixed)
            }

            logger.log_step(iteration + 1, log_metrics, current_lr)

        # Periodic rendering
        if (iteration + 1) % render_interval == 0 or iteration == 0:
            logger.logger.info(f"  → Rendering at iteration {iteration + 1}...")

            render_H, render_W = 400, 400
            scale_factor = render_H / H
            render_focal = test_focal * scale_factor

            with torch.no_grad():
                encoder.eval()
                field.eval()
                rgb_head.eval()

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

                encoder.train()
                field.train()
                rgb_head.train()

            img_np = rendered_image.cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(render_dir / f"iter_{iteration + 1:06d}.png")

            # Compute PSNR against ground truth
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
            logger.logger.info(f"  → Render PSNR: {render_psnr:.2f} dB (Best: {best_render_psnr:.2f})")

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

    logger.logger.info(f"\n{'='*60}")
    logger.logger.info(f"Training Complete!")
    logger.logger.info(f"Best Batch PSNR: {best_psnr:.2f} dB")
    logger.logger.info(f"Best Render PSNR: {best_render_psnr:.2f} dB")
    logger.logger.info(f"Final Avg PSNR (last 100): {final_metrics['avg_psnr_final_100']:.2f} dB")
    logger.logger.info(f"{'='*60}\n")

    return best_render_psnr


if __name__ == "__main__":
    import sys
    scene = sys.argv[1] if len(sys.argv) > 1 else "lego"
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    print(f"\n{'='*60}")
    print(f"Starting IMPROVED NGP Training")
    print(f"Scene: {scene}")
    print(f"Iterations: {num_iters}")
    print(f"{'='*60}\n")

    best_psnr = train_nerf(scene, num_iters)

    print(f"\nFinal Best Render PSNR: {best_psnr:.2f} dB")
    print(f"Expected for Instant-NGP: >29 dB")

