"""Simple training example for NGP baseline."""
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
    cfg.train.lr = 5e-3
    cfg.train.iters = num_iters

    # Initialize comprehensive logger
    log_dir = Path("outputs") / scene / "logs"
    logger = NGPLogger(log_dir, cfg, run_name=f"{scene}_{num_iters}iters")

    # Log system information
    logger.log_system_info()

    # Log complete configuration
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
    logger.logger.info("Creating model...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    optimizer = create_optimizer(encoder, field, rgb_head, cfg)

    # Log model architecture and parameters
    logger.log_model_info(encoder, field, rgb_head, occ_grid)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-5
    )

    # Create trainer
    trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)

    # Generate all training rays once
    logger.logger.info("Generating training rays...")
    H, W = 800, 800
    rays = make_rays(H, W, cameras, None, device)
    logger.logger.info(f"Generated {rays.N:,} rays")

    # Load ground truth images
    logger.logger.info("Loading ground truth images...")
    from ngp_baseline_torch.data_utils import load_nerf_synthetic_images
    try:
        images, _ = load_nerf_synthetic_images(scene_path, "train", device)
        logger.logger.info(f"Loaded {images.shape[0]} images with shape {images.shape[1:3]}")

        target_rgb = images.reshape(-1, 3)
        assert target_rgb.shape[0] == rays.N, \
            f"Ray count {rays.N} doesn't match image pixels {target_rgb.shape[0]}"

        logger.logger.info(f"Target RGB range: [{target_rgb.min():.3f}, {target_rgb.max():.3f}]")
        logger.log_dataset_info(images.shape[0], (H, W), rays.N)

        gt_image = images[0].cpu().numpy()

    except Exception as e:
        logger.logger.warning(f"Could not load images ({e}), using synthetic target")
        coords = torch.arange(rays.N, device=device).float() / rays.N
        target_rgb = torch.stack([
            coords,
            1.0 - coords,
            torch.ones_like(coords) * 0.5
        ], dim=-1)
        logger.logger.info("Using synthetic gradient pattern for testing")
        gt_image = None

    # 创建输出目录
    render_dir = Path("outputs") / scene / "training_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    if gt_image is not None:
        gt_img_pil = Image.fromarray((gt_image * 255).astype(np.uint8))
        gt_img_pil.save(render_dir / "ground_truth.png")
        logger.logger.info(f"Saved ground truth reference to {render_dir / 'ground_truth.png'}")

    test_pose = cameras.poses[0]
    test_focal = cameras.focal
    test_near = cameras.near
    test_far = cameras.far

    render_interval = min(100, num_iters // 20)
    logger.logger.info(f"Will render intermediate results every {render_interval} iterations")

    best_psnr = 0.0
    best_render_psnr = 0.0
    losses = []
    psnrs = []

    # Start training
    logger.log_training_start()
    logger.logger.info(f"Initial LR: {optimizer.param_groups[0]['lr']:.6f}")

    for iteration in range(num_iters):
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

        metrics = trainer.step(ray_batch, target_batch)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

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

            if (iteration + 1) % 500 == 0:
                logger.log_gradient_stats(encoder, field, rgb_head)
                logger.log_weight_stats(encoder, field, rgb_head)

        if (iteration + 1) % 1000 == 0 and iteration > 0:
            start_idx = max(0, len(losses) - 1000)
            epoch_metrics = {
                'loss': sum(losses[start_idx:]) / len(losses[start_idx:]),
                'psnr': sum(psnrs[start_idx:]) / len(psnrs[start_idx:]),
            }
            logger.log_epoch_summary(iteration - 999, iteration + 1, epoch_metrics)

        if (iteration + 1) % render_interval == 0 or iteration == 0:
            logger.logger.info(f"  → Rendering intermediate result at iteration {iteration + 1}...")

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

                logger.log_rendering_stats(aux)

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

    logger.logger.info(f"\nTraining complete!")
    logger.logger.info(f"Best batch PSNR: {best_psnr:.2f} dB")
    if gt_image is not None:
        logger.logger.info(f"Best render PSNR: {best_render_psnr:.2f} dB")
    logger.logger.info(f"Logs saved to: {log_dir}")
    logger.logger.info(f"Renders saved to: {render_dir}")

    return encoder, field, rgb_head, occ_grid


if __name__ == "__main__":
    train_nerf(scene="lego", num_iters=10000)
