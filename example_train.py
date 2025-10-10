"""Simple training example for NGP baseline."""
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays
from ngp_baseline_torch.runtime import Trainer, render_image
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.device import optimize_cuda
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
    cfg.train.lr = 5e-3  # 降低学习率从 1e-2 到 5e-3
    cfg.train.iters = num_iters

    # Seed for reproducibility
    seed_everything(cfg.train.seed, deterministic=False)
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

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-5
    )

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

        # 保存第一张训练图像作为参考
        gt_image = images[0].cpu().numpy()

    except Exception as e:
        print(f"Warning: Could not load images ({e}), using synthetic target")
        # Fallback: create a simple pattern for testing
        coords = torch.arange(rays.N, device=device).float() / rays.N
        target_rgb = torch.stack([
            coords,
            1.0 - coords,
            torch.ones_like(coords) * 0.5
        ], dim=-1)
        print("Using synthetic gradient pattern for testing")
        gt_image = None

    # 创建输出目录用于保存中间渲染结果
    render_dir = Path("outputs") / scene / "training_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    # 保存ground truth参考图
    if gt_image is not None:
        gt_img_pil = Image.fromarray((gt_image * 255).astype(np.uint8))
        gt_img_pil.save(render_dir / "ground_truth.png")
        print(f"Saved ground truth reference to {render_dir / 'ground_truth.png'}")

    # 选择一个测试视角用于中间渲染（使用第一个训练视角）
    test_pose = cameras.poses[0]
    test_focal = cameras.focal
    test_near = cameras.near
    test_far = cameras.far

    # 渲染间隔（每隔多少步渲染一次）
    render_interval = min(100, num_iters // 20)  # 至少渲染20次
    print(f"Will render intermediate results every {render_interval} iterations")

    # 添加训练统计
    best_psnr = 0.0
    best_render_psnr = 0.0  # 基于完整渲染的PSNR
    losses = []
    psnrs = []

    # Training loop
    print(f"\nTraining for {num_iters} iterations...")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.6f}")

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

        # 更新学习率
        scheduler.step()

        # 记录统计
        losses.append(metrics['loss'])
        psnrs.append(metrics['psnr'])

        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']

        # Log
        if (iteration + 1) % cfg.train.log_every == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            avg_psnr = sum(psnrs[-100:]) / min(100, len(psnrs))
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Iter {iteration + 1}/{num_iters}: "
                  f"Loss={metrics['loss']:.6f} (avg={avg_loss:.6f}), "
                  f"PSNR={metrics['psnr']:.2f} dB (avg={avg_psnr:.2f}, best={best_psnr:.2f}), "
                  f"LR={current_lr:.6f}")

        # 渲染中间结果
        if (iteration + 1) % render_interval == 0 or iteration == 0:
            print(f"  → Rendering intermediate result at iteration {iteration + 1}...")

            # 使用较低分辨率加快渲染，但保持相同的视场角
            render_H, render_W = 400, 400
            # 重要：按比例缩放focal length以保持相同的FOV
            scale_factor = render_H / H  # 400 / 800 = 0.5
            render_focal = test_focal * scale_factor

            with torch.no_grad():
                rendered_image = render_image(
                    H=render_H,
                    W=render_W,
                    pose=test_pose,
                    focal=render_focal,  # 使用缩放后的focal
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

                # 计算渲染图像的真实PSNR（与ground truth对比）
                if gt_image is not None:
                    # 调整ground truth到相同分辨率
                    gt_resized = torch.nn.functional.interpolate(
                        torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).to(device),
                        size=(render_H, render_W),
                        mode='bilinear',
                        align_corners=False
                    )
                    gt_resized = gt_resized.squeeze(0).permute(1, 2, 0)

                    # 计算MSE和PSNR
                    mse_render = torch.mean((rendered_image - gt_resized) ** 2).item()
                    render_psnr = -10.0 * torch.log10(torch.tensor(mse_render + 1e-8)).item()

                    if render_psnr > best_render_psnr:
                        best_render_psnr = render_psnr

                    print(f"  → Render PSNR: {render_psnr:.2f} dB (best: {best_render_psnr:.2f} dB)")

            # 转换为numpy并保存
            img_np = rendered_image.cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

            # 保存图像，文件名包含渲染PSNR
            if gt_image is not None:
                output_path = render_dir / f"iter_{iteration + 1:06d}_psnr_{render_psnr:.2f}dB.png"
            else:
                output_path = render_dir / f"iter_{iteration + 1:06d}.png"
            img_pil.save(output_path)
            print(f"  → Saved to {output_path}")

    print(f"\nTraining complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Final PSNR: {psnrs[-1]:.2f} dB")

    # 渲染最终的高分辨率图像
    print("\nRendering final high-resolution image...")
    with torch.no_grad():
        final_image = render_image(
            H=800,
            W=800,
            pose=test_pose,
            focal=test_focal,
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

    final_img_np = final_image.cpu().numpy()
    final_img_np = np.clip(final_img_np, 0, 1)
    final_img_pil = Image.fromarray((final_img_np * 255).astype(np.uint8))
    final_img_pil.save(render_dir / "final_render.png")
    print(f"Saved final render to {render_dir / 'final_render.png'}")

    # Save checkpoint
    from ngp_baseline_torch.artifact import export
    output_dir = Path("outputs") / scene
    export(encoder, field, rgb_head, cfg, output_dir, occ_grid)
    print(f"\nModel saved to {output_dir}")
    print(f"Training renders saved to {render_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="lego", help="Scene name")
    parser.add_argument("--iters", type=int, default=10000, help="Training iterations")
    args = parser.parse_args()

    train_nerf(args.scene, args.iters)
