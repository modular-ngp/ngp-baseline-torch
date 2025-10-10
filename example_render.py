"""Simple inference/rendering example."""
import torch
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all
from ngp_baseline_torch.artifact import load
from ngp_baseline_torch.rays import load_nerf_synthetic
from ngp_baseline_torch.runtime import render_image


def render_from_checkpoint(checkpoint_dir: str, scene: str = "lego", output_dir: str = "renders"):
    """Render images from a trained checkpoint.

    Args:
        checkpoint_dir: Directory containing saved model
        scene: Scene name
        output_dir: Output directory for rendered images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    cfg = Config()
    cfg.dataset.scene = scene

    # Create model
    print("Creating model...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)

    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    print(f"Loading checkpoint from {checkpoint_path}")
    encoder, field, rgb_head, occ_grid = load(
        checkpoint_path, encoder, field, rgb_head, device, occ_grid
    )

    # Load camera poses
    scene_path = Path(cfg.dataset.root) / cfg.dataset.scene
    print(f"Loading cameras from {scene_path}")

    # Try test set first, fallback to validation
    try:
        cameras = load_nerf_synthetic(scene_path, cfg.dataset.test_split)
        print(f"Loaded {cameras.N} test cameras")
    except:
        cameras = load_nerf_synthetic(scene_path, cfg.dataset.val_split)
        print(f"Loaded {cameras.N} validation cameras")

    cameras.poses = cameras.poses.to(device)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Render all views
    print(f"\nRendering {cameras.N} images...")
    for i in range(cameras.N):
        print(f"Rendering view {i+1}/{cameras.N}...", end=' ')

        image = render_image(
            H=cameras.height,
            W=cameras.width,
            pose=cameras.poses[i],
            focal=cameras.focal,
            near=cameras.near,
            far=cameras.far,
            encoder=encoder,
            field=field,
            rgb_head=rgb_head,
            cfg=cfg,
            device=device,
            chunk_size=4096
        )

        # Save image
        import numpy as np
        from PIL import Image

        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img.save(output_path / f"render_{i:04d}.png")
        print("Done")

    print(f"\nAll images saved to {output_path}")


def render_video_360(checkpoint_dir: str, scene: str = "lego",
                     output_file: str = "render_360.mp4", n_frames: int = 120):
    """Render a 360-degree video around the scene.

    Args:
        checkpoint_dir: Directory containing saved model
        scene: Scene name
        output_file: Output video filename
        n_frames: Number of frames in video
    """
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    cfg = Config()
    cfg.dataset.scene = scene

    # Create and load model
    print("Loading model...")
    encoder, field, rgb_head, occ_grid = create_all(cfg, device)
    encoder, field, rgb_head, occ_grid = load(
        checkpoint_dir, encoder, field, rgb_head, device, occ_grid
    )

    # Load reference camera
    scene_path = Path(cfg.dataset.root) / cfg.dataset.scene
    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")

    # Generate circular camera path
    radius = 4.0
    height = 0.0
    frames = []

    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames

        # Camera position
        cam_x = radius * np.cos(angle)
        cam_z = radius * np.sin(angle)
        cam_y = height

        # Look at origin
        forward = -np.array([cam_x, cam_y, cam_z])
        forward = forward / np.linalg.norm(forward)

        # Up vector
        up = np.array([0, 1, 0])

        # Right vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Recompute up
        up = np.cross(forward, right)

        # Construct pose matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward
        pose[:3, 3] = [cam_x, cam_y, cam_z]

        pose_tensor = torch.from_numpy(pose).to(device)

        # Render
        print(f"Frame {i+1}/{n_frames}...", end=' ')
        image = render_image(
            H=400, W=400,
            pose=pose_tensor,
            focal=cameras.focal,
            near=2.0,
            far=6.0,
            encoder=encoder,
            field=field,
            rgb_head=rgb_head,
            cfg=cfg,
            device=device,
            chunk_size=4096
        )

        frames.append((image.cpu().numpy() * 255).astype(np.uint8))
        print("Done")

    # Save video
    try:
        import cv2

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"\nVideo saved to {output_file}")
    except ImportError:
        print("\nOpenCV not installed. Saving frames as images instead...")
        output_dir = Path("video_frames")
        output_dir.mkdir(exist_ok=True)

        from PIL import Image
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(output_dir / f"frame_{i:04d}.png")

        print(f"Frames saved to {output_dir}/")
        print("Install OpenCV to generate video: pip install opencv-python")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--scene", type=str, default="lego", help="Scene name")
    parser.add_argument("--mode", type=str, choices=["render", "video"], default="render",
                       help="Render mode: 'render' for all views, 'video' for 360 video")
    parser.add_argument("--output", type=str, default="renders", help="Output directory or file")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames for video")

    args = parser.parse_args()

    if args.mode == "render":
        render_from_checkpoint(args.checkpoint, args.scene, args.output)
    else:
        render_video_360(args.checkpoint, args.scene, args.output, args.frames)

