"""Camera loading from NeRF synthetic dataset."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class CameraData:
    """Camera intrinsics and extrinsics."""
    poses: torch.Tensor  # [N, 4, 4] camera-to-world matrices
    focal: float
    width: int
    height: int
    near: float
    far: float

    @property
    def N(self) -> int:
        return self.poses.shape[0]


def load_nerf_synthetic(scene_path: str | Path, split_file: str = "transforms_train.json") -> CameraData:
    """Load camera data from NeRF synthetic dataset.

    Args:
        scene_path: Path to scene directory (e.g., "data/nerf-synthetic/lego")
        split_file: Transform JSON filename

    Returns:
        CameraData with loaded poses and intrinsics
    """
    scene_path = Path(scene_path)
    transforms_path = scene_path / split_file

    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    # Extract intrinsics
    camera_angle_x = meta['camera_angle_x']

    # Load first frame to get resolution
    frames = meta['frames']
    if not frames:
        raise ValueError(f"No frames found in {transforms_path}")

    # Try to infer resolution from image file or use default
    first_frame = frames[0]
    img_path = scene_path / (first_frame['file_path'].replace('./', '') + '.png')

    # Default resolution for NeRF synthetic
    width = height = 800

    # Calculate focal length
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    # Load all poses
    poses_list = []
    for frame in frames:
        pose_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        poses_list.append(pose_matrix)

    poses = torch.from_numpy(np.stack(poses_list, axis=0))  # [N, 4, 4]

    # Default near/far for NeRF synthetic
    near = 2.0
    far = 6.0

    return CameraData(
        poses=poses,
        focal=focal,
        width=width,
        height=height,
        near=near,
        far=far
    )


def estimate_scene_aabb(
    cameras: CameraData,
    near: float | None = None,
    far: float | None = None,
    margin: float = 0.05,
    samples_per_axis: int = 8,
) -> tuple[float, float, float, float, float, float]:
    """Estimate an axis-aligned bounding box that encloses the scene.

    The bounds are inferred from each camera frustum by projecting the
    extremal image-plane corners at the given near/far planes.

    Args:
        cameras: Loaded camera data.
        near: Optional override for near plane distance.
        far: Optional override for far plane distance.
        margin: Fractional padding applied to each axis extent.
        samples_per_axis: Number of evenly spaced samples along each image axis.

    Returns:
        Tuple (xmin, ymin, zmin, xmax, ymax, zmax) covering the scene.
    """
    poses = cameras.poses
    device = poses.device
    dtype = poses.dtype

    H = float(cameras.height)
    W = float(cameras.width)
    focal = float(cameras.focal)
    near = float(near if near is not None else cameras.near)
    far = float(far if far is not None else cameras.far)

    samples_per_axis = max(int(samples_per_axis), 2)
    i_vals = torch.linspace(0.0, H - 1.0, steps=samples_per_axis, device=device, dtype=dtype)
    j_vals = torch.linspace(0.0, W - 1.0, steps=samples_per_axis, device=device, dtype=dtype)
    ii, jj = torch.meshgrid(i_vals, j_vals, indexing='ij')

    pixels = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=-1)
    i = pixels[:, 0]
    j = pixels[:, 1]

    dirs_cam = torch.stack(
        [
            (j - W * 0.5) / focal,
            -(i - H * 0.5) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

    # Transform directions into world space for every camera pose.
    dirs_cam = dirs_cam.unsqueeze(0).expand(poses.shape[0], -1, -1)  # [N, C, 3]
    rot = poses[:, None, :3, :3]  # [N, 1, 3, 3]
    dirs_world = torch.sum(dirs_cam.unsqueeze(-2) * rot, dim=-1)  # [N, C, 3]
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

    origins = poses[:, None, :3, 3]  # [N, 1, 3]

    depths = torch.tensor([near, far], device=device, dtype=dtype)  # [2]
    points = origins.unsqueeze(2) + dirs_world.unsqueeze(2) * depths.view(1, 1, -1, 1)
    points = points.reshape(-1, 3)

    mins = points.min(dim=0).values
    maxs = points.max(dim=0).values

    extent = maxs - mins
    padding = torch.max(extent * margin, torch.full_like(extent, 1e-3))

    mins = (mins - padding).cpu().tolist()
    maxs = (maxs + padding).cpu().tolist()

    return (
        mins[0],
        mins[1],
        mins[2],
        maxs[0],
        maxs[1],
        maxs[2],
    )

