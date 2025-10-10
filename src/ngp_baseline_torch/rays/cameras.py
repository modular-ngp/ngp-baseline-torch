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

