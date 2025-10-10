"""Ray generation for rendering."""
from __future__ import annotations
import torch
from ..types import RayBatch
from .cameras import CameraData


def make_rays(H: int, W: int, cameras: CameraData, indices: torch.Tensor | None = None,
              device: torch.device | None = None) -> RayBatch:
    """Generate rays for given camera(s).

    Args:
        H: Image height
        W: Image width
        cameras: Camera data with poses and intrinsics
        indices: Optional camera indices to sample from [K], if None uses all cameras
        device: Target device, if None uses cameras.poses.device

    Returns:
        RayBatch in SoA format
    """
    if device is None:
        device = cameras.poses.device

    # Determine which cameras to use
    if indices is None:
        poses = cameras.poses.to(device)  # [N, 4, 4]
        N = cameras.N
    else:
        poses = cameras.poses[indices].to(device)  # [K, 4, 4]
        N = indices.shape[0]

    # Generate pixel coordinates [H, W, 2]
    i, j = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Camera coordinates (centered, y-down convention)
    focal = cameras.focal
    dirs = torch.stack([
        (j - W * 0.5) / focal,
        -(i - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]

    # Flatten to [H*W, 3]
    dirs = dirs.reshape(-1, 3)

    # Expand for all cameras: [N, H*W, 3]
    dirs = dirs.unsqueeze(0).expand(N, -1, -1)

    # Transform ray directions by camera rotation
    # poses[:, :3, :3] is rotation matrix [N, 3, 3]
    # dirs [N, H*W, 3] @ R^T = [N, H*W, 3]
    rays_d = torch.sum(dirs[..., None, :] * poses[:, None, :3, :3], dim=-1)  # [N, H*W, 3]

    # Ray origins from camera translation
    rays_o = poses[:, None, :3, 3].expand(-1, H * W, -1)  # [N, H*W, 3]

    # Normalize directions
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Flatten across cameras and pixels: [N*H*W, 3]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    # Create tmin/tmax
    num_rays = rays_o.shape[0]
    tmin = torch.full((num_rays,), cameras.near, device=device, dtype=torch.float32)
    tmax = torch.full((num_rays,), cameras.far, device=device, dtype=torch.float32)

    # Convert to SoA format
    return RayBatch(
        orig_x=rays_o[:, 0],
        orig_y=rays_o[:, 1],
        orig_z=rays_o[:, 2],
        dir_x=rays_d[:, 0],
        dir_y=rays_d[:, 1],
        dir_z=rays_d[:, 2],
        tmin=tmin,
        tmax=tmax,
        mask=None
    )


def make_rays_single(H: int, W: int, pose: torch.Tensor, focal: float,
                     near: float, far: float, device: torch.device) -> RayBatch:
    """Generate rays for a single camera pose (optimized).

    Args:
        H: Image height
        W: Image width
        pose: Single camera pose [4, 4]
        focal: Focal length
        near: Near plane
        far: Far plane
        device: Target device

    Returns:
        RayBatch in SoA format
    """
    pose = pose.to(device)

    # Generate pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Camera coordinates
    dirs = torch.stack([
        (j - W * 0.5) / focal,
        -(i - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]

    # Transform by rotation
    rays_d = torch.sum(dirs[..., None, :] * pose[None, None, :3, :3], dim=-1)  # [H, W, 3]
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins
    rays_o = pose[None, None, :3, 3].expand(H, W, -1)  # [H, W, 3]

    # Flatten
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    num_rays = rays_o.shape[0]
    tmin = torch.full((num_rays,), near, device=device, dtype=torch.float32)
    tmax = torch.full((num_rays,), far, device=device, dtype=torch.float32)

    return RayBatch(
        orig_x=rays_o[:, 0],
        orig_y=rays_o[:, 1],
        orig_z=rays_o[:, 2],
        dir_x=rays_d[:, 0],
        dir_y=rays_d[:, 1],
        dir_z=rays_d[:, 2],
        tmin=tmin,
        tmax=tmax,
        mask=None
    )

