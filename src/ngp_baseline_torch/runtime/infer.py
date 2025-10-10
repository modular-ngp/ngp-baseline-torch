"""Inference/rendering runtime."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..types import RayBatch
from ..config import Config
from ..integrator import render_batch as integrator_render_batch


@torch.no_grad()
def render_batch(
    rays: RayBatch,
    encoder: nn.Module,
    field: nn.Module,
    rgb_head: nn.Module,
    cfg: Config,
    occupancy_grid: nn.Module | None = None
) -> tuple[torch.Tensor, dict]:
    """Render a batch of rays (inference mode).

    Args:
        rays: Ray batch to render
        encoder: Coordinate encoder
        field: Density/feature MLP
        rgb_head: RGB head
        cfg: Configuration
        occupancy_grid: Optional occupancy grid

    Returns:
        rgb: Rendered RGB [N, 3]
        aux: Auxiliary outputs
    """
    encoder.eval()
    field.eval()
    rgb_head.eval()

    rgb, aux = integrator_render_batch(
        rays=rays,
        encoder=encoder,
        field=field,
        rgb_head=rgb_head,
        cfg=cfg.integrator,
        occupancy_grid=occupancy_grid
    )

    return rgb, aux


@torch.no_grad()
def render_image(
    H: int,
    W: int,
    pose: torch.Tensor,
    focal: float,
    near: float,
    far: float,
    encoder: nn.Module,
    field: nn.Module,
    rgb_head: nn.Module,
    cfg: Config,
    device: torch.device,
    chunk_size: int = 4096,
    occupancy_grid: nn.Module | None = None
) -> tuple[torch.Tensor, dict]:
    """Render a full image in chunks.

    Args:
        H, W: Image dimensions
        pose: Camera pose [4, 4]
        focal: Focal length
        near, far: Near/far planes
        encoder, field, rgb_head: Model components
        cfg: Configuration
        device: Computation device
        chunk_size: Number of rays per chunk
        occupancy_grid: Optional occupancy grid

    Returns:
        Rendered image [H, W, 3]
        aux: Aggregated auxiliary outputs
    """
    from ..rays import make_rays_single

    # Generate all rays
    rays = make_rays_single(H, W, pose, focal, near, far, device)

    # Render in chunks
    N = rays.N
    rgb_chunks = []
    aux_chunks = []

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)

        # Extract chunk
        ray_chunk = RayBatch(
            orig_x=rays.orig_x[i:end],
            orig_y=rays.orig_y[i:end],
            orig_z=rays.orig_z[i:end],
            dir_x=rays.dir_x[i:end],
            dir_y=rays.dir_y[i:end],
            dir_z=rays.dir_z[i:end],
            tmin=rays.tmin[i:end],
            tmax=rays.tmax[i:end],
            mask=rays.mask[i:end] if rays.mask is not None else None
        )

        rgb_chunk, aux_chunk = render_batch(ray_chunk, encoder, field, rgb_head, cfg, occupancy_grid)
        rgb_chunks.append(rgb_chunk)
        aux_chunks.append(aux_chunk)

    # Concatenate and reshape
    rgb = torch.cat(rgb_chunks, dim=0)
    rgb = rgb.reshape(H, W, 3)

    # Aggregate auxiliary outputs
    aux_aggregated = {}
    if len(aux_chunks) > 0:
        # Average numeric values across chunks
        for key in aux_chunks[0].keys():
            if key == 'num_steps':
                # Average number of steps
                aux_aggregated[key] = sum(aux[key] for aux in aux_chunks) / len(aux_chunks)
            elif isinstance(aux_chunks[0][key], torch.Tensor):
                # Concatenate tensors
                aux_aggregated[key] = torch.cat([aux[key] for aux in aux_chunks], dim=0)

    return rgb, aux_aggregated
