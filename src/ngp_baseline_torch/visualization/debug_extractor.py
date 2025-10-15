"""
Debug data extraction utilities for NGP training visualization.

Provides functions to extract and sample debug data from NGP models during training.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any


def sample_density_grid(
    encoder: torch.nn.Module,
    field: torch.nn.Module,
    rgb_head: torch.nn.Module,
    bbox_min: np.ndarray = None,
    bbox_max: np.ndarray = None,
    num_samples: int = 100_000,
    device: torch.device = None,
    batch_size: int = 8192,
    viewdir: torch.Tensor = None,
    use_view_dependent: bool = False,  # 默认禁用视角依赖
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample the learned density and color field at random 3D positions.

    Args:
        encoder: Position encoder module
        field: Neural field module
        rgb_head: RGB output head module
        bbox_min: Minimum bounding box coordinates [3,] (default: [-1, -1, -1])
        bbox_max: Maximum bounding box coordinates [3,] (default: [1, 1, 1])
        num_samples: Number of 3D points to sample
        device: Computation device
        batch_size: Batch size for inference
        viewdir: Optional view direction for view-dependent rendering [3,] or None
        use_view_dependent: Whether to use view-dependent rendering (may be slower/complex)

    Returns:
        positions: [N, 3] sampled positions
        colors: [N, 3] RGB colors
        densities: [N, 1] density values
    """
    if bbox_min is None:
        bbox_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    if bbox_max is None:
        bbox_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sample positions uniformly in bounding box
    positions = torch.rand(num_samples, 3, device=device)
    bbox_min_t = torch.tensor(bbox_min, device=device)
    bbox_max_t = torch.tensor(bbox_max, device=device)
    positions = positions * (bbox_max_t - bbox_min_t) + bbox_min_t

    # Query network in batches
    all_colors = []
    all_densities = []

    encoder.eval()
    field.eval()
    rgb_head.eval()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_pos = positions[i:i + batch_size]
            batch_size_actual = batch_pos.shape[0]

            # Forward pass
            try:
                # Encode positions
                encoded = encoder(batch_pos)

                # Extract tensor from EncodedFeat if needed
                if hasattr(encoded, 'feat'):
                    encoded_tensor = encoded.feat
                else:
                    encoded_tensor = encoded

                # Get density and features from field
                field_output = field(encoded_tensor)

                # Handle different field output formats
                if isinstance(field_output, tuple):
                    density = field_output[0]
                    rgb_features = field_output[1] if len(field_output) > 1 else None
                else:
                    density = field_output
                    rgb_features = None

                # Get RGB - handle view-dependent vs view-independent
                if use_view_dependent and hasattr(rgb_head, 'view_dependent') and rgb_head.view_dependent:
                    # View-dependent path (complex, may have dimension issues)
                    # For now, skip view-dependent rendering in visualization
                    print(f"[DebugExtractor] Warning: View-dependent rendering skipped for visualization")
                    rgb = torch.ones(batch_size_actual, 3, device=device) * 0.5  # Gray placeholder
                else:
                    # View-independent path or force non-view-dependent rendering
                    # Use rgb_features if available, otherwise generate dummy colors based on density
                    if rgb_features is not None:
                        try:
                            # Try to call RGB head without viewdir
                            rgb = rgb_head(rgb_features, None)
                        except:
                            # If that fails, generate colors from density
                            density_normalized = torch.clamp(density.squeeze(-1) if density.dim() == 2 else density, 0, 1)
                            rgb = density_normalized.unsqueeze(-1).expand(-1, 3)
                    else:
                        # Generate colors from density
                        density_normalized = torch.clamp(density.squeeze(-1) if density.dim() == 2 else density, 0, 1)
                        rgb = density_normalized.unsqueeze(-1).expand(-1, 3)

                # Ensure proper shapes
                if density.dim() == 1:
                    density = density.unsqueeze(-1)

                all_colors.append(rgb)
                all_densities.append(density)

            except Exception as e:
                print(f"[DebugExtractor] Error during sampling: {e}")
                # Fallback to dummy data
                all_colors.append(torch.zeros(batch_size_actual, 3, device=device))
                all_densities.append(torch.zeros(batch_size_actual, 1, device=device))

    colors = torch.cat(all_colors, dim=0)
    densities = torch.cat(all_densities, dim=0)

    # Clamp colors to [0, 1]
    colors = torch.clamp(colors, 0, 1)

    return positions, colors, densities


def sample_rays_along_camera(
    rays: Any,
    num_rays: int = 1024,
    num_samples_per_ray: int = 64,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points along camera rays for visualization.

    Args:
        rays: Ray batch object
        num_rays: Number of rays to sample
        num_samples_per_ray: Number of samples along each ray
        device: Computation device

    Returns:
        ray_origins: [num_rays, 3] ray origin positions
        ray_directions: [num_rays, 3] ray directions
        sample_positions: [num_rays * num_samples_per_ray, 3] sample positions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sample random rays
    indices = torch.randperm(rays.N, device=device)[:num_rays]

    # Get ray data
    origins = torch.stack([
        rays.orig_x[indices],
        rays.orig_y[indices],
        rays.orig_z[indices]
    ], dim=-1)

    directions = torch.stack([
        rays.dir_x[indices],
        rays.dir_y[indices],
        rays.dir_z[indices]
    ], dim=-1)

    tmin = rays.tmin[indices]
    tmax = rays.tmax[indices]

    # Sample along rays
    t_vals = torch.linspace(0, 1, num_samples_per_ray, device=device)
    t_vals = tmin.unsqueeze(-1) + (tmax - tmin).unsqueeze(-1) * t_vals.unsqueeze(0)

    # Compute sample positions: o + t * d
    sample_positions = origins.unsqueeze(1) + directions.unsqueeze(1) * t_vals.unsqueeze(-1)
    sample_positions = sample_positions.reshape(-1, 3)

    return origins, directions, sample_positions


def extract_occupancy_grid_data(
    occupancy_grid: Optional[torch.nn.Module],
    device: torch.device = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract occupancy grid data for visualization.

    Args:
        occupancy_grid: Occupancy grid module
        device: Computation device

    Returns:
        Dictionary with grid data or None if not available
    """
    if occupancy_grid is None:
        return None

    try:
        # Try to access grid data (depends on implementation)
        if hasattr(occupancy_grid, 'grid'):
            grid_data = occupancy_grid.grid
            if isinstance(grid_data, torch.Tensor):
                return {
                    'grid': grid_data.cpu(),
                    'shape': grid_data.shape,
                    'resolution': grid_data.shape[0] if grid_data.dim() >= 1 else 0,
                }

        return None

    except Exception as e:
        print(f"[DebugExtractor] Could not extract occupancy grid: {e}")
        return None


def compute_camera_matrix(pose: torch.Tensor, focal: float, width: int, height: int) -> np.ndarray:
    """
    Compute camera matrix from pose and intrinsics.

    Args:
        pose: [4, 4] camera-to-world transformation matrix
        focal: Focal length
        width: Image width
        height: Image height

    Returns:
        [4, 4] camera matrix (numpy array)
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()

    return pose.astype(np.float32)


def extract_training_metrics(
    metrics: Dict[str, float],
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Extract training metrics for visualization.

    Args:
        metrics: Dictionary of training metrics
        optimizer: Optimizer to extract learning rate from

    Returns:
        Dictionary with extracted metrics
    """
    result = {}

    # Extract loss and PSNR
    if 'loss' in metrics:
        result['loss'] = float(metrics['loss'])

    if 'psnr' in metrics:
        result['psnr'] = float(metrics['psnr'])

    # Extract learning rate
    if optimizer is not None and len(optimizer.param_groups) > 0:
        result['learning_rate'] = float(optimizer.param_groups[0]['lr'])

    return result


def filter_by_density_threshold(
    positions,
    colors,
    densities,
    threshold: float = 0.01,
    max_points: Optional[int] = None,
):
    """
    Filter points by density threshold and optionally limit count.

    Args:
        positions: [N, 3] positions (torch.Tensor or numpy array)
        colors: [N, 3] colors (torch.Tensor or numpy array)
        densities: [N, 1] or [N,] densities (torch.Tensor or numpy array)
        threshold: Minimum density threshold
        max_points: Maximum number of points to keep (random sample if exceeded)

    Returns:
        Filtered positions, colors, densities (same type as input)
    """
    # Determine if input is torch or numpy
    is_torch = False
    try:
        import torch
        if isinstance(positions, torch.Tensor):
            is_torch = True
    except ImportError:
        pass

    if is_torch:
        # PyTorch path
        # Ensure densities is 1D
        if densities.dim() == 2:
            densities = densities.squeeze(-1)

        # Filter by threshold
        mask = densities >= threshold
        positions = positions[mask]
        colors = colors[mask]
        densities = densities[mask]

        # Limit count if needed
        if max_points is not None and positions.shape[0] > max_points:
            indices = torch.randperm(positions.shape[0])[:max_points]
            positions = positions[indices]
            colors = colors[indices]
            densities = densities[indices]

        # Reshape densities to [N, 1]
        if densities.dim() == 1:
            densities = densities.unsqueeze(-1)
    else:
        # NumPy path
        import numpy as np

        # Convert to numpy if needed
        positions = np.asarray(positions)
        colors = np.asarray(colors)
        densities = np.asarray(densities)

        # Ensure densities is 1D
        if densities.ndim == 2:
            densities = densities.squeeze(-1)

        # Filter by threshold
        mask = densities >= threshold
        positions = positions[mask]
        colors = colors[mask]
        densities = densities[mask]

        # Limit count if needed
        if max_points is not None and positions.shape[0] > max_points:
            indices = np.random.permutation(positions.shape[0])[:max_points]
            positions = positions[indices]
            colors = colors[indices]
            densities = densities[indices]

        # Reshape densities to [N, 1]
        if densities.ndim == 1:
            densities = densities[:, np.newaxis]

    return positions, colors, densities
