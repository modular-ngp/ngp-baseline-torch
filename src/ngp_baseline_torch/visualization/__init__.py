"""Visualization module for real-time NGP debugging via shared memory."""

from .debug_server import NGPDebugServer
from .debug_extractor import (
    sample_density_grid,
    sample_rays_along_camera,
    extract_occupancy_grid_data,
    compute_camera_matrix,
    extract_training_metrics,
    filter_by_density_threshold,
)

__all__ = [
    'NGPDebugServer',
    'sample_density_grid',
    'sample_rays_along_camera',
    'extract_occupancy_grid_data',
    'compute_camera_matrix',
    'extract_training_metrics',
    'filter_by_density_threshold',
]

