"""Occupancy grid for empty space skipping."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..types import GridState
from ..config import GridConfig


class OccupancyGrid(nn.Module):
    """Occupancy grid with EMA updates for skipping empty space.

    Maintains a coarse voxel grid that tracks which regions contain geometry.
    """

    def __init__(
        self,
        cfg: GridConfig,
        aabb: tuple[float, float, float, float, float, float] = (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5)
    ):
        """Initialize occupancy grid.

        Args:
            cfg: Grid configuration
            aabb: Axis-aligned bounding box
        """
        super().__init__()
        self.cfg = cfg
        self.resolution = cfg.resolution

        # AABB
        self.register_buffer('aabb_min', torch.tensor([aabb[0], aabb[1], aabb[2]], dtype=torch.float32))
        self.register_buffer('aabb_max', torch.tensor([aabb[3], aabb[4], aabb[5]], dtype=torch.float32))

        # Initialize grid as all occupied (conservative)
        # Use float for EMA, will threshold to binary for queries
        grid_size = (cfg.levels, cfg.resolution, cfg.resolution, cfg.resolution)
        self.register_buffer('grid', torch.ones(grid_size, dtype=torch.float32))

        # EMA state
        self.register_buffer('grid_ema', torch.ones(grid_size, dtype=torch.float32))

        self.step_count = 0

    def query(self, xyz: torch.Tensor) -> torch.Tensor:
        """Query occupancy at 3D positions.

        Args:
            xyz: World coordinates [B, 3]

        Returns:
            mask: Boolean mask [B] indicating occupied voxels
        """
        B = xyz.shape[0]
        device = xyz.device

        # Normalize to [0, 1]
        xyz_normalized = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min)
        xyz_normalized = torch.clamp(xyz_normalized, 0.0, 1.0)

        # Scale to grid coordinates
        grid_coords = xyz_normalized * self.resolution
        grid_coords = torch.clamp(grid_coords.long(), 0, self.resolution - 1)

        # Query grid (use first level for now)
        level = 0
        occupancy = self.grid[level, grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]

        # Threshold
        mask = occupancy > self.cfg.threshold

        return mask

    def update(self, positions: torch.Tensor, densities: torch.Tensor) -> None:
        """Update grid with new density samples.

        Args:
            positions: Sample positions [N, 3]
            densities: Density values [N]
        """
        if self.step_count < self.cfg.warmup_steps:
            self.step_count += 1
            return

        # Normalize positions
        xyz_normalized = (positions - self.aabb_min) / (self.aabb_max - self.aabb_min)
        xyz_normalized = torch.clamp(xyz_normalized, 0.0, 1.0)

        # Get grid coordinates
        grid_coords = xyz_normalized * self.resolution
        grid_coords = torch.clamp(grid_coords.long(), 0, self.resolution - 1)

        # Accumulate density into grid
        level = 0
        grid_update = torch.zeros_like(self.grid[level])
        grid_count = torch.zeros_like(self.grid[level])

        # Scatter densities into grid
        for i in range(positions.shape[0]):
            x, y, z = grid_coords[i]
            grid_update[x, y, z] += densities[i].item()
            grid_count[x, y, z] += 1

        # Average
        valid = grid_count > 0
        grid_update[valid] /= grid_count[valid]

        # EMA update
        self.grid_ema[level] = self.cfg.ema_tau * self.grid_ema[level] + (1 - self.cfg.ema_tau) * grid_update

        # Threshold for binary grid
        self.grid[level] = (self.grid_ema[level] > self.cfg.threshold).float()

        self.step_count += 1

    def get_state(self) -> GridState:
        """Get current grid state.

        Returns:
            GridState with current occupancy data
        """
        # Convert to packed bitset (simplified: use uint8)
        bitset = (self.grid > self.cfg.threshold).to(torch.uint8)
        bitset = bitset.reshape(self.cfg.levels, -1)

        return GridState(
            bitset=bitset,
            ema_tau=self.cfg.ema_tau,
            threshold=self.cfg.threshold,
            levels=self.cfg.levels,
            resolution=self.resolution
        )


def query(xyz: torch.Tensor, grid: OccupancyGrid) -> torch.Tensor:
    """Functional interface for occupancy query.

    Args:
        xyz: Positions [B, 3]
        grid: OccupancyGrid instance

    Returns:
        Occupancy mask [B]
    """
    return grid.query(xyz)


def update(stats: dict, grid: OccupancyGrid) -> GridState:
    """Functional interface for grid update.

    Args:
        stats: Dictionary with 'positions' and 'densities'
        grid: OccupancyGrid instance

    Returns:
        Updated GridState
    """
    if 'positions' in stats and 'densities' in stats:
        grid.update(stats['positions'], stats['densities'])
    return grid.get_state()

