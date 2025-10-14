"""Test occupancy grid."""
import pytest
import torch
from ngp_baseline_torch.grid.occupancy import OccupancyGrid
from ngp_baseline_torch.config import GridConfig


class TestOccupancyGrid:
    """Test occupancy grid."""

    def test_creation(self, device):
        """Test grid creation."""
        cfg = GridConfig(resolution=64, levels=2)
        grid = OccupancyGrid(cfg).to(device)

        assert grid.grid.shape == (2, 64, 64, 64)
        assert grid.grid.device.type == device.type

    def test_query_inside_aabb(self, device, set_seed):
        """Test querying points inside AABB."""
        cfg = GridConfig(resolution=32, levels=1)
        grid = OccupancyGrid(cfg, aabb=(-1, -1, -1, 1, 1, 1)).to(device)

        # Points inside AABB
        xyz = torch.rand(100, 3, device=device) * 2 - 1  # [-1, 1]
        mask = grid.query(xyz)

        assert mask.shape == (100,)
        assert mask.dtype == torch.bool

    def test_query_outside_aabb(self, device):
        """Test querying points outside AABB."""
        cfg = GridConfig(resolution=32, levels=1)
        grid = OccupancyGrid(cfg, aabb=(-1, -1, -1, 1, 1, 1)).to(device)

        # Points outside AABB
        xyz = torch.rand(50, 3, device=device) * 4 + 2  # [2, 6]
        mask = grid.query(xyz)

        # Points outside should be masked out
        assert mask.shape == (50,)

    def test_update(self, device, set_seed):
        """Test grid update."""
        cfg = GridConfig(resolution=32, levels=1, ema_tau=0.9)
        grid = OccupancyGrid(cfg).to(device)

        # Create some density values
        xyz = torch.rand(100, 3, device=device) * 2 - 1
        sigma = torch.rand(100, device=device)

        grid.update(xyz, sigma)

        # Check that grid was updated
        assert grid.step_count == 1
