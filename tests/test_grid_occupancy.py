"""Test occupancy grid functionality."""
import pytest
import torch
from ngp_baseline_torch.grid import OccupancyGrid
from ngp_baseline_torch.config import GridConfig


@pytest.mark.quick
def test_occupancy_grid_query(device):
    """Test occupancy grid query."""
    cfg = GridConfig(resolution=64, ema_tau=0.95, threshold=0.01)
    grid = OccupancyGrid(cfg, aabb=(-1.5, -1.5, -1.5, 1.5, 1.5, 1.5))
    grid.to(device)

    # Query some positions
    xyz = torch.rand(100, 3, device=device) * 2 - 1  # [-1, 1]
    mask = grid.query(xyz)

    assert mask.shape == (100,)
    assert mask.dtype == torch.bool


@pytest.mark.quick
def test_occupancy_grid_update(device):
    """Test occupancy grid update mechanism."""
    cfg = GridConfig(resolution=32, ema_tau=0.95, threshold=0.01, warmup_steps=0)
    grid = OccupancyGrid(cfg, aabb=(-1.5, -1.5, -1.5, 1.5, 1.5, 1.5))
    grid.to(device)

    # Create high density samples
    positions = torch.rand(1000, 3, device=device) * 2 - 1
    densities = torch.ones(1000, device=device) * 5.0

    # Update grid
    grid.update(positions, densities)

    # Grid should have been updated
    assert grid.step_count > 0


@pytest.mark.quick
def test_occupancy_grid_state(device):
    """Test getting grid state."""
    cfg = GridConfig(resolution=32)
    grid = OccupancyGrid(cfg)
    grid.to(device)

    state = grid.get_state()

    assert state.levels == cfg.levels
    assert state.resolution == cfg.resolution
    assert state.ema_tau == cfg.ema_tau
    assert state.threshold == cfg.threshold

