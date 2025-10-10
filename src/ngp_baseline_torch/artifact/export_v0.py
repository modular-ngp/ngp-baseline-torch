"""Artifact export for deployment."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from ..types import ArtifactMeta
from ..config import Config


def export(
    encoder: nn.Module,
    field: nn.Module,
    rgb_head: nn.Module,
    cfg: Config,
    out_dir: str | Path,
    occupancy_grid: nn.Module | None = None
) -> None:
    """Export trained model to artifact directory.

    Args:
        encoder: Trained encoder
        field: Trained field MLP
        rgb_head: Trained RGB head
        cfg: Configuration
        out_dir: Output directory path
        occupancy_grid: Optional occupancy grid
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    meta = ArtifactMeta(
        version="v0.1.0",
        endianness=sys.byteorder,
        alignment_bytes=256,
        arch_req="cuda" if torch.cuda.is_available() else "cpu",
        precision={
            'weights': cfg.precision.param_dtype,
            'accum': cfg.precision.accum_dtype,
        },
        seed=cfg.train.seed
    )

    # Save metadata
    with open(out_dir / "meta.json", 'w') as f:
        json.dump(meta.to_dict(), f, indent=2)

    # Save model topology/config
    topo = {
        'encoder_type': encoder.__class__.__name__,
        'field_type': field.__class__.__name__,
        'model_config': {
            'mlp_width': cfg.model.mlp_width,
            'mlp_depth': cfg.model.mlp_depth,
            'activation': cfg.model.activation,
        },
        'integrator_config': {
            'n_steps': cfg.integrator.n_steps_fixed,
            'strategy': cfg.integrator.step_strategy,
        }
    }

    with open(out_dir / "topo.json", 'w') as f:
        json.dump(topo, f, indent=2)

    # Save model parameters
    torch.save({
        'encoder': encoder.state_dict(),
        'field': field.state_dict(),
        'rgb_head': rgb_head.state_dict(),
    }, out_dir / "params.pth")

    # Save hash grid separately if applicable
    if hasattr(encoder, 'hash_tables'):
        hash_data = {
            'num_levels': encoder.num_levels,
            'resolutions': encoder.resolutions.cpu(),
            'tables': [t.data.cpu() for t in encoder.hash_tables],
        }
        torch.save(hash_data, out_dir / "hashgrid.pth")

    # Save occupancy grid if present
    if occupancy_grid is not None:
        occgrid_data = {
            'grid': occupancy_grid.grid.cpu(),
            'grid_ema': occupancy_grid.grid_ema.cpu(),
            'resolution': occupancy_grid.resolution,
        }
        torch.save(occgrid_data, out_dir / "occgrid.pth")

    print(f"Exported artifact to {out_dir}")


def load(
    artifact_dir: str | Path,
    encoder: nn.Module,
    field: nn.Module,
    rgb_head: nn.Module,
    device: torch.device,
    occupancy_grid: nn.Module | None = None
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]:
    """Load model from artifact directory.

    Args:
        artifact_dir: Directory containing exported artifact
        encoder, field, rgb_head: Model instances to load into
        device: Target device
        occupancy_grid: Optional occupancy grid to load

    Returns:
        Loaded model components
    """
    artifact_dir = Path(artifact_dir)

    # Load parameters
    checkpoint = torch.load(artifact_dir / "params.pth", map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    field.load_state_dict(checkpoint['field'])
    rgb_head.load_state_dict(checkpoint['rgb_head'])

    # Load occupancy grid if present
    if occupancy_grid is not None and (artifact_dir / "occgrid.pth").exists():
        occgrid_data = torch.load(artifact_dir / "occgrid.pth", map_location=device)
        occupancy_grid.grid.copy_(occgrid_data['grid'])
        occupancy_grid.grid_ema.copy_(occgrid_data['grid_ema'])

    return encoder, field, rgb_head, occupancy_grid

