"""Factory for creating model components from configuration."""
from __future__ import annotations
import torch
import torch.nn as nn
from .config import Config
from .encoder.pe import PositionalEncoder
from .encoder.hashgrid_torch import HashGridEncoder
from .field.mlp import NGP_MLP, TinyMLP
from .field.heads import RGBHead
from .grid.occupancy import OccupancyGrid
from .opt.adam import make_optimizer
from .device import get_dtype


def create_encoder(cfg: Config, device) -> nn.Module:
    """Create encoder from config.

    Args:
        cfg: Configuration
        device: Target device

    Returns:
        Encoder module (PE or HashGrid)
    """
    if cfg.model.hash_levels > 0:
        # Use hash grid encoder
        encoder = HashGridEncoder(
            num_levels=cfg.model.hash_levels,
            base_resolution=cfg.model.hash_res0,
            max_resolution=int(cfg.model.hash_res0 * (cfg.model.hash_per_level_scale ** (cfg.model.hash_levels - 1))),
            features_per_level=cfg.model.hash_features_per_level,
            log2_hashmap_size=19,
            aabb=cfg.dataset.aabb
        )
    else:
        # Use positional encoding only
        encoder = PositionalEncoder(
            num_bands=cfg.model.pe_bands,
            include_input=True
        )

    return encoder.to(device)


def create_field(cfg: Config, encoder: nn.Module, device) -> nn.Module:
    """Create field MLP from config.

    Args:
        cfg: Configuration
        encoder: Encoder to determine input dimension
        device: Target device

    Returns:
        Field MLP module
    """
    input_dim = encoder.output_dim

    field = NGP_MLP(
        input_dim=input_dim,
        hidden_dim=cfg.model.mlp_width,
        num_layers=cfg.model.mlp_depth,
        output_dim=16,  # RGB feature dimension
        activation=cfg.model.activation,
        bias=True,
        density_activation=cfg.model.density_activation
    )

    return field.to(device)


def create_rgb_head(cfg: Config, device) -> nn.Module:
    """Create RGB head from config.

    Args:
        cfg: Configuration
        device: Target device

    Returns:
        RGB head module
    """
    rgb_head = RGBHead(
        rgb_feat_dim=16,
        view_dependent=cfg.model.view_dependent,
        viewdir_dim=3 + cfg.model.pe_viewdir_bands * 3 * 2
    )

    return rgb_head.to(device)


def create_occupancy_grid(cfg: Config, device) -> OccupancyGrid | None:
    """Create occupancy grid from config.

    Args:
        cfg: Configuration
        device: Target device

    Returns:
        OccupancyGrid or None if not using grid strategy
    """
    if cfg.integrator.step_strategy == "grid":
        grid = OccupancyGrid(
            cfg=cfg.grid,
            aabb=cfg.dataset.aabb
        )
        return grid.to(device)
    return None


def create_all(cfg: Config, device) -> tuple[nn.Module, nn.Module, nn.Module, OccupancyGrid | None]:
    """Create all model components.

    Args:
        cfg: Configuration
        device: Target device

    Returns:
        encoder, field, rgb_head, occupancy_grid
    """
    encoder = create_encoder(cfg, device)
    field = create_field(cfg, encoder, device)
    rgb_head = create_rgb_head(cfg, device)
    occupancy_grid = create_occupancy_grid(cfg, device)

    return encoder, field, rgb_head, occupancy_grid


def create_optimizer(encoder: nn.Module, field: nn.Module, rgb_head: nn.Module, cfg: Config):
    """Create optimizer with separate learning rates for encoder and MLP.

    Args:
        encoder, field, rgb_head: Model components
        cfg: Configuration

    Returns:
        Optimizer with parameter groups
    """
    # Separate parameter groups with different learning rates
    param_groups = [
        {'params': encoder.parameters(), 'lr': cfg.train.lr_encoder, 'name': 'encoder'},
        {'params': field.parameters(), 'lr': cfg.train.lr_mlp, 'name': 'field'},
        {'params': rgb_head.parameters(), 'lr': cfg.train.lr_mlp, 'name': 'rgb_head'},
    ]

    optimizer = torch.optim.Adam(
        param_groups,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
        weight_decay=cfg.train.weight_decay
    )

    return optimizer
