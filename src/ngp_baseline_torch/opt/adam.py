"""Optimizer configuration and wrappers."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config import TrainConfig


def wrap(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Create Adam optimizer with configuration.
    
    Args:
        params: Model parameters (iterable or param groups)
        cfg: Training configuration
        
    Returns:
        Configured Adam optimizer
    """
    optimizer = torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay
    )
    return optimizer


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Create optimizer for a model.
    
    Args:
        model: Neural network model
        cfg: Training configuration
        
    Returns:
        Configured optimizer
    """
    return wrap(model.parameters(), cfg)

