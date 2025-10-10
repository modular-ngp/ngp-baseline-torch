"""Training runtime with AMP support."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..types import RayBatch
from ..config import Config
from ..integrator import render_batch as integrator_render_batch
from ..loss import l2 as rgb_loss_l2


def train_step(
    rays: RayBatch,
    target_rgb: torch.Tensor,
    encoder: nn.Module,
    field: nn.Module,
    rgb_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    scaler: torch.cuda.amp.GradScaler | None = None,
    occupancy_grid: nn.Module | None = None
) -> dict[str, float]:
    """Execute a single training step.

    Args:
        rays: Ray batch
        target_rgb: Ground truth RGB [N, 3]
        encoder, field, rgb_head: Model components
        optimizer: Optimizer
        cfg: Configuration
        scaler: AMP gradient scaler (if using AMP)
        occupancy_grid: Optional occupancy grid

    Returns:
        Metrics dictionary with losses and statistics
    """
    encoder.train()
    field.train()
    rgb_head.train()

    optimizer.zero_grad()

    # Forward pass with optional AMP
    if cfg.precision.use_amp and scaler is not None:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred_rgb, aux = integrator_render_batch(
                rays=rays,
                encoder=encoder,
                field=field,
                rgb_head=rgb_head,
                cfg=cfg.integrator,
                occupancy_grid=occupancy_grid
            )
            loss = rgb_loss_l2(pred_rgb, target_rgb)

        # Backward with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred_rgb, aux = integrator_render_batch(
            rays=rays,
            encoder=encoder,
            field=field,
            rgb_head=rgb_head,
            cfg=cfg.integrator,
            occupancy_grid=occupancy_grid
        )
        loss = rgb_loss_l2(pred_rgb, target_rgb)

        # Standard backward
        loss.backward()
        optimizer.step()

    # Compute metrics
    with torch.no_grad():
        mse = loss.item()
        psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-8)).item()

    metrics = {
        'loss': mse,
        'psnr': psnr,
        'num_steps': aux.get('num_steps', 0),
    }

    return metrics


class Trainer:
    """Training manager with state."""

    def __init__(
        self,
        encoder: nn.Module,
        field: nn.Module,
        rgb_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: Config,
        device: torch.device,
        occupancy_grid: nn.Module | None = None
    ):
        """Initialize trainer.

        Args:
            encoder, field, rgb_head: Model components
            optimizer: Optimizer
            cfg: Configuration
            device: Computation device
            occupancy_grid: Optional occupancy grid
        """
        self.encoder = encoder
        self.field = field
        self.rgb_head = rgb_head
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.occupancy_grid = occupancy_grid

        # AMP scaler
        self.scaler = None
        if cfg.precision.use_amp and device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()

        self.step_count = 0

    def step(self, rays: RayBatch, target_rgb: torch.Tensor) -> dict[str, float]:
        """Execute training step.

        Args:
            rays: Ray batch
            target_rgb: Target RGB [N, 3]

        Returns:
            Metrics dictionary
        """
        metrics = train_step(
            rays=rays,
            target_rgb=target_rgb,
            encoder=self.encoder,
            field=self.field,
            rgb_head=self.rgb_head,
            optimizer=self.optimizer,
            cfg=self.cfg,
            scaler=self.scaler,
            occupancy_grid=self.occupancy_grid
        )

        self.step_count += 1
        return metrics

