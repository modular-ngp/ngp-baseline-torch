"""Ray marching and volume integration."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..types import RayBatch
from ..config import IntegratorConfig
from . import compositor


class RayMarcher(nn.Module):
    """Ray marching with volume rendering.

    Samples points along rays, evaluates encoder+field, and composites RGB.
    """

    def __init__(self, cfg: IntegratorConfig):
        """Initialize ray marcher.

        Args:
            cfg: Integrator configuration
        """
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        rays: RayBatch,
        encoder: nn.Module,
        field: nn.Module,
        rgb_head: nn.Module,
        occupancy_grid: nn.Module | None = None
    ) -> tuple[torch.Tensor, dict]:
        """March along rays and render RGB.

        Args:
            rays: Ray batch in SoA format
            encoder: Coordinate encoder (PE or HashGrid)
            field: MLP field (sigma + rgb features)
            rgb_head: RGB head (features to RGB)
            occupancy_grid: Optional occupancy grid for skipping

        Returns:
            rgb: Rendered RGB [N, 3]
            aux: Auxiliary outputs (depths, weights, etc.)
        """
        N = rays.N
        device = rays.device

        # Sample points along rays
        if self.cfg.step_strategy == "fixed":
            t_samples, dt = self._sample_fixed_steps(rays)
        elif self.cfg.step_strategy == "grid":
            t_samples, dt = self._sample_grid_steps(rays, occupancy_grid)
        else:
            raise ValueError(f"Unknown step strategy: {self.cfg.step_strategy}")

        # t_samples: [N, S]
        S = t_samples.shape[1]

        # Compute 3D positions
        # rays.orig_* [N], rays.dir_* [N], t_samples [N, S]
        # positions [N, S, 3]
        positions = torch.stack([
            rays.orig_x.unsqueeze(-1) + rays.dir_x.unsqueeze(-1) * t_samples,
            rays.orig_y.unsqueeze(-1) + rays.dir_y.unsqueeze(-1) * t_samples,
            rays.orig_z.unsqueeze(-1) + rays.dir_z.unsqueeze(-1) * t_samples,
        ], dim=-1)

        # Flatten for network evaluation [N*S, 3]
        positions_flat = positions.reshape(-1, 3)

        # Encode positions
        encoded = encoder(positions_flat)

        # Evaluate field
        sigma_flat, rgb_feat_flat = field(encoded.feat)

        # Reshape back to [N, S]
        sigma = sigma_flat.reshape(N, S)
        rgb_feat = rgb_feat_flat.reshape(N, S, -1)

        # Convert features to RGB
        rgb_steps = rgb_head(rgb_feat.reshape(-1, rgb_feat.shape[-1]), None)
        rgb_steps = rgb_steps.reshape(N, S, 3)

        # Composite
        if self.cfg.early_stop_T > 0:
            rgb_out, T_out, _ = compositor.compose_with_early_stop(
                sigma, rgb_steps, dt, self.cfg.early_stop_T
            )
        else:
            rgb_out, T_out = compositor.compose(sigma, rgb_steps, dt, 0.0)

        # Auxiliary outputs
        aux = {
            'transmittance': T_out,
            'sigma': sigma,
            'num_steps': S,
        }

        return rgb_out, aux

    def _sample_fixed_steps(self, rays: RayBatch) -> tuple[torch.Tensor, float]:
        """Sample uniformly spaced steps along rays.

        Args:
            rays: Ray batch

        Returns:
            t_samples: Sample positions [N, S]
            dt: Step size (scalar)
        """
        N = rays.N
        S = self.cfg.n_steps_fixed
        device = rays.device

        # Linear spacing from tmin to tmax
        t_samples = torch.linspace(0, 1, S, device=device, dtype=torch.float32)
        t_samples = rays.tmin.unsqueeze(-1) + t_samples.unsqueeze(0) * (rays.tmax - rays.tmin).unsqueeze(-1)

        # Compute step size
        dt = (rays.tmax - rays.tmin).mean().item() / S

        return t_samples, dt

    def _sample_grid_steps(
        self,
        rays: RayBatch,
        occupancy_grid: nn.Module | None
    ) -> tuple[torch.Tensor, float]:
        """Sample steps with occupancy grid guidance.

        For now, fallback to fixed steps. Grid-guided sampling can be added later.

        Args:
            rays: Ray batch
            occupancy_grid: Occupancy grid for skipping

        Returns:
            t_samples: Sample positions [N, S]
            dt: Step size
        """
        # TODO: Implement occupancy-guided sampling
        # For now, use fixed steps as fallback
        return self._sample_fixed_steps(rays)


def render_batch(
    rays: RayBatch,
    encoder: nn.Module,
    field: nn.Module,
    rgb_head: nn.Module,
    cfg: IntegratorConfig,
    occupancy_grid: nn.Module | None = None
) -> tuple[torch.Tensor, dict]:
    """Functional interface for rendering a ray batch.

    Args:
        rays: Ray batch
        encoder: Encoder module
        field: Field MLP
        rgb_head: RGB head
        cfg: Integrator config
        occupancy_grid: Optional occupancy grid

    Returns:
        rgb: Rendered RGB [N, 3]
        aux: Auxiliary data
    """
    marcher = RayMarcher(cfg)
    marcher.to(rays.device)
    return marcher(rays, encoder, field, rgb_head, occupancy_grid)

