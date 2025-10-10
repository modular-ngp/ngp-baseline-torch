"""Volume rendering compositor with alpha compositing."""
from __future__ import annotations
import torch


def compose(
    sigma_steps: torch.Tensor,
    rgb_steps: torch.Tensor,
    dt: torch.Tensor | float,
    T_threshold: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composite RGB along rays using volumetric rendering equation.

    Args:
        sigma_steps: Density values [N, S] for N rays and S steps
        rgb_steps: RGB values [N, S, 3]
        dt: Step size, scalar or [N, S]
        T_threshold: Early stopping threshold for transmittance

    Returns:
        rgb_out: Composited RGB [N, 3]
        T_out: Final transmittance [N]
    """
    N, S = sigma_steps.shape
    device = sigma_steps.device

    # Convert density to alpha: alpha = 1 - exp(-sigma * dt)
    if isinstance(dt, float):
        alpha = 1.0 - torch.exp(-sigma_steps * dt)  # [N, S]
    else:
        alpha = 1.0 - torch.exp(-sigma_steps * dt)  # [N, S]

    # Clamp alpha to [0, 1]
    alpha = torch.clamp(alpha, 0.0, 1.0)

    # Compute transmittance: T_i = prod(1 - alpha_j) for j < i
    # Use cumulative product for efficiency
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)  # [N, S]

    # Shift transmittance: T[i] should be product up to i-1
    T = torch.cat([torch.ones(N, 1, device=device), T[:, :-1]], dim=-1)  # [N, S]

    # Weight for each step: w_i = T_i * alpha_i
    weights = T * alpha  # [N, S]

    # Composite RGB: rgb = sum(w_i * rgb_i)
    rgb_out = (weights.unsqueeze(-1) * rgb_steps).sum(dim=1)  # [N, 3]

    # Final transmittance
    T_out = T[:, -1] * (1.0 - alpha[:, -1])  # [N]

    # Add white background for NeRF synthetic scenes
    rgb_out = rgb_out + T_out.unsqueeze(-1)

    return rgb_out, T_out


def compose_with_early_stop(
    sigma_steps: torch.Tensor,
    rgb_steps: torch.Tensor,
    dt: torch.Tensor | float,
    T_threshold: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite with early stopping when transmittance is low.

    Returns additional mask indicating where computation can stop.

    Args:
        sigma_steps: Density values [N, S]
        rgb_steps: RGB values [N, S, 3]
        dt: Step size
        T_threshold: Stop when T < threshold

    Returns:
        rgb_out: Composited RGB [N, 3]
        T_out: Final transmittance [N]
        stop_mask: Boolean mask [N, S] indicating stopped samples (True where stopped)
    """
    N, S = sigma_steps.shape
    device = sigma_steps.device

    # Convert to alpha
    if isinstance(dt, float):
        alpha = 1.0 - torch.exp(-sigma_steps * dt)
    else:
        alpha = 1.0 - torch.exp(-sigma_steps * dt)
    alpha = torch.clamp(alpha, 0.0, 1.0)

    # Initialize outputs
    rgb_out = torch.zeros(N, 3, device=device, dtype=rgb_steps.dtype)
    T = torch.ones(N, device=device, dtype=sigma_steps.dtype)
    stop_mask = torch.zeros(N, S, device=device, dtype=torch.bool)

    # Track which rays have stopped
    ray_stopped = torch.zeros(N, device=device, dtype=torch.bool)

    # Sequential accumulation with early stopping
    for s in range(S):
        # Check if we should continue (T > threshold and not already stopped)
        active = (T > T_threshold) & (~ray_stopped)

        if not active.any():
            # All rays stopped, mark remaining samples as stopped
            stop_mask[:, s:] = True
            break

        # Compute weights for active rays
        w = T * alpha[:, s]  # [N]

        # Accumulate RGB (only for active rays)
        rgb_out = rgb_out + (w.unsqueeze(-1) * rgb_steps[:, s, :]) * active.unsqueeze(-1).float()

        # Update transmittance
        T = T * (1.0 - alpha[:, s])

        # Update stopped status: rays that just went below threshold
        newly_stopped = (T <= T_threshold) & (~ray_stopped)
        ray_stopped = ray_stopped | newly_stopped

        # Mark this sample as stopped for rays that have stopped
        stop_mask[:, s] = ray_stopped

    # Add white background
    rgb_out = rgb_out + T.unsqueeze(-1)

    return rgb_out, T, stop_mask


def composite_test(sigma: torch.Tensor, rgb: torch.Tensor, dt: float) -> torch.Tensor:
    """Simple test compositor for validation.

    Args:
        sigma: [N, S]
        rgb: [N, S, 3]
        dt: scalar step size

    Returns:
        Composited RGB [N, 3]
    """
    rgb_out, _ = compose(sigma, rgb, dt, T_threshold=0.0)
    return rgb_out
