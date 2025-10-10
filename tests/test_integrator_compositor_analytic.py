"""Test volume rendering compositor."""
import pytest
import torch
import numpy as np
from ngp_baseline_torch.integrator.compositor import compose, compose_with_early_stop


@pytest.mark.quick
def test_compositor_analytic_constant(device):
    """Test compositor against analytic solution for constant sigma/rgb."""
    N = 100  # rays
    S = 64   # steps
    dt = 0.1

    # Constant sigma and RGB
    sigma = torch.ones(N, S, device=device) * 2.0
    rgb = torch.ones(N, S, 3, device=device) * 0.5

    # Analytical solution for constant sigma:
    # alpha = 1 - exp(-sigma * dt)
    # T_k = exp(-sigma * dt * k)
    # rgb_accumulated = rgb * sum(T_k * alpha) for k in range(S)

    alpha = 1.0 - np.exp(-2.0 * dt)
    expected_weight_sum = sum(np.exp(-2.0 * dt * k) * alpha for k in range(S))
    expected_rgb = 0.5 * expected_weight_sum

    # Final transmittance
    T_final = np.exp(-2.0 * dt * S)
    expected_rgb_with_bg = expected_rgb + T_final  # white background

    rgb_out, T_out = compose(sigma, rgb, dt, T_threshold=0.0)

    # Check RGB is close to analytical - convert to same dtype
    expected_tensor = torch.tensor(expected_rgb_with_bg, device=device, dtype=rgb_out.dtype)
    assert torch.allclose(rgb_out[:, 0], expected_tensor, atol=1e-4)

    # Check transmittance
    expected_T = torch.tensor(T_final, device=device, dtype=T_out.dtype)
    assert torch.allclose(T_out, expected_T, atol=1e-4)


@pytest.mark.quick
def test_compositor_empty_space(device):
    """Test compositor with zero density (empty space)."""
    N = 50
    S = 32
    dt = 0.1

    # Zero density everywhere
    sigma = torch.zeros(N, S, device=device)
    rgb = torch.ones(N, S, 3, device=device) * 0.5

    rgb_out, T_out = compose(sigma, rgb, dt)

    # With zero density, transmittance should be 1.0
    assert torch.allclose(T_out, torch.ones(N, device=device), atol=1e-5)

    # Output should be white background (1.0)
    assert torch.allclose(rgb_out, torch.ones(N, 3, device=device), atol=1e-5)


@pytest.mark.quick
def test_compositor_early_stop(device):
    """Test early stopping when transmittance drops below threshold."""
    N = 100
    S = 128
    dt = 0.05  # Smaller step to accumulate more
    T_threshold = 1e-2  # Higher threshold for easier early stopping

    # Very high density at start to ensure early stopping
    sigma = torch.ones(N, S, device=device) * 20.0
    rgb = torch.ones(N, S, 3, device=device) * 0.5

    rgb_out, T_out, stop_mask = compose_with_early_stop(sigma, rgb, dt, T_threshold)

    assert rgb_out.shape == (N, 3)
    assert T_out.shape == (N,)
    assert stop_mask.shape == (N, S)

    # With very high density, most rays should stop early (T drops below threshold)
    # Check that at least some steps were marked as stopped
    stopped_steps = stop_mask.sum().item()
    assert stopped_steps > 0, f"Expected some early stopping, got {stopped_steps} stopped steps"


@pytest.mark.quick
def test_compositor_output_range(device):
    """Test that compositor produces valid RGB range."""
    N = 100
    S = 64

    sigma = torch.rand(N, S, device=device) * 10.0
    rgb = torch.rand(N, S, 3, device=device)

    rgb_out, T_out = compose(sigma, rgb, 0.1)

    # RGB should be in valid range [0, ~1] (can be slightly > 1 due to background)
    assert torch.all(rgb_out >= 0)
    assert torch.all(rgb_out <= 1.1)  # Allow slight overflow from background

    # Transmittance in [0, 1]
    assert torch.all(T_out >= 0)
    assert torch.all(T_out <= 1)
