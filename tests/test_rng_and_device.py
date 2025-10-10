"""Test RNG and device setup."""
import pytest
import torch
import numpy as np
from ngp_baseline_torch.rng import seed_everything, make_generator
from ngp_baseline_torch.device import get_device, setup_device
from ngp_baseline_torch.config import PrecisionConfig


@pytest.mark.quick
def test_seed_determinism(device, seed):
    """Test that seeding produces identical results."""
    # First run
    seed_everything(seed, deterministic=True)
    x1 = torch.randn(100, device=device)
    n1 = np.random.randn(100)

    # Second run with same seed
    seed_everything(seed, deterministic=True)
    x2 = torch.randn(100, device=device)
    n2 = np.random.randn(100)

    # Should be identical
    assert torch.allclose(x1, x2)
    assert np.allclose(n1, n2)


@pytest.mark.quick
def test_generator_seeding(device, seed):
    """Test generator seeding."""
    gen1 = make_generator(seed, device)
    gen2 = make_generator(seed, device)

    x1 = torch.randn(100, device=device, generator=gen1)
    x2 = torch.randn(100, device=device, generator=gen2)

    assert torch.allclose(x1, x2)


@pytest.mark.quick
def test_device_selection():
    """Test device selection logic."""
    device = get_device(prefer_cuda=True)
    assert device.type in ('cuda', 'cpu')

    cpu_device = get_device(prefer_cuda=False)
    assert cpu_device.type == 'cpu'


@pytest.mark.quick
def test_dtype_setup():
    """Test dtype configuration."""
    cfg = PrecisionConfig(param_dtype="float32", compute_dtype="float32")
    device, param_dtype, compute_dtype = setup_device(cfg, prefer_cuda=False)

    assert param_dtype == torch.float32
    assert compute_dtype == torch.float32


@pytest.mark.quick
def test_amp_policy():
    """Test AMP configuration toggles."""
    cfg_amp = PrecisionConfig(use_amp=True, compute_dtype="float16")
    cfg_no_amp = PrecisionConfig(use_amp=False, compute_dtype="float32")

    assert cfg_amp.use_amp is True
    assert cfg_no_amp.use_amp is False

