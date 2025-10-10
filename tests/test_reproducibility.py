"""Test reproducibility with fixed seeds."""
import pytest
import torch
from ngp_baseline_torch.rng import seed_everything
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all


@pytest.mark.quick
def test_reproducibility_same_seed(device, seed):
    """Test that same seed produces identical results."""
    cfg = Config()
    cfg.model.hash_levels = 4

    # Run 1
    seed_everything(seed, deterministic=True)
    encoder1, field1, rgb_head1, _ = create_all(cfg, device)

    xyz = torch.randn(100, 3, device=device)

    encoded1 = encoder1(xyz)
    sigma1, rgb_feat1 = field1(encoded1.feat)

    # Run 2 with same seed
    seed_everything(seed, deterministic=True)
    encoder2, field2, rgb_head2, _ = create_all(cfg, device)

    encoded2 = encoder2(xyz)
    sigma2, rgb_feat2 = field2(encoded2.feat)

    # Should be identical
    assert torch.allclose(sigma1, sigma2, atol=1e-7)
    assert torch.allclose(rgb_feat1, rgb_feat2, atol=1e-7)


@pytest.mark.quick
def test_reproducibility_different_seed(device):
    """Test that different seeds produce different results."""
    cfg = Config()
    cfg.model.hash_levels = 4

    # Run 1
    seed_everything(1337, deterministic=True)
    encoder1, field1, _, _ = create_all(cfg, device)

    # Run 2 with different seed
    seed_everything(42, deterministic=True)
    encoder2, field2, _, _ = create_all(cfg, device)

    # Use different random input for each run to ensure different results
    seed_everything(1337, deterministic=True)
    xyz1 = torch.randn(100, 3, device=device)

    seed_everything(42, deterministic=True)
    xyz2 = torch.randn(100, 3, device=device)

    # Make sure inputs are different
    assert not torch.allclose(xyz1, xyz2, atol=1e-3)

    encoded1 = encoder1(xyz1)
    sigma1, _ = field1(encoded1.feat)

    encoded2 = encoder2(xyz2)
    sigma2, _ = field2(encoded2.feat)

    # With different seeds and inputs, outputs should be different
    assert not torch.allclose(sigma1, sigma2, atol=1e-2)
