"""Test RNG utilities."""
import pytest
import torch
import numpy as np
from ngp_baseline_torch.rng import seed_everything, make_generator


class TestRNG:
    """Test random number generator utilities."""

    def test_seed_everything(self):
        """Test seeding all RNGs."""
        seed = 42
        seed_everything(seed, deterministic=False)

        # Generate some random numbers
        torch_val1 = torch.rand(1).item()
        np_val1 = np.random.rand()

        # Reset seed
        seed_everything(seed, deterministic=False)

        # Generate again - should match
        torch_val2 = torch.rand(1).item()
        np_val2 = np.random.rand()

        assert torch_val1 == torch_val2
        assert np_val1 == np_val2

    def test_different_seeds(self):
        """Test different seeds produce different results."""
        seed_everything(42, deterministic=False)
        val1 = torch.rand(1).item()

        seed_everything(123, deterministic=False)
        val2 = torch.rand(1).item()

        assert val1 != val2

    def test_make_generator(self, device):
        """Test generator creation."""
        gen = make_generator(42, device)

        assert isinstance(gen, torch.Generator)
        assert gen.device == device
