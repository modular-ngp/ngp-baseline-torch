"""Test loss functions."""
import pytest
import torch
from ngp_baseline_torch.loss.rgb import l2, l1, huber


class TestLossFunctions:
    """Test RGB loss functions."""

    def test_l2_loss(self, device, set_seed):
        """Test L2 (MSE) loss."""
        pred = torch.rand(100, 3, device=device)
        target = torch.rand(100, 3, device=device)

        loss = l2(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_l2_perfect_match(self, device):
        """Test L2 loss with perfect match."""
        pred = torch.rand(50, 3, device=device)
        target = pred.clone()

        loss = l2(pred, target)

        assert loss < 1e-6

    def test_l1_loss(self, device, set_seed):
        """Test L1 loss."""
        pred = torch.rand(100, 3, device=device)
        target = torch.rand(100, 3, device=device)

        loss = l1(pred, target)

        assert loss.ndim == 0
        assert loss >= 0

    def test_huber_loss(self, device, set_seed):
        """Test Huber loss."""
        pred = torch.rand(100, 3, device=device)
        target = torch.rand(100, 3, device=device)

        loss = huber(pred, target, delta=0.1)

        assert loss.ndim == 0
        assert loss >= 0

    def test_reduction_modes(self, device, set_seed):
        """Test different reduction modes."""
        pred = torch.rand(50, 3, device=device)
        target = torch.rand(50, 3, device=device)

        loss_mean = l2(pred, target, reduction='mean')
        loss_sum = l2(pred, target, reduction='sum')
        loss_none = l2(pred, target, reduction='none')

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.shape == (50, 3)

    def test_gradient_flow(self, device, set_seed):
        """Test gradient backpropagation."""
        pred = torch.rand(50, 3, device=device, requires_grad=True)
        target = torch.rand(50, 3, device=device)

        loss = l2(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

