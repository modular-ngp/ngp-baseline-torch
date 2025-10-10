"""Test RGB loss functions."""
import pytest
import torch
from ngp_baseline_torch.loss import l2, huber, l1


@pytest.mark.quick
def test_l2_loss(device):
    """Test L2 loss computation."""
    pred = torch.rand(100, 3, device=device)
    target = torch.rand(100, 3, device=device)

    loss = l2(pred, target, reduction='mean')

    assert loss.item() >= 0
    assert not torch.isnan(loss)


@pytest.mark.quick
def test_loss_zero_when_equal(device):
    """Test that loss is zero for identical pred and target."""
    rgb = torch.rand(100, 3, device=device)

    loss = l2(rgb, rgb, reduction='mean')

    assert torch.allclose(loss, torch.tensor(0.0, device=device), atol=1e-7)


@pytest.mark.quick
def test_loss_reduction_modes(device):
    """Test different reduction modes."""
    pred = torch.rand(100, 3, device=device)
    target = torch.rand(100, 3, device=device)

    loss_mean = l2(pred, target, reduction='mean')
    loss_sum = l2(pred, target, reduction='sum')
    loss_none = l2(pred, target, reduction='none')

    assert loss_mean.ndim == 0  # scalar
    assert loss_sum.ndim == 0   # scalar
    assert loss_none.shape == (100, 3)  # per-element


@pytest.mark.quick
def test_huber_loss(device):
    """Test Huber loss."""
    pred = torch.rand(100, 3, device=device)
    target = torch.rand(100, 3, device=device)

    loss = huber(pred, target, delta=0.1)

    assert loss.item() >= 0
    assert not torch.isnan(loss)

