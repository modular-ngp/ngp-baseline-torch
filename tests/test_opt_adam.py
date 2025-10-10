"""Test optimizer wrapper."""
import pytest
import torch
from ngp_baseline_torch.config import TrainConfig
from ngp_baseline_torch.opt import wrap, make_optimizer


@pytest.mark.quick
def test_optimizer_creation():
    """Test optimizer creation from config."""
    cfg = TrainConfig(lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-6)

    # Create simple parameter
    param = torch.nn.Parameter(torch.randn(10, 10))

    optimizer = wrap([param], cfg)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == 1e-3


@pytest.mark.quick
def test_optimizer_step(device):
    """Test that optimizer updates parameters."""
    cfg = TrainConfig(lr=1e-2)

    param = torch.nn.Parameter(torch.randn(10, 10, device=device))
    initial_value = param.data.clone()

    optimizer = wrap([param], cfg)

    # Compute dummy loss and step
    loss = (param ** 2).sum()
    loss.backward()
    optimizer.step()

    # Parameter should have changed
    assert not torch.allclose(param.data, initial_value)


@pytest.mark.quick
def test_make_optimizer_for_model(device):
    """Test optimizer creation for full model."""
    from ngp_baseline_torch.field.mlp import NGP_MLP

    cfg = TrainConfig(lr=1e-3)
    model = NGP_MLP(input_dim=32, hidden_dim=64, num_layers=2)
    model.to(device)

    optimizer = make_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) > 0

