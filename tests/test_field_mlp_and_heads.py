"""Test field MLP and RGB heads."""
import pytest
import torch
from ngp_baseline_torch.field.mlp import NGP_MLP, TinyMLP
from ngp_baseline_torch.field.heads import RGBHead


@pytest.mark.quick
def test_mlp_forward(device):
    """Test MLP forward pass."""
    B = 100
    input_dim = 32
    hidden_dim = 64
    output_dim = 16

    mlp = NGP_MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        output_dim=output_dim
    )
    mlp.to(device)

    feat = torch.randn(B, input_dim, device=device)
    sigma, rgb_feat = mlp(feat)

    assert sigma.shape == (B,)
    assert rgb_feat.shape == (B, output_dim)
    assert not torch.isnan(sigma).any()
    assert not torch.isnan(rgb_feat).any()


@pytest.mark.quick
def test_mlp_sigma_positivity(device):
    """Test that sigma output is positive (via softplus)."""
    mlp = NGP_MLP(input_dim=32, hidden_dim=64, num_layers=2)
    mlp.to(device)

    feat = torch.randn(100, 32, device=device)
    sigma, _ = mlp(feat)

    # Sigma should be non-negative
    assert torch.all(sigma >= 0)


@pytest.mark.quick
def test_mlp_gradients(device):
    """Test that MLP produces gradients for all parameters."""
    mlp = NGP_MLP(input_dim=32, hidden_dim=64, num_layers=2)
    mlp.to(device)

    feat = torch.randn(10, 32, device=device)
    sigma, rgb_feat = mlp(feat)

    loss = sigma.sum() + rgb_feat.sum()
    loss.backward()

    # Check all parameters have gradients
    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


@pytest.mark.quick
def test_rgb_head_output_range(device):
    """Test RGB head outputs values in [0, 1]."""
    rgb_head = RGBHead(rgb_feat_dim=16, view_dependent=False)
    rgb_head.to(device)

    rgb_feat = torch.randn(100, 16, device=device)
    rgb = rgb_head(rgb_feat, None)

    assert rgb.shape == (100, 3)
    assert torch.all(rgb >= 0)
    assert torch.all(rgb <= 1)


@pytest.mark.quick
def test_rgb_head_view_dependent(device):
    """Test view-dependent RGB head."""
    rgb_head = RGBHead(rgb_feat_dim=16, view_dependent=True, viewdir_dim=27)
    rgb_head.to(device)

    rgb_feat = torch.randn(100, 16, device=device)
    viewdir = torch.randn(100, 27, device=device)

    rgb = rgb_head(rgb_feat, viewdir)

    assert rgb.shape == (100, 3)
    assert torch.all(rgb >= 0)
    assert torch.all(rgb <= 1)


@pytest.mark.quick
def test_tiny_mlp(device):
    """Test compact TinyMLP variant."""
    mlp = TinyMLP(input_dim=32, hidden_dim=32, output_dim=16)
    mlp.to(device)

    feat = torch.randn(50, 32, device=device)
    sigma, rgb_feat = mlp(feat)

    assert sigma.shape == (50,)
    assert rgb_feat.shape == (50, 16)
    assert torch.all(sigma >= 0)

