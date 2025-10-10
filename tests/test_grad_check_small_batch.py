"""Gradient checking for small batches."""
import pytest
import torch
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all
from ngp_baseline_torch.types import RayBatch


@pytest.mark.quick
def test_grad_flow_encoder_to_loss(device):
    """Test that gradients flow from loss back to encoder."""
    cfg = Config()
    cfg.model.hash_levels = 4
    cfg.model.mlp_width = 32
    cfg.integrator.n_steps_fixed = 16
    
    encoder, field, rgb_head, _ = create_all(cfg, device)
    
    # Create small ray batch
    N = 32
    rays = RayBatch(
        orig_x=torch.zeros(N, device=device),
        orig_y=torch.zeros(N, device=device),
        orig_z=torch.zeros(N, device=device),
        dir_x=torch.zeros(N, device=device),
        dir_y=torch.zeros(N, device=device),
        dir_z=torch.ones(N, device=device),
        tmin=torch.ones(N, device=device) * 2.0,
        tmax=torch.ones(N, device=device) * 6.0,
    )
    
    # Forward pass
    from ngp_baseline_torch.integrator import render_batch
    rgb_pred, _ = render_batch(rays, encoder, field, rgb_head, cfg.integrator)
    
    # Compute loss
    target = torch.ones(N, 3, device=device)
    loss = torch.nn.functional.mse_loss(rgb_pred, target)
    
    # Backward
    loss.backward()
    
    # Check encoder has gradients
    has_grad = False
    for param in encoder.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Encoder parameters have no gradients"
    
    # Check field has gradients
    has_grad = False
    for param in field.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Field parameters have no gradients"
    
    # Check RGB head has gradients
    has_grad = False
    for param in rgb_head.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "RGB head parameters have no gradients"


@pytest.mark.quick
def test_grad_finite_difference_mlp(device):
    """Test MLP gradients against finite difference."""
    from ngp_baseline_torch.field.mlp import NGP_MLP
    
    # Small MLP for testing
    mlp = NGP_MLP(input_dim=32, hidden_dim=16, num_layers=1, output_dim=8)
    mlp.to(device)
    
    # Input
    x = torch.randn(10, 32, device=device, requires_grad=True)
    
    # Forward
    sigma, rgb_feat = mlp(x)
    loss = sigma.sum() + rgb_feat.sum()
    
    # Backward
    loss.backward()
    
    # Check gradients exist and are finite
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()
    
    # Finite difference check for first weight
    eps = 1e-4
    param = list(mlp.parameters())[0]
    grad_analytical = param.grad.clone()
    
    # Perturb one element
    idx = (0, 0)
    original_val = param.data[idx].item()
    
    # Forward +eps
    param.data[idx] = original_val + eps
    with torch.no_grad():
        sigma_plus, rgb_feat_plus = mlp(x)
        loss_plus = sigma_plus.sum() + rgb_feat_plus.sum()
    
    # Forward -eps
    param.data[idx] = original_val - eps
    with torch.no_grad():
        sigma_minus, rgb_feat_minus = mlp(x)
        loss_minus = sigma_minus.sum() + rgb_feat_minus.sum()
    
    # Restore
    param.data[idx] = original_val
    
    # Finite difference
    grad_numerical = (loss_plus - loss_minus) / (2 * eps)
    grad_analytical_val = grad_analytical[idx].item()
    
    # Compare (allow some tolerance)
    rel_error = abs(grad_numerical - grad_analytical_val) / (abs(grad_numerical) + 1e-8)
    print(f"\nNumerical: {grad_numerical:.6f}, Analytical: {grad_analytical_val:.6f}, Rel error: {rel_error:.6f}")
    
    assert rel_error < 0.01, f"Gradient mismatch: numerical={grad_numerical:.6f}, analytical={grad_analytical_val:.6f}"


@pytest.mark.quick
def test_grad_check_hash_encoder(device):
    """Test hash encoder gradients."""
    from ngp_baseline_torch.encoder.hashgrid_torch import HashGridEncoder
    
    encoder = HashGridEncoder(
        num_levels=4,
        base_resolution=16,
        max_resolution=128,
        features_per_level=2
    )
    encoder.to(device)
    
    # Input positions - make them leaf tensors
    xyz = torch.rand(16, 3, device=device) * 2 - 1
    xyz.requires_grad = True

    # Forward
    encoded = encoder(xyz)
    loss = encoded.feat.sum()
    
    # Backward
    loss.backward()
    
    # Check input gradients
    assert xyz.grad is not None
    assert torch.isfinite(xyz.grad).all()
    assert not torch.isnan(xyz.grad).any()
    
    # Check hash table gradients
    for level, table in enumerate(encoder.hash_tables):
        assert table.grad is not None, f"Level {level} hash table has no gradient"
        assert torch.isfinite(table.grad).all(), f"Level {level} has non-finite gradients"


@pytest.mark.quick
def test_grad_vanishing_check(device):
    """Test that gradients don't vanish through deep network."""
    cfg = Config()
    cfg.model.hash_levels = 8
    cfg.model.mlp_width = 64
    cfg.model.mlp_depth = 3  # Deeper network
    
    encoder, field, rgb_head, _ = create_all(cfg, device)
    
    # Small batch
    xyz = torch.rand(32, 3, device=device, requires_grad=True) * 2 - 1
    
    # Forward through encoder
    encoded = encoder(xyz)
    
    # Forward through field
    sigma, rgb_feat = field(encoded.feat)
    
    # Forward through RGB head
    rgb = rgb_head(rgb_feat, None)
    
    # Loss
    target = torch.ones_like(rgb)
    loss = torch.nn.functional.mse_loss(rgb, target)
    
    # Backward
    loss.backward()
    
    # Check input gradients didn't vanish
    assert xyz.grad is not None
    grad_norm = xyz.grad.norm().item()
    assert grad_norm > 1e-6, f"Gradients vanished: norm={grad_norm}"
    
    # Check first layer of field has reasonable gradients
    first_param = list(field.parameters())[0]
    assert first_param.grad is not None
    first_grad_norm = first_param.grad.norm().item()
    assert first_grad_norm > 1e-6, f"First layer gradients vanished: norm={first_grad_norm}"


@pytest.mark.quick
def test_grad_exploding_check(device):
    """Test that gradients don't explode."""
    cfg = Config()
    cfg.model.hash_levels = 16
    cfg.model.mlp_width = 128
    cfg.model.mlp_depth = 4
    
    encoder, field, rgb_head, _ = create_all(cfg, device)
    
    # Batch
    xyz = torch.rand(64, 3, device=device, requires_grad=True) * 2 - 1
    
    # Forward
    encoded = encoder(xyz)
    sigma, rgb_feat = field(encoded.feat)
    rgb = rgb_head(rgb_feat, None)
    
    # Loss
    target = torch.rand_like(rgb)
    loss = torch.nn.functional.mse_loss(rgb, target)
    
    # Backward
    loss.backward()
    
    # Check gradients aren't exploding
    max_grad = 0.0
    for param in list(encoder.parameters()) + list(field.parameters()) + list(rgb_head.parameters()):
        if param.grad is not None:
            max_grad = max(max_grad, param.grad.abs().max().item())
    
    assert max_grad < 1e3, f"Gradients exploding: max={max_grad}"
    assert torch.isfinite(xyz.grad).all(), "Input gradients contain inf/nan"


@pytest.mark.quick
def test_second_order_grad(device):
    """Test that second-order gradients work (for some optimizers)."""
    from ngp_baseline_torch.field.mlp import TinyMLP
    
    mlp = TinyMLP(input_dim=16, hidden_dim=16, output_dim=8)
    mlp.to(device)
    
    x = torch.randn(8, 16, device=device, requires_grad=True)
    
    # First forward/backward
    sigma1, rgb_feat1 = mlp(x)
    loss1 = sigma1.sum()
    loss1.backward(create_graph=True)
    
    # Second backward (through the gradients)
    if x.grad is not None:
        grad_sum = x.grad.sum()
        grad_sum.backward()
    
    # Should not crash and should produce finite values
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.quick
def test_grad_accumulation(device):
    """Test gradient accumulation across multiple batches."""
    from ngp_baseline_torch.field.mlp import NGP_MLP
    
    mlp = NGP_MLP(input_dim=32, hidden_dim=32, num_layers=1, output_dim=8)
    mlp.to(device)
    
    # Accumulate gradients over 3 batches
    for i in range(3):
        x = torch.randn(8, 32, device=device)
        sigma, rgb_feat = mlp(x)
        loss = sigma.sum() + rgb_feat.sum()
        loss.backward()
    
    # Check gradients accumulated
    for param in mlp.parameters():
        assert param.grad is not None
        assert param.grad.abs().sum() > 0
        assert torch.isfinite(param.grad).all()


@pytest.mark.quick
def test_zero_grad_resets(device):
    """Test that zero_grad properly resets gradients."""
    from ngp_baseline_torch.field.mlp import NGP_MLP
    
    mlp = NGP_MLP(input_dim=16, hidden_dim=16, num_layers=1, output_dim=4)
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters())
    
    # First backward
    x = torch.randn(4, 16, device=device)
    sigma, rgb_feat = mlp(x)
    loss = sigma.sum()
    loss.backward()
    
    # Store gradient
    first_grad = list(mlp.parameters())[0].grad.clone()
    
    # Zero grad
    optimizer.zero_grad()
    
    # Check all gradients are zero or None
    for param in mlp.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() == 0
    
    # Second backward with different input
    x2 = torch.randn(4, 16, device=device) + 10.0  # Very different input
    sigma2, rgb_feat2 = mlp(x2)
    loss2 = sigma2.sum()
    loss2.backward()
    
    # New gradient should be different
    second_grad = list(mlp.parameters())[0].grad
    assert not torch.allclose(first_grad, second_grad, atol=1e-6)
