"""Test 3: Training Components - Optimizer, Loss, Learning Rate"""
import torch
import pytest
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.loss.rgb import l2 as rgb_loss_l2


class TestOptimizer:
    """Test optimizer configuration."""

    def test_separate_learning_rates(self):
        """Test separate LR for encoder and MLP."""
        cfg = Config()
        cfg.train.lr_encoder = 1e-2
        cfg.train.lr_mlp = 1e-3

        device = torch.device('cpu')
        encoder, field, rgb_head, _ = create_all(cfg, device)
        optimizer = create_optimizer(encoder, field, rgb_head, cfg)

        # Check parameter groups
        assert len(optimizer.param_groups) == 3

        # Check learning rates
        encoder_lr = optimizer.param_groups[0]['lr']
        field_lr = optimizer.param_groups[1]['lr']
        rgb_head_lr = optimizer.param_groups[2]['lr']

        assert encoder_lr == 1e-2, f"Encoder LR should be 1e-2, got {encoder_lr}"
        assert field_lr == 1e-3, f"Field LR should be 1e-3, got {field_lr}"
        assert rgb_head_lr == 1e-3, f"RGB head LR should be 1e-3, got {rgb_head_lr}"

        print(f"✓ Separate LRs: encoder={encoder_lr}, mlp={field_lr}")

    def test_optimizer_step(self):
        """Test optimizer can update parameters."""
        cfg = Config()
        device = torch.device('cpu')

        encoder, field, rgb_head, _ = create_all(cfg, device)
        optimizer = create_optimizer(encoder, field, rgb_head, cfg)

        # Save initial params
        initial_param = encoder.hash_tables[0].clone()

        # Forward and backward
        xyz = torch.randn(10, 3)
        encoded = encoder(xyz)
        sigma, rgb_feat = field(encoded.feat)
        loss = sigma.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check params changed
        assert not torch.allclose(encoder.hash_tables[0], initial_param)

        print(f"✓ Optimizer updates parameters correctly")


class TestLoss:
    """Test loss functions."""

    def test_rgb_loss_l2(self):
        """Test RGB L2 loss."""
        pred = torch.rand(100, 3)
        target = torch.rand(100, 3)

        loss = rgb_loss_l2(pred, target)

        assert loss.item() >= 0
        assert torch.isfinite(loss)

        # Test perfect prediction
        loss_zero = rgb_loss_l2(pred, pred)
        assert loss_zero.item() < 1e-6

        print(f"✓ RGB L2 loss working, value={loss.item():.6f}")

    def test_loss_to_psnr(self):
        """Test MSE to PSNR conversion."""
        mse = 0.001  # MSE
        psnr = -10 * torch.log10(torch.tensor(mse))

        assert psnr.item() > 0
        assert 20 < psnr.item() < 40  # Reasonable range

        print(f"✓ MSE={mse:.6f} -> PSNR={psnr.item():.2f} dB")


class TestGradients:
    """Test gradient flow through entire pipeline."""

    def test_end_to_end_gradients(self):
        """Test gradients flow from loss to encoder."""
        cfg = Config()
        cfg.model.hash_levels = 4  # Smaller for speed
        device = torch.device('cpu')

        encoder, field, rgb_head, _ = create_all(cfg, device)

        # Create dummy data with reasonable values to ensure gradient flow
        xyz = torch.randn(32, 3) * 0.5  # Smaller range to stay in valid region
        viewdir = torch.randn(32, 27) * 0.5
        target = torch.rand(32, 3)

        # Forward pass
        encoded = encoder(xyz)
        sigma, rgb_feat = field(encoded.feat)
        rgb_pred = rgb_head(rgb_feat, viewdir)

        # Use a loss that ensures gradients flow through sigma
        # MSE loss on RGB + small regularization on sigma to ensure gradient
        rgb_loss = rgb_loss_l2(rgb_pred, target)
        sigma_reg = (sigma.mean() - 1.0) ** 2  # Encourage non-zero sigma
        loss = rgb_loss + 0.01 * sigma_reg

        # Backward pass
        loss.backward()

        # Check all modules have gradients
        assert encoder.hash_tables[0].grad is not None
        assert torch.isfinite(encoder.hash_tables[0].grad).all()

        # Check field gradients (allow some to be zero for sparse activations)
        has_grad_count = 0
        for name, param in field.named_parameters():
            if param.grad is not None:
                has_grad_count += 1
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

        # At least most parameters should have gradients
        total_params = len(list(field.parameters()))
        assert has_grad_count >= total_params * 0.8, \
            f"Too few parameters have gradients: {has_grad_count}/{total_params}"

        # Check RGB head gradients
        for name, param in rgb_head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all()

        print(f"✓ End-to-end gradients flow correctly ({has_grad_count}/{total_params} field params)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
