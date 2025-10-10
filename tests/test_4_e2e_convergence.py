"""Test 4: End-to-End Convergence Test - CRITICAL TEST"""
import torch
import pytest
import numpy as np
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays
from ngp_baseline_torch.data_utils import load_nerf_synthetic_images
from ngp_baseline_torch.types import RayBatch
from ngp_baseline_torch.integrator import render_batch
from ngp_baseline_torch.loss.rgb import l2 as rgb_loss_l2


@pytest.fixture(scope="module")
def scene_data():
    """Load lego scene data once for all tests."""
    scene_path = Path("data/nerf-synthetic/lego")

    if not scene_path.exists():
        pytest.skip(f"Dataset not found at {scene_path}")

    # Load cameras and images
    cameras = load_nerf_synthetic(scene_path, "transforms_train.json")
    images, _ = load_nerf_synthetic_images(scene_path, "train", torch.device('cpu'))

    return {
        'cameras': cameras,
        'images': images,
        'scene_path': scene_path
    }


class TestQuickOverfit:
    """Test that model can overfit a single image quickly."""

    def test_single_image_overfit(self, scene_data):
        """Test overfitting single image in 500 iterations."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")

        # Config
        cfg = Config()
        cfg.model.hash_levels = 16
        cfg.model.density_activation = "trunc_exp"
        cfg.integrator.perturb = True
        cfg.integrator.n_steps_fixed = 64  # Fewer samples for speed
        cfg.train.lr_encoder = 1e-2
        cfg.train.lr_mlp = 1e-3
        cfg.train.batch_rays = 4096

        # Create model
        encoder, field, rgb_head, _ = create_all(cfg, device)
        optimizer = create_optimizer(encoder, field, rgb_head, cfg)

        # Use single image
        cameras = scene_data['cameras']
        cameras.poses = cameras.poses[[0]].to(device)  # First image only
        images = scene_data['images'][[0]].to(device)

        # Generate rays
        H, W = 800, 800
        rays = make_rays(H, W, cameras, None, device)
        target_rgb = images.reshape(-1, 3)

        print(f"Training on single image: {rays.N} rays")

        # Training loop
        encoder.train()
        field.train()
        rgb_head.train()

        psnrs = []
        for i in range(500):
            # Sample random rays
            idx = torch.randperm(rays.N, device=device)[:cfg.train.batch_rays]
            ray_batch = RayBatch(
                orig_x=rays.orig_x[idx], orig_y=rays.orig_y[idx], orig_z=rays.orig_z[idx],
                dir_x=rays.dir_x[idx], dir_y=rays.dir_y[idx], dir_z=rays.dir_z[idx],
                tmin=rays.tmin[idx], tmax=rays.tmax[idx]
            )

            # Forward
            pred_rgb, _ = render_batch(ray_batch, encoder, field, rgb_head, cfg.integrator)
            loss = rgb_loss_l2(pred_rgb, target_rgb[idx])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            psnr = -10 * torch.log10(loss).item()
            psnrs.append(psnr)

            if (i + 1) % 100 == 0:
                print(f"  Iter {i+1}: Loss={loss.item():.6f}, PSNR={psnr:.2f} dB")

        final_psnr = psnrs[-1]

        # Assertions
        assert final_psnr > 25.0, f"Failed to overfit single image: PSNR={final_psnr:.2f} < 25.0 dB"
        print(f"\n✓ Single image overfit: Final PSNR={final_psnr:.2f} dB (>25 dB)")


class TestFastConvergence:
    """Test convergence on full training set."""

    @pytest.mark.slow
    def test_1000_iterations_convergence(self, scene_data):
        """Test reaching ~22 dB in 1000 iterations."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        print(f"\nDevice: {device}")

        # Config with all fixes
        cfg = Config()
        cfg.model.hash_levels = 16
        cfg.model.density_activation = "trunc_exp"
        cfg.integrator.perturb = True
        cfg.integrator.n_steps_fixed = 128
        cfg.train.lr_encoder = 1e-2
        cfg.train.lr_mlp = 1e-3
        cfg.train.batch_rays = 4096

        # Create model
        encoder, field, rgb_head, _ = create_all(cfg, device)
        optimizer = create_optimizer(encoder, field, rgb_head, cfg)

        # Load full training set
        cameras = scene_data['cameras']
        cameras.poses = cameras.poses.to(device)
        images = scene_data['images'].to(device)

        H, W = 800, 800
        rays = make_rays(H, W, cameras, None, device)
        target_rgb = images.reshape(-1, 3)

        print(f"Training on {cameras.N} images: {rays.N} total rays")

        # Training
        encoder.train()
        field.train()
        rgb_head.train()

        psnrs = []
        for i in range(1000):
            idx = torch.randperm(rays.N, device=device)[:cfg.train.batch_rays]
            ray_batch = RayBatch(
                orig_x=rays.orig_x[idx], orig_y=rays.orig_y[idx], orig_z=rays.orig_z[idx],
                dir_x=rays.dir_x[idx], dir_y=rays.dir_y[idx], dir_z=rays.dir_z[idx],
                tmin=rays.tmin[idx], tmax=rays.tmax[idx]
            )

            pred_rgb, _ = render_batch(ray_batch, encoder, field, rgb_head, cfg.integrator)
            loss = rgb_loss_l2(pred_rgb, target_rgb[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = -10 * torch.log10(loss).item()
            psnrs.append(psnr)

            if (i + 1) % 200 == 0:
                avg_psnr = np.mean(psnrs[-100:])
                print(f"  Iter {i+1}: PSNR={psnr:.2f} dB, Avg(100)={avg_psnr:.2f} dB")

        final_avg_psnr = np.mean(psnrs[-100:])

        # Instant-NGP should reach ~22 dB in 1000 iterations
        assert final_avg_psnr > 20.0, f"Convergence too slow: PSNR={final_avg_psnr:.2f} < 20.0 dB at 1000 iters"
        print(f"\n✓ 1000 iterations: Avg PSNR={final_avg_psnr:.2f} dB (>20 dB)")


class TestFullConvergence:
    """Test full convergence to expected PSNR."""

    @pytest.mark.slow
    def test_5000_iterations_target_psnr(self, scene_data):
        """CRITICAL TEST: Must reach 28+ dB in 5000 iterations on Lego scene.

        Instant-NGP paper reports:
        - Lego scene: 32+ dB at convergence
        - Should reach 28+ dB within 5000 iterations
        - Should reach 30+ dB within 20000 iterations
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable training time")

        print(f"\n{'='*70}")
        print(f"CRITICAL CONVERGENCE TEST - 5000 Iterations on Lego Scene")
        print(f"Target: 28+ dB (Instant-NGP baseline)")
        print(f"Device: {device}")
        print(f"{'='*70}\n")

        # Optimal config
        cfg = Config()
        cfg.model.hash_levels = 16
        cfg.model.hash_per_level_scale = 1.5
        cfg.model.mlp_width = 64
        cfg.model.mlp_depth = 2
        cfg.model.density_activation = "trunc_exp"
        cfg.model.view_dependent = True
        cfg.integrator.perturb = True
        cfg.integrator.n_steps_fixed = 128
        cfg.integrator.early_stop_T = 1e-4
        cfg.train.lr_encoder = 1e-2
        cfg.train.lr_mlp = 1e-3
        cfg.train.batch_rays = 4096
        cfg.precision.use_amp = False

        # Create model
        encoder, field, rgb_head, _ = create_all(cfg, device)
        optimizer = create_optimizer(encoder, field, rgb_head, cfg)

        # Load data
        cameras = scene_data['cameras']
        cameras.poses = cameras.poses.to(device)
        images = scene_data['images'].to(device)

        H, W = 800, 800
        rays = make_rays(H, W, cameras, None, device)
        target_rgb = images.reshape(-1, 3)

        print(f"Training set: {cameras.N} images, {rays.N:,} rays\n")

        # Training loop with LR decay
        encoder.train()
        field.train()
        rgb_head.train()

        initial_lrs = [group['lr'] for group in optimizer.param_groups]

        psnrs = []
        milestones = {100: 15.0, 500: 20.0, 1000: 22.0, 2000: 25.0, 5000: 28.0}

        for i in range(5000):
            # Learning rate decay
            if i > 1000:
                progress = (i - 1000) / 4000
                decay = 0.1 ** progress
                for idx, group in enumerate(optimizer.param_groups):
                    group['lr'] = initial_lrs[idx] * decay

            # Sample and train
            idx = torch.randperm(rays.N, device=device)[:cfg.train.batch_rays]
            ray_batch = RayBatch(
                orig_x=rays.orig_x[idx], orig_y=rays.orig_y[idx], orig_z=rays.orig_z[idx],
                dir_x=rays.dir_x[idx], dir_y=rays.dir_y[idx], dir_z=rays.dir_z[idx],
                tmin=rays.tmin[idx], tmax=rays.tmax[idx]
            )

            pred_rgb, _ = render_batch(ray_batch, encoder, field, rgb_head, cfg.integrator)
            loss = rgb_loss_l2(pred_rgb, target_rgb[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = -10 * torch.log10(loss).item()
            psnrs.append(psnr)

            # Logging
            if (i + 1) in milestones or (i + 1) % 500 == 0:
                avg_psnr_100 = np.mean(psnrs[-100:]) if len(psnrs) >= 100 else np.mean(psnrs)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Iter {i+1:4d}: PSNR={psnr:.2f} dB, Avg(100)={avg_psnr_100:.2f} dB, LR={current_lr:.6f}")

                # Check milestone
                if (i + 1) in milestones:
                    expected = milestones[i + 1]
                    if avg_psnr_100 < expected - 2.0:
                        print(f"    ⚠ Warning: Below expected {expected:.1f} dB")

        # Final evaluation
        final_psnr = np.mean(psnrs[-100:])
        max_psnr = max(psnrs[-500:])

        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  Final Avg PSNR (last 100): {final_psnr:.2f} dB")
        print(f"  Max PSNR (last 500):       {max_psnr:.2f} dB")
        print(f"  Target:                     28.00 dB")
        print(f"{'='*70}\n")

        # CRITICAL ASSERTION
        assert final_psnr >= 28.0, (
            f"CONVERGENCE FAILED!\n"
            f"Expected: ≥28.0 dB (Instant-NGP baseline)\n"
            f"Achieved: {final_psnr:.2f} dB\n"
            f"This indicates a fundamental issue with the implementation."
        )

        print(f"✓ CONVERGENCE TEST PASSED: {final_psnr:.2f} dB ≥ 28.0 dB")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_4_e2e_convergence.py -v -s
    pytest.main([__file__, "-v", "-s", "-k", "test_5000"])
"""Test 1: Core Components - Hash Grid, MLP, RGB Head"""
import torch
import pytest
from ngp_baseline_torch.encoder.hashgrid_torch import HashGridEncoder
from ngp_baseline_torch.encoder.pe import PositionalEncoder
from ngp_baseline_torch.field.mlp import NGP_MLP
from ngp_baseline_torch.field.heads import RGBHead


class TestHashGridEncoder:
    """Test hash grid encoder with critical fixes."""

    def test_initialization_range(self):
        """CRITICAL: Hash grid must initialize in [-1e-4, 1e-4] range."""
        encoder = HashGridEncoder(num_levels=16, base_resolution=16)

        for level, table in enumerate(encoder.hash_tables):
            min_val = table.min().item()
            max_val = table.max().item()

            # Must be within uniform[-1e-4, 1e-4]
            assert -1e-4 <= min_val <= 1e-4, f"Level {level}: min={min_val} out of range"
            assert -1e-4 <= max_val <= 1e-4, f"Level {level}: max={max_val} out of range"

        print(f"✓ Hash grid initialization correct: [{min_val:.6f}, {max_val:.6f}]")

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = HashGridEncoder(num_levels=16, features_per_level=2)
        xyz = torch.randn(128, 3)

        output = encoder(xyz)

        # Should output padded features
        assert output.feat.shape[0] == 128
        assert output.feat.shape[1] == encoder.output_dim
        assert encoder.output_dim % 16 == 0, "Output must be padded to 16"

        print(f"✓ Hash grid output shape: {output.feat.shape}")

    def test_gradient_flow(self):
        """Test gradients flow through hash grid."""
        encoder = HashGridEncoder(num_levels=4, base_resolution=16)
        xyz = torch.randn(32, 3, requires_grad=True)

        output = encoder(xyz)
        loss = output.feat.sum()
        loss.backward()

        # Check gradients exist
        assert xyz.grad is not None
        assert torch.isfinite(xyz.grad).all()

        # Check hash table gradients
        for table in encoder.hash_tables:
            assert table.grad is not None
            assert torch.isfinite(table.grad).all()

        print(f"✓ Hash grid gradients flow correctly")


class TestMLP:
    """Test MLP with truncated exponential activation."""

    def test_density_activation_trunc_exp(self):
        """CRITICAL: Must use truncated exponential for density."""
        mlp = NGP_MLP(32, 64, 2, 16, density_activation="trunc_exp")

        assert mlp.density_activation == "trunc_exp"

        x = torch.randn(64, 32)
        sigma, rgb_feat = mlp(x)

        # Sigma must be non-negative
        assert (sigma >= 0).all()
        assert torch.isfinite(sigma).all()

        # Should have reasonable range
        assert sigma.max() < 1e6, "Density too large, clipping failed"

        print(f"✓ Truncated exponential activation working, sigma range: [{sigma.min():.3f}, {sigma.max():.3f}]")

    def test_sigma_bias_initialization(self):
        """CRITICAL: Sigma head bias should be -1.5 for sparse initialization."""
        mlp = NGP_MLP(32, 64, 2, 16, density_activation="trunc_exp")

        bias = mlp.sigma_head.bias.item()
        assert abs(bias - (-1.5)) < 0.01, f"Sigma bias should be -1.5, got {bias}"

        # Test that initial densities are low
        x = torch.zeros(100, 32)  # Zero input
        sigma, _ = mlp(x)

        mean_sigma = sigma.mean().item()
        assert mean_sigma < 1.0, f"Initial sigma too high: {mean_sigma}"

        print(f"✓ Sigma bias: {bias:.3f}, initial mean sigma: {mean_sigma:.6f}")

    def test_forward_shapes(self):
        """Test MLP output shapes."""
        mlp = NGP_MLP(input_dim=48, hidden_dim=64, num_layers=2, output_dim=16)

        x = torch.randn(128, 48)
        sigma, rgb_feat = mlp(x)

        assert sigma.shape == (128,)
        assert rgb_feat.shape == (128, 16)

        print(f"✓ MLP output shapes correct")


class TestRGBHead:
    """Test RGB head with view-dependent rendering."""

    def test_view_dependent_mode(self):
        """Test view-dependent RGB head."""
        rgb_head = RGBHead(rgb_feat_dim=16, view_dependent=True, viewdir_dim=27)

        rgb_feat = torch.randn(64, 16)
        viewdir = torch.randn(64, 27)

        rgb = rgb_head(rgb_feat, viewdir)

        assert rgb.shape == (64, 3)
        assert (rgb >= 0).all() and (rgb <= 1).all(), "RGB must be in [0, 1]"

        print(f"✓ View-dependent RGB head working, range: [{rgb.min():.3f}, {rgb.max():.3f}]")

    def test_view_independent_mode(self):
        """Test view-independent RGB head."""
        rgb_head = RGBHead(rgb_feat_dim=16, view_dependent=False)

        rgb_feat = torch.randn(64, 16)
        rgb = rgb_head(rgb_feat, None)

        assert rgb.shape == (64, 3)
        assert (rgb >= 0).all() and (rgb <= 1).all()

        print(f"✓ View-independent RGB head working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

