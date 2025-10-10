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


def train_with_milestones(
    scene_data,
    num_iters: int,
    milestones: dict[int, float],
    device,
    cfg: Config
):
    """Shared training function with milestone validation.
    
    Args:
        scene_data: Scene data fixture
        num_iters: Total number of iterations
        milestones: Dict of {iteration: expected_psnr_dB}
        device: Training device
        cfg: Configuration
        
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*70}")
    print(f"CONVERGENCE TEST - {num_iters} Iterations on Lego Scene")
    print(f"Device: {device}")
    print(f"Milestones: {milestones}")
    print(f"{'='*70}\n")

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

    print(f"Training set: {cameras.N} images, {rays.N:,} rays")
    print(f"Config: hash_levels={cfg.model.hash_levels}, n_steps={cfg.integrator.n_steps_fixed}")
    print(f"Learning rates: encoder={cfg.train.lr_encoder}, mlp={cfg.train.lr_mlp}\n")

    # Training loop with LR decay
    encoder.train()
    field.train()
    rgb_head.train()

    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    psnrs = []
    milestone_results = {}

    for i in range(num_iters):
        # Learning rate decay (start after warmup)
        warmup_iters = max(500, num_iters // 20)
        if i > warmup_iters:
            progress = (i - warmup_iters) / (num_iters - warmup_iters)
            # Exponential decay from 1.0 to 0.1
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

        # Logging and milestone checking
        iter_num = i + 1
        if iter_num in milestones or iter_num % max(100, num_iters // 20) == 0:
            avg_psnr_100 = np.mean(psnrs[-100:]) if len(psnrs) >= 100 else np.mean(psnrs)
            current_lr = optimizer.param_groups[0]['lr']
            
            status = ""
            if iter_num in milestones:
                expected = milestones[iter_num]
                milestone_results[iter_num] = avg_psnr_100
                diff = avg_psnr_100 - expected
                if diff >= 0:
                    status = f" ✓ (target: {expected:.1f} dB)"
                elif diff >= -1.0:
                    status = f" ~ (target: {expected:.1f} dB, diff: {diff:.1f})"
                else:
                    status = f" ✗ (target: {expected:.1f} dB, diff: {diff:.1f})"
            
            print(f"  Iter {iter_num:5d}: PSNR={psnr:.2f} dB, Avg(100)={avg_psnr_100:.2f} dB, LR={current_lr:.6f}{status}")

    # Final evaluation
    final_psnr = np.mean(psnrs[-100:])
    max_psnr = max(psnrs[-500:]) if len(psnrs) >= 500 else max(psnrs)
    min_psnr_last_1000 = min(psnrs[-1000:]) if len(psnrs) >= 1000 else min(psnrs)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Final Avg PSNR (last 100): {final_psnr:.2f} dB")
    print(f"  Max PSNR (last 500):       {max_psnr:.2f} dB")
    print(f"  Min PSNR (last 1000):      {min_psnr_last_1000:.2f} dB")
    print(f"  Milestone Results:")
    for milestone, achieved in sorted(milestone_results.items()):
        expected = milestones[milestone]
        diff = achieved - expected
        status = "✓" if diff >= 0 else ("~" if diff >= -1.0 else "✗")
        print(f"    {milestone:5d} iters: {achieved:.2f} dB (expected {expected:.1f} dB) {status}")
    print(f"{'='*70}\n")

    return {
        'psnrs': psnrs,
        'final_psnr': final_psnr,
        'max_psnr': max_psnr,
        'milestone_results': milestone_results
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

        # Assertions - should achieve 25+ dB on single image
        assert final_psnr > 25.0, f"Failed to overfit single image: PSNR={final_psnr:.2f} < 25.0 dB"
        print(f"\n✓ Single image overfit: Final PSNR={final_psnr:.2f} dB (>25 dB)")


class TestStageConvergence:
    """Test convergence at specific iteration milestones.
    
    Based on Instant-NGP paper benchmarks for Lego scene:
    - 500 iters: ~20-22 dB
    - 1000 iters: ~23-25 dB
    - 2000 iters: ~26-28 dB
    - 5000 iters: ~30-31 dB
    """

    def _get_optimal_config(self) -> Config:
        """Get optimal configuration for convergence."""
        cfg = Config()
        cfg.model.hash_levels = 16
        cfg.model.hash_per_level_scale = 1.5
        cfg.model.hash_features_per_level = 2
        cfg.model.mlp_width = 64
        cfg.model.mlp_depth = 2
        cfg.model.density_activation = "trunc_exp"
        cfg.model.view_dependent = True
        cfg.model.pe_viewdir_bands = 4
        cfg.integrator.perturb = True
        cfg.integrator.n_steps_fixed = 128
        cfg.integrator.early_stop_T = 1e-4
        cfg.train.lr_encoder = 1e-2
        cfg.train.lr_mlp = 1e-3
        cfg.train.batch_rays = 4096
        cfg.precision.use_amp = False
        return cfg

    @pytest.mark.slow
    def test_500_iterations(self, scene_data):
        """Test 500 iterations - should reach ~20 dB."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        cfg = self._get_optimal_config()
        milestones = {
            100: 15.0,
            500: 20.0
        }

        results = train_with_milestones(scene_data, 500, milestones, device, cfg)
        
        # Should reach at least 19 dB (allowing 1 dB tolerance)
        assert results['final_psnr'] >= 19.0, (
            f"500 iterations failed: {results['final_psnr']:.2f} dB < 19.0 dB"
        )
        print(f"✓ 500 iterations PASSED: {results['final_psnr']:.2f} dB >= 19.0 dB")

    @pytest.mark.slow
    def test_1000_iterations(self, scene_data):
        """Test 1000 iterations - should reach ~23 dB."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        cfg = self._get_optimal_config()
        milestones = {
            100: 15.0,
            500: 20.0,
            1000: 23.0
        }

        results = train_with_milestones(scene_data, 1000, milestones, device, cfg)
        
        # Should reach at least 22 dB
        assert results['final_psnr'] >= 22.0, (
            f"1000 iterations failed: {results['final_psnr']:.2f} dB < 22.0 dB"
        )
        print(f"✓ 1000 iterations PASSED: {results['final_psnr']:.2f} dB >= 22.0 dB")

    @pytest.mark.slow
    def test_2000_iterations(self, scene_data):
        """Test 2000 iterations - should reach ~26 dB."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        cfg = self._get_optimal_config()
        milestones = {
            500: 20.0,
            1000: 23.0,
            2000: 26.0
        }

        results = train_with_milestones(scene_data, 2000, milestones, device, cfg)
        
        # Should reach at least 25 dB
        assert results['final_psnr'] >= 25.0, (
            f"2000 iterations failed: {results['final_psnr']:.2f} dB < 25.0 dB"
        )
        print(f"✓ 2000 iterations PASSED: {results['final_psnr']:.2f} dB >= 25.0 dB")

    @pytest.mark.slow
    def test_5000_iterations(self, scene_data):
        """Test 5000 iterations - should reach ~30 dB."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        cfg = self._get_optimal_config()
        milestones = {
            500: 20.0,
            1000: 23.0,
            2000: 26.0,
            5000: 30.0
        }

        results = train_with_milestones(scene_data, 5000, milestones, device, cfg)
        
        # Should reach at least 29 dB
        assert results['final_psnr'] >= 29.0, (
            f"5000 iterations failed: {results['final_psnr']:.2f} dB < 29.0 dB"
        )
        print(f"✓ 5000 iterations PASSED: {results['final_psnr']:.2f} dB >= 29.0 dB")


class TestFullConvergence:
    """Test full convergence to match Instant-NGP paper results."""

    @pytest.mark.slow
    def test_10000_iterations_high_quality(self, scene_data):
        """Test 10000 iterations - should reach ~31 dB."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        cfg = Config()
        cfg.model.hash_levels = 16
        cfg.model.hash_per_level_scale = 1.5
        cfg.model.hash_features_per_level = 2
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

        milestones = {
            500: 20.0,
            1000: 23.0,
            2000: 26.0,
            5000: 30.0,
            10000: 31.0
        }

        results = train_with_milestones(scene_data, 10000, milestones, device, cfg)
        
        # Should reach at least 30 dB
        assert results['final_psnr'] >= 30.0, (
            f"10000 iterations failed: {results['final_psnr']:.2f} dB < 30.0 dB"
        )
        print(f"✓ 10000 iterations PASSED: {results['final_psnr']:.2f} dB >= 30.0 dB")

    @pytest.mark.slow
    def test_20000_iterations_full_convergence(self, scene_data):
        """CRITICAL TEST: Full convergence at 20000 iterations - should reach ~32 dB.
        
        This matches the Instant-NGP paper's reported results for Lego scene.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            pytest.skip("Requires CUDA for reasonable speed")

        cfg = Config()
        cfg.model.hash_levels = 16
        cfg.model.hash_per_level_scale = 1.5
        cfg.model.hash_features_per_level = 2
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

        milestones = {
            500: 20.0,
            1000: 23.0,
            2000: 26.0,
            5000: 30.0,
            10000: 31.0,
            20000: 32.0
        }

        results = train_with_milestones(scene_data, 20000, milestones, device, cfg)
        
        # CRITICAL: Should reach at least 31 dB for full convergence
        assert results['final_psnr'] >= 31.0, (
            f"FULL CONVERGENCE FAILED!\n"
            f"Expected: ≥31.0 dB at 20000 iterations (Instant-NGP baseline)\n"
            f"Achieved: {results['final_psnr']:.2f} dB\n"
            f"This indicates the implementation is not matching paper results."
        )
        print(f"✓ FULL CONVERGENCE PASSED: {results['final_psnr']:.2f} dB >= 31.0 dB")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_4_e2e_convergence.py -v -s
    # For specific tests: python -m pytest tests/test_4_e2e_convergence.py::TestStageConvergence::test_2000_iterations -v -s
    pytest.main([__file__, "-v", "-s"])
