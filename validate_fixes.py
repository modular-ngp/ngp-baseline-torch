"""Quick validation of critical fixes."""
import torch
import sys
sys.path.insert(0, 'src')

from ngp_baseline_torch.encoder.hashgrid_torch import HashGridEncoder
from ngp_baseline_torch.field.mlp import NGP_MLP
from ngp_baseline_torch.config import IntegratorConfig
from ngp_baseline_torch.integrator.marcher import RayMarcher
from ngp_baseline_torch.types import RayBatch

print("="*70)
print("VALIDATING CRITICAL FIXES")
print("="*70)

# Test 1: Hash Grid Initialization
print("\n[Test 1] Hash Grid Initialization")
encoder = HashGridEncoder(num_levels=16)
min_val = encoder.hash_tables[0].min().item()
max_val = encoder.hash_tables[0].max().item()
print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
assert -1e-4 <= min_val <= 1e-4, f"Min out of range: {min_val}"
assert -1e-4 <= max_val <= 1e-4, f"Max out of range: {max_val}"
print("  ✓ PASS - Uniform initialization in [-1e-4, 1e-4]")

# Test 2: MLP Density Activation
print("\n[Test 2] MLP Density Activation")
mlp = NGP_MLP(32, 64, 2, 16, density_activation='trunc_exp')
print(f"  Activation: {mlp.density_activation}")
print(f"  Sigma bias: {mlp.sigma_head.bias.item():.3f}")
assert mlp.density_activation == 'trunc_exp', "Wrong activation"
assert abs(mlp.sigma_head.bias.item() - (-1.5)) < 0.01, "Wrong bias"
x = torch.randn(10, 32)
sigma, _ = mlp(x)
assert (sigma >= 0).all(), "Negative sigma"
assert sigma.max() < 1e6, "Sigma too large"
print("  ✓ PASS - Truncated exponential with bias=-1.5")

# Test 3: Sample Jittering
print("\n[Test 3] Sample Jittering")
cfg = IntegratorConfig(perturb=True, n_steps_fixed=64)
marcher = RayMarcher(cfg)
rays = RayBatch(
    orig_x=torch.zeros(5), orig_y=torch.zeros(5), orig_z=torch.zeros(5),
    dir_x=torch.ones(5), dir_y=torch.zeros(5), dir_z=torch.zeros(5),
    tmin=torch.ones(5)*2, tmax=torch.ones(5)*6
)

marcher.train()
t1, _ = marcher._sample_fixed_steps(rays)
t2, _ = marcher._sample_fixed_steps(rays)
train_diff = (t1 - t2).abs().mean().item()

marcher.eval()
t3, _ = marcher._sample_fixed_steps(rays)
t4, _ = marcher._sample_fixed_steps(rays)
eval_diff = (t3 - t4).abs().mean().item()

print(f"  Training mode diff: {train_diff:.6f}")
print(f"  Eval mode diff: {eval_diff:.6f}")
assert train_diff > 0, "No jittering in training mode"
assert eval_diff == 0, "Jittering in eval mode"
print("  ✓ PASS - Jittering enabled in training, disabled in eval")

# Test 4: Forward Pass
print("\n[Test 4] End-to-End Forward Pass")
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.config import Config

cfg = Config()
cfg.model.hash_levels = 4
cfg.model.density_activation = 'trunc_exp'
cfg.integrator.perturb = True

device = torch.device('cpu')
encoder, field, rgb_head, _ = create_all(cfg, device)

xyz = torch.randn(10, 3)
viewdir = torch.randn(10, 27)

encoded = encoder(xyz)
sigma, rgb_feat = field(encoded.feat)
rgb = rgb_head(rgb_feat, viewdir)

assert sigma.shape == (10,), f"Wrong sigma shape: {sigma.shape}"
assert rgb.shape == (10, 3), f"Wrong RGB shape: {rgb.shape}"
assert (rgb >= 0).all() and (rgb <= 1).all(), "RGB out of [0,1]"
print(f"  Sigma range: [{sigma.min():.3f}, {sigma.max():.3f}]")
print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
print("  ✓ PASS - Forward pass working correctly")

# Test 5: Optimizer Setup
print("\n[Test 5] Optimizer Configuration")
optimizer = create_optimizer(encoder, field, rgb_head, cfg)
print(f"  Encoder LR: {optimizer.param_groups[0]['lr']}")
print(f"  MLP LR: {optimizer.param_groups[1]['lr']}")
assert optimizer.param_groups[0]['lr'] == cfg.train.lr_encoder
assert optimizer.param_groups[1]['lr'] == cfg.train.lr_mlp
print("  ✓ PASS - Separate learning rates configured")

print("\n" + "="*70)
print("ALL CRITICAL VALIDATIONS PASSED ✓")
print("="*70)
print("\nThe implementation includes all key fixes:")
print("  1. Hash grid uniform initialization [-1e-4, 1e-4]")
print("  2. Truncated exponential density activation")
print("  3. Sigma bias initialization to -1.5")
print("  4. Sample jittering during training")
print("  5. Separate learning rates for encoder/MLP")
print("\nReady to run full convergence test!")

