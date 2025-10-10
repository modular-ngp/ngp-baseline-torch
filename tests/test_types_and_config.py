"""Test types and configuration dataclasses."""
import pytest
import torch
from ngp_baseline_torch.types import RayBatch, EncodedFeat, assert_ray_batch_valid, pad_to_multiple
from ngp_baseline_torch.config import Config, DatasetConfig, ModelConfig


@pytest.mark.quick
def test_config_defaults():
    """Test that configuration has valid defaults."""
    cfg = Config()

    assert cfg.dataset.scene == "lego"
    assert cfg.model.mlp_width == 64
    assert cfg.train.seed == 1337
    assert cfg.precision.accum_dtype == "float32"


@pytest.mark.quick
def test_ray_batch_soa_structure(device):
    """Test RayBatch SoA structure and validation."""
    N = 100

    # Valid ray batch
    rays = RayBatch(
        orig_x=torch.randn(N, device=device),
        orig_y=torch.randn(N, device=device),
        orig_z=torch.randn(N, device=device),
        dir_x=torch.randn(N, device=device),
        dir_y=torch.randn(N, device=device),
        dir_z=torch.randn(N, device=device),
        tmin=torch.ones(N, device=device) * 2.0,
        tmax=torch.ones(N, device=device) * 6.0,
    )

    assert rays.N == N
    # Compare device types, not exact device instances
    assert rays.device.type == device.type
    assert_ray_batch_valid(rays)


@pytest.mark.quick
def test_ray_batch_invalid_length():
    """Test that mismatched SoA lengths are caught."""
    with pytest.raises(AssertionError):
        RayBatch(
            orig_x=torch.randn(100),
            orig_y=torch.randn(100),
            orig_z=torch.randn(100),
            dir_x=torch.randn(100),
            dir_y=torch.randn(100),
            dir_z=torch.randn(99),  # Wrong length
            tmin=torch.ones(100) * 2.0,
            tmax=torch.ones(100) * 6.0,
        )


@pytest.mark.quick
def test_encoded_feat_padding():
    """Test EncodedFeat padding validation."""
    # Valid: multiple of 16
    feat = EncodedFeat(feat=torch.randn(10, 32))
    assert feat.feat.shape[1] == 32

    # Invalid: not multiple of 16
    with pytest.raises(AssertionError):
        EncodedFeat(feat=torch.randn(10, 31))


@pytest.mark.quick
def test_pad_to_multiple():
    """Test padding utility function."""
    assert pad_to_multiple(10, 16) == 16
    assert pad_to_multiple(16, 16) == 16
    assert pad_to_multiple(17, 16) == 32
    assert pad_to_multiple(31, 16) == 32
    assert pad_to_multiple(32, 16) == 32
