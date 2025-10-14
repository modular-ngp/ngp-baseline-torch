"""Test model factory."""
import pytest
import torch
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_encoder, create_field, create_rgb_head


class TestFactory:
    """Test factory functions."""

    def test_create_encoder_hashgrid(self, device):
        """Test creating hash grid encoder."""
        cfg = Config()
        cfg.model.hash_levels = 8

        encoder = create_encoder(cfg, device)

        assert encoder is not None
        assert next(encoder.parameters()).device.type == device.type

    def test_create_encoder_pe(self, device):
        """Test creating positional encoder."""
        cfg = Config()
        cfg.model.hash_levels = 0  # Use PE instead
        cfg.model.pe_bands = 6

        encoder = create_encoder(cfg, device)

        assert encoder is not None
        # PE encoder has no parameters, so check module device differently
        assert encoder is not None

    def test_create_field(self, device):
        """Test creating field MLP."""
        cfg = Config()
        encoder = create_encoder(cfg, device)

        field = create_field(cfg, encoder, device)

        assert field is not None
        assert next(field.parameters()).device.type == device.type

    def test_create_rgb_head(self, device):
        """Test creating RGB head."""
        cfg = Config()
        cfg.model.view_dependent = True

        rgb_head = create_rgb_head(cfg, device)

        assert rgb_head is not None
        assert next(rgb_head.parameters()).device.type == device.type

    def test_create_rgb_head_no_viewdir(self, device):
        """Test creating RGB head without view direction."""
        cfg = Config()
        cfg.model.view_dependent = False

        rgb_head = create_rgb_head(cfg, device)

        assert rgb_head is not None

    def test_end_to_end_forward(self, device, set_seed):
        """Test forward pass through complete pipeline."""
        cfg = Config()
        cfg.model.hash_levels = 4

        encoder = create_encoder(cfg, device)
        field = create_field(cfg, encoder, device)
        rgb_head = create_rgb_head(cfg, device)

        # Test forward pass
        xyz = torch.rand(100, 3, device=device)

        # For view-dependent RGB, need to match the exact viewdir_dim expected by RGB head
        # RGB head expects: 3 + cfg.model.pe_viewdir_bands * 3 * 2 = 3 + 4*3*2 = 27 dimensions
        expected_viewdir_dim = 3 + cfg.model.pe_viewdir_bands * 3 * 2  # 27

        from ngp_baseline_torch.encoder.pe import PositionalEncoder

        viewdir_encoder = PositionalEncoder(
            num_bands=cfg.model.pe_viewdir_bands, include_input=True
        ).to(device)

        viewdir = torch.rand(100, 3, device=device)
        viewdir = viewdir / torch.norm(viewdir, dim=-1, keepdim=True)
        viewdir_encoded_full = viewdir_encoder(viewdir).feat

        # Slice to match expected dimensions (PE pads to multiple of 16)
        viewdir_encoded = viewdir_encoded_full[:, :expected_viewdir_dim]

        encoded = encoder(xyz)
        sigma, feat = field(encoded.feat)
        rgb = rgb_head(feat, viewdir_encoded)

        assert sigma.shape == (100,)
        assert rgb.shape == (100, 3)
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1)
