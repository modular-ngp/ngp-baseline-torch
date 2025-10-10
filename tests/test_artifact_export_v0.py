"""Test artifact export functionality."""
import pytest
import torch
import tempfile
from pathlib import Path
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all
from ngp_baseline_torch.artifact import export, load


@pytest.mark.quick
def test_artifact_export_files(device):
    """Test that export creates all required files."""
    cfg = Config()
    cfg.model.hash_levels = 4

    encoder, field, rgb_head, occ_grid = create_all(cfg, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        export(encoder, field, rgb_head, cfg, tmpdir, occ_grid)

        out_path = Path(tmpdir)

        # Check required files exist
        assert (out_path / "meta.json").exists()
        assert (out_path / "topo.json").exists()
        assert (out_path / "params.pth").exists()
        assert (out_path / "hashgrid.pth").exists()


@pytest.mark.quick
def test_artifact_metadata_fields(device):
    """Test that metadata contains required fields."""
    import json

    cfg = Config()
    encoder, field, rgb_head, _ = create_all(cfg, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        export(encoder, field, rgb_head, cfg, tmpdir)

        with open(Path(tmpdir) / "meta.json", 'r') as f:
            meta = json.load(f)

        # Check required fields
        assert 'version' in meta
        assert 'endianness' in meta
        assert 'alignment_bytes' in meta
        assert 'arch_req' in meta
        assert 'precision' in meta
        assert 'seed' in meta

        # Check alignment is reasonable
        assert meta['alignment_bytes'] >= 128


@pytest.mark.quick
def test_artifact_load_roundtrip(device):
    """Test that exported artifact can be loaded back."""
    cfg = Config()
    cfg.model.hash_levels = 4

    encoder, field, rgb_head, _ = create_all(cfg, device)

    # Get initial output
    xyz = torch.randn(10, 3, device=device)
    with torch.no_grad():
        encoded1 = encoder(xyz)
        sigma1, _ = field(encoded1.feat)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        export(encoder, field, rgb_head, cfg, tmpdir)

        # Create new model and load
        encoder2, field2, rgb_head2, _ = create_all(cfg, device)
        encoder2, field2, rgb_head2, _ = load(tmpdir, encoder2, field2, rgb_head2, device)

        # Check outputs match
        with torch.no_grad():
            encoded2 = encoder2(xyz)
            sigma2, _ = field2(encoded2.feat)

        assert torch.allclose(sigma1, sigma2, atol=1e-6)

