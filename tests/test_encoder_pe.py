"""Test positional encoding."""
import pytest
import torch
import numpy as np
from ngp_baseline_torch.encoder.pe import PositionalEncoder, encode


@pytest.mark.quick
def test_pe_output_shape(device):
    """Test positional encoding output shape."""
    B = 100
    num_bands = 10

    encoder = PositionalEncoder(num_bands=num_bands, include_input=True)
    encoder.to(device)

    xyz = torch.randn(B, 3, device=device)
    encoded = encoder(xyz)

    # Check output is EncodedFeat
    assert encoded.feat.ndim == 2
    assert encoded.feat.shape[0] == B
    assert encoded.feat.shape[1] % 16 == 0


@pytest.mark.quick
def test_pe_numpy_reference(device):
    """Compare PE implementation against numpy reference."""
    B = 10
    num_bands = 4

    encoder = PositionalEncoder(num_bands=num_bands, include_input=True)
    encoder.to(device)

    xyz = torch.randn(B, 3, device=device)
    encoded = encoder(xyz)

    # Numpy reference implementation
    xyz_np = xyz.cpu().numpy()
    output_np = [xyz_np]

    for i in range(num_bands):
        freq = 2.0 ** i
        # Match encoder order: sin then cos for all components
        xyz_freq = xyz_np * freq
        output_np.append(np.sin(xyz_freq))
        output_np.append(np.cos(xyz_freq))

    expected = np.concatenate(output_np, axis=-1)

    # Compare (ignoring padding)
    actual = encoded.feat[:, :expected.shape[1]].cpu().numpy()
    # Use slightly larger tolerance for accumulated floating point errors
    assert np.allclose(actual, expected, atol=1e-5)


@pytest.mark.quick
def test_pe_bands_scaling():
    """Test that doubling bands increases output dimension."""
    encoder_small = PositionalEncoder(num_bands=5, include_input=True)
    encoder_large = PositionalEncoder(num_bands=10, include_input=True)

    assert encoder_large.out_dim_raw > encoder_small.out_dim_raw
