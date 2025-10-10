"""Positional encoding (frequency encoding)."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..types import EncodedFeat, pad_to_multiple


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for coordinates.

    Encodes xyz with sin/cos at multiple frequency bands.
    """

    def __init__(self, num_bands: int = 10, include_input: bool = True):
        """Initialize positional encoder.

        Args:
            num_bands: Number of frequency bands (L in paper)
            include_input: Whether to include raw input coordinates
        """
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer('freq_bands', freq_bands)

        # Calculate output dimension and pad to multiple of 16
        out_dim = 3 if include_input else 0
        out_dim += num_bands * 3 * 2  # sin and cos for each band and xyz
        self.out_dim_padded = pad_to_multiple(out_dim, 16)
        self.out_dim_raw = out_dim

    def forward(self, xyz: torch.Tensor) -> EncodedFeat:
        """Encode xyz coordinates.

        Args:
            xyz: Input coordinates [B, 3]

        Returns:
            EncodedFeat with encoded features [B, F] where F is padded to multiple of 16
        """
        B = xyz.shape[0]
        device = xyz.device
        dtype = xyz.dtype

        outputs = []

        # Include raw input if specified
        if self.include_input:
            outputs.append(xyz)

        # Apply sinusoidal encoding for each frequency band
        # xyz: [B, 3] -> [B, 3, 1]
        # freq_bands: [L] -> [1, 1, L]
        # xyz_freq: [B, 3, L]
        xyz_freq = xyz.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)

        # Flatten to [B, 3*L]
        xyz_freq = xyz_freq.reshape(B, -1)

        # Apply sin and cos
        outputs.append(torch.sin(xyz_freq))
        outputs.append(torch.cos(xyz_freq))

        # Concatenate all features
        feat = torch.cat(outputs, dim=-1)  # [B, out_dim_raw]

        # Pad to multiple of 16
        if self.out_dim_padded > self.out_dim_raw:
            padding = torch.zeros(B, self.out_dim_padded - self.out_dim_raw,
                                device=device, dtype=dtype)
            feat = torch.cat([feat, padding], dim=-1)

        return EncodedFeat(feat=feat)

    @property
    def output_dim(self) -> int:
        """Get padded output dimension."""
        return self.out_dim_padded


def encode(xyz: torch.Tensor, num_bands: int = 10, include_input: bool = True) -> EncodedFeat:
    """Functional interface for positional encoding.

    Args:
        xyz: Input coordinates [B, 3]
        num_bands: Number of frequency bands
        include_input: Whether to include raw input

    Returns:
        EncodedFeat with encoded features
    """
    encoder = PositionalEncoder(num_bands, include_input)
    encoder.to(xyz.device)
    return encoder(xyz)

