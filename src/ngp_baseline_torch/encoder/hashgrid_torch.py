"""Multi-resolution hash grid encoder (PyTorch implementation)."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..types import EncodedFeat, pad_to_multiple


class HashGridEncoder(nn.Module):
    """Multi-resolution hash grid encoder with trilinear interpolation.

    This is a pure PyTorch implementation approximating Instant-NGP's hash encoding.
    Uses dense tensors for simplicity; can be replaced with CUDA kernel later.
    """

    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        aabb: tuple[float, float, float, float, float, float] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    ):
        """Initialize hash grid encoder.

        Args:
            num_levels: Number of resolution levels
            base_resolution: Coarsest resolution
            max_resolution: Finest resolution
            features_per_level: Feature vector size per level
            log2_hashmap_size: Log2 of hash table size
            aabb: Axis-aligned bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
        """
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size

        # AABB for coordinate normalization
        self.register_buffer('aabb_min', torch.tensor([aabb[0], aabb[1], aabb[2]], dtype=torch.float32))
        self.register_buffer('aabb_max', torch.tensor([aabb[3], aabb[4], aabb[5]], dtype=torch.float32))

        # Calculate resolution for each level using geometric progression
        b = torch.exp2(torch.log2(torch.tensor(max_resolution / base_resolution)) / (num_levels - 1))
        resolutions = (base_resolution * b ** torch.arange(num_levels)).int()
        self.register_buffer('resolutions', resolutions)

        # Create hash tables for each level
        # For simplicity, use dense tables when resolution^3 < hashmap_size, else hash
        self.hash_tables = nn.ParameterList()
        for level in range(num_levels):
            res = resolutions[level].item()
            grid_size = min(res ** 3, self.hashmap_size)

            # Initialize with small random values
            table = nn.Parameter(torch.randn(grid_size, features_per_level) * 1e-4)
            self.hash_tables.append(table)

        # Output dimension (padded)
        out_dim = num_levels * features_per_level
        self.out_dim_padded = pad_to_multiple(out_dim, 16)
        self.out_dim_raw = out_dim

    def _hash_index(self, coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """Hash 3D coordinates to table indices.

        Args:
            coords: Integer grid coordinates [B, 8, 3]
            resolution: Grid resolution

        Returns:
            Hash indices [B, 8]
        """
        # Primes for hashing (from Instant-NGP)
        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64, device=coords.device)

        # Hash function: (x * p1 ^ y * p2 ^ z * p3) % table_size
        coords_i64 = coords.long()
        hashed = (coords_i64[..., 0] * primes[0]) ^ \
                 (coords_i64[..., 1] * primes[1]) ^ \
                 (coords_i64[..., 2] * primes[2])

        grid_size = min(resolution ** 3, self.hashmap_size)
        return hashed % grid_size

    def forward(self, xyz: torch.Tensor) -> EncodedFeat:
        """Encode coordinates with multi-resolution hash grid.

        Args:
            xyz: World coordinates [B, 3]

        Returns:
            EncodedFeat with concatenated features from all levels
        """
        B = xyz.shape[0]
        device = xyz.device
        dtype = xyz.dtype

        # Normalize coordinates to [0, 1]
        xyz_normalized = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min)
        xyz_normalized = torch.clamp(xyz_normalized, 0.0, 1.0)

        level_features = []

        for level in range(self.num_levels):
            res = self.resolutions[level].item()

            # Scale to grid coordinates [0, res]
            xyz_grid = xyz_normalized * res

            # Get integer coordinates for 8 corners of cube
            xyz_floor = torch.floor(xyz_grid).int()
            xyz_ceil = xyz_floor + 1

            # Trilinear interpolation weights
            xyz_frac = xyz_grid - xyz_floor.float()

            # Clamp to grid bounds
            xyz_floor = torch.clamp(xyz_floor, 0, res - 1)
            xyz_ceil = torch.clamp(xyz_ceil, 0, res)

            # Generate 8 corner coordinates [B, 8, 3]
            corners = torch.stack([
                torch.stack([xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]], dim=-1),
                torch.stack([xyz_ceil[:, 0],  xyz_floor[:, 1], xyz_floor[:, 2]], dim=-1),
                torch.stack([xyz_floor[:, 0], xyz_ceil[:, 1],  xyz_floor[:, 2]], dim=-1),
                torch.stack([xyz_ceil[:, 0],  xyz_ceil[:, 1],  xyz_floor[:, 2]], dim=-1),
                torch.stack([xyz_floor[:, 0], xyz_floor[:, 1], xyz_ceil[:, 2]], dim=-1),
                torch.stack([xyz_ceil[:, 0],  xyz_floor[:, 1], xyz_ceil[:, 2]], dim=-1),
                torch.stack([xyz_floor[:, 0], xyz_ceil[:, 1],  xyz_ceil[:, 2]], dim=-1),
                torch.stack([xyz_ceil[:, 0],  xyz_ceil[:, 1],  xyz_ceil[:, 2]], dim=-1),
            ], dim=1)  # [B, 8, 3]

            # Hash corners to get indices
            indices = self._hash_index(corners, res)  # [B, 8]

            # Lookup features from hash table
            table = self.hash_tables[level]
            corner_features = table[indices]  # [B, 8, F]

            # Trilinear interpolation weights for 8 corners
            wx = xyz_frac[:, 0:1]  # [B, 1]
            wy = xyz_frac[:, 1:2]
            wz = xyz_frac[:, 2:3]

            weights = torch.stack([
                (1 - wx) * (1 - wy) * (1 - wz),
                wx * (1 - wy) * (1 - wz),
                (1 - wx) * wy * (1 - wz),
                wx * wy * (1 - wz),
                (1 - wx) * (1 - wy) * wz,
                wx * (1 - wy) * wz,
                (1 - wx) * wy * wz,
                wx * wy * wz,
            ], dim=1)  # [B, 8, 1]

            # Weighted sum
            level_feat = (corner_features * weights).sum(dim=1)  # [B, F]
            level_features.append(level_feat)

        # Concatenate all levels
        feat = torch.cat(level_features, dim=-1)  # [B, num_levels * F]

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


def encode(xyz: torch.Tensor, state: HashGridEncoder) -> EncodedFeat:
    """Functional interface for hash grid encoding.

    Args:
        xyz: Input coordinates [B, 3]
        state: Initialized HashGridEncoder

    Returns:
        EncodedFeat with encoded features
    """
    return state(xyz)

