"""Output heads for converting features to RGB."""
from __future__ import annotations
import torch
import torch.nn as nn


class RGBHead(nn.Module):
    """Convert RGB features to final RGB values.

    Optionally incorporates view direction for view-dependent effects.
    """

    def __init__(
        self,
        rgb_feat_dim: int,
        view_dependent: bool = False,
        viewdir_dim: int = 27,  # 3 + 4*3*2 for PE with 4 bands
    ):
        """Initialize RGB head.

        Args:
            rgb_feat_dim: Dimension of RGB features from MLP
            view_dependent: Whether to use view direction
            viewdir_dim: Dimension of encoded view direction (if used)
        """
        super().__init__()
        self.view_dependent = view_dependent

        if view_dependent:
            # Combine RGB features with view direction
            self.rgb_net = nn.Sequential(
                nn.Linear(rgb_feat_dim + viewdir_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 3),
                nn.Sigmoid()
            )
        else:
            # Simple linear projection
            self.rgb_net = nn.Sequential(
                nn.Linear(rgb_feat_dim, 3),
                nn.Sigmoid()
            )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb_feat: torch.Tensor, viewdir: torch.Tensor | None = None) -> torch.Tensor:
        """Convert features to RGB.

        Args:
            rgb_feat: RGB features [B, rgb_feat_dim]
            viewdir: Optional encoded view directions [B, viewdir_dim]

        Returns:
            rgb: RGB values [B, 3] in range [0, 1]
        """
        if self.view_dependent:
            assert viewdir is not None, "View direction required for view-dependent head"
            feat = torch.cat([rgb_feat, viewdir], dim=-1)
        else:
            feat = rgb_feat

        return self.rgb_net(feat)


def rgb(rgb_feat: torch.Tensor, viewdir: torch.Tensor | None = None,
        head: RGBHead | None = None) -> torch.Tensor:
    """Functional interface for RGB head.

    Args:
        rgb_feat: RGB features [B, F]
        viewdir: Optional view directions [B, V]
        head: RGBHead instance, if None creates default

    Returns:
        RGB values [B, 3]
    """
    if head is None:
        head = RGBHead(rgb_feat.shape[-1], view_dependent=(viewdir is not None))
        head.to(rgb_feat.device)

    return head(rgb_feat, viewdir)

