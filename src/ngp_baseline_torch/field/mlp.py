"""MLP for density and color prediction."""
from __future__ import annotations
import torch
import torch.nn as nn


class NGP_MLP(nn.Module):
    """Compact MLP for Instant-NGP style NeRF.

    Takes encoded features and outputs density (sigma) and RGB features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 16,
        activation: str = "relu",
        bias: bool = True
    ):
        """Initialize MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
            output_dim: Output feature dimension (for RGB head)
            activation: Activation function ("relu", "silu", "softplus")
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Select activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)
        elif activation == "softplus":
            self.activation = nn.Softplus(beta=100)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        layers.append(self.activation)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(self.activation)

        self.net = nn.Sequential(*layers)

        # Separate heads for sigma and RGB features
        self.sigma_head = nn.Linear(hidden_dim, 1, bias=bias)
        self.rgb_head = nn.Linear(hidden_dim, output_dim, bias=bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            feat: Input features [B, input_dim]

        Returns:
            sigma: Density [B]
            rgb_feat: RGB features [B, output_dim]
        """
        x = self.net(feat)

        # Sigma with softplus to ensure positivity
        sigma = self.sigma_head(x).squeeze(-1)  # [B]
        sigma = torch.nn.functional.softplus(sigma - 1.0)  # shift for better initialization

        # RGB features (no activation, will be processed by head)
        rgb_feat = self.rgb_head(x)  # [B, output_dim]

        return sigma, rgb_feat


class TinyMLP(nn.Module):
    """Ultra-compact MLP for maximum speed.

    Single hidden layer with minimal parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.sigma_head = nn.Linear(hidden_dim, 1, bias=False)
        self.rgb_head = nn.Linear(hidden_dim, output_dim, bias=False)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(feat)
        sigma = torch.nn.functional.softplus(self.sigma_head(x).squeeze(-1) - 1.0)
        rgb_feat = self.rgb_head(x)
        return sigma, rgb_feat

