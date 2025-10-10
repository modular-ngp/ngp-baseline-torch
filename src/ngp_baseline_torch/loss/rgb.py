"""RGB loss functions."""
from __future__ import annotations
import torch


def l2(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """L2 (MSE) loss for RGB prediction.

    Args:
        pred: Predicted RGB [N, 3]
        target: Target RGB [N, 3]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss scalar or per-sample losses
    """
    loss = torch.nn.functional.mse_loss(pred, target, reduction=reduction)
    return loss


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1, reduction: str = 'mean') -> torch.Tensor:
    """Huber loss for RGB prediction (more robust to outliers).

    Args:
        pred: Predicted RGB [N, 3]
        target: Target RGB [N, 3]
        delta: Threshold for switching from L2 to L1
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss scalar or per-sample losses
    """
    return torch.nn.functional.huber_loss(pred, target, delta=delta, reduction=reduction)


def l1(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """L1 loss for RGB prediction.

    Args:
        pred: Predicted RGB [N, 3]
        target: Target RGB [N, 3]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss scalar or per-sample losses
    """
    return torch.nn.functional.l1_loss(pred, target, reduction=reduction)

