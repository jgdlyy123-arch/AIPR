"""
AIPR Regularizer (PyTorch) — Adaptive Isometric Policy Regularization

DfI(W) = ||W^T W - I||_F^2

λ_t = λ_0 · sigmoid((DfI_avg − τ) / α)

Applied to linear weight matrices (and 2D-unfolded conv kernels) of Q-networks.
"""

import math
import torch
import torch.nn as nn


def compute_dfi(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the Deviation from Isometry for a weight tensor.

    For 2D weights (out_features, in_features): DfI = ||W^T W - I||_F^2
    For 4D conv weights (out_ch, in_ch, kH, kW): reshape to 2D first.
    """
    if weight.ndim == 4:
        # Conv layer: reshape (out_ch, in_ch*kH*kW)
        w = weight.view(weight.size(0), -1)
    elif weight.ndim == 2:
        w = weight
    else:
        return weight.new_zeros(1).squeeze()

    # Use the smaller dimension for the identity size to keep DfI meaningful
    d = min(w.shape)
    if w.shape[0] <= w.shape[1]:
        # W is fat (or square): compute W W^T - I  (shape: out × out)
        gram = w @ w.t()
    else:
        # W is tall: compute W^T W - I  (shape: in × in)
        gram = w.t() @ w

    identity = torch.eye(d, device=w.device, dtype=w.dtype)
    diff = gram - identity
    return (diff * diff).sum()


def compute_aipr_loss(
    model: nn.Module,
    lambda_0: float = 0.01,
    tau: float = 1.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the adaptive AIPR regularization loss for a PyTorch model.

    Returns (aipr_loss, info_dict).
    """
    dfi_values = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            dfi = compute_dfi(module.weight)
            dfi_values.append(dfi)

    if not dfi_values:
        device = next(model.parameters()).device
        zero = torch.zeros(1, device=device).squeeze()
        return zero, {"aipr/lambda": 0.0, "aipr/dfi_mean": 0.0, "aipr/loss": 0.0}

    dfi_sum = sum(dfi_values)
    dfi_avg = dfi_sum / len(dfi_values)

    # Adaptive lambda: near 0 when DfI < tau, strong when DfI >> tau
    lambda_t = lambda_0 * torch.sigmoid((dfi_avg - tau) / alpha)
    aipr_loss = lambda_t * dfi_sum

    info = {
        "aipr/lambda": float(lambda_t.item()),
        "aipr/dfi_mean": float(dfi_avg.item()),
        "aipr/loss": float(aipr_loss.item()),
    }
    return aipr_loss, info
