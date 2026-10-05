"""Physics-informed loss function for battery degradation models.

Combines standard empirical loss with thermodynamic constraints:
1. Irreversibility / Monotonicity penalty: SOH must not spontaneously increase.
2. Thermodynamic boundary penalty: SOH must reside in physically plausible bounds [0, 1.05].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):
    """Loss function incorporating electrochemical domain knowledge into neural networks.

    L_total = L_data + lambda_mono * L_monotonicity + lambda_bound * L_boundary
    """

    def __init__(
        self,
        lambda_mono: float = 0.5,
        lambda_bound: float = 1.0,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_bound = lambda_bound

        if loss_type == "huber":
            self.base_loss = nn.SmoothL1Loss()
        else:
            self.base_loss = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        cycle_order: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute physics-informed composite loss.

        Args:
            y_pred: Predicted SOH (B, 1) or (B,).
            y_true: Ground truth SOH (B, 1) or (B,).
            cycle_order: Optional sequence index or cycle number to enforce monotonic decay.

        Returns:
            Dictionary with 'loss' (total scalar tensor), 'data_loss', 'mono_loss', 'bound_loss'.
        """
        yp = y_pred.view(-1)
        yt = y_true.view(-1)

        # 1. Empirical data fidelity loss
        data_loss = self.base_loss(yp, yt)

        # 2. Monotonicity constraint (irreversibility of capacity fade)
        if len(yp) > 1:
            if cycle_order is not None:
                # Sort predictions by cycle index to evaluate chronological delta
                idx = torch.argsort(cycle_order.view(-1))
                yp_sorted = yp[idx]
            else:
                yp_sorted = yp

            # SOH(t) - SOH(t-1): positive values indicate impossible capacity gain
            diffs = yp_sorted[1:] - yp_sorted[:-1]
            unphysical_gains = torch.relu(diffs)
            mono_loss = torch.mean(unphysical_gains**2)
        else:
            mono_loss = torch.tensor(0.0, device=yp.device)

        # 3. Thermodynamic boundary constraint: SOH in [0.0, 1.05]
        upper_violation = torch.relu(yp - 1.05)
        lower_violation = torch.relu(-yp)
        bound_loss = torch.mean(upper_violation**2 + lower_violation**2)

        total_loss = data_loss + (self.lambda_mono * mono_loss) + (self.lambda_bound * bound_loss)

        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "mono_loss": mono_loss,
            "bound_loss": bound_loss,
        }
