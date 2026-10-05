"""PyTorch LSTM architecture for battery degradation sequence modeling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BatteryLSTM(nn.Module):
    """Recurrent LSTM neural network for SOH degradation sequence modeling.

    Maps sliding sequence windows of cycle features (N, W, D) to current SOH.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Predicted SOH scalar tensor of shape (batch_size, 1).
        """
        lstm_out, _ = self.lstm(x)
        # Take the output representation of the last cycle in the window
        last_step = lstm_out[:, -1, :]
        out = self.regressor(last_step)
        return out

    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        """Convenience method for evaluation from numpy array."""
        self.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X.astype(np.float32))
            preds = self.forward(tensor_x).view(-1).cpu().numpy()
        return np.clip(preds, 0.0, 1.2)
