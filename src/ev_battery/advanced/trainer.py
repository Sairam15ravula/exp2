"""Deterministic trainer and checkpoint persistence for BatteryLSTM (Rule 6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ev_battery.advanced.loss import PhysicsInformedLoss
from ev_battery.advanced.model import BatteryLSTM


def compute_tensor_sha256(X: np.ndarray, y: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of sequence data (Rule 6)."""
    hasher = hashlib.sha256()
    hasher.update(X.tobytes())
    hasher.update(y.tobytes())
    return hasher.hexdigest()


class LSTMTrainer:
    """Trainer for BatteryLSTM with physics-informed loss and Rule 6 metadata tracking."""

    def __init__(
        self,
        model: BatteryLSTM,
        loss_fn: nn.Module | None = None,
        lr: float = 0.005,
        epochs: int = 50,
        batch_size: int = 32,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else PhysicsInformedLoss()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        # Rule 6: Fix random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        self.history: list[dict[str, float]] = []
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_version: str = "v1.0-seq-synthetic",
    ) -> LSTMTrainer:
        """Train the LSTM model on sequence windows."""
        self.model.train()

        tensor_x = torch.from_numpy(X.astype(np.float32))
        tensor_y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_mono_loss = 0.0
            n_batches = 0

            for bx, by in loader:
                self.optimizer.zero_grad()
                pred = self.model(bx)

                if isinstance(self.loss_fn, PhysicsInformedLoss):
                    loss_dict = self.loss_fn(pred, by)
                    loss = loss_dict["loss"]
                    d_loss = float(loss_dict["data_loss"].item())
                    m_loss = float(loss_dict["mono_loss"].item())
                else:
                    loss = self.loss_fn(pred, by)
                    d_loss = float(loss.item())
                    m_loss = 0.0

                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.item())
                epoch_data_loss += d_loss
                epoch_mono_loss += m_loss
                n_batches += 1

            self.history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss / max(1, n_batches),
                "data_loss": epoch_data_loss / max(1, n_batches),
                "mono_loss": epoch_mono_loss / max(1, n_batches),
            })

        import sklearn

        self.training_metadata = {
            "dataset_version": dataset_version,
            "data_sha256": compute_tensor_sha256(X, y),
            "random_seed": self.random_seed,
            "hyperparameters": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            },
            "final_loss": self.history[-1]["loss"] if self.history else None,
            "library_versions": {
                "torch": torch.__version__,
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
            },
        }

        return self


def save_lstm_checkpoint(
    trainer: LSTMTrainer,
    directory: str | Path,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save PyTorch weights and Rule 6 metadata sidecar."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)

    weights_path = dest / "battery_lstm.pt"
    torch.save(trainer.model.state_dict(), weights_path)

    metadata = dict(trainer.training_metadata)
    if metrics:
        metadata["metrics"] = metrics

    meta_path = dest / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return dest


def load_lstm_checkpoint(directory: str | Path) -> tuple[BatteryLSTM, dict[str, Any]]:
    """Load BatteryLSTM from checkpoint directory."""
    dest = Path(directory)
    weights_path = dest / "battery_lstm.pt"
    meta_path = dest / "metadata.json"

    if not weights_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint or metadata missing in {directory}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    hp = metadata["hyperparameters"]
    model = BatteryLSTM(
        input_dim=hp["input_dim"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    return model, metadata
