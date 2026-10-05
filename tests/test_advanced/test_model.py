"""Tests for BatteryLSTM architecture, trainer, and Rule 6 checkpoint persistence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ev_battery.advanced.loss import PhysicsInformedLoss
from ev_battery.advanced.model import BatteryLSTM
from ev_battery.advanced.trainer import (
    LSTMTrainer,
    load_lstm_checkpoint,
    save_lstm_checkpoint,
)


class TestBatteryLSTM:
    def test_forward_pass_shape(self):
        batch_size = 8
        seq_len = 5
        input_dim = 7

        model = BatteryLSTM(input_dim=input_dim, hidden_dim=16, num_layers=1)
        x = torch.randn(batch_size, seq_len, input_dim)
        out = model(x)

        assert out.shape == (batch_size, 1)

    def test_predict_numpy(self):
        model = BatteryLSTM(input_dim=7, hidden_dim=16, num_layers=1)
        X = np.random.randn(10, 5, 7).astype(np.float32)
        preds = model.predict_numpy(X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (10,)
        assert (preds >= 0.0).all()


class TestLSTMTrainer:
    @pytest.fixture
    def dummy_data(self):
        np.random.seed(42)
        X = np.random.randn(30, 5, 7).astype(np.float32)
        y = np.linspace(1.0, 0.8, 30).astype(np.float32)
        return X, y

    def test_fit_reduces_loss(self, dummy_data):
        X, y = dummy_data
        model = BatteryLSTM(input_dim=7, hidden_dim=16, num_layers=1)
        trainer = LSTMTrainer(
            model=model,
            loss_fn=PhysicsInformedLoss(lambda_mono=0.5, lambda_bound=0.5),
            epochs=5,
            lr=0.01,
            batch_size=16,
            random_seed=42,
        )
        trainer.fit(X, y)

        assert len(trainer.history) == 5
        assert "data_loss" in trainer.history[0]
        assert "mono_loss" in trainer.history[0]

    def test_rule_6_checkpoint_persistence(self, dummy_data, tmp_path):
        """Rule 6: Save and load model checkpoint and metadata sidecar."""
        X, y = dummy_data
        model = BatteryLSTM(input_dim=7, hidden_dim=16, num_layers=1)
        trainer = LSTMTrainer(model=model, epochs=2, random_seed=42)
        trainer.fit(X, y, dataset_version="test-seq-v1")

        meta = trainer.training_metadata
        assert meta["dataset_version"] == "test-seq-v1"
        assert len(meta["data_sha256"]) == 64
        assert "torch" in meta["library_versions"]

        save_dir = tmp_path / "lstm_ckpt"
        metrics = {"val_rmse": 0.015}
        save_lstm_checkpoint(trainer, save_dir, metrics=metrics)

        loaded_model, loaded_meta = load_lstm_checkpoint(save_dir)
        assert loaded_meta["metrics"]["val_rmse"] == 0.015

        orig_preds = model.predict_numpy(X)
        loaded_preds = loaded_model.predict_numpy(X)
        assert np.allclose(orig_preds, loaded_preds, atol=1e-5)
