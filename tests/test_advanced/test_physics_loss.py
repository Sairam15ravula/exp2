"""Tests for Physics-Informed Loss module."""

from __future__ import annotations

import pytest
import torch

from ev_battery.advanced.loss import PhysicsInformedLoss


class TestPhysicsInformedLoss:
    def test_monotonic_decay_has_zero_penalty(self):
        """When predicted SOH strictly decreases (physically valid), mono_loss is 0."""
        loss_fn = PhysicsInformedLoss(lambda_mono=1.0, lambda_bound=1.0)

        # Monotonically decaying predictions
        y_pred = torch.tensor([1.0, 0.95, 0.90, 0.85, 0.80])
        y_true = torch.tensor([1.0, 0.95, 0.90, 0.85, 0.80])

        out = loss_fn(y_pred, y_true)
        assert out["mono_loss"].item() == pytest.approx(0.0)
        assert out["bound_loss"].item() == pytest.approx(0.0)
        assert out["data_loss"].item() == pytest.approx(0.0)
        assert out["loss"].item() == pytest.approx(0.0)

    def test_unphysical_capacity_gain_penalized(self):
        """When predicted SOH increases (capacity gain), mono_loss > 0."""
        loss_fn = PhysicsInformedLoss(lambda_mono=2.0, lambda_bound=0.0)

        # Non-monotonic predictions (jump from 0.80 back to 0.90)
        y_pred = torch.tensor([0.90, 0.80, 0.90])
        y_true = torch.tensor([0.90, 0.85, 0.80])

        out = loss_fn(y_pred, y_true)
        assert out["mono_loss"].item() > 0.0
        assert out["loss"].item() > out["data_loss"].item()

    def test_boundary_violations_penalized(self):
        """Predictions > 1.05 or < 0.0 incur boundary loss."""
        loss_fn = PhysicsInformedLoss(lambda_mono=0.0, lambda_bound=1.0)

        # Predictions exceeding physical boundaries
        y_pred = torch.tensor([1.20, -0.10])
        y_true = torch.tensor([1.00, 0.00])

        out = loss_fn(y_pred, y_true)
        assert out["bound_loss"].item() > 0.0
        assert out["loss"].item() > out["data_loss"].item()

    def test_gradient_flow_through_physics_loss(self):
        """Gradients propagate through both data and physics penalty terms."""
        loss_fn = PhysicsInformedLoss(lambda_mono=1.0, lambda_bound=1.0)

        y_pred = torch.tensor([0.80, 0.90], requires_grad=True)
        y_true = torch.tensor([0.85, 0.80])

        out = loss_fn(y_pred, y_true)
        out["loss"].backward()

        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
