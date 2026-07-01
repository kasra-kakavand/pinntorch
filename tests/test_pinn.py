"""
Unit tests for the PINN class.

These test the *machinery* — construction, training loop, prediction,
output shapes — not solution accuracy (the examples demonstrate accuracy).
A tiny network and few epochs keep these fast; we only check that things
run and produce sane shapes/finite values.
"""

import torch
import pytest

import pinntorch as pt


def _simple_pinn():
    """A minimal heat-equation PINN for testing machinery."""
    def pde(u, x, t):
        u_t = pt.diff(u, t, order=1)
        u_xx = pt.diff(u, x, order=2)
        return u_t - 0.1 * u_xx

    return pt.PINN(
        pde=pde,
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        initial_condition=lambda x: torch.sin(torch.pi * x),
        boundary_condition=lambda t: torch.zeros_like(t),
        hidden_layers=[16, 16],
        activation="tanh",
    )


class TestConstruction:
    def test_builds(self):
        model = _simple_pinn()
        assert model.dim == 2
        assert model.coordinate_names == ["x", "t"]

    def test_network_has_parameters(self):
        model = _simple_pinn()
        assert model.network.count_parameters() > 0


class TestPrediction:
    def test_predict_returns_tensor(self):
        model = _simple_pinn()
        out = model.predict(x=0.5, t=0.5)
        assert isinstance(out, torch.Tensor)

    def test_predict_is_finite(self):
        model = _simple_pinn()
        out = model.predict(x=0.5, t=0.5)
        assert torch.isfinite(out).all()

    def test_predict_missing_coordinate_raises(self):
        model = _simple_pinn()
        with pytest.raises(ValueError):
            model.predict(x=0.5)  # missing t


class TestTraining:
    def test_train_runs_and_reduces_loss(self):
        model = _simple_pinn()
        history = model.train(epochs=200, verbose=False, print_every=1000)
        # loss should be recorded for every epoch
        assert len(history["total_loss"]) == 200
        # and it should generally go down: last 20 avg < first 20 avg
        first = sum(history["total_loss"][:20]) / 20
        last = sum(history["total_loss"][-20:]) / 20
        assert last < first

    def test_history_has_all_components(self):
        model = _simple_pinn()
        history = model.train(epochs=50, verbose=False, print_every=1000)
        for key in ["total_loss", "pde_loss", "ic_loss", "bc_loss"]:
            assert key in history
            assert len(history[key]) == 50

    def test_lbfgs_phase_runs(self):
        model = _simple_pinn()
        history = model.train(
            epochs=50, lbfgs_steps=20, verbose=False, print_every=1000
        )
        # LBFGS appends extra entries beyond the 50 Adam epochs
        assert len(history["total_loss"]) > 50
        assert torch.isfinite(torch.tensor(history["total_loss"][-1]))
