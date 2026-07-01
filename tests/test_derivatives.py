"""
Unit tests for the differentiation operators.

These are the mathematical core of PINNTorch. We test each operator
against functions whose derivatives are known analytically, so a
failure here means the autograd plumbing is wrong — not a training issue.
"""

import torch
import pytest

from pinntorch.core.derivatives import diff, gradient, laplacian, divergence, curl


def _grid(n=50, low=0.0, high=1.0):
    """A 1D grid with gradients enabled."""
    x = torch.linspace(low, high, n).reshape(-1, 1)
    x.requires_grad_(True)
    return x


class TestDiff:
    """Test the core `diff` operator against known derivatives."""

    def test_first_derivative_of_square(self):
        # u = x^2  ->  du/dx = 2x
        x = _grid()
        u = x ** 2
        du = diff(u, x, order=1)
        assert torch.allclose(du, 2 * x, atol=1e-5)

    def test_second_derivative_of_square(self):
        # u = x^2  ->  d2u/dx2 = 2
        x = _grid()
        u = x ** 2
        d2u = diff(u, x, order=2)
        assert torch.allclose(d2u, 2 * torch.ones_like(x), atol=1e-5)

    def test_derivative_of_sin(self):
        # u = sin(x)  ->  du/dx = cos(x)
        x = _grid(low=0.0, high=6.28)
        u = torch.sin(x)
        du = diff(u, x, order=1)
        assert torch.allclose(du, torch.cos(x), atol=1e-4)

    def test_second_derivative_of_sin(self):
        # u = sin(x)  ->  d2u/dx2 = -sin(x)
        x = _grid(low=0.0, high=6.28)
        u = torch.sin(x)
        d2u = diff(u, x, order=2)
        assert torch.allclose(d2u, -torch.sin(x), atol=1e-4)

    def test_third_derivative_of_cube(self):
        # u = x^3 -> d3u/dx3 = 6
        x = _grid()
        u = x ** 3
        d3u = diff(u, x, order=3)
        assert torch.allclose(d3u, 6 * torch.ones_like(x), atol=1e-4)

    def test_requires_grad_error(self):
        # Input without requires_grad should raise a clear error
        x = torch.linspace(0, 1, 10).reshape(-1, 1)
        u = x ** 2
        with pytest.raises(ValueError):
            diff(u, x)

    def test_invalid_order(self):
        x = _grid()
        u = x ** 2
        with pytest.raises(ValueError):
            diff(u, x, order=0)


class TestGradient:
    """Test the gradient operator."""

    def test_gradient_single_input(self):
        x = _grid()
        u = x ** 2
        g = gradient(u, x)
        assert torch.allclose(g, 2 * x, atol=1e-5)

    def test_gradient_multiple_inputs(self):
        # u = x*y -> du/dx = y, du/dy = x
        x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        y = torch.tensor([[4.0], [5.0], [6.0]], requires_grad=True)
        u = x * y
        g = gradient(u, [x, y])
        assert torch.allclose(g[0], y, atol=1e-5)  # du/dx = y
        assert torch.allclose(g[1], x, atol=1e-5)  # du/dy = x


class TestLaplacian:
    """Test the Laplacian operator."""

    def test_laplacian_1d(self):
        # u = x^2 -> laplacian = 2
        x = _grid()
        u = x ** 2
        lap = laplacian(u, x)
        assert torch.allclose(lap, 2 * torch.ones_like(x), atol=1e-5)

    def test_laplacian_2d(self):
        # u = x^2 + y^2 -> laplacian = 2 + 2 = 4
        x = torch.tensor([[1.0], [2.0]], requires_grad=True)
        y = torch.tensor([[3.0], [4.0]], requires_grad=True)
        u = x ** 2 + y ** 2
        lap = laplacian(u, [x, y])
        assert torch.allclose(lap, 4 * torch.ones_like(x), atol=1e-5)


class TestDivergence:
    """Test the divergence operator."""

    def test_divergence_2d(self):
        # F = [x, y] -> div = 1 + 1 = 2
        x = torch.tensor([[1.0], [2.0]], requires_grad=True)
        y = torch.tensor([[3.0], [4.0]], requires_grad=True)
        F = [x, y]
        div = divergence(F, [x, y])
        assert torch.allclose(div, 2 * torch.ones_like(x), atol=1e-5)


class TestCurl:
    """Test the curl operator."""

    def test_curl_requires_3d(self):
        x = torch.tensor([[1.0]], requires_grad=True)
        y = torch.tensor([[2.0]], requires_grad=True)
        with pytest.raises(ValueError):
            curl([x, y], [x, y])
