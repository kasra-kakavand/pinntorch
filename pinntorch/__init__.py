"""
PINNTorch: Physics-Informed Neural Networks, made simple.

A Python library for solving partial differential equations using
neural networks with PyTorch.

Example:
    >>> import pinntorch as pt
    >>> import torch
    >>>
    >>> def heat_equation(u, x, t):
    ...     u_t = pt.diff(u, t)
    ...     u_xx = pt.diff(u, x, order=2)
    ...     return u_t - 0.1 * u_xx
    >>>
    >>> model = pt.PINN(
    ...     pde=heat_equation,
    ...     domain={'x': (0, 1), 't': (0, 1)},
    ... )
    >>> model.train(epochs=10000)
"""

from pinntorch.version import __version__

# Core API
from pinntorch.core.pinn import PINN
from pinntorch.core.derivatives import diff, gradient, laplacian, divergence, curl
from pinntorch.conditions.initial import InitialCondition
from pinntorch.conditions.boundary import BoundaryCondition
from pinntorch.networks.mlp import MLP

# Visualization (optional - requires matplotlib)
try:
    from pinntorch.visualization.plots import (
        plot_solution,
        plot_solution_1d,
        plot_training_history,
        plot_comparison,
        plot_error,
    )
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

__all__ = [
    "__version__",
    # Core
    "PINN",
    # Derivatives
    "diff",
    "gradient",
    "laplacian",
    "divergence",
    "curl",
    # Conditions
    "InitialCondition",
    "BoundaryCondition",
    # Networks
    "MLP",
]

# Add visualization to __all__ if available
if _HAS_VIZ:
    __all__.extend([
        "plot_solution",
        "plot_solution_1d",
        "plot_training_history",
        "plot_comparison",
        "plot_error",
    ])

__author__ = "Kasra Kakavand"
__email__ = "kasrakakavand@gmail.com"
__license__ = "MIT"
__description__ = "Physics-Informed Neural Networks for PyTorch."
