"""
Initial condition handlers for time-dependent PDEs.

Initial conditions specify the solution at t = 0 (or other initial time).
For example, the heat equation u_t = α*u_xx requires an initial
temperature distribution u(x, 0).
"""

from typing import Callable, Optional

import torch


class InitialCondition:
    """
    Represents an initial condition for a time-dependent PDE.

    The initial condition specifies the value of u at the initial time:
        u(x, t=0) = f(x)

    Example:
        Create an initial condition u(x, 0) = sin(πx):

        >>> import torch
        >>> from pinntorch import InitialCondition
        >>>
        >>> ic = InitialCondition(
        ...     function=lambda x: torch.sin(torch.pi * x),
        ...     time_value=0.0
        ... )
    """

    def __init__(
        self,
        function: Callable,
        time_value: float = 0.0,
        weight: float = 1.0,
    ):
        """
        Args:
            function: Function that returns u given spatial coordinates
                Signature: function(*x) -> u
            time_value: The time at which the condition applies (default: 0.0)
            weight: Loss weight for this condition (default: 1.0)
        """
        self.function = function
        self.time_value = time_value
        self.weight = weight

    def __call__(self, *spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the initial condition at given spatial coordinates.

        Args:
            *spatial_coords: Spatial coordinate tensors

        Returns:
            Initial condition values
        """
        return self.function(*spatial_coords)

    def __repr__(self) -> str:
        return (
            f"InitialCondition(time_value={self.time_value}, "
            f"weight={self.weight})"
        )
