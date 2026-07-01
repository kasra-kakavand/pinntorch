"""
Boundary condition handlers for PDEs.

Boundary conditions specify the solution behavior at the domain boundaries.
Common types include:
- Dirichlet: u = g (fixed value)
- Neumann: ∂u/∂n = g (fixed flux)
- Periodic: u(x_left) = u(x_right)
"""

from typing import Callable, Optional, Tuple

import torch


class BoundaryCondition:
    """
    Represents a boundary condition for a PDE.

    Supports Dirichlet (value), Neumann (derivative), and periodic
    boundary conditions.

    Example:
        Dirichlet condition u = 0 at boundaries:

        >>> import torch
        >>> from pinntorch import BoundaryCondition
        >>>
        >>> bc = BoundaryCondition(
        ...     function=lambda t: torch.zeros_like(t),
        ...     condition_type='dirichlet'
        ... )
    """

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"

    def __init__(
        self,
        function: Callable,
        condition_type: str = "dirichlet",
        boundary: str = "both",
        weight: float = 1.0,
    ):
        """
        Args:
            function: Function that returns boundary value given time
                Signature: function(t) -> u_boundary
            condition_type: Type of condition
                Options: 'dirichlet', 'neumann', 'periodic'
            boundary: Which boundary to apply to
                Options: 'left', 'right', 'both'
            weight: Loss weight for this condition (default: 1.0)
        """
        if condition_type not in [self.DIRICHLET, self.NEUMANN, self.PERIODIC]:
            raise ValueError(
                f"Unknown condition_type: {condition_type}. "
                f"Must be 'dirichlet', 'neumann', or 'periodic'"
            )

        if boundary not in ["left", "right", "both"]:
            raise ValueError(
                f"Unknown boundary: {boundary}. "
                f"Must be 'left', 'right', or 'both'"
            )
        if condition_type != self.DIRICHLET:
            raise NotImplementedError(
                f"'{condition_type}' boundary conditions are not yet supported. "
                f"Only 'dirichlet' is currently implemented. "
                f"Neumann and periodic are planned for a future release."
            )
        self.function = function
        self.condition_type = condition_type
        self.boundary = boundary
        self.weight = weight

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the boundary condition at given time.

        Args:
            t: Time coordinate tensor

        Returns:
            Boundary condition values
        """
        return self.function(t)

    def __repr__(self) -> str:
        return (
            f"BoundaryCondition(type='{self.condition_type}', "
            f"boundary='{self.boundary}', weight={self.weight})"
        )
