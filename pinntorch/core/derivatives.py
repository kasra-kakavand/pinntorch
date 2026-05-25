"""
Automatic differentiation utilities for Physics-Informed Neural Networks.

This module provides the core differentiation operators used to compute
PDE residuals. By leveraging PyTorch's autograd engine, we obtain exact
derivatives of neural network outputs with respect to their inputs.

The functions in this module are the mathematical foundation of PINNs:
they enable us to express PDEs as differentiable loss functions.

Example:
    >>> import torch
    >>> from pinntorch import diff
    >>>
    >>> x = torch.linspace(0, 1, 100, requires_grad=True).reshape(-1, 1)
    >>> u = x ** 2  # u = x^2
    >>> du_dx = diff(u, x)        # du/dx = 2x
    >>> d2u_dx2 = diff(u, x, order=2)  # d^2u/dx^2 = 2
"""

from typing import List, Optional, Union

import torch


def diff(
    output: torch.Tensor,
    input: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Compute the derivative of `output` with respect to `input`.

    This is the fundamental operation for PINNs. It uses PyTorch's
    automatic differentiation to compute exact derivatives of any order.

    Args:
        output: The tensor to differentiate (e.g., neural network output)
        input: The tensor to differentiate with respect to
        order: Order of the derivative (1 for first, 2 for second, etc.)

    Returns:
        The derivative tensor with the same shape as `output`

    Example:
        Computing du/dx and d²u/dx² for u(x) = sin(x):

        >>> import torch
        >>> x = torch.linspace(0, 2 * torch.pi, 100, requires_grad=True).reshape(-1, 1)
        >>> u = torch.sin(x)
        >>> du_dx = diff(u, x)         # Should be cos(x)
        >>> d2u_dx2 = diff(u, x, order=2)  # Should be -sin(x)

    Notes:
        - `input` must have `requires_grad=True`
        - For higher-order derivatives, intermediate derivatives are
          computed with `create_graph=True` to enable further differentiation
    """
    if order < 1:
        raise ValueError(f"Order must be >= 1, got {order}")

    if not input.requires_grad:
        raise ValueError(
            "Input tensor must have requires_grad=True. "
            "Set it with: input.requires_grad_(True)"
        )

    # Compute the derivative recursively
    result = output
    for i in range(order):
        # For higher-order derivatives, we need create_graph=True
        create_graph = (i < order - 1) or torch.is_grad_enabled()

        result = torch.autograd.grad(
            outputs=result,
            inputs=input,
            grad_outputs=torch.ones_like(result),
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    return result


def gradient(
    output: torch.Tensor,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the gradient of `output` with respect to multiple inputs.

    The gradient is the vector of first-order partial derivatives.
    For a function u(x, y, z), gradient returns [du/dx, du/dy, du/dz].

    Args:
        output: The tensor to differentiate
        inputs: Either a single tensor or a list of tensors

    Returns:
        If `inputs` is a single tensor, returns a single derivative tensor.
        If `inputs` is a list, returns a list of derivative tensors.

    Example:
        Computing the gradient of u(x, t) = x*t:

        >>> import torch
        >>> x = torch.tensor([1.0], requires_grad=True)
        >>> t = torch.tensor([2.0], requires_grad=True)
        >>> u = x * t
        >>> grad = gradient(u, [x, t])
        >>> # grad[0] = du/dx = t = 2.0
        >>> # grad[1] = du/dt = x = 1.0
    """
    if isinstance(inputs, torch.Tensor):
        return diff(output, inputs, order=1)

    if isinstance(inputs, list):
        return [diff(output, inp, order=1) for inp in inputs]

    raise TypeError(
        f"inputs must be a Tensor or list of Tensors, got {type(inputs).__name__}"
    )


def laplacian(
    output: torch.Tensor,
    inputs: Union[torch.Tensor, List[torch.Tensor]],
) -> torch.Tensor:
    """
    Compute the Laplacian of `output` with respect to inputs.

    The Laplacian is the sum of second-order partial derivatives:
        Δu = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z² + ...

    This operator appears in many fundamental PDEs:
        - Poisson equation: Δu = f
        - Heat equation: ∂u/∂t = αΔu
        - Wave equation: ∂²u/∂t² = c²Δu

    Args:
        output: The tensor to differentiate
        inputs: A single tensor or list of tensors representing spatial coordinates

    Returns:
        The Laplacian as a single tensor

    Example:
        Computing the Laplacian of u(x, y) = x² + y²:

        >>> import torch
        >>> x = torch.tensor([1.0], requires_grad=True)
        >>> y = torch.tensor([2.0], requires_grad=True)
        >>> u = x**2 + y**2
        >>> lap = laplacian(u, [x, y])
        >>> # lap = d²u/dx² + d²u/dy² = 2 + 2 = 4
    """
    if isinstance(inputs, torch.Tensor):
        return diff(output, inputs, order=2)

    if isinstance(inputs, list):
        result = torch.zeros_like(output)
        for inp in inputs:
            result = result + diff(output, inp, order=2)
        return result

    raise TypeError(
        f"inputs must be a Tensor or list of Tensors, got {type(inputs).__name__}"
    )


def divergence(
    vector_field: List[torch.Tensor],
    inputs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute the divergence of a vector field.

    For a vector field F = [F_x, F_y, F_z] and inputs [x, y, z]:
        div(F) = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z

    This operator appears in:
        - Continuity equations
        - Maxwell's equations
        - Conservation laws

    Args:
        vector_field: List of tensors representing vector components
        inputs: List of tensors representing coordinates

    Returns:
        The divergence as a single tensor

    Example:
        Computing div(F) where F = [x, y]:

        >>> import torch
        >>> x = torch.tensor([1.0], requires_grad=True)
        >>> y = torch.tensor([2.0], requires_grad=True)
        >>> F = [x, y]
        >>> div = divergence(F, [x, y])
        >>> # div = dx/dx + dy/dy = 1 + 1 = 2
    """
    if len(vector_field) != len(inputs):
        raise ValueError(
            f"vector_field and inputs must have same length, "
            f"got {len(vector_field)} and {len(inputs)}"
        )

    result = torch.zeros_like(vector_field[0])
    for component, coord in zip(vector_field, inputs):
        result = result + diff(component, coord, order=1)

    return result


def curl(
    vector_field: List[torch.Tensor],
    inputs: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Compute the curl of a 3D vector field.

    For F = [F_x, F_y, F_z] and inputs [x, y, z]:
        curl(F) = [∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y]

    This operator appears in:
        - Electromagnetic field equations
        - Fluid dynamics (vorticity)
        - Differential geometry

    Args:
        vector_field: List of 3 tensors representing vector components
        inputs: List of 3 tensors representing coordinates [x, y, z]

    Returns:
        The curl as a list of 3 tensors
    """
    if len(vector_field) != 3 or len(inputs) != 3:
        raise ValueError(
            "Curl is only defined for 3D vector fields. "
            f"Got vector_field of length {len(vector_field)} "
            f"and inputs of length {len(inputs)}"
        )

    Fx, Fy, Fz = vector_field
    x, y, z = inputs

    curl_x = diff(Fz, y, order=1) - diff(Fy, z, order=1)
    curl_y = diff(Fx, z, order=1) - diff(Fz, x, order=1)
    curl_z = diff(Fy, x, order=1) - diff(Fx, y, order=1)

    return [curl_x, curl_y, curl_z]
