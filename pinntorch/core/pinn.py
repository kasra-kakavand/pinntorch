"""
The main PINN class for solving partial differential equations.

This module provides the high-level PINN class that integrates:
- Neural network approximation
- PDE residual computation via autograd
- Boundary and initial condition enforcement
- Training and prediction
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from pinntorch.networks.mlp import MLP


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.

    A PINN approximates the solution of a partial differential equation
    using a neural network. The network is trained to minimize:

        L_total = L_PDE + λ_ic * L_IC + λ_bc * L_BC

    where:
        - L_PDE is the PDE residual loss
        - L_IC is the initial condition loss
        - L_BC is the boundary condition loss

    Example:
        Solve the 1D heat equation ∂u/∂t = 0.1 * ∂²u/∂x²:

        >>> import torch
        >>> import pinntorch as pt
        >>>
        >>> def heat_pde(u, x, t):
        ...     u_t = pt.diff(u, t)
        ...     u_xx = pt.diff(u, x, order=2)
        ...     return u_t - 0.1 * u_xx
        >>>
        >>> def ic(x):
        ...     return torch.sin(torch.pi * x)
        >>>
        >>> def bc(t):
        ...     return torch.zeros_like(t)
        >>>
        >>> model = pt.PINN(
        ...     pde=heat_pde,
        ...     domain={'x': (0, 1), 't': (0, 1)},
        ...     initial_condition=ic,
        ...     boundary_condition=bc,
        ... )
        >>> model.train(epochs=10000)
    """

    def __init__(
        self,
        pde: Callable,
        domain: Dict[str, Tuple[float, float]],
        initial_condition: Optional[Callable] = None,
        boundary_condition: Optional[Callable] = None,
        network: Optional[nn.Module] = None,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the PINN.

        Args:
            pde: Function that computes the PDE residual.
                Signature: pde(u, *coordinates) -> residual
            domain: Dictionary mapping coordinate names to (min, max) tuples
                Example: {'x': (0, 1), 't': (0, 1)}
            initial_condition: Optional function for initial condition
                Signature: ic(x, ...) -> u_initial
            boundary_condition: Optional function for boundary condition
                Signature: bc(t, ...) -> u_boundary
            network: Optional custom neural network (overrides hidden_layers/activation)
            hidden_layers: List of hidden layer sizes (default: [64, 64, 64])
            activation: Activation function name (default: 'tanh')
            device: PyTorch device (default: auto-detect)
        """
        super().__init__()

        self.pde = pde
        self.domain = domain
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition

        # Coordinate names (e.g., ['x', 't'])
        self.coordinate_names = list(domain.keys())
        self.dim = len(self.coordinate_names)

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Build or use provided network
        if network is not None:
            self.network = network
        else:
            self.network = MLP(
                input_dim=self.dim,
                output_dim=1,
                hidden_layers=hidden_layers,
                activation=activation,
            )

        self.network = self.network.to(device)

        # Loss weights
        self.lambda_pde = 1.0
        self.lambda_ic = 1.0
        self.lambda_bc = 1.0

        # Training history
        self.history = {
            "total_loss": [],
            "pde_loss": [],
            "ic_loss": [],
            "bc_loss": [],
        }

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute u given coordinates.

        Args:
            *coords: Coordinate tensors (e.g., x, t)

        Returns:
            Predicted u values
        """
        # Stack coordinates into a single input tensor
        inputs = torch.cat(coords, dim=-1)
        return self.network(inputs)

    def predict(self, **coords) -> torch.Tensor:
        """
        Predict u at given coordinates.

        Args:
            **coords: Named coordinates (e.g., x=0.5, t=0.3)

        Returns:
            Predicted u value

        Example:
            >>> u = model.predict(x=0.5, t=0.3)
        """
        coord_tensors = []
        for name in self.coordinate_names:
            if name not in coords:
                raise ValueError(f"Missing coordinate: {name}")

            value = coords[name]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([[float(value)]], dtype=torch.float32)
            else:
                value = value.float().reshape(-1, 1)

            value = value.to(self.device)
            coord_tensors.append(value)

        self.network.eval()
        with torch.no_grad():
            return self.forward(*coord_tensors)

    def _sample_domain(self, n_points: int) -> List[torch.Tensor]:
        """
        Sample random points from the domain interior.

        Args:
            n_points: Number of points to sample

        Returns:
            List of coordinate tensors, each with shape (n_points, 1)
        """
        coords = []
        for name in self.coordinate_names:
            low, high = self.domain[name]
            sampled = torch.rand(n_points, 1, device=self.device) * (high - low) + low
            sampled.requires_grad_(True)
            coords.append(sampled)
        return coords

    def _compute_pde_loss(self, n_points: int = 1000) -> torch.Tensor:
        """Compute the PDE residual loss."""
        coords = self._sample_domain(n_points)
        u = self.forward(*coords)
        residual = self.pde(u, *coords)
        return torch.mean(residual ** 2)

    def _compute_ic_loss(self, n_points: int = 100) -> torch.Tensor:
        """Compute the initial condition loss."""
        if self.initial_condition is None:
            return torch.tensor(0.0, device=self.device)

        # IC: t = 0, x varies
        # Assumes last coordinate is time
        space_coords = []
        for name in self.coordinate_names[:-1]:
            low, high = self.domain[name]
            sampled = torch.rand(n_points, 1, device=self.device) * (high - low) + low
            space_coords.append(sampled)

        # t = 0
        t_zero = torch.zeros(n_points, 1, device=self.device)

        all_coords = space_coords + [t_zero]
        u_pred = self.forward(*all_coords)
        u_true = self.initial_condition(*space_coords).reshape(-1, 1)

        return torch.mean((u_pred - u_true) ** 2)

    def _compute_bc_loss(self, n_points: int = 100) -> torch.Tensor:
        """Compute the boundary condition loss."""
        if self.boundary_condition is None:
            return torch.tensor(0.0, device=self.device)

        # BC: x = boundary, t varies
        # Assumes first coordinate is space, last is time
        x_name = self.coordinate_names[0]
        x_low, x_high = self.domain[x_name]

        # Sample t values
        t_low, t_high = self.domain[self.coordinate_names[-1]]
        t_sampled = torch.rand(n_points, 1, device=self.device) * (t_high - t_low) + t_low

        # Left boundary (x = x_low)
        x_left = torch.full((n_points, 1), x_low, device=self.device)
        u_left = self.forward(x_left, t_sampled)
        u_left_true = self.boundary_condition(t_sampled).reshape(-1, 1)

        # Right boundary (x = x_high)
        x_right = torch.full((n_points, 1), x_high, device=self.device)
        u_right = self.forward(x_right, t_sampled)
        u_right_true = self.boundary_condition(t_sampled).reshape(-1, 1)

        loss_left = torch.mean((u_left - u_left_true) ** 2)
        loss_right = torch.mean((u_right - u_right_true) ** 2)

        return loss_left + loss_right

    def train(
        self,
        epochs: int = 10000,
        learning_rate: float = 1e-3,
        n_pde_points: int = 1000,
        n_ic_points: int = 100,
        n_bc_points: int = 100,
        lbfgs_steps: int = 0,
        verbose: bool = True,
        print_every: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train the PINN.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
            n_pde_points: Number of PDE collocation points per epoch
            n_ic_points: Number of initial condition points per epoch
            n_bc_points: Number of boundary condition points per epoch
            verbose: Whether to print training progress
            print_every: Print frequency (every N epochs)

        Returns:
            Training history dictionary
        """
        self.network.train()
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        if verbose:
            print(f"Training PINN on device: {self.device}")
            print(f"Network: {self.network}")
            print(f"Domain: {self.domain}")
            print("-" * 60)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute losses
            pde_loss = self._compute_pde_loss(n_pde_points)
            ic_loss = self._compute_ic_loss(n_ic_points)
            bc_loss = self._compute_bc_loss(n_bc_points)

            # Total loss
            total_loss = (
                self.lambda_pde * pde_loss
                + self.lambda_ic * ic_loss
                + self.lambda_bc * bc_loss
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Record history
            self.history["total_loss"].append(total_loss.item())
            self.history["pde_loss"].append(pde_loss.item())
            self.history["ic_loss"].append(ic_loss.item())
            self.history["bc_loss"].append(bc_loss.item())

            # Print progress
            if verbose and (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Total: {total_loss.item():.6f} | "
                    f"PDE: {pde_loss.item():.6f} | "
                    f"IC: {ic_loss.item():.6f} | "
                    f"BC: {bc_loss.item():.6f}"
                )

        # ------------------------------------------------------------------
        # Optional second-stage LBFGS refinement
        # ------------------------------------------------------------------
        if lbfgs_steps > 0:
            if verbose:
                print("-" * 60)
                print(f"Refining with LBFGS for {lbfgs_steps} steps...")

            # Fix the collocation points for LBFGS (it needs a stable objective,
            # unlike Adam which can tolerate fresh random samples each step).
            pde_coords = self._sample_domain(n_pde_points)

            lbfgs = optim.LBFGS(
                self.network.parameters(),
                max_iter=lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad()
                u = self.forward(*pde_coords)
                pde_loss = torch.mean(self.pde(u, *pde_coords) ** 2)
                ic_loss = self._compute_ic_loss(n_ic_points)
                bc_loss = self._compute_bc_loss(n_bc_points)
                loss = (
                    self.lambda_pde * pde_loss
                    + self.lambda_ic * ic_loss
                    + self.lambda_bc * bc_loss
                )
                loss.backward()
                # record history for plotting
                self.history["total_loss"].append(loss.item())
                self.history["pde_loss"].append(pde_loss.item())
                self.history["ic_loss"].append(ic_loss.item())
                self.history["bc_loss"].append(bc_loss.item())
                return loss

            lbfgs.step(closure)

        final_loss = self.history["total_loss"][-1]
        if verbose:
            print("-" * 60)
            print(f"Training complete! Final loss: {final_loss:.6f}")

        return self.history
