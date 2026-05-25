"""
Multi-Layer Perceptron architectures for PINNs.

This module provides customizable MLP architectures designed for use
with Physics-Informed Neural Networks. The architectures support various
activation functions and initialization schemes optimized for PDE solving.
"""

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn


# Activation functions commonly used in PINNs
ACTIVATIONS = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "sin": lambda x: torch.sin(x),
    "swish": nn.SiLU(),
}


class SinActivation(nn.Module):
    """Sinusoidal activation function for SIREN-like architectures."""

    def __init__(self, omega: float = 1.0):
        """
        Args:
            omega: Frequency parameter for the sine activation
        """
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Physics-Informed Neural Networks.

    A fully-connected feed-forward network with customizable depth,
    width, and activation functions. The architecture is designed
    specifically for PDE solving with PINNs.

    Attributes:
        input_dim: Number of input features (e.g., 2 for (x, t))
        output_dim: Number of output features (e.g., 1 for scalar u)
        hidden_layers: List of hidden layer sizes
        activation: Activation function name

    Example:
        Create a network for solving u(x, t):

        >>> import torch
        >>> from pinntorch.networks import MLP
        >>>
        >>> # Network: (x, t) -> u
        >>> net = MLP(
        ...     input_dim=2,
        ...     output_dim=1,
        ...     hidden_layers=[64, 64, 64],
        ...     activation='tanh'
        ... )
        >>>
        >>> # Forward pass
        >>> x = torch.tensor([[0.5, 0.3]])  # (batch, [x, t])
        >>> u = net(x)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        init_method: str = "xavier",
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output features (default: 1)
            hidden_layers: List of hidden layer sizes (default: [64, 64, 64])
            activation: Activation function name
                       Options: 'tanh', 'relu', 'gelu', 'silu', 'sin', 'swish'
            init_method: Weight initialization method
                       Options: 'xavier', 'he', 'normal'
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.init_method = init_method

        # Get activation function
        if activation == "sin":
            self.activation = SinActivation()
        elif activation in ACTIVATIONS:
            self.activation = ACTIVATIONS[activation]
        else:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {list(ACTIVATIONS.keys())}"
            )

        # Build layers
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1])
             for i in range(len(layer_sizes) - 1)]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using the specified method."""
        for layer in self.layers:
            if self.init_method == "xavier":
                nn.init.xavier_normal_(layer.weight)
            elif self.init_method == "he":
                nn.init.kaiming_normal_(layer.weight)
            elif self.init_method == "normal":
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            else:
                raise ValueError(f"Unknown init method: {self.init_method}")

            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"MLP(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"activation='{self.activation_name}', "
            f"parameters={self.count_parameters():,})"
        )
