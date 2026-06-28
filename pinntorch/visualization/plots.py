"""
Visualization utilities for Physics-Informed Neural Network solutions.

This module provides plotting functions to visualize PDE solutions,
training progress, and comparisons with analytical solutions.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Verify matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_solution(
    model,
    n_points: int = 100,
    title: str = "PINN Solution",
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: str = "viridis",
) -> "Figure":
    """
    Plot the 2D PINN solution as a heatmap.

    For PINNs with 2D input (e.g., (x, t)), creates a heatmap showing
    the solution across the entire domain.

    Args:
        model: Trained PINN model
        n_points: Number of points per dimension (default: 100)
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the figure
        cmap: Colormap name

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    if len(model.coordinate_names) != 2:
        raise ValueError(
            f"plot_solution requires 2D input, got {len(model.coordinate_names)}D"
        )

    # Get coordinate names and ranges
    name1, name2 = model.coordinate_names
    range1 = model.domain[name1]
    range2 = model.domain[name2]

    # Create grid
    coord1 = np.linspace(range1[0], range1[1], n_points)
    coord2 = np.linspace(range2[0], range2[1], n_points)
    C1, C2 = np.meshgrid(coord1, coord2)

    # Predict on grid
    u_pred = np.zeros_like(C1)
    for i in range(n_points):
        for j in range(n_points):
            u_pred[i, j] = model.predict(
                **{name1: float(C1[i, j]), name2: float(C2[i, j])}
            ).item()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(C1, C2, u_pred, cmap=cmap, shading="auto")
    plt.colorbar(im, ax=ax, label="u")

    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_solution_1d(
    model,
    fixed_coord: Dict[str, float],
    varying_coord: str,
    n_points: int = 200,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> "Figure":
    """
    Plot 1D slice of the PINN solution.

    Useful for visualizing the solution at a specific time or position.

    Args:
        model: Trained PINN model
        fixed_coord: Dict of fixed coordinates (e.g., {'t': 0.5})
        varying_coord: Name of the varying coordinate (e.g., 'x')
        n_points: Number of points along varying coordinate
        title: Plot title (auto-generated if None)
        save_path: Optional path to save
        show: Whether to display

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    # Get varying coordinate range
    var_range = model.domain[varying_coord]
    var_values = np.linspace(var_range[0], var_range[1], n_points)

    # Predict
    u_values = []
    for v in var_values:
        coords = {**fixed_coord, varying_coord: float(v)}
        u_values.append(model.predict(**coords).item())

    u_values = np.array(u_values)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(var_values, u_values, "b-", linewidth=2, label="PINN solution")

    ax.set_xlabel(varying_coord, fontsize=12)
    ax.set_ylabel("u", fontsize=12)

    if title is None:
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_coord.items())
        title = f"Solution at {fixed_str}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
    show: bool = True,
    log_scale: bool = True,
) -> "Figure":
    """
    Plot training loss curves.

    Shows total loss and individual loss components (PDE, IC, BC) over training.

    Args:
        history: Training history dict from model.train()
        title: Plot title
        save_path: Optional path to save
        show: Whether to display
        log_scale: Use logarithmic y-axis (recommended)

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["total_loss"]) + 1)

    ax.plot(epochs, history["total_loss"], "b-", linewidth=2, label="Total Loss")
    ax.plot(epochs, history["pde_loss"], "r--", linewidth=1.5, label="PDE Loss")
    ax.plot(epochs, history["ic_loss"], "g--", linewidth=1.5, label="IC Loss")
    ax.plot(epochs, history["bc_loss"], "m--", linewidth=1.5, label="BC Loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_comparison(
    model,
    analytical: Callable,
    n_points: int = 100,
    title: str = "PINN vs Analytical Solution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> "Figure":
    """
    Compare PINN solution to analytical solution side-by-side.

    Args:
        model: Trained PINN model
        analytical: Function computing analytical solution
            Signature: analytical(x, t) -> u
        n_points: Number of points per dimension
        title: Plot title
        save_path: Optional path to save
        show: Whether to display

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    if len(model.coordinate_names) != 2:
        raise ValueError("plot_comparison requires 2D input")

    name1, name2 = model.coordinate_names
    range1 = model.domain[name1]
    range2 = model.domain[name2]

    coord1 = np.linspace(range1[0], range1[1], n_points)
    coord2 = np.linspace(range2[0], range2[1], n_points)
    C1, C2 = np.meshgrid(coord1, coord2)

    # Predict and compute analytical
    u_pred = np.zeros_like(C1)
    u_true = np.zeros_like(C1)

    for i in range(n_points):
        for j in range(n_points):
            u_pred[i, j] = model.predict(
                **{name1: float(C1[i, j]), name2: float(C2[i, j])}
            ).item()
            u_true[i, j] = analytical(C1[i, j], C2[i, j])

    error = np.abs(u_pred - u_true)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PINN solution
    im0 = axes[0].pcolormesh(C1, C2, u_pred, cmap="viridis", shading="auto")
    axes[0].set_title("PINN Solution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(name1)
    axes[0].set_ylabel(name2)
    plt.colorbar(im0, ax=axes[0])

    # Analytical solution
    im1 = axes[1].pcolormesh(C1, C2, u_true, cmap="viridis", shading="auto")
    axes[1].set_title("Analytical Solution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(name1)
    axes[1].set_ylabel(name2)
    plt.colorbar(im1, ax=axes[1])

    # Error
    im2 = axes[2].pcolormesh(C1, C2, error, cmap="hot", shading="auto")
    axes[2].set_title("Absolute Error", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(name1)
    axes[2].set_ylabel(name2)
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_error(
    model,
    analytical: Callable,
    n_points: int = 100,
    title: str = "Absolute Error",
    save_path: Optional[str] = None,
    show: bool = True,
) -> "Figure":
    """
    Plot the absolute error between PINN and analytical solutions.

    Args:
        model: Trained PINN model
        analytical: Analytical solution function
        n_points: Number of points per dimension
        title: Plot title
        save_path: Optional path to save
        show: Whether to display

    Returns:
        Matplotlib figure object
    """
    _check_matplotlib()

    if len(model.coordinate_names) != 2:
        raise ValueError("plot_error requires 2D input")

    name1, name2 = model.coordinate_names
    range1 = model.domain[name1]
    range2 = model.domain[name2]

    coord1 = np.linspace(range1[0], range1[1], n_points)
    coord2 = np.linspace(range2[0], range2[1], n_points)
    C1, C2 = np.meshgrid(coord1, coord2)

    error = np.zeros_like(C1)
    for i in range(n_points):
        for j in range(n_points):
            u_pred = model.predict(
                **{name1: float(C1[i, j]), name2: float(C2[i, j])}
            ).item()
            u_true = analytical(C1[i, j], C2[i, j])
            error[i, j] = abs(u_pred - u_true)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(C1, C2, error, cmap="hot", shading="auto")
    plt.colorbar(im, ax=ax, label="|Error|")

    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig
