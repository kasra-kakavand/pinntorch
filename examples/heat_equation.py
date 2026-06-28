"""
Solve the 1D Heat Equation using PINNTorch.

This example demonstrates how to solve the heat equation:

    ∂u/∂t = α * ∂²u/∂x²

with:
    - Domain: x ∈ [0, 1], t ∈ [0, 1]
    - α = 0.1 (thermal diffusivity)
    - Initial condition: u(x, 0) = sin(πx)
    - Boundary conditions: u(0, t) = u(1, t) = 0

The analytical solution is:
    u(x, t) = sin(πx) * exp(-α * π² * t)

We'll train a PINN and compare to the analytical solution.
"""

import numpy as np
import torch

import pinntorch as pt


def main():
    """Train a PINN to solve the heat equation."""

    print("=" * 70)
    print("PINNTorch Example: 1D Heat Equation")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define the PDE: ∂u/∂t - α * ∂²u/∂x² = 0
    alpha = 0.1

    def heat_equation(u, x, t):
        u_t = pt.diff(u, t, order=1)
        u_xx = pt.diff(u, x, order=2)
        return u_t - alpha * u_xx

    # Initial condition: u(x, 0) = sin(πx)
    def initial_condition(x):
        return torch.sin(torch.pi * x)

    # Boundary conditions: u(0, t) = u(1, t) = 0
    def boundary_condition(t):
        return torch.zeros_like(t)

    # Create the PINN
    model = pt.PINN(
        pde=heat_equation,
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        initial_condition=initial_condition,
        boundary_condition=boundary_condition,
        hidden_layers=[64, 64, 64, 64],
        activation="tanh",
    )

    print(f"\nModel created!")
    print(f"Network parameters: {model.network.count_parameters():,}")

    # Train the PINN
    print("\nStarting training...")
    history = model.train(
        epochs=5000,
        learning_rate=1e-3,
        n_pde_points=1000,
        n_ic_points=100,
        n_bc_points=100,
        verbose=True,
        print_every=500,
    )

    # Evaluate at specific points
    print("\n" + "=" * 70)
    print("Evaluation: PINN vs Analytical Solution")
    print("=" * 70)

    test_points = [
        (0.5, 0.0),   # u(0.5, 0) = sin(π/2) = 1.0
        (0.5, 0.5),   # u(0.5, 0.5) = sin(π/2) * exp(-0.1π² * 0.5)
        (0.25, 0.5),  # u(0.25, 0.5) = sin(π/4) * exp(-0.1π² * 0.5)
        (0.5, 1.0),   # u(0.5, 1.0) = sin(π/2) * exp(-0.1π²)
    ]

    print(f"\n{'Point (x, t)':<20} {'PINN':<15} {'Analytical':<15} {'Error':<15}")
    print("-" * 70)

    for x_val, t_val in test_points:
        # PINN prediction
        u_pinn = model.predict(x=x_val, t=t_val).item()

        # Analytical solution
        u_exact = np.sin(np.pi * x_val) * np.exp(-alpha * np.pi**2 * t_val)

        # Error
        error = abs(u_pinn - u_exact)

        print(f"({x_val}, {t_val}){'':<10} {u_pinn:<15.6f} {u_exact:<15.6f} {error:<15.6e}")

    # Compute overall error on a grid
    print("\n" + "=" * 70)
    print("Overall Performance")
    print("=" * 70)

    n_test = 50
    x_test = np.linspace(0, 1, n_test)
    t_test = np.linspace(0, 1, n_test)

    errors = []
    for x_val in x_test:
        for t_val in t_test:
            u_pinn = model.predict(x=x_val, t=t_val).item()
            u_exact = np.sin(np.pi * x_val) * np.exp(-alpha * np.pi**2 * t_val)
            errors.append(abs(u_pinn - u_exact))

    errors = np.array(errors)

    print(f"\nMean Absolute Error:    {np.mean(errors):.6e}")
    print(f"Max Absolute Error:     {np.max(errors):.6e}")
    print(f"L2 Error:               {np.sqrt(np.mean(errors**2)):.6e}")

    print("\nTraining complete! The PINN successfully solved the heat equation.")
    print("Final loss components:")
    print(f"  Total: {history['total_loss'][-1]:.6e}")
    print(f"  PDE:   {history['pde_loss'][-1]:.6e}")
    print(f"  IC:    {history['ic_loss'][-1]:.6e}")
    print(f"  BC:    {history['bc_loss'][-1]:.6e}")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print("\nGenerating visualizations...")

    # Analytical solution for comparison
    def analytical(x, t):
        return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

    # 1. Training history (loss curves)
    pt.plot_training_history(
        history,
        title="Heat Equation: Training History",
        save_path="heat_training_history.png",
        show=False,
    )

    # 2. PINN solution heatmap
    pt.plot_solution(
        model,
        title="Heat Equation: PINN Solution u(x, t)",
        save_path="heat_solution.png",
        show=False,
    )

    # 3. PINN vs analytical comparison (3 panels)
    pt.plot_comparison(
        model,
        analytical=analytical,
        title="Heat Equation: PINN vs Analytical",
        save_path="heat_comparison.png",
        show=False,
    )

    # 4. 1D slice at t = 0.5
    pt.plot_solution_1d(
        model,
        fixed_coord={"t": 0.5},
        varying_coord="x",
        title="Heat Equation: Solution at t = 0.5",
        save_path="heat_slice_t0.5.png",
        show=False,
    )

    print("Saved 4 figures:")
    print("  - heat_training_history.png")
    print("  - heat_solution.png")
    print("  - heat_comparison.png")
    print("  - heat_slice_t0.5.png")


if __name__ == "__main__":
    main()
