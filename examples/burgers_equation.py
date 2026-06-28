"""
Solve the 1D viscous Burgers' equation using PINNTorch.

    ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²

with:
    - Domain: x ∈ [-1, 1], t ∈ [0, 1]
    - ν = 0.01 / π  (viscosity)
    - Initial condition: u(x, 0) = -sin(πx)
    - Boundary conditions: u(-1, t) = u(1, t) = 0

This is a classic PINN benchmark (Raissi et al., 2019). The nonlinear
advection term causes a sharp gradient to form near x = 0 over time.
There is no simple closed-form solution, so we rely on the PDE residual,
initial condition, and boundary conditions alone.
"""

import numpy as np
import torch

import pinntorch as pt


def main():
    print("=" * 70)
    print("PINNTorch Example: 1D Burgers' Equation")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    nu = 0.01 / np.pi

    # PDE residual: u_t + u * u_x - nu * u_xx = 0
    def burgers(u, x, t):
        u_t = pt.diff(u, t, order=1)
        u_x = pt.diff(u, x, order=1)
        u_xx = pt.diff(u, x, order=2)
        return u_t + u * u_x - nu * u_xx

    # Initial condition: u(x, 0) = -sin(pi x)
    def initial_condition(x):
        return -torch.sin(torch.pi * x)

    # Boundary conditions: u = 0 at both ends
    def boundary_condition(t):
        return torch.zeros_like(t)

    model = pt.PINN(
        pde=burgers,
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        initial_condition=initial_condition,
        boundary_condition=boundary_condition,
        hidden_layers=[64, 64, 64, 64],
        activation="tanh",
    )

    print(f"\nModel created!")
    print(f"Network parameters: {model.network.count_parameters():,}")

    print("\nStarting training...")
    model.lambda_ic = 10.0
    model.lambda_bc = 10.0
    history = model.train(
        epochs=20000,          # was 8000
        learning_rate=1e-3,
        n_pde_points=4000,     # was 2000
        n_ic_points=400,       # was 200
        n_bc_points=400,       # was 200
        verbose=True,
        print_every=2000,
    )

    print("\nTraining complete!")
    print(f"  Total loss: {history['total_loss'][-1]:.6e}")
    print(f"  PDE:        {history['pde_loss'][-1]:.6e}")
    print(f"  IC:         {history['ic_loss'][-1]:.6e}")
    print(f"  BC:         {history['bc_loss'][-1]:.6e}")

    # Spot-check a few values
    print("\nSample predictions:")
    for x_val, t_val in [(-0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.0, 1.0)]:
        u_val = model.predict(x=x_val, t=t_val).item()
        print(f"  u({x_val:>4}, {t_val}) = {u_val:.6f}")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print("\nGenerating visualizations...")

    pt.plot_training_history(
        history,
        title="Burgers' Equation: Training History",
        save_path="burgers_training_history.png",
        show=False,
    )

    pt.plot_solution(
        model,
        title="Burgers' Equation: PINN Solution u(x, t)",
        save_path="burgers_solution.png",
        show=False,
    )

    # Snapshots at several times to see the shock forming
    for t_snap in [0.0, 0.25, 0.5, 0.75]:
        pt.plot_solution_1d(
            model,
            fixed_coord={"t": t_snap},
            varying_coord="x",
            title=f"Burgers' Equation: Solution at t = {t_snap}",
            save_path=f"burgers_slice_t{t_snap}.png",
            show=False,
        )

    print("Saved figures:")
    print("  - burgers_training_history.png")
    print("  - burgers_solution.png")
    print("  - burgers_slice_t0.0.png ... t0.75.png")


if __name__ == "__main__":
    main()
