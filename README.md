<div align="center">

# 🧮 PINNTorch

### Physics-Informed Neural Networks, made simple.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

**Solve partial differential equations with neural networks using PyTorch.**

</div>

---

## What is PINNTorch?

PINNTorch is an open-source Python library for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs). It leverages PyTorch's automatic differentiation to provide a clean, modern API for scientific machine learning.

## Why PINNTorch?

Traditional PDE solvers (finite differences, finite elements) require discretization and struggle with high-dimensional problems. PINNs use neural networks to approximate solutions, offering:

- **Mesh-free** solutions
- **Continuous** representations
- **High-dimensional** problem handling
- **Mesh-free** solutions
- **Continuous** representations
- **Automatic differentiation** for exact PDE residuals
- **Simple, PyTorch-native API**


## Key Features

- **Simple API** — define PDEs as plain Python functions
- **Validated solvers** — heat equation (validated) and Burgers' equation (with documented limitations)
- **Dirichlet boundary conditions** and first-order initial conditions
- **Two-stage optimization** — Adam followed by optional LBFGS refinement
- **Automatic differentiation** — exact derivatives via PyTorch autograd
- **Built-in visualization** — solution heatmaps, 1D slices, training curves

### Planned

- Neumann and periodic boundary conditions
- Second-order-in-time PDEs (wave equation)
- Inverse problems (parameter discovery from data)
- Adaptive collocation sampling

## Installation

> **Note:** PINNTorch is currently in active development. PyPI release coming soon!

```bash
# Coming soon
pip install pinntorch

# Development version
git clone https://github.com/kasra-kakavand/pinntorch.git
cd pinntorch
pip install -e .
```

### Example: 1D Heat Equation

Solve the heat equation `∂u/∂t = α·∂²u/∂x²` with PINNTorch:

```python
import pinntorch as pt
import torch

# Define the PDE
def heat_equation(u, x, t):
    alpha = 0.1
    u_t = pt.diff(u, t)
    u_xx = pt.diff(u, x, order=2)
    return u_t - alpha * u_xx

# Define conditions
def initial_condition(x):
    return torch.sin(torch.pi * x)

def boundary_condition(t):
    return torch.zeros_like(t)

# Create and train the PINN
model = pt.PINN(
    pde=heat_equation,
    domain={'x': (0, 1), 't': (0, 1)},
    initial_condition=initial_condition,
    boundary_condition=boundary_condition,
)

model.train(epochs=10000, learning_rate=1e-3)

# Visualize the solution
pt.plot_solution(model)
```

## Supported Equations

| Equation | Description | Status |
|----------|-------------|--------|
| Heat | 1D diffusion; validated against analytical solution (L2 error ~3e-3) | ✅ |
| Burgers | 1D nonlinear advection-diffusion; captures shock, mild center drift | ✅ |
| Wave | Second-order in time | 📋 Planned |
| Schrödinger | Quantum mechanics | 📋 Planned |
| Poisson | Electrostatics | 📋 Planned |

## Results & Limitations

PINNTorch is an early-stage library. Here is an honest account of what it does well and where it falls short.

### Heat equation (validated)

The 1D heat equation solver is validated against the known analytical
solution `u(x,t) = sin(πx)·exp(-απ²t)`:

- Mean absolute error: ~2.6e-3
- L2 error: ~2.8e-3

The PINN solution matches the analytical solution closely across the
entire domain.

### Burgers' equation (works, with a known limitation)

The nonlinear Burgers' equation is solved using two-stage optimization
(Adam followed by LBFGS refinement). The solver captures the characteristic
shock formation near x = 0.

**Known limitation:** under uniform random collocation sampling, the
solution exhibits mild drift at the domain center at later times, where
the shock gradient is steepest and uniform sampling under-resolves it.
Adaptive collocation sampling — the standard remedy — is planned for a
future release.

### Optimization

Training uses the standard PINN recipe: Adam to reach a good region of
parameter space, then optional LBFGS refinement (`lbfgs_steps` argument)
for stable convergence. LBFGS uses fixed collocation points, as its line
search requires a stable objective.


## Why Physics-Informed Neural Networks?

PINNs represent a paradigm shift in scientific computing:

> "PINNs enable us to solve PDEs that were previously intractable, particularly in high dimensions and with complex geometries."

**Applications include:**
- Climate modeling
- Fluid dynamics simulation
- Quantum mechanics
- Materials science
- Biomedical engineering

## Roadmap

- [x] Project foundation
- [x] Core PINN class with autograd
- [x] Heat equation solver (validated)
- [x] Burgers equation solver
- [x] Adam + LBFGS optimization
- [x] Visualization module
- [x] Test suite
- [ ] PyPI release (v0.1.0)
- [ ] Neumann / periodic boundary conditions
- [ ] Wave equation (second-order in time)
- [ ] Inverse problem support
- [ ] Adaptive collocation sampling
- [ ] Documentation site

## Citation

If you use PINNTorch in your research, please cite:

```bibtex
@software{kakavand2026pinntorch,
  title={PINNTorch: A Python Library for Physics-Informed Neural Networks},
  author={Kakavand, Kasra},
  year={2026},
  url={https://github.com/kasra-kakavand/pinntorch}
}
```

## References

This library implements methods from:

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## License

PINNTorch is released under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Kasra Kakavand**
- GitHub: [@kasra-kakavand](https://github.com/kasra-kakavand)
- Other projects: [FairCheck](https://github.com/kasra-kakavand/faircheck) | [Deepfake Fairness](https://github.com/kasra-kakavand/deepfake-fairness)

## Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, or improving documentation, please feel free to open an issue or pull request.

---

<div align="center">

**Building the future of scientific machine learning.**

If you find PINNTorch useful, please consider giving it a star ⭐

</div>
