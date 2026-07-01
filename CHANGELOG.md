# Changelog

All notable changes to PINNTorch are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core `PINN` class for solving PDEs with neural networks
- Differentiation operators: `diff`, `gradient`, `laplacian`, `divergence`, `curl`
- Two-stage optimization: Adam followed by optional LBFGS refinement
- `InitialCondition` and `BoundaryCondition` (Dirichlet) handlers
- `MLP` network with configurable depth, width, activation, and initialization
- Visualization module: solution heatmaps, 1D slices, training curves, comparison plots
- Validated 1D heat equation example (L2 error ~2.8e-3)
- Burgers' equation example with documented center-drift limitation
- Test suite: 21 tests covering differentiation operators and PINN machinery

### Known Limitations
- Only Dirichlet boundary conditions are implemented; Neumann and periodic
  raise `NotImplementedError`
- Burgers' equation exhibits mild center drift under uniform collocation sampling
- Second-order-in-time PDEs (e.g. wave equation) not yet supported
- Inverse problems not yet supported

## [0.1.0] - 2026-06-28

### Added
- Initial project structure, packaging, and MIT license
