# Examples

NLvib-py includes several canonical examples demonstrating nonlinear vibration analysis workflows.

## Overview

| # | System | Nonlinearity | Method | Location |
|---|--------|-------------|--------|----------|
| 01 | Duffing (SDOF) | Cubic spring | HB + Shooting | `examples/01_Duffing/` |
| 02 | 2-DOF chain | Cubic spring | HB | `examples/02_two_dof_cubic/` |
| 03 | 2-DOF chain | Unilateral spring | HB + Shooting | `examples/03_two_dof_unilateral/` |
| 04 | 2-DOF chain | Tanh dry friction | NMA backbone | `examples/04_two_dof_tanh_friction/` |
| 05 | Single DOF geometric | Polynomial stiffness | HB + NMA | `examples/05_geometric_nonlinearity/` |
| 06 | 3-DOF chain | Elastic dry friction (Jenkins) | HB | `examples/06_multi_dof_multi_nl/` |
| 07 | FE beam | Tanh dry friction at tip | HB | `examples/07_beam_tanh_friction/` |
| 08 | FE beam | Cubic spring at tip | NMA backbone | `examples/08_beam_cubic_spring_nma/` |

## Running Examples

Each example can be run as a Python script:

```bash
python examples/01_Duffing/run.py
```

Or explored interactively in tutorial notebooks:

```bash
jupyter notebook examples/demo/
```

## Example Categories

### Forced Response (Harmonic Balance)

Examples demonstrating frequency response analysis with external forcing:
- **01 Duffing**: Classic hardening spring behavior
- **02 Two-DOF Cubic**: Multi-DOF forced response
- **03 Two-DOF Unilateral**: Nonsmooth contact nonlinearity
- **06 Multi-DOF Jenkins**: Dry friction in 3-DOF system
- **07 Beam Tanh Friction**: Continuous structure with friction

### Nonlinear Normal Modes (NMA)

Examples computing backbone curves without external forcing:
- **04 Two-DOF Tanh Friction**: Energy-dependent friction behavior
- **05 Geometric Nonlinearity**: Polynomial stiffness
- **08 Beam Cubic NMA**: Continuous structure backbone

### Finite Element Systems

Examples using FE discretization:
- **07 Beam Tanh Friction**: Euler-Bernoulli beam with tip friction
- **08 Beam Cubic NMA**: Euler-Bernoulli beam with tip spring

## Validation

All examples are validated against MATLAB reference outputs. See:
- `examples/comparison/` - Jupyter notebooks comparing Python vs MATLAB results
- [Validation page](../validation.md) - Quantitative error metrics

## Tutorial Notebooks

The `examples/demo/` directory contains tutorial notebooks covering:
- `00_quickstart.ipynb` - Quick introduction
- `01_nonlinear_elements.ipynb` - Defining nonlinearities
- `02_mechanical_systems.ipynb` - Building systems
- `03_harmonic_balance.ipynb` - HB solver usage
- `04_shooting.ipynb` - Shooting method
- `05_continuation.ipynb` - Arc-length continuation
- `06_visualization.ipynb` - Plotting results
- `07_parameter_study.ipynb` - Parametric sweeps
- `08_cms_reduction.ipynb` - Component mode synthesis

## Example Structure

Each example directory contains:
```
01_Duffing/
├── run.py              # Main Python script
├── system_config.py    # System definition
└── README.md           # Example description
```

## Next Steps

1. Start with the [Duffing oscillator](01_duffing.md) for a simple introduction
2. Explore [tutorial notebooks](../../examples/demo/) for guided learning
3. Review the [User Guide](../user-guide/nonlinear-elements.md) for concepts
4. Check [API Reference](../api/nonlinearities.md) for detailed documentation
