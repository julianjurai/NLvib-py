# Getting Started

## Installation

NLvib-py requires Python ≥ 3.11.

### From Source

```bash
git clone https://github.com/julianjurai/NLvib-py.git
cd NLvib-py
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

### Dependencies

Core dependencies:
- `numpy` - Numerical computing
- `scipy` - Scientific computing and sparse matrices
- `matplotlib` - Visualization

Development dependencies:
- `pytest` - Testing
- `mypy` - Type checking
- `ruff` - Linting

## Quick Start

### 1. Import the Library

```python
from nlvib.systems import ChainOfOscillators
from nlvib.nonlinearities import CubicSpring
from nlvib.solvers import harmonic_balance
from nlvib.continuation import arc_length_continuation
```

### 2. Define a Nonlinear System

```python
# Create a 2-DOF oscillator with cubic spring
system = ChainOfOscillators(
    n_dof=2,
    mass=[1.0, 1.0],
    stiffness=[1.0, 1.0],
    damping=[0.01, 0.01]
)

# Add cubic nonlinearity
nl_element = CubicSpring(
    dof=0,
    stiffness=0.5,
    name="cubic_spring"
)
system.add_nonlinearity(nl_element)
```

### 3. Solve with Harmonic Balance

```python
# Solve at a specific frequency
omega = 1.0  # rad/s
forcing_amplitude = 0.1
H = 5  # Number of harmonics

solution = harmonic_balance(
    system=system,
    omega=omega,
    forcing=forcing_amplitude,
    n_harmonics=H
)

print(f"Solution amplitude: {solution.amplitude}")
```

### 4. Perform Continuation

```python
# Trace the frequency response curve
omega_range = (0.5, 2.0)

results = arc_length_continuation(
    system=system,
    omega_start=omega_range[0],
    omega_end=omega_range[1],
    forcing=forcing_amplitude,
    n_harmonics=H,
    max_steps=100
)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results.omega, results.amplitude, 'b-', label='Stable')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Amplitude')
plt.title('Frequency Response Curve')
plt.legend()
plt.grid(True)
plt.show()
```

## Next Steps

- Explore the [User Guide](user-guide/nonlinear-elements.md) for detailed explanations
- Check out the [Examples](examples/index.md) for complete workflows
- Review the [API Reference](api/nonlinearities.md) for all available functions

## Running Tests

```bash
pytest
```

## Running Examples

Each example can be run as a Python script:

```bash
python examples/01_Duffing/run.py
```

Or explored in tutorial notebooks:

```bash
jupyter notebook examples/demo/
```

## Getting Help

- **Documentation**: https://nlvib-py.readthedocs.io (coming soon)
- **Issues**: https://github.com/julianjurai/NLvib-py/issues
- **Original MATLAB toolbox**: https://www.ila.uni-stuttgart.de/nlvib/
