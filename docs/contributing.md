# Contributing

Thank you for your interest in contributing to NLvib-py!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/julianjurai/NLvib-py.git
   cd NLvib-py
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**
   ```bash
   pytest
   ```

## Code Standards

### Type Annotations

All public functions must have type annotations:

```python
def harmonic_balance(
    system: MechanicalSystem,
    omega: float,
    forcing: float,
    n_harmonics: int = 5
) -> HarmonicBalanceResult:
    """Solve harmonic balance equations."""
    ...
```

### Docstrings

Use Google-style docstrings with equation references:

```python
def cubic_force(x: np.ndarray, k3: float) -> np.ndarray:
    """Compute cubic spring force.

    Implements Eq. (2.15) from Krack & Gross (2019):

        f_nl(x) = k3 * x^3

    Args:
        x: Displacement array
        k3: Cubic stiffness coefficient

    Returns:
        Nonlinear force array

    References:
        Krack M. & Gross J. (2019). Harmonic Balance for Nonlinear
        Vibration Problems. Springer. doi:10.1007/978-3-030-14023-6
    """
    return k3 * x**3
```

### Code Style

- **Linting**: Run `ruff check` before committing
- **Type checking**: Run `mypy src/nlvib --strict`
- **Formatting**: Use 4 spaces for indentation
- **Line length**: Max 100 characters
- **Imports**: Group stdlib, third-party, local

### Performance

- **No Python loops** in hot numerical paths
- Use **NumPy vectorization** throughout
- Use **scipy.sparse** for matrices ≥ 10×10
- Profile before optimizing

## Testing

### Test Structure

```
tests/
├── unit/           # Per-function unit tests
├── integration/    # End-to-end solver tests
└── validation/     # MATLAB comparison tests
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_nonlinearities.py

# With coverage
pytest --cov=nlvib --cov-report=html
```

### Writing Tests

```python
import pytest
import numpy as np
from nlvib.nonlinearities import CubicSpring

def test_cubic_spring_force():
    """Test cubic spring force calculation."""
    nl = CubicSpring(dof=0, stiffness=1.0)
    x = np.array([1.0, 2.0, 3.0])

    force = nl.force(x)
    expected = np.array([1.0, 8.0, 27.0])

    np.testing.assert_allclose(force, expected)
```

## Validation Against MATLAB

All new features must match MATLAB outputs within tolerance:

```python
# Load MATLAB fixture
matlab_data = np.load('tests/fixtures/duffing_hb.npz')
matlab_amplitude = matlab_data['amplitude']

# Run Python implementation
result = harmonic_balance(system, ...)

# Compare
rel_error = np.abs(result.amplitude - matlab_amplitude) / matlab_amplitude
assert np.max(rel_error) < 1e-6
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes**
   - Write code with type annotations
   - Add/update docstrings
   - Write tests

3. **Run checks**
   ```bash
   pytest
   mypy src/nlvib --strict
   ruff check src/nlvib
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```
   Then create a pull request on GitHub.

## Agent Framework (Internal)

For contributors working with the agentic development workflow:

- **Agent files**: See `agents/AGENTS.md` for agent roles and protocols
- **Task tracking**: `agents/TASKS.md` maintains the project task list
- **PM guide**: `agents/PM.md` for project management procedures

## Documentation

### Building Docs Locally

```bash
mkdocs serve
```

Then visit http://localhost:8000

### Adding Documentation

- User guides go in `docs/user-guide/`
- Examples go in `docs/examples/`
- API reference is auto-generated from docstrings

## Questions?

- Open an issue on GitHub
- Check existing issues and discussions
- Review the [original MATLAB toolbox documentation](../matlab_src/DOC/NLvibManual.pdf)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
