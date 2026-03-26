# NLvib — Nonlinear Vibration Analysis Toolbox

NLvib is a Python port of the MATLAB NLvib toolbox by Krack & Gross (2019). It provides
algorithms for computing periodic solutions of nonlinear mechanical systems via:

- **Harmonic Balance (HB)** — frequency-domain residual formulation with Alternating
  Frequency-Time (AFT) transform
- **Shooting** — time-domain periodic boundary-value formulation with Newmark integration
- **Arc-length continuation** — robust branch tracing with fold detection and adaptive step size

## Installation

```bash
pip install nlvib
```

To install with documentation build dependencies:

```bash
pip install "nlvib[docs]"
```

## Quick-Start

### Duffing oscillator frequency response (Harmonic Balance)

```python
import numpy as np
from nlvib.systems.oscillators import SingleMassOscillator
from nlvib.nonlinearities.elements import cubic_spring
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions
from nlvib.visualization.plots import plot_frf

# Define system: m=1, d=0.01, k=1 with cubic spring k3=1
system = SingleMassOscillator(m=1.0, d=0.01, k=1.0)
system.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=0))

# Excitation amplitude
f_ext = np.array([1.0])

# Wrap residual for continuation
def residual(x, omega):
    Q = x[:-1]
    return hb_residual(Q, omega, system, n_harmonics=3, excitation=f_ext)

# Trace the frequency response branch
options = ContinuationOptions(
    omega_range=(0.5, 1.8),
    ds=0.01,
    max_steps=500,
)
solver = ContinuationSolver()
result = solver.run(hb_residual, system, n_harmonics=3,
                    excitation=f_ext, options=options)

# Plot
fig = plot_frf(result, dof=0, harmonic=1)
fig.savefig("duffing_frf.png", dpi=150)
```

### Shooting method

```python
from nlvib.solvers.shooting import shooting_residual, newmark_step

# Integrate one period and compute shooting residual + monodromy matrix
R, J = shooting_residual(y0, omega=1.2, system=system, n_periods=1, n_steps=500)
```

## Package Structure

| Module | Contents |
|--------|----------|
| `nlvib.nonlinearities` | `NonlinearElement` dataclass and 5 factory functions |
| `nlvib.systems` | `MechanicalSystem` base class and 6 concrete system classes |
| `nlvib.solvers` | HB residual (forced + NMA), shooting residual, Newmark step |
| `nlvib.continuation` | `ContinuationSolver` with arc-length parametrisation |
| `nlvib.systems.cms` | Craig-Bampton and Rubin model reduction |
| `nlvib.visualization` | 8 matplotlib/plotly plot functions |
| `nlvib.io` | CalculiX FRD/mesh reader and writer |
| `nlvib.utils` | FFT transforms, AFT, dynamic scaling, arc length |

## Reference

Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*.
Springer. [https://doi.org/10.1007/978-3-030-14023-6](https://doi.org/10.1007/978-3-030-14023-6)
