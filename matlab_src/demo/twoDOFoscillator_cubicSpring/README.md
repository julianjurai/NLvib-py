# Two-DOF Oscillator with Cubic Spring — MATLAB/Octave Demo

Reference run of the original NLvib MATLAB example for comparison against the Python port.

## What it does

Computes the nonlinear frequency response of a 2-DOF chain oscillator with a cubic spring nonlinearity using:
- **Harmonic Balance (HB)** — fast, ~0.2s
- **Shooting method** — slower, ~16s, used to verify HB
- **Floquet stability analysis** — detects turning points and Neimark-Sacker bifurcations

Output: `frequency_response.png`

## Requirements

No full MATLAB install is needed. **GNU Octave** works fine.

```bash
brew install octave   # if not already installed
```

## Run

```bash
cd matlab/demo/twoDOFoscillator_cubicSpring
octave --no-gui twoDOFoscillator_cubicSpring.m
```

Or from the repo root:

```bash
octave --no-gui matlab/demo/twoDOFoscillator_cubicSpring/twoDOFoscillator_cubicSpring.m
```

Expected runtime: ~22 seconds. The plot is saved as `frequency_response.png` in this directory.

## Files

| File | Description |
|------|-------------|
| `twoDOFoscillator_cubicSpring.m` | Main example script |
| `HB_residual.m` | Harmonic Balance residual + AFT |
| `shooting_residual.m` | Shooting residual + Newmark integrator |
| `solve_and_continue.m` | Arc-length continuation solver |
| `ChainOfOscillators.m` | 2-DOF chain system class |
| `MechanicalSystem.m` | Base mechanical system class |
| `SingleMassOscillator.m` | SDOF oscillator class |
| `FE_EulerBernoulliBeam.m` | FE beam class |
| `FE_ElasticRod.m` | FE rod class |
| `FEmodel.m` | FE model base class |
| `CMS_ROM.m` | Craig-Bampton/Rubin model reduction |
| `System_with_PolynomialStiffnessNonlinearity.m` | Polynomial stiffness system |
| `frequency_response.png` | Output plot from last run |

## Compare against Python

The Python equivalent is at `examples/02_two_dof_cubic_spring/run.py`:

```bash
python examples/02_two_dof_cubic_spring/run.py
```

Output saved to `examples/02_two_dof_cubic_spring/output/`.
