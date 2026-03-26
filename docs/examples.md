# Examples

The `examples/` directory contains eight runnable scripts that reproduce the benchmark
problems from Krack & Gross (2019). Each script:

1. Constructs the mechanical system and attaches nonlinear elements
2. Runs the full continuation analysis (HB or Shooting)
3. Saves all required plots to `examples/<name>/output/` as PNG

| # | Name | System | Method | Key Plots |
|---|------|--------|--------|-----------|
| 01 | Duffing oscillator | `SingleMassOscillator` | HB + Shooting | FRF, harmonic content, time series |
| 02 | Two-DOF cubic spring | `ChainOfOscillators` | HB | FRF (both DOFs), harmonic content |
| 03 | Two-DOF unilateral spring | `ChainOfOscillators` | HB | FRF, phase portrait |
| 04 | Two-DOF tanh friction NM | `ChainOfOscillators` | HB (NMA) | Backbone curve, FRF |
| 05 | Geometric nonlinearity | `ChainOfOscillators` | HB | FRF (hardening/softening) |
| 06 | Multi-DOF multi-NL | `ChainOfOscillators` | HB | FRF all DOFs, convergence |
| 07 | Beam tanh friction | `FE_EulerBernoulliBeam` | HB | FRF, mode shape at resonance |
| 08 | Beam cubic spring NM | `FE_EulerBernoulliBeam` | HB (NMA) | Backbone curve, mode shape |

## Running an Example

```bash
cd examples/01_duffing
python run.py
# Plots saved to examples/01_duffing/output/
```

## Reference

Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*.
Springer. [https://doi.org/10.1007/978-3-030-14023-6](https://doi.org/10.1007/978-3-030-14023-6)
