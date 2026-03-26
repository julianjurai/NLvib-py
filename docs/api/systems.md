# Mechanical Systems

The `nlvib.systems` package provides the `MechanicalSystem` base class and five concrete
system implementations covering lumped-parameter models and finite-element discretisations.

All system matrices (M, D, K) are stored as `scipy.sparse.csr_matrix`. Nonlinear elements
are attached via `add_nonlinear_element()` and evaluated by `eval_nonlinear_forces()`.

---

## MechanicalSystem

Base class that owns the mass, damping, and stiffness matrices, manages nonlinear element
attachment, and provides the force evaluation interface required by all solvers.

::: nlvib.systems.base.MechanicalSystem

---

## SingleMassOscillator

::: nlvib.systems.oscillators.SingleMassOscillator

---

## ChainOfOscillators

::: nlvib.systems.oscillators.ChainOfOscillators

---

## FE_EulerBernoulliBeam

::: nlvib.systems.fe_beam.FE_EulerBernoulliBeam

---

## FE_ElasticRod

::: nlvib.systems.fe_rod.FE_ElasticRod

---

## System_with_PolynomialStiffness

::: nlvib.systems.polynomial.System_with_PolynomialStiffness
