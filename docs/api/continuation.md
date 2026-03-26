# Continuation

The `nlvib.continuation` module implements arc-length continuation for tracing solution
branches of parameter-dependent systems of nonlinear equations.

Key features:

- Tangent predictor with arc-length parametrisation (Krack & Gross (2019) §4)
- Schur complement bordered system solve for the extended Jacobian
- Fold detection via sign change of the tangent component in the parameter direction
- Adaptive step size: doubles if Newton converges in <5 iterations, halves if >9

---

## ContinuationOptions

::: nlvib.continuation.solver.ContinuationOptions

---

## ContinuationResult

::: nlvib.continuation.solver.ContinuationResult

---

## ContinuationSolver

::: nlvib.continuation.solver.ContinuationSolver
