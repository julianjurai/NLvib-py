# Solvers

The `nlvib.solvers` package provides two families of periodic-solution solvers:

- **Harmonic Balance** (`nlvib.solvers.harmonic_balance`) — frequency-domain formulation
  using the Alternating Frequency-Time (AFT) transform. Returns residual and Jacobian
  suitable for Newton iteration or arc-length continuation.
- **Shooting** (`nlvib.solvers.shooting`) — time-domain formulation using Newmark
  time integration. Returns periodic boundary-value residual and monodromy matrix.

Equation references: Krack & Gross (2019) §2 (HB/AFT), §3.2 (Shooting/Newmark).

---

## Harmonic Balance

### hb_residual

::: nlvib.solvers.harmonic_balance.hb_residual

---

### hb_residual_nma

::: nlvib.solvers.harmonic_balance.hb_residual_nma

---

## Shooting

### newmark_step

::: nlvib.solvers.shooting.newmark_step

---

### shooting_residual

::: nlvib.solvers.shooting.shooting_residual
