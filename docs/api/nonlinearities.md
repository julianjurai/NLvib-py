# Nonlinear Elements

The `nlvib.nonlinearities` module defines the `NonlinearElement` dataclass and six factory
functions that construct standard nonlinear force/velocity elements for use with any
`MechanicalSystem` subclass.

Each element encapsulates a force function `f(q, dq) -> (f_nl, df_dq, df_ddq)` and the
DOF index (or indices) it acts on. Element implementations follow the conventions of
Krack & Gross (2019), Appendix C, Table C.1.

---

## NonlinearElement

::: nlvib.nonlinearities.elements.NonlinearElement

---

## Factory Functions

### cubic_spring

::: nlvib.nonlinearities.elements.cubic_spring

---

### quadratic_damper

::: nlvib.nonlinearities.elements.quadratic_damper

---

### tanh_dry_friction

::: nlvib.nonlinearities.elements.tanh_dry_friction

---

### unilateral_spring

::: nlvib.nonlinearities.elements.unilateral_spring

---

### polynomial_stiffness

::: nlvib.nonlinearities.elements.polynomial_stiffness

---

### elastic_dry_friction

::: nlvib.nonlinearities.elements.elastic_dry_friction
