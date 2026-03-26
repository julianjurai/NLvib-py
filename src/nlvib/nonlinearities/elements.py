"""
Nonlinear element functions for NLvib.

Each factory function returns a :class:`NonlinearElement` whose ``eval(q, dq)``
method returns a 3-tuple ``(f, df_dq, df_ddq)``:

- ``f``       – scalar nonlinear force contribution
- ``df_dq``   – gradient of *f* w.r.t. **q** (1-D array, same length as *q*)
- ``df_ddq``  – gradient of *f* w.r.t. **dq** (1-D array, same length as *dq*)

Equation references: Krack & Gross (2019) *Harmonic Balance for Nonlinear Vibration
Problems*, Appendix C, Table C.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
EvalFn = Callable[[FloatArray, FloatArray], tuple[float, FloatArray, FloatArray]]


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonlinearElement:
    """Container for a nonlinear element evaluation function.

    Parameters
    ----------
    eval:
        Callable ``eval(q, dq) -> (f, df_dq, df_ddq)`` where

        - *q*  is the displacement state vector (shape ``(n_dof,)``),
        - *dq* is the velocity state vector (shape ``(n_dof,)``),
        - *f*  is the scalar nonlinear force,
        - *df_dq*  is the gradient of *f* w.r.t. *q* (shape ``(n_dof,)``),
        - *df_ddq* is the gradient of *f* w.r.t. *dq* (shape ``(n_dof,)``).

    label:
        Human-readable identifier (used in repr and logging).
    """

    eval: EvalFn
    label: str = "NonlinearElement"

    def __call__(
        self, q: FloatArray, dq: FloatArray
    ) -> tuple[float, FloatArray, FloatArray]:
        """Convenience: call ``self.eval(q, dq)`` directly."""
        return self.eval(q, dq)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def cubic_spring(k3: float, dof_index: int) -> NonlinearElement:
    """Create a cubic (Duffing-type) spring attached to one DOF.

    The restoring force is

    .. math::

        f = k_3 \\, q_i^3

    with Jacobians

    .. math::

        \\frac{\\partial f}{\\partial q_i} = 3 k_3 q_i^2, \\quad
        \\frac{\\partial f}{\\partial \\dot{q}_j} = 0 \\; \\forall j.

    Reference: Krack & Gross (2019) Appendix C, Table C.1 — Cubic spring.

    Parameters
    ----------
    k3:
        Cubic stiffness coefficient [N/m³].
    dof_index:
        Zero-based index of the DOF to which this spring is attached.

    Returns
    -------
    NonlinearElement
        Element with ``eval(q, dq) -> (f, df_dq, df_ddq)``.
    """

    def _eval(q: FloatArray, dq: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        qi = q[dof_index]
        f: float = float(k3 * qi**3)
        df_dq = np.zeros_like(q)
        df_dq[dof_index] = 3.0 * k3 * qi**2
        df_ddq = np.zeros_like(dq)
        return f, df_dq, df_ddq

    return NonlinearElement(eval=_eval, label=f"cubic_spring(k3={k3}, dof={dof_index})")


def quadratic_damper(c2: float, dof_index: int) -> NonlinearElement:
    r"""Create a quadratic (velocity-squared) damper attached to one DOF.

    The dissipative force is

    .. math::

        f = c_2 \, \dot{q}_i \, |\dot{q}_i|

    with Jacobians

    .. math::

        \frac{\partial f}{\partial q_j} = 0 \; \forall j, \quad
        \frac{\partial f}{\partial \dot{q}_i} = 2 c_2 |\dot{q}_i|.

    Note: the sign convention preserves the direction of the damping force
    (opposing velocity) via :math:`\dot{q}_i |\dot{q}_i|` rather than
    :math:`|\dot{q}_i|^2`.

    Reference: Krack & Gross (2019) Appendix C, Table C.1 — Quadratic damper.

    Parameters
    ----------
    c2:
        Quadratic damping coefficient [N·s²/m²].
    dof_index:
        Zero-based index of the DOF to which this damper is attached.

    Returns
    -------
    NonlinearElement
        Element with ``eval(q, dq) -> (f, df_dq, df_ddq)``.
    """

    def _eval(q: FloatArray, dq: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        dqi = dq[dof_index]
        f: float = float(c2 * dqi * np.abs(dqi))
        df_dq = np.zeros_like(q)
        df_ddq = np.zeros_like(dq)
        df_ddq[dof_index] = 2.0 * c2 * np.abs(dqi)
        return f, df_dq, df_ddq

    return NonlinearElement(
        eval=_eval, label=f"quadratic_damper(c2={c2}, dof={dof_index})"
    )


def tanh_dry_friction(f0: float, c: float, dof_index: int) -> NonlinearElement:
    r"""Create a hyperbolic-tangent (smooth dry-friction) element on one DOF.

    The friction force is

    .. math::

        f = f_0 \, \tanh(c \, \dot{q}_i)

    with Jacobians

    .. math::

        \frac{\partial f}{\partial q_j} = 0 \; \forall j, \quad
        \frac{\partial f}{\partial \dot{q}_i} =
            f_0 \, c \, \operatorname{sech}^2(c \, \dot{q}_i).

    The parameter *c* controls the sharpness of the transition from sticking
    to sliding; large *c* approximates ideal Coulomb friction.

    Reference: Krack & Gross (2019) Appendix C, Table C.1 — Tanh dry friction.

    Parameters
    ----------
    f0:
        Maximum (slip) friction force [N].
    c:
        Regularisation slope [s/m].  Large values ≈ ideal Coulomb friction.
    dof_index:
        Zero-based index of the DOF to which this element is attached.

    Returns
    -------
    NonlinearElement
        Element with ``eval(q, dq) -> (f, df_dq, df_ddq)``.
    """

    def _eval(q: FloatArray, dq: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        dqi = dq[dof_index]
        tanh_val = np.tanh(c * dqi)
        f: float = float(f0 * tanh_val)
        df_dq = np.zeros_like(q)
        df_ddq = np.zeros_like(dq)
        # sech²(x) = 1 - tanh²(x)
        df_ddq[dof_index] = f0 * c * (1.0 - tanh_val**2)
        return f, df_dq, df_ddq

    return NonlinearElement(
        eval=_eval, label=f"tanh_dry_friction(f0={f0}, c={c}, dof={dof_index})"
    )


def unilateral_spring(k: float, gap: float, dof_index: int) -> NonlinearElement:
    r"""Create a unilateral (contact) spring attached to one DOF.

    The contact force is active only when the displacement exceeds the gap:

    .. math::

        f = k \, \max(q_i - \delta, \; 0)

    with Jacobians

    .. math::

        \frac{\partial f}{\partial q_i} =
            \begin{cases} k & q_i > \delta \\ 0 & q_i \leq \delta \end{cases},
        \quad
        \frac{\partial f}{\partial \dot{q}_j} = 0 \; \forall j.

    At the contact point :math:`q_i = \delta` the derivative is taken as 0
    (sub-gradient choice consistent with the MATLAB NLvib implementation).

    Reference: Krack & Gross (2019) Appendix C, Table C.1 — Unilateral spring.

    Parameters
    ----------
    k:
        Contact stiffness [N/m].
    gap:
        Gap (clearance) :math:`\delta` [m].  Contact occurs when
        ``q[dof_index] > gap``.
    dof_index:
        Zero-based index of the DOF to which this spring is attached.

    Returns
    -------
    NonlinearElement
        Element with ``eval(q, dq) -> (f, df_dq, df_ddq)``.
    """

    def _eval(q: FloatArray, dq: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        qi = q[dof_index]
        penetration = qi - gap
        in_contact = penetration > 0.0
        f: float = float(k * penetration) if in_contact else 0.0
        df_dq = np.zeros_like(q)
        if in_contact:
            df_dq[dof_index] = k
        df_ddq = np.zeros_like(dq)
        return f, df_dq, df_ddq

    return NonlinearElement(
        eval=_eval, label=f"unilateral_spring(k={k}, gap={gap}, dof={dof_index})"
    )


def polynomial_stiffness(
    exponents: NDArray[np.intp],
    coefficients: NDArray[np.float64],
    dof_indices: NDArray[np.intp],
) -> NonlinearElement:
    r"""Create a polynomial stiffness element spanning multiple DOFs.

    The force is a sum of monomials in the displacements of the specified DOFs:

    .. math::

        f = \sum_{m=1}^{M} c_m \prod_{l=1}^{L} q_{i_l}^{e_{m,l}}

    where :math:`M` is the number of terms, :math:`L` is the number of
    participating DOFs, :math:`c_m` are the coefficients, :math:`e_{m,l}` are
    the (integer) exponents, and :math:`i_l` are the global DOF indices.

    Jacobian w.r.t. :math:`q_{i_k}`:

    .. math::

        \frac{\partial f}{\partial q_{i_k}} =
            \sum_{m=1}^{M} c_m \, e_{m,k} \, q_{i_k}^{e_{m,k}-1}
            \prod_{l \neq k} q_{i_l}^{e_{m,l}}

    All velocity Jacobians are zero.

    Reference: Krack & Gross (2019) Appendix C, Table C.1 — Polynomial stiffness.

    Parameters
    ----------
    exponents:
        Integer array of shape ``(M, L)`` where *M* is the number of monomial
        terms and *L* is the number of participating DOFs.  Each row gives the
        exponents for one monomial.
    coefficients:
        Float array of shape ``(M,)`` — one coefficient per monomial term.
    dof_indices:
        Integer array of shape ``(L,)`` — global (zero-based) DOF indices of
        the participating DOFs, in the same column order as *exponents*.

    Returns
    -------
    NonlinearElement
        Element with ``eval(q, dq) -> (f, df_dq, df_ddq)``.

    Raises
    ------
    ValueError
        If array shapes are inconsistent.
    """
    exponents_arr = np.asarray(exponents, dtype=np.intp)
    coefficients_arr = np.asarray(coefficients, dtype=np.float64)
    dof_indices_arr = np.asarray(dof_indices, dtype=np.intp)

    n_terms, n_local_dofs = exponents_arr.shape
    if coefficients_arr.shape != (n_terms,):
        raise ValueError(
            f"coefficients must have shape ({n_terms},), "
            f"got {coefficients_arr.shape}"
        )
    if dof_indices_arr.shape != (n_local_dofs,):
        raise ValueError(
            f"dof_indices must have shape ({n_local_dofs},), "
            f"got {dof_indices_arr.shape}"
        )

    def _eval(q: FloatArray, dq: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        # q_local: shape (n_local_dofs,)
        q_local = q[dof_indices_arr]

        # term_powers[m, l] = q_local[l] ** exponents[m, l]
        # Shape: (n_terms, n_local_dofs)  — exponents_arr is (n_terms, n_local_dofs)
        term_powers: NDArray[np.float64] = (
            q_local[np.newaxis, :] ** exponents_arr.astype(np.float64)
        )

        # monomial_values[m] = prod_l term_powers[m, l]
        monomial_values: NDArray[np.float64] = np.prod(term_powers, axis=1)  # (n_terms,)

        f: float = float(np.dot(coefficients_arr, monomial_values))

        df_dq = np.zeros_like(q)

        # Jacobian w.r.t. q_local[k]:
        #   df/dq[i_k] = sum_m c_m * e[m,k] * q_local[k]^(e[m,k]-1)
        #                         * prod_{l!=k} q_local[l]^e[m,l]
        #
        # Implementation: compute deriv_factor[m,k] = e[m,k] * q_local[k]^(e[m,k]-1)
        # using the recurrence: deriv_factor[m,k] = e[m,k] * term_powers[m,k] / q_local[k]
        # but handle q_local[k]==0 carefully with case analysis on exponent:
        #   e==0: derivative is 0                    (q^-1 * 0 = 0)
        #   e==1: derivative is 1                    (q^0 = 1)
        #   e>=2: derivative is e * q_local[k]^(e-1) (→ 0 when q_local[k]==0)
        # All cases collapse to: e[m,k] * q_local[k]^max(e[m,k]-1, 0) when q_local[k]=0
        # requires special treatment only for e==0 (0^(-1) is undefined but multiplied by 0).
        deriv_exp: NDArray[np.float64] = np.maximum(
            exponents_arr.astype(np.float64) - 1.0, 0.0
        )  # (n_terms, n_local_dofs)

        # q_local_safe: replace 0 with 1 to avoid 0^negative_power; zeros corrected below.
        q_local_safe = np.where(q_local == 0.0, 1.0, q_local)  # (n_local_dofs,)

        # deriv_factor[m,k] = e[m,k] * q_local[k]^(e[m,k]-1)
        # Handles q_local[k]==0 correctly:
        #   e==0: coefficient multiplied later is 0 anyway (masked out below)
        #   e==1: q_local[k]^0 = 1 ✓ (safe_q gives 1 when q_local==0, deriv_exp=0, so 1^0=1)
        #   e>=2: q_local[k]^(e-1) = 0 ✓ when q_local[k]==0 — BUT deriv_exp>=1 so
        #         q_local_safe^(e-1) = 1 when q_local==0. We correct this via rest_product mask.
        deriv_factor: NDArray[np.float64] = (
            exponents_arr.astype(np.float64) * q_local_safe[np.newaxis, :] ** deriv_exp
        )  # (n_terms, n_local_dofs)

        # rest_product[m,k] = prod_{l!=k} q_local[l]^e[m,l]
        #
        # General formula: monomial_values[m] / term_powers[m,k], but this breaks when
        # term_powers[m,k] == 0 (i.e. q_local[k]==0 and e[m,k]>=1) because the monomial
        # is 0 yet the product of the OTHER factors may be non-zero.
        #
        # Fix: when term_powers[m,k] == 0 AND e[m,k] >= 1, compute rest_product directly
        # as the product of all columns except k using log-sum or explicit division.
        # We use a cumulative-product left/right pass to avoid any loop over k.
        #
        # cumprod_left[m, k]  = prod_{l < k}  term_powers[m, l]
        # cumprod_right[m, k] = prod_{l > k}  term_powers[m, l]
        # rest_product[m, k]  = cumprod_left[m, k] * cumprod_right[m, k]
        cumprod_left: NDArray[np.float64] = np.ones((n_terms, n_local_dofs))
        cumprod_right: NDArray[np.float64] = np.ones((n_terms, n_local_dofs))
        if n_local_dofs > 1:
            cumprod_left[:, 1:] = np.cumprod(term_powers[:, :-1], axis=1)
            cumprod_right[:, :-1] = np.cumprod(term_powers[:, :0:-1], axis=1)[:, ::-1]
        rest_product: NDArray[np.float64] = cumprod_left * cumprod_right  # (n_terms, n_local_dofs)

        deriv_contributions: NDArray[np.float64] = (
            coefficients_arr[:, np.newaxis] * deriv_factor * rest_product
        )  # (n_terms, n_local_dofs)

        # Zero out contributions where e[m,k]==0 (constant w.r.t. that DOF)
        # and where e[m,k]>=2 AND q_local[k]==0 (deriv_factor incorrectly = e*1 via safe_q)
        # In both these cases the true derivative is 0.
        zero_mask = (exponents_arr == 0) | (
            (exponents_arr >= 2) & (q_local[np.newaxis, :] == 0.0)
        )
        deriv_contributions = np.where(zero_mask, 0.0, deriv_contributions)

        # Sum over monomial terms → shape (n_local_dofs,)
        local_jac: NDArray[np.float64] = np.sum(deriv_contributions, axis=0)

        # Scatter to global df_dq using indexed assignment (no loop)
        df_dq[dof_indices_arr] += local_jac

        df_ddq = np.zeros_like(dq)
        return f, df_dq, df_ddq

    return NonlinearElement(
        eval=_eval,
        label=(
            f"polynomial_stiffness(n_terms={n_terms}, dofs={list(dof_indices_arr)})"
        ),
    )
