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

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Optional Numba JIT for performance-critical inner loops
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _jenkins_loop(qnl_2: "NDArray[np.float64]", k_slip: float, f_lim: float) -> "NDArray[np.float64]":
        """Numba-JIT Jenkins state machine (sequential, cannot be vectorised)."""
        n2 = len(qnl_2)
        fnl_2 = np.zeros(n2)
        qsl = 0.0
        for ij in range(1, n2):
            f_pred = k_slip * (qnl_2[ij] - qsl)
            if abs(f_pred) >= f_lim:
                if f_pred >= 0.0:
                    fnl_2[ij] = f_lim
                else:
                    fnl_2[ij] = -f_lim
                qsl = qnl_2[ij] - fnl_2[ij] / k_slip
            else:
                fnl_2[ij] = f_pred
        return fnl_2

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    _jenkins_loop = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
EvalFn = Callable[[FloatArray, FloatArray], tuple[float, FloatArray, FloatArray]]
# Batch eval: (q_time, dq_time) both (n_dof, n_time) → f_time (n_dof, n_time)
EvalBatchFn = Callable[[FloatArray, FloatArray], FloatArray]


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

    eval_batch:
        Optional vectorised form: ``eval_batch(q_time, dq_time) -> f_time``
        where *q_time* and *dq_time* have shape ``(n_dof, n_time)`` and
        *f_time* has shape ``(n_dof, n_time)``.  When present the AFT loop
        uses this instead of calling ``eval`` once per time sample.

    target_dof:
        Optional explicit index of the DOF that receives this element's
        scalar force.  When set, :meth:`MechanicalSystem.eval_nonlinear_forces`
        uses this directly instead of inferring the target from the gradient.
        Required for multi-DOF elements (e.g. ``polynomial_stiffness``) where
        the gradient spans several DOFs but the force belongs to one specific row.
        Ignored when *force_direction* is set.

    force_direction:
        Optional unit (or weighted) direction vector of shape ``(n_dof,)``.
        When set, the scalar force *f* returned by ``eval`` is distributed
        to all DOFs as ``f_global += f * force_direction``, and the Jacobian
        row is ``df_global[i, :] += force_direction[i] * df_dq``.  Used for
        elements acting along an arbitrary direction (e.g. the Jenkins element
        with W = [−1, 1, 0] for a relative DOF spring).  For such elements,
        ``eval_batch`` is the primary evaluation path; ``eval`` returns the
        projected scalar force for the assembly fallback only.

    label:
        Human-readable identifier (used in repr and logging).
    """

    eval: EvalFn
    eval_batch: EvalBatchFn | None = None
    target_dof: int | None = None
    force_direction: FloatArray | None = field(default=None, compare=False)
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

    def _eval_batch(q_time: FloatArray, dq_time: FloatArray) -> FloatArray:
        f_time = np.zeros_like(q_time)
        f_time[dof_index, :] = k3 * q_time[dof_index, :] ** 3
        return f_time

    return NonlinearElement(eval=_eval, eval_batch=_eval_batch, label=f"cubic_spring(k3={k3}, dof={dof_index})")


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

    def _eval_batch(q_time: FloatArray, dq_time: FloatArray) -> FloatArray:
        f_time = np.zeros_like(q_time)
        dqi_t = dq_time[dof_index, :]
        f_time[dof_index, :] = c2 * dqi_t * np.abs(dqi_t)
        return f_time

    return NonlinearElement(
        eval=_eval, eval_batch=_eval_batch, label=f"quadratic_damper(c2={c2}, dof={dof_index})"
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

    def _eval_batch(q_time: FloatArray, dq_time: FloatArray) -> FloatArray:
        f_time = np.zeros_like(q_time)
        f_time[dof_index, :] = f0 * np.tanh(c * dq_time[dof_index, :])
        return f_time

    return NonlinearElement(
        eval=_eval, eval_batch=_eval_batch, label=f"tanh_dry_friction(f0={f0}, c={c}, dof={dof_index})"
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

    def _eval_batch(q_time: FloatArray, dq_time: FloatArray) -> FloatArray:
        f_time = np.zeros_like(q_time)
        penetration = q_time[dof_index, :] - gap
        f_time[dof_index, :] = k * np.maximum(penetration, 0.0)
        return f_time

    return NonlinearElement(
        eval=_eval, eval_batch=_eval_batch, label=f"unilateral_spring(k={k}, gap={gap}, dof={dof_index})"
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

    # The first entry of dof_indices is the primary DOF that receives the force.
    # This is the convention for multi-DOF polynomial elements: dof_indices[0] is
    # where the scalar force f is applied; the full gradient vector covers all
    # participating DOFs so the Jacobian row is correctly assembled.
    _target_dof = int(dof_indices_arr[0])

    def _eval_batch(q_time: FloatArray, dq_time: FloatArray) -> FloatArray:
        f_time = np.zeros_like(q_time)
        # q_local_t: shape (n_local_dofs, n_time)
        q_local_t = q_time[dof_indices_arr, :]
        # term_powers_t[m, l, t] = q_local_t[l, t]^e[m,l]
        # shape: (n_terms, n_local_dofs, n_time)
        term_powers_t = (
            q_local_t[np.newaxis, :, :]
            ** exponents_arr[:, :, np.newaxis].astype(np.float64)
        )
        # monomial_values_t[m, t] = prod_l term_powers_t[m, l, t]
        monomial_values_t = np.prod(term_powers_t, axis=1)  # (n_terms, n_time)
        # f_values_t[t] = sum_m c_m * monomial_values_t[m, t]
        f_values_t = coefficients_arr @ monomial_values_t  # (n_time,)
        # Force goes to the primary DOF (dof_indices[0])
        f_time[_target_dof, :] = f_values_t
        return f_time

    return NonlinearElement(
        eval=_eval,
        eval_batch=_eval_batch,
        target_dof=_target_dof,
        label=(
            f"polynomial_stiffness(n_terms={n_terms}, dofs={list(dof_indices_arr)})"
        ),
    )


def elastic_dry_friction(
    k_slip: float,
    f_lim: float,
    dof_index: int | None = None,
    force_direction: NDArray[np.float64] | None = None,
) -> NonlinearElement:
    r"""Create a Jenkins/Masing elastic dry friction (hysteretic) element.

    The element models a spring (stiffness *k_slip*) in series with a Coulomb
    slider (slip force *f_lim*).  The force law is:

    .. math::

        f = k_{\mathrm{slip}} \,(q_{\mathrm{nl}} - z)

    where :math:`q_{\mathrm{nl}}` is the projected displacement and :math:`z`
    is the internal slider position governed by Masing's rule:

    - *Stuck*: :math:`|f| < f_{\mathrm{lim}}` → :math:`z` unchanged,
      :math:`f = k_{\mathrm{slip}}(q - z)`.
    - *Sliding*: :math:`|f| \ge f_{\mathrm{lim}}` → :math:`f = \pm f_{\mathrm{lim}}`,
      :math:`z = q \mp f_{\mathrm{lim}} / k_{\mathrm{slip}}`.

    The AFT evaluation integrates two periods from zero initial conditions
    and uses the second (settled) period for the Fourier transform, matching
    the MATLAB NLvib ``elasticDryFriction`` implementation.

    Reference: Jenkins (1962); Masing (1926); Krack & Gross (2019) §C.2.

    Parameters
    ----------
    k_slip:
        Elastic stiffness of the stuck spring [N/m].
    f_lim:
        Coulomb slip-force limit [N].
    dof_index:
        Zero-based DOF index for an axis-aligned single-DOF element.
        Exactly one of *dof_index* or *force_direction* must be provided.
    force_direction:
        Direction vector of shape ``(n_dof,)`` for a multi-DOF element
        (e.g. ``[-1, 1, 0]`` for a relative-displacement Jenkins spring
        between DOF 0 and DOF 1).  The projected displacement is
        :math:`q_{\mathrm{nl}} = w^T q` and the global force contribution
        is :math:`\Delta f = w \cdot f_{\mathrm{nl}}`.
        Exactly one of *dof_index* or *force_direction* must be provided.

    Returns
    -------
    NonlinearElement
        Element with ``eval`` and ``eval_batch`` methods.

    Raises
    ------
    ValueError
        If neither or both of *dof_index* and *force_direction* are supplied.
    """
    if (dof_index is None) == (force_direction is None):
        raise ValueError(
            "Exactly one of dof_index or force_direction must be provided."
        )

    if force_direction is not None:
        w: NDArray[np.float64] = np.asarray(force_direction, dtype=np.float64)
    else:
        # Build axis-aligned direction from dof_index
        # (direction is used only by force_direction path; dof_index path uses
        #  the dof_index directly for performance)
        w = None  # type: ignore[assignment]

    def _jenkins_time_series(qnl_1period: FloatArray) -> FloatArray:
        """Run Jenkins state machine for two periods; return second period.

        Marches through ``[qnl_1period, qnl_1period]`` (2 concatenated
        periods) starting from zero initial conditions, then returns the
        forces for the second period only (stabilised hysteresis loop).

        Equation reference: Krack & Gross (2019) §C.2, MATLAB HB_residual.m
        ``elasticDryFriction`` branch.

        Uses Numba JIT when available for ~50× speedup on the sequential loop.
        """
        n = len(qnl_1period)
        qnl_2 = np.concatenate([qnl_1period, qnl_1period])
        if _HAVE_NUMBA:
            fnl_2 = _jenkins_loop(qnl_2, k_slip, f_lim)
        else:
            fnl_2 = np.zeros(2 * n, dtype=np.float64)
            qsl: float = 0.0  # slider position (internal state)
            for ij in range(1, 2 * n):
                f_pred = k_slip * (qnl_2[ij] - qsl)
                if abs(f_pred) >= f_lim:
                    fnl_2[ij] = f_lim * float(np.sign(f_pred))
                    qsl = qnl_2[ij] - fnl_2[ij] / k_slip
                else:
                    fnl_2[ij] = f_pred
                    # qsl unchanged (stuck)
        return fnl_2[n:]  # settled second period

    if dof_index is not None:
        # -----------------------------------------------------------------
        # Single-DOF axis-aligned element
        # -----------------------------------------------------------------
        def _eval_single(
            q: FloatArray, dq: FloatArray
        ) -> tuple[float, FloatArray, FloatArray]:
            qi = float(q[dof_index])
            f_pred = k_slip * qi
            if abs(f_pred) >= f_lim:
                fnl = float(f_lim * np.sign(f_pred))
                effective_k = 0.0  # sliding: no stiffness increment
            else:
                fnl = f_pred
                effective_k = k_slip
            df_dq = np.zeros_like(q)
            df_dq[dof_index] = effective_k
            df_ddq = np.zeros_like(dq)
            return fnl, df_dq, df_ddq

        def _eval_batch_single(
            q_time: FloatArray, dq_time: FloatArray
        ) -> FloatArray:
            qnl = q_time[dof_index, :]  # (n_time,)
            fnl = _jenkins_time_series(qnl)
            f_time = np.zeros_like(q_time)
            f_time[dof_index, :] = fnl
            return f_time

        return NonlinearElement(
            eval=_eval_single,
            eval_batch=_eval_batch_single,
            target_dof=dof_index,
            label=f"elastic_dry_friction(k={k_slip}, f_lim={f_lim}, dof={dof_index})",
        )

    else:
        # -----------------------------------------------------------------
        # Multi-DOF force-direction element  (e.g. W = [-1, 1, 0])
        # -----------------------------------------------------------------
        def _eval_dir(
            q: FloatArray, dq: FloatArray
        ) -> tuple[float, FloatArray, FloatArray]:
            # Projected displacement: qnl = w' * q
            qnl = float(w @ q)
            f_pred = k_slip * qnl
            if abs(f_pred) >= f_lim:
                fnl = float(f_lim * np.sign(f_pred))
                effective_k = 0.0
            else:
                fnl = f_pred
                effective_k = k_slip
            # Gradient of scalar fnl w.r.t. q: dFnl/dq = effective_k * w
            df_dq = w * effective_k
            df_ddq = np.zeros_like(dq)
            # Return scalar fnl; assembly distributes via force_direction
            return fnl, df_dq, df_ddq

        def _eval_batch_dir(
            q_time: FloatArray, dq_time: FloatArray
        ) -> FloatArray:
            # qnl = w' * q_time, shape (n_time,)
            qnl = w @ q_time  # (n_time,)
            fnl = _jenkins_time_series(qnl)  # (n_time,)
            # Distribute: f_time[i, :] = w[i] * fnl
            return np.outer(w, fnl)  # (n_dof, n_time)

        return NonlinearElement(
            eval=_eval_dir,
            eval_batch=_eval_batch_dir,
            target_dof=None,
            force_direction=w,
            label=(
                f"elastic_dry_friction(k={k_slip}, f_lim={f_lim}, "
                f"dir={list(force_direction)})"  # type: ignore[arg-type]
            ),
        )
