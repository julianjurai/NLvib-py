"""
Shooting method and Newmark time integrator for NLvib.

This module provides:

- :func:`newmark_step` — single time step of the Newmark average constant
  acceleration scheme (β = 1/4, γ = 1/2), displacement-form formulation.
- :func:`shooting_residual` — periodicity condition
  ``R = y(T) - y(0)`` and the monodromy-based Jacobian for use in a
  Newton continuation loop.

Displacement-form Newmark
--------------------------
Given :math:`q_n, \\dot q_n, \\ddot q_n` and :math:`f_{n+1}`, solve

.. math::

    K_{\\mathrm{eff}}\\, q_{n+1}
    = f_{n+1} - f_{\\mathrm{nl}}(q^*_{n+1}, \\dot q^*_{n+1})
      + \\left(\\frac{M}{\\beta\\Delta t^2}
        + \\frac{\\gamma}{\\beta\\Delta t} D\\right) q^*_{n+1}
      - D\\, \\dot q^*_{n+1}

where the predictors are

.. math::

    q^*_{n+1} &= q_n + \\Delta t\\,\\dot q_n
                + \\Delta t^2\\left(\\frac12-\\beta\\right)\\ddot q_n \\\\
    \\dot q^*_{n+1} &= \\dot q_n
                + \\Delta t\\left(1-\\gamma\\right)\\ddot q_n

and

.. math::

    K_{\\mathrm{eff}} = \\frac{M}{\\beta\\Delta t^2}
        + \\frac{\\gamma}{\\beta\\Delta t} D + K.

Then recover

.. math::

    \\ddot q_{n+1} &= \\frac{q_{n+1} - q^*_{n+1}}{\\beta\\Delta t^2} \\\\
    \\dot q_{n+1}  &= \\dot q^*_{n+1} + \\gamma\\Delta t\\,\\ddot q_{n+1}.

Equation references
-------------------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  ISBN 978-3-030-14022-9.

- Newmark scheme:    K&G §3.2, equations (3.9)–(3.11)
- Shooting residual: K&G §3.2, equation (3.7)
- Monodromy matrix:  K&G §3.2, equations (3.12)–(3.14)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from nlvib.systems.base import MechanicalSystem

__all__ = ["newmark_step", "shooting_residual"]

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Named constants — Newmark average constant acceleration coefficients
# ---------------------------------------------------------------------------

#: Newmark β parameter — average constant acceleration (unconditionally stable).
NEWMARK_BETA: float = 0.25

#: Newmark γ parameter — average constant acceleration (second-order accurate).
NEWMARK_GAMMA: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_dense(mat: FloatArray | sp.spmatrix) -> FloatArray:
    """Convert a sparse or dense matrix to a dense NumPy float64 array."""
    if sp.issparse(mat):
        sparse_mat: sp.spmatrix = mat
        result: FloatArray = sparse_mat.toarray()
        return result
    return np.asarray(mat, dtype=np.float64)


# ---------------------------------------------------------------------------
# Newmark single-step integrator  (displacement-form)
# ---------------------------------------------------------------------------


def newmark_step(
    y: FloatArray,
    f_ext: FloatArray,
    M: FloatArray | sp.spmatrix,
    D: FloatArray | sp.spmatrix,
    K: FloatArray | sp.spmatrix,
    nonlinear_forces_fn: Callable[[FloatArray, FloatArray], FloatArray],
    dt: float,
    beta: float = NEWMARK_BETA,
    gamma: float = NEWMARK_GAMMA,
    ddq_n: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Advance the state by one time step using the displacement-form Newmark scheme.

    Implements the Newmark average constant acceleration method (β = 1/4,
    γ = 1/2) applied to

    .. math::

        M \\ddot{q} + D \\dot{q} + K q + f_{\\mathrm{nl}}(q, \\dot{q})
        = f_{\\mathrm{ext}}

    The **displacement form** is used: :math:`q_{n+1}` is the primary unknown,
    solved from (K&G §3.2, eq. 3.11):

    .. math::

        K_{\\mathrm{eff}} q_{n+1}
        = f_{n+1} - f_{\\mathrm{nl}}(q^*, \\dot{q}^*)
          + \\left(\\frac{M}{\\beta \\Delta t^2}
            + \\frac{\\gamma}{\\beta \\Delta t} D\\right) q^*
          - D \\dot{q}^*

    where the predictors are

    .. math::

        q^* &= q_n + \\Delta t \\dot{q}_n
               + \\Delta t^2 (\\tfrac{1}{2}-\\beta) \\ddot{q}_n \\\\
        \\dot{q}^* &= \\dot{q}_n + \\Delta t (1-\\gamma) \\ddot{q}_n

    and :math:`K_{\\mathrm{eff}} = M/(\\beta \\Delta t^2)
    + \\gamma/(\\beta \\Delta t) D + K`.

    Then the corrector recovers

    .. math::

        \\ddot{q}_{n+1} &= (q_{n+1} - q^*) / (\\beta \\Delta t^2) \\\\
        \\dot{q}_{n+1} &= \\dot{q}^* + \\gamma \\Delta t \\ddot{q}_{n+1}

    Parameters
    ----------
    y:
        State vector ``[q, dq]`` of shape ``(2 * n_dof,)`` at time :math:`t_n`.
    f_ext:
        External force vector of shape ``(n_dof,)`` evaluated at
        :math:`t_{n+1}`.
    M:
        Mass matrix of shape ``(n_dof, n_dof)``.
    D:
        Damping matrix of shape ``(n_dof, n_dof)``.
    K:
        Stiffness matrix of shape ``(n_dof, n_dof)``.
    nonlinear_forces_fn:
        Callable ``(q, dq) -> f_nl`` returning the nonlinear force vector of
        shape ``(n_dof,)`` (e.g. ``system.eval_nonlinear_forces`` wrapped to
        return only the force).
    dt:
        Time step :math:`\\Delta t` [s].
    beta:
        Newmark :math:`\\beta` parameter.  Default ``0.25``.
    gamma:
        Newmark :math:`\\gamma` parameter.  Default ``0.5``.
    ddq_n:
        Acceleration :math:`\\ddot{q}_n` of shape ``(n_dof,)`` at
        :math:`t_n`.  If ``None``, it is computed from the EOM at :math:`t_n`
        using :math:`f_{n+1}` as the force (a good approximation when
        :math:`\\Delta t` is small or forces vary slowly).  For accurate
        chained integration, pass the ``ddq_next`` returned by the previous
        call.

    Returns
    -------
    y_next : ndarray, shape ``(2 * n_dof,)``
        State vector ``[q_{n+1}, dq_{n+1}]`` at time :math:`t_{n+1}`.
    ddq_next : ndarray, shape ``(n_dof,)``
        Acceleration :math:`\\ddot{q}_{n+1}` at time :math:`t_{n+1}`.
        Pass back as ``ddq_n`` on the subsequent call.
    """
    y_arr: FloatArray = np.asarray(y, dtype=np.float64)
    n_dof: int = y_arr.size // 2
    q_n: FloatArray = y_arr[:n_dof]
    dq_n: FloatArray = y_arr[n_dof:]
    f_ext_arr: FloatArray = np.asarray(f_ext, dtype=np.float64)

    M_d: FloatArray = _to_dense(M)
    D_d: FloatArray = _to_dense(D)
    K_d: FloatArray = _to_dense(K)

    # Obtain ddq_n.  Callers that chain steps should pass ddq_next from the
    # previous call for exact second-order accuracy.
    if ddq_n is not None:
        ddq_n_arr: FloatArray = np.asarray(ddq_n, dtype=np.float64)
    else:
        # K&G §3.2: M*ddq_n = f_n - D*dq_n - K*q_n - f_nl(q_n, dq_n)
        # Approximate f_n ≈ f_{n+1} (suitable when Δt is small).
        f_nl_0: FloatArray = nonlinear_forces_fn(q_n, dq_n)
        ddq_n_arr = la.solve(
            M_d, f_ext_arr - D_d @ dq_n - K_d @ q_n - f_nl_0
        )

    dt2: float = dt * dt

    # Predictors (K&G §3.2, eq. 3.10)
    q_pred: FloatArray = q_n + dt * dq_n + dt2 * (0.5 - beta) * ddq_n_arr
    dq_pred: FloatArray = dq_n + dt * (1.0 - gamma) * ddq_n_arr

    # Effective stiffness (K&G §3.2, eq. 3.11)
    K_eff: FloatArray = M_d / (beta * dt2) + (gamma / (beta * dt)) * D_d + K_d

    # Effective RHS — displacement form (K&G §3.2, eq. 3.11):
    #   K_eff * q_{n+1} = f_{n+1} - f_nl(q_pred, dq_pred)
    #                   + [M/(β*dt²) + (γ/(β*dt))*D] * q_pred - D * dq_pred
    f_nl_pred: FloatArray = nonlinear_forces_fn(q_pred, dq_pred)
    A_pred: FloatArray = M_d / (beta * dt2) + (gamma / (beta * dt)) * D_d
    rhs_eff: FloatArray = (
        f_ext_arr - f_nl_pred + A_pred @ q_pred - D_d @ dq_pred
    )

    q_next: FloatArray = la.solve(K_eff, rhs_eff)

    # Recover acceleration and velocity from corrector (K&G §3.2, eq. 3.9)
    ddq_next: FloatArray = (q_next - q_pred) / (beta * dt2)
    dq_next: FloatArray = dq_pred + gamma * dt * ddq_next

    y_next: FloatArray = np.concatenate([q_next, dq_next])
    return y_next, ddq_next


# ---------------------------------------------------------------------------
# Shooting residual
# ---------------------------------------------------------------------------


def shooting_residual(
    y0: FloatArray,
    omega: float,
    system: MechanicalSystem,
    n_periods: int = 1,
    n_steps: int = 200,
    f_ext_fn: Callable[[float], FloatArray] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute the shooting residual and monodromy-based Jacobian.

    Integrates the system for ``n_periods`` forcing periods using
    :func:`newmark_step` and evaluates the periodicity condition

    .. math::

        R(y_0) = y(T_{\\mathrm{total}}) - y_0

    where :math:`T_{\\mathrm{total}} = n_{\\mathrm{periods}} \\cdot T` and
    :math:`T = 2\\pi / \\omega`.

    The Jacobian of the residual is

    .. math::

        J = \\Phi - I

    where :math:`\\Phi` is the monodromy matrix (sensitivity of the final
    state to the initial state), computed by simultaneously integrating the
    variational equations (K&G §3.2, eq. 3.12–3.14):

    .. math::

        \\dot{\\Phi} = A(t)\\, \\Phi, \\quad \\Phi(0) = I

    At each Newmark step the linearised state-transition map
    :math:`S_n = \\partial y_{n+1} / \\partial y_n` (K&G §3.2, eq. 3.14) is

    .. math::

        S_n = \\begin{bmatrix}
            I - \\beta\\Delta t^2 K_{\\mathrm{eff}}^{-1} K_{\\mathrm{tan}}
            & \\Delta t I - \\beta\\Delta t^2 K_{\\mathrm{eff}}^{-1} D_{\\mathrm{tan}} \\\\
            -\\gamma\\Delta t\\, K_{\\mathrm{eff}}^{-1} K_{\\mathrm{tan}}
            & I - \\gamma\\Delta t\\, K_{\\mathrm{eff}}^{-1} D_{\\mathrm{tan}}
        \\end{bmatrix}

    where :math:`K_{\\mathrm{tan}} = K + \\partial f_{\\mathrm{nl}}/\\partial q`
    and :math:`D_{\\mathrm{tan}} = D + \\partial f_{\\mathrm{nl}}/\\partial \\dot q`.

    Parameters
    ----------
    y0:
        Initial state ``[q_0, dq_0]`` of shape ``(2 * n_dof,)``.
    omega:
        Forcing angular frequency :math:`\\Omega` [rad/s].
        Period :math:`T = 2\\pi / \\Omega`.
    system:
        A :class:`~nlvib.systems.base.MechanicalSystem` instance.
    n_periods:
        Number of forcing periods to integrate over.
    n_steps:
        Number of Newmark time steps per period.
    f_ext_fn:
        Optional callable ``(t: float) -> ndarray(n_dof,)`` returning the
        external force vector at time *t*.  If ``None`` (default), the system
        is treated as autonomous (no external forcing).

    Returns
    -------
    R : ndarray, shape ``(2 * n_dof,)``
        Shooting residual :math:`y(T_{\\mathrm{total}}) - y_0`.
    J : ndarray, shape ``(2 * n_dof, 2 * n_dof)``
        Jacobian of the residual, :math:`\\Phi - I` (monodromy matrix minus
        identity).

    References
    ----------
    K&G §3.2, equations (3.7), (3.12)–(3.14).
    """
    y0_arr: FloatArray = np.asarray(y0, dtype=np.float64)
    n_dof: int = system.n_dof
    n_state: int = 2 * n_dof

    T_period: float = 2.0 * np.pi / omega
    T_total: float = n_periods * T_period
    n_steps_total: int = n_periods * n_steps
    dt: float = T_total / n_steps_total
    dt2: float = dt * dt

    # Dense system matrices (re-used at every step)
    M_d: FloatArray = system.M.toarray()
    D_d: FloatArray = system.D.toarray()
    K_d: FloatArray = system.K.toarray()

    beta: float = NEWMARK_BETA
    gamma: float = NEWMARK_GAMMA

    # Effective stiffness and its inverse (constant for linear part)
    K_eff: FloatArray = M_d / (beta * dt2) + (gamma / (beta * dt)) * D_d + K_d
    K_eff_inv: FloatArray = la.inv(K_eff)

    # Inertia-damping predictor coefficient A_pred = M/(β*dt²) + (γ/(β*dt))*D
    A_pred_mat: FloatArray = M_d / (beta * dt2) + (gamma / (beta * dt)) * D_d

    _zero_force: FloatArray = np.zeros(n_dof, dtype=np.float64)

    def _f_ext_at(t: float) -> FloatArray:
        if f_ext_fn is not None:
            return f_ext_fn(t)
        return _zero_force

    def _fnl_full(
        q: FloatArray, dq: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return nonlinear force and both Jacobians."""
        return system.eval_nonlinear_forces(q, dq)

    # -----------------------------------------------------------------------
    # Initialise ddq_0 from the EOM at the initial state
    # K&G §3.2, eq. 3.9: M*ddq_0 = f_ext(0) - D*dq_0 - K*q_0 - f_nl(q_0,dq_0)
    # -----------------------------------------------------------------------
    q0: FloatArray = y0_arr[:n_dof]
    dq0: FloatArray = y0_arr[n_dof:]
    f_nl_0, df_dq_0, df_ddq_0 = _fnl_full(q0, dq0)
    M_inv: FloatArray = la.inv(M_d)
    ddq_curr: FloatArray = M_inv @ (
        _f_ext_at(0.0) - D_d @ dq0 - K_d @ q0 - f_nl_0
    )

    # -----------------------------------------------------------------------
    # Extended-state monodromy propagation (K&G §3.2, eq. 3.12–3.14)
    #
    # We track two quantities:
    #   Phi       (n_state × n_state) — sensitivity ∂y_n/∂y_0
    #   Phi_ddq   (n_dof   × n_state) — sensitivity ∂ddq_n/∂y_0
    #
    # At step 0, ddq_0 is computed from the EOM, so its sensitivity is
    #   ∂ddq_0/∂q_0  = −M⁻¹*(K + ∂f_nl/∂q)|_{q_0}
    #   ∂ddq_0/∂dq_0 = −M⁻¹*(D + ∂f_nl/∂dq)|_{q_0}
    #
    # This formulation correctly accounts for the fact that ddq_n at step
    # n > 0 comes from the Newmark corrector of the previous step (not from
    # re-solving the EOM), so its sensitivity must be propagated through the
    # chain rather than recomputed analytically at each step.
    # -----------------------------------------------------------------------
    Phi: FloatArray = np.eye(n_state, dtype=np.float64)

    # Initial sensitivity of ddq_0 w.r.t. y_0 — shape (n_dof, n_state)
    K_tan_0: FloatArray = K_d + df_dq_0
    D_tan_0: FloatArray = D_d + df_ddq_0
    Phi_ddq: FloatArray = np.hstack(
        [-M_inv @ K_tan_0, -M_inv @ D_tan_0]
    )  # (n_dof, n_state)

    y_curr: FloatArray = y0_arr.copy()
    t_curr: float = 0.0

    for _ in range(n_steps_total):
        q_n: FloatArray = y_curr[:n_dof]
        dq_n: FloatArray = y_curr[n_dof:]

        # Predictors (K&G §3.2, eq. 3.10)
        q_pred: FloatArray = q_n + dt * dq_n + dt2 * (0.5 - beta) * ddq_curr
        dq_pred: FloatArray = dq_n + dt * (1.0 - gamma) * ddq_curr

        # Nonlinear forces and Jacobians at predictor state
        f_nl_pred, df_dq_pred, df_ddq_pred = _fnl_full(q_pred, dq_pred)

        # External force at the next time step t_{n+1}
        t_next: float = t_curr + dt
        f_ext_np1: FloatArray = _f_ext_at(t_next)

        # Effective RHS — displacement form (K&G §3.2, eq. 3.11)
        rhs_eff: FloatArray = (
            f_ext_np1 - f_nl_pred + A_pred_mat @ q_pred - D_d @ dq_pred
        )
        q_next: FloatArray = K_eff_inv @ rhs_eff

        # Corrector (K&G §3.2, eq. 3.9)
        ddq_next: FloatArray = (q_next - q_pred) / (beta * dt2)
        dq_next: FloatArray = dq_pred + gamma * dt * ddq_next

        # -------------------------------------------------------------------
        # Monodromy update (K&G §3.2, eq. 3.14) — extended-state propagation.
        #
        # Sensitivities of predictors w.r.t. y_0, using chain rule through
        # the Newmark predictor and the propagated ∂ddq_n/∂y_0 = Phi_ddq:
        #
        #   ∂q_pred/∂y_0 = Phi[:n_dof,:] + dt*Phi[n_dof:,:]
        #                  + dt²*(0.5-β)*Phi_ddq
        #   ∂dq_pred/∂y_0 = Phi[n_dof:,:] + dt*(1-γ)*Phi_ddq
        # -------------------------------------------------------------------
        dqpred_dy0: FloatArray = (
            Phi[:n_dof, :]
            + dt * Phi[n_dof:, :]
            + dt2 * (0.5 - beta) * Phi_ddq
        )
        ddqpred_dy0: FloatArray = Phi[n_dof:, :] + dt * (1.0 - gamma) * Phi_ddq

        # Sensitivities of q_next w.r.t. predictor states — displacement form:
        #   ∂rhs_eff/∂q_pred  = A_pred_mat − ∂f_nl/∂q  →  C_q
        #   ∂rhs_eff/∂dq_pred = −D − ∂f_nl/∂dq         →  C_v
        C_q: FloatArray = K_eff_inv @ (A_pred_mat - df_dq_pred)
        C_v: FloatArray = K_eff_inv @ (-D_d - df_ddq_pred)

        dqnext_dy0: FloatArray = C_q @ dqpred_dy0 + C_v @ ddqpred_dy0

        # Sensitivity of ddq_next = (q_next − q_pred) / (β Δt²)
        dddqnext_dy0: FloatArray = (
            (dqnext_dy0 - dqpred_dy0) / (beta * dt2)
        )

        # Sensitivity of dq_next = dq_pred + γ Δt ddq_next
        dvnext_dy0: FloatArray = (
            ddqpred_dy0 + gamma * dt * dddqnext_dy0
        )

        # Assemble updated monodromy: Φ_{n+1} = ∂y_{n+1}/∂y_0
        Phi = np.vstack([dqnext_dy0, dvnext_dy0])   # (n_state, n_state)
        Phi_ddq = dddqnext_dy0                        # (n_dof, n_state)

        y_curr = np.concatenate([q_next, dq_next])
        ddq_curr = ddq_next
        t_curr = t_next

    # Shooting residual: R = y(T) - y(0)  (K&G §3.2, eq. 3.7)
    R: FloatArray = y_curr - y0_arr

    # Jacobian of the residual: J = Φ − I  (monodromy minus identity)
    J: FloatArray = Phi - np.eye(n_state, dtype=np.float64)

    return R, J
