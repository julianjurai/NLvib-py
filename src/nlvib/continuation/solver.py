"""
Arc-length continuation solver for NLvib.

Implements pseudo-arc-length (Keller's) continuation to trace solution
branches of parameterised nonlinear systems.  The solver is generic: it
accepts any residual function ``R(x, λ) = 0`` where **x** is the solution
vector and **λ** is the continuation parameter (e.g. frequency Ω for FRF).

Algorithm reference
-------------------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer. §4 — continuation method.

Predictor–corrector scheme (K&G §4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Predictor** (K&G §4.2, eq. 4.3):
   Solve the bordered linear system::

       [J   ∂R/∂λ ] [t_x]   [0]
       [tᵀ  t_λ   ] [t_λ] = [1]

   for the unit tangent ``(t_x, t_λ)``, then extrapolate::

       x_pred  = x  + ds · t_x
       λ_pred  = λ  + ds · t_λ

2. **Corrector** (K&G §4.3):
   Newton iteration on the augmented system::

       G(x, λ) = [R(x, λ)               ]   =  0
                 [(x−x_p)·t_x+(λ−λ_p)·t_λ]

   where ``(x_p, λ_p)`` is the predicted point.

3. **Adaptive step size** (K&G §4.4):
   - If Newton converged in ``< 5`` iterations: double ``ds``.
   - If Newton needed ``> 9`` iterations: halve ``ds``.
   - Clamp to ``[ds_min, ds_max]``.

4. **Stability** (fold detection):
   A sign change of ``t_λ`` indicates a fold point (turning point) on the
   branch.  Steps are flagged as *potentially unstable* between consecutive
   fold points (K&G §4.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["ContinuationOptions", "ContinuationResult", "ContinuationSolver"]

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

#: Newton iteration threshold below which step size is doubled.
_STEP_DOUBLE_THRESHOLD: int = 5

#: Newton iteration threshold above which step size is halved.
_STEP_HALVE_THRESHOLD: int = 9

#: Step size growth factor.
_STEP_GROWTH_FACTOR: float = 2.0

#: Step size reduction factor.
_STEP_REDUCE_FACTOR: float = 0.5


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContinuationOptions:
    """Configuration for :class:`ContinuationSolver`.

    Attributes
    ----------
    ds_initial:
        Initial arc-length step size (K&G §4.2).
    ds_min:
        Minimum permitted arc-length step.  The solver aborts if ``ds``
        falls below this value (K&G §4.4).
    ds_max:
        Maximum permitted arc-length step (K&G §4.4).
    max_steps:
        Maximum number of continuation steps.
    max_newton_iter:
        Maximum Newton iterations per corrector step (K&G §4.3).
    newton_tol:
        Convergence tolerance for the Newton corrector: iteration stops when
        ``‖G‖₂ < newton_tol`` (K&G §4.3).
    adapt_step:
        If ``True`` (default), adapt the step size based on the number of
        Newton iterations (K&G §4.4).
    lambda_min:
        Stop continuation when ``λ < lambda_min``.  ``None`` = no limit.
    lambda_max:
        Stop continuation when ``λ > lambda_max``.  ``None`` = no limit.
    callback:
        Optional callable invoked at the end of each accepted step with the
        current augmented state ``z = [x; λ]``.  If the callback returns
        ``False`` the continuation is stopped.
    """

    ds_initial: float = 0.01
    ds_min: float = 1e-6
    ds_max: float = 0.1
    max_steps: int = 500
    max_newton_iter: int = 20
    newton_tol: float = 1e-10
    adapt_step: bool = True
    lambda_min: float | None = None
    lambda_max: float | None = None
    callback: Callable[[FloatArray], bool | None] | None = None


@dataclass
class ContinuationResult:
    """Results returned by :class:`ContinuationSolver`.

    Attributes
    ----------
    solutions:
        Array of accepted augmented solution vectors ``[x; λ]``.
        Shape ``(n_steps, n_vars + 1)``.
    stability:
        Boolean array of shape ``(n_steps,)``.  ``True`` indicates the step
        is in a potentially unstable segment (between two consecutive fold
        points, K&G §4.5).
    ds_history:
        Arc-length step used at each accepted step, shape ``(n_steps,)``.
    n_steps:
        Number of accepted steps (equals ``len(solutions)``).
    converged:
        ``True`` if the solver reached a terminal condition gracefully
        (parameter limit or callback stop).  ``False`` if it hit
        ``max_steps`` or ``ds < ds_min``.
    message:
        Human-readable termination message.
    """

    solutions: FloatArray
    stability: NDArray[np.bool_]
    ds_history: FloatArray
    n_steps: int
    converged: bool
    message: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _solve_bordered(
    J: FloatArray,
    dR_dlam: FloatArray,
    t_x: FloatArray,
    t_lam: float,
    rhs_top: FloatArray,
    rhs_bot: float,
) -> tuple[FloatArray, float]:
    """Solve the 2×2 bordered linear system (K&G §4.2, eq. 4.3).

    The bordered system is::

        [J        dR/dλ] [dx  ]   [rhs_top]
        [t_xᵀ    t_λ  ] [dlam] = [rhs_bot]

    Solved via block elimination (Schur complement):

    1. Solve  J · u = rhs_top  (``u``) and  J · v = dR/dλ  (``v``).
    2. Schur complement: ``s = t_λ − t_xᵀ · v``.
    3. ``dlam = (rhs_bot − t_xᵀ · u) / s``.
    4. ``dx   = u − dlam · v``.

    Parameters
    ----------
    J:
        Jacobian ∂R/∂x, shape ``(n, n)``.
    dR_dlam:
        Partial derivative ∂R/∂λ, shape ``(n,)``.
    t_x:
        Current tangent in x-direction, shape ``(n,)``.
    t_lam:
        Current tangent in λ-direction, scalar.
    rhs_top:
        Right-hand side for the physical equations, shape ``(n,)``.
    rhs_bot:
        Right-hand side for the arc-length equation, scalar.

    Returns
    -------
    dx : FloatArray, shape ``(n,)``
    dlam : float
    """
    # Step 1: Solve J u = rhs_top  and  J v = dR/dλ
    u = np.linalg.solve(J, rhs_top)
    v = np.linalg.solve(J, dR_dlam)

    # Step 2: Schur complement  s = t_λ − t_xᵀ v
    s = t_lam - float(t_x @ v)
    if abs(s) < np.finfo(float).tiny:
        # Near-singular bordered system — fall back to least-squares
        n = J.shape[0]
        A_aug = np.zeros((n + 1, n + 1), dtype=np.float64)
        A_aug[:n, :n] = J
        A_aug[:n, n] = dR_dlam
        A_aug[n, :n] = t_x
        A_aug[n, n] = t_lam
        b_aug = np.append(rhs_top, rhs_bot)
        sol = np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]
        return sol[:n], float(sol[n])

    # Step 3: dlam = (rhs_bot − t_xᵀ u) / s
    dlam = (rhs_bot - float(t_x @ u)) / s

    # Step 4: dx = u − dlam v
    dx = u - dlam * v
    return dx, dlam


def _compute_tangent(
    J: FloatArray,
    dR_dlam: FloatArray,
    t_x_prev: FloatArray,
    t_lam_prev: float,
) -> tuple[FloatArray, float]:
    """Compute the unit tangent at the current point (K&G §4.2, eq. 4.3).

    Solves the bordered linear system with right-hand side ``[0; 1]``::

        [J        dR/dλ] [t_x ]   [0]
        [t_xᵀ    t_λ  ] [t_λ ] = [1]

    The solution is then normalised to unit arc-length norm:
    ``‖[t_x; t_λ]‖₂ = 1``.

    The sign of the tangent is chosen to maintain consistent direction of
    traversal (positive dot product with the previous tangent vector).

    Parameters
    ----------
    J:
        Jacobian ∂R/∂x, shape ``(n, n)``.
    dR_dlam:
        ∂R/∂λ, shape ``(n,)``.
    t_x_prev:
        Previous tangent x-component, shape ``(n,)``.  Used for sign selection.
    t_lam_prev:
        Previous tangent λ-component.  Used for sign selection.

    Returns
    -------
    t_x : FloatArray, shape ``(n,)``
        Unit tangent in x-direction.
    t_lam : float
        Unit tangent in λ-direction.
    """
    n = J.shape[0]
    rhs_top = np.zeros(n, dtype=np.float64)
    rhs_bot = 1.0

    t_x_raw, t_lam_raw = _solve_bordered(
        J, dR_dlam, t_x_prev, t_lam_prev, rhs_top, rhs_bot
    )

    # Normalise to unit arc-length norm (K&G §4.2, eq. 4.3)
    norm = float(np.sqrt(t_x_raw @ t_x_raw + t_lam_raw**2))
    if norm < np.finfo(float).tiny:
        # Degenerate — return previous tangent unchanged
        return t_x_prev.copy(), t_lam_prev

    t_x = t_x_raw / norm
    t_lam = t_lam_raw / norm

    # Enforce consistent traversal direction
    dot = float(t_x_prev @ t_x) + t_lam_prev * t_lam
    if dot < 0.0:
        t_x = -t_x
        t_lam = -t_lam

    return t_x, t_lam


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------


class ContinuationSolver:
    """Pseudo-arc-length continuation solver (K&G §4).

    Traces solution branches of parameterised nonlinear systems::

        R(x, λ) = 0

    using a tangent predictor and a Newton corrector with the arc-length
    parametrisation constraint (K&G §4.3):

        (x − x_pred)·t_x + (λ − λ_pred)·t_λ = 0

    Examples
    --------
    Trace the Duffing FRF from ω = 0.5 to ω = 1.5::

        from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

        def residual_fn(x, lam):
            return hb_residual(x, lam, system, n_harmonics, excitation)

        solver = ContinuationSolver()
        opts = ContinuationOptions(ds_initial=0.02, lambda_max=1.5)
        result = solver.run(residual_fn, x0, omega0, opts)
    """

    def run(
        self,
        residual_fn: Callable[[FloatArray, float], tuple[FloatArray, FloatArray]],
        x0: FloatArray,
        lambda0: float,
        options: ContinuationOptions | None = None,
    ) -> ContinuationResult:
        """Trace a solution branch by arc-length continuation (K&G §4).

        Starting from the initial solution point ``(x0, lambda0)`` (which
        must satisfy ``R(x0, lambda0) ≈ 0``), the solver iterates:

        1. **Predictor**: compute the tangent ``(t_x, t_λ)`` and predict
           ``(x_pred, λ_pred) = (x + ds·t_x, λ + ds·t_λ)``  (K&G §4.2).
        2. **Corrector**: Newton iterations on the augmented system
           ``G = [R; arc-length constraint] = 0``  (K&G §4.3).
        3. **Step size adaptation**: halve/double ``ds`` based on iteration
           count  (K&G §4.4).
        4. Append accepted solution to branch; check termination criteria.

        Parameters
        ----------
        residual_fn:
            Callable ``(x, lambda) -> (R, J)`` where:

            - ``R`` is the residual vector, shape ``(n,)``.
            - ``J`` is the Jacobian ∂R/∂x, shape ``(n, n)``.

        x0:
            Initial solution vector, shape ``(n,)``.
        lambda0:
            Initial continuation parameter value.
        options:
            Solver options.  If ``None``, defaults are used.

        Returns
        -------
        ContinuationResult
            Accepted solution branch with stability flags and step history.

        Notes
        -----
        Equation references:

        - K&G §4.2, eq. (4.3): tangent predictor (bordered linear system).
        - K&G §4.3: augmented Newton corrector.
        - K&G §4.4: adaptive step size control.
        - K&G §4.5: fold (turning point) detection via sign of ``t_λ``.
        """
        if options is None:
            options = ContinuationOptions()

        x_arr = np.asarray(x0, dtype=np.float64).copy()
        lam = float(lambda0)
        n = x_arr.size

        ds = float(options.ds_initial)

        # Storage for accepted solutions
        solutions_list: list[FloatArray] = []
        stability_list: list[bool] = []
        ds_hist_list: list[float] = []

        # ---------------------------------------------------------------------------
        # Bootstrap: compute initial Jacobian and tangent direction
        # ---------------------------------------------------------------------------
        R0, J0 = residual_fn(x_arr, lam)

        # ∂R/∂λ by central FD (K&G §4.2)
        h_lam = float(np.sqrt(np.finfo(float).eps)) * max(abs(lam), 1.0)
        R_p, _ = residual_fn(x_arr, lam + h_lam)
        R_m, _ = residual_fn(x_arr, lam - h_lam)
        dR_dlam: FloatArray = (R_p - R_m) / (2.0 * h_lam)

        # Initial tangent: start with λ increasing (t_lam_prev = +1.0)
        t_x_prev = np.zeros(n, dtype=np.float64)
        t_lam_prev = 1.0

        t_x, t_lam = _compute_tangent(J0, dR_dlam, t_x_prev, t_lam_prev)

        # Store initial point
        z0: FloatArray = np.append(x_arr, lam)
        solutions_list.append(z0.copy())
        stability_list.append(False)
        ds_hist_list.append(0.0)

        # Track fold crossings for stability flagging (K&G §4.5)
        t_lam_prev_step = t_lam
        unstable_flag: bool = False

        # ---------------------------------------------------------------------------
        # Main continuation loop
        # ---------------------------------------------------------------------------
        step: int = 0
        converged: bool = False
        message: str = "max_steps reached"

        while step < options.max_steps:

            # ---------------------------------------------------------------
            # Predictor (K&G §4.2, eq. 4.3)
            # ---------------------------------------------------------------
            x_pred = x_arr + ds * t_x
            lam_pred = lam + ds * t_lam

            # ---------------------------------------------------------------
            # Corrector: Newton on augmented system G = [R; arc-constraint] = 0
            # (K&G §4.3)
            # ---------------------------------------------------------------
            x_c = x_pred.copy()
            lam_c = float(lam_pred)

            newton_ok = False
            n_iter: int = 0

            for n_iter in range(1, options.max_newton_iter + 1):
                R_c, J_c = residual_fn(x_c, lam_c)

                # ∂R/∂λ at corrector point
                h_lam_c = float(np.sqrt(np.finfo(float).eps)) * max(abs(lam_c), 1.0)
                Rp_c, _ = residual_fn(x_c, lam_c + h_lam_c)
                Rm_c, _ = residual_fn(x_c, lam_c - h_lam_c)
                dR_dlam_c: FloatArray = (Rp_c - Rm_c) / (2.0 * h_lam_c)

                # Arc-length constraint: g = (x_c − x_pred)·t_x + (λ_c − λ_pred)·t_λ
                g_arc = float((x_c - x_pred) @ t_x) + (lam_c - lam_pred) * t_lam

                # Augmented residual norm check
                G_norm = float(
                    np.sqrt(float(R_c @ R_c) + g_arc**2)
                )
                if G_norm < options.newton_tol:
                    newton_ok = True
                    break

                # Newton step: solve bordered system with RHS = -[R_c; g_arc]
                # (K&G §4.3)
                dx, dlam = _solve_bordered(
                    J_c,
                    dR_dlam_c,
                    t_x,
                    t_lam,
                    -R_c,
                    -g_arc,
                )

                x_c = x_c + dx
                lam_c = lam_c + dlam

            # ---------------------------------------------------------------
            # Step size adaptation (K&G §4.4)
            # ---------------------------------------------------------------
            if options.adapt_step:
                if n_iter < _STEP_DOUBLE_THRESHOLD and newton_ok:
                    ds = min(ds * _STEP_GROWTH_FACTOR, options.ds_max)
                elif n_iter > _STEP_HALVE_THRESHOLD or not newton_ok:
                    ds = ds * _STEP_REDUCE_FACTOR

            if not newton_ok:
                # Step failed to converge — shrink ds and retry
                if ds < options.ds_min:
                    message = (
                        f"ds = {ds:.3e} < ds_min = {options.ds_min:.3e}; "
                        "continuation aborted"
                    )
                    converged = False
                    break
                # Retry from the same base point with reduced ds
                continue

            # ---------------------------------------------------------------
            # Accept the step
            # ---------------------------------------------------------------
            x_arr = x_c
            lam = lam_c
            step += 1

            # ---------------------------------------------------------------
            # Update tangent for next step (K&G §4.2)
            # ---------------------------------------------------------------
            R_new, J_new = residual_fn(x_arr, lam)
            h_lam_new = float(np.sqrt(np.finfo(float).eps)) * max(abs(lam), 1.0)
            Rp_n, _ = residual_fn(x_arr, lam + h_lam_new)
            Rm_n, _ = residual_fn(x_arr, lam - h_lam_new)
            dR_dlam_new: FloatArray = (Rp_n - Rm_n) / (2.0 * h_lam_new)

            t_x, t_lam = _compute_tangent(J_new, dR_dlam_new, t_x, t_lam)

            # ---------------------------------------------------------------
            # Fold / stability detection (K&G §4.5):
            # Sign change of t_λ → fold point → toggle stability
            # ---------------------------------------------------------------
            if t_lam_prev_step * t_lam < 0.0:
                unstable_flag = not unstable_flag
            t_lam_prev_step = t_lam

            # ---------------------------------------------------------------
            # Record accepted step
            # ---------------------------------------------------------------
            z_new: FloatArray = np.append(x_arr, lam)
            solutions_list.append(z_new.copy())
            stability_list.append(unstable_flag)
            ds_hist_list.append(ds)

            # ---------------------------------------------------------------
            # Callback
            # ---------------------------------------------------------------
            if options.callback is not None:
                cb_result = options.callback(z_new)
                if cb_result is False:
                    message = "callback returned False"
                    converged = True
                    break

            # ---------------------------------------------------------------
            # Termination: parameter bounds (K&G §4.4)
            # ---------------------------------------------------------------
            if options.lambda_min is not None and lam < options.lambda_min:
                message = f"lambda = {lam:.6g} < lambda_min = {options.lambda_min:.6g}"
                converged = True
                break
            if options.lambda_max is not None and lam > options.lambda_max:
                message = f"lambda = {lam:.6g} > lambda_max = {options.lambda_max:.6g}"
                converged = True
                break

            # Check ds bounds after adaptation
            if ds < options.ds_min:
                message = (
                    f"ds = {ds:.3e} < ds_min = {options.ds_min:.3e}; "
                    "continuation aborted"
                )
                converged = False
                break

        else:
            # Loop exhausted max_steps
            message = f"max_steps = {options.max_steps} reached"
            converged = False

        # ---------------------------------------------------------------------------
        # Assemble result arrays
        # ---------------------------------------------------------------------------
        solutions_arr: FloatArray = np.array(solutions_list, dtype=np.float64)
        stability_arr: NDArray[np.bool_] = np.array(stability_list, dtype=bool)
        ds_history_arr: FloatArray = np.array(ds_hist_list, dtype=np.float64)

        return ContinuationResult(
            solutions=solutions_arr,
            stability=stability_arr,
            ds_history=ds_history_arr,
            n_steps=len(solutions_list),
            converged=converged,
            message=message,
        )
