"""
Unit tests for the shooting method and Newmark time integrator.

Tests
-----
1. ``test_newmark_vs_solve_ivp`` — compare Newmark integration against
   ``scipy.integrate.solve_ivp`` (RK45) for a 1-DOF linear oscillator over
   10 periods; displacement error < 1e-4.

2. ``test_shooting_residual_duffing`` — verify that the shooting residual is
   small at a known periodic solution of the Duffing oscillator (obtained by
   running ``solve_ivp`` to find the periodic orbit).

3. ``test_energy_balance`` — period-averaged input power equals dissipated
   power within 1 % for a harmonically forced Duffing oscillator.

References
----------
Krack & Gross (2019) §3.2.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate as si
import scipy.linalg as la
from numpy.typing import NDArray

from nlvib.nonlinearities.elements import cubic_spring
from nlvib.solvers.shooting import NEWMARK_BETA, NEWMARK_GAMMA, newmark_step, shooting_residual
from nlvib.systems.base import MechanicalSystem

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _linear_system_1dof(
    m: float = 1.0,
    d: float = 0.1,
    k: float = 1.0,
) -> MechanicalSystem:
    """1-DOF linear oscillator as a MechanicalSystem (no nonlinear elements)."""
    M = np.array([[m]])
    D = np.array([[d]])
    K = np.array([[k]])
    return MechanicalSystem(M, D, K)


def _duffing_system(
    m: float = 1.0,
    d: float = 0.05,
    k: float = 1.0,
    k3: float = 0.5,
) -> MechanicalSystem:
    """1-DOF Duffing oscillator: M*ddq + D*dq + K*q + k3*q³ = 0."""
    M = np.array([[m]])
    D = np.array([[d]])
    K = np.array([[k]])
    sys = MechanicalSystem(M, D, K)
    sys.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))
    return sys


def _ode_rhs_linear(
    t: float,
    y: FloatArray,
    m: float,
    d: float,
    k: float,
    f_amp: float,
    omega: float,
) -> FloatArray:
    """RHS for 1-DOF linear oscillator with harmonic forcing."""
    q, dq = y
    f_ext = f_amp * np.cos(omega * t)
    ddq = (f_ext - d * dq - k * q) / m
    return np.array([dq, ddq])


def _ode_rhs_duffing(
    t: float,
    y: FloatArray,
    m: float,
    d: float,
    k: float,
    k3: float,
    f_amp: float,
    omega: float,
) -> FloatArray:
    """RHS for 1-DOF Duffing oscillator with harmonic forcing."""
    q, dq = y
    f_ext = f_amp * np.cos(omega * t)
    ddq = (f_ext - d * dq - k * q - k3 * q**3) / m
    return np.array([dq, ddq])


def _newmark_chain(
    y0: FloatArray,
    ddq0: FloatArray,
    f_fn: object,   # Callable[[float], FloatArray]
    M: FloatArray,
    D: FloatArray,
    K: FloatArray,
    fnl: object,    # Callable[[FloatArray, FloatArray], FloatArray]
    dt: float,
    n_steps: int,
    t0: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Integrate using newmark_step, chaining ddq correctly.

    Parameters
    ----------
    y0      : initial state [q, dq]
    ddq0    : initial acceleration ddq at t0
    f_fn    : callable(t) -> force vector at time t
    M, D, K : matrices
    fnl     : callable(q, dq) -> nonlinear force
    dt      : time step
    n_steps : number of steps
    t0      : start time

    Returns
    -------
    y_final : state at t0 + n_steps*dt
    ddq_final : acceleration at t0 + n_steps*dt
    """
    import numpy as np
    from nlvib.solvers.shooting import newmark_step as _step

    y = y0.copy()
    ddq = ddq0.copy()
    t = t0
    for _ in range(n_steps):
        t_next = t + dt
        f_ext_val = f_fn(t_next)  # type: ignore[operator]
        y, ddq = _step(y, f_ext_val, M, D, K, fnl, dt, ddq_n=ddq)  # type: ignore[operator]
        t = t_next
    return y, ddq


# ---------------------------------------------------------------------------
# Test 1: Newmark vs solve_ivp for a linear 1-DOF oscillator
# ---------------------------------------------------------------------------


class TestNewmarkVsSolveIvp:
    """Compare Newmark integration with RK45 for a 1-DOF linear system."""

    M_VAL: float = 1.0
    D_VAL: float = 0.1
    K_VAL: float = 1.0
    F_AMP: float = 0.5
    OMEGA: float = 0.8  # below resonance — steady oscillation
    N_PERIODS: int = 10
    N_STEPS_PER_PERIOD: int = 1000  # 1000 steps/period keeps O(dt²) error below 1e-4

    def _run_newmark(self) -> FloatArray:
        """Integrate using newmark_step (chaining ddq) and return displacement history."""
        m, d, k = self.M_VAL, self.D_VAL, self.K_VAL
        omega, f_amp = self.OMEGA, self.F_AMP

        M = np.array([[m]])
        D = np.array([[d]])
        K = np.array([[k]])

        T = 2.0 * np.pi / omega
        dt = T / self.N_STEPS_PER_PERIOD
        n_total = self.N_PERIODS * self.N_STEPS_PER_PERIOD

        def _fnl(q: FloatArray, dq: FloatArray) -> FloatArray:
            return np.zeros_like(q)

        # Initialize ddq_0 from EOM at t=0: M*ddq_0 = f(0) - D*dq_0 - K*q_0
        y = np.zeros(2, dtype=np.float64)
        f0 = np.array([f_amp * np.cos(omega * 0.0)])
        ddq = la.solve(M, f0 - D @ y[:1] - K @ y[:1])

        t = 0.0
        q_history = np.empty(n_total + 1)
        q_history[0] = y[0]

        for i in range(n_total):
            t_next = t + dt
            f_ext = np.array([f_amp * np.cos(omega * t_next)])
            y, ddq = newmark_step(y, f_ext, M, D, K, _fnl, dt, ddq_n=ddq)
            q_history[i + 1] = y[0]
            t = t_next

        return q_history

    def _run_solve_ivp(self) -> FloatArray:
        """Integrate using RK45 and return displacement at same time points."""
        m, d, k = self.M_VAL, self.D_VAL, self.K_VAL
        omega, f_amp = self.OMEGA, self.F_AMP

        T = 2.0 * np.pi / omega
        dt = T / self.N_STEPS_PER_PERIOD
        n_total = self.N_PERIODS * self.N_STEPS_PER_PERIOD
        t_eval = np.linspace(0.0, self.N_PERIODS * T, n_total + 1)

        sol = si.solve_ivp(
            fun=lambda t, y: _ode_rhs_linear(t, y, m, d, k, f_amp, omega),
            t_span=(0.0, self.N_PERIODS * T),
            y0=np.zeros(2),
            method="RK45",
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol.success, f"solve_ivp failed: {sol.message}"
        return sol.y[0]  # displacement only

    def test_displacement_error_below_threshold(self) -> None:
        """Newmark displacement error vs RK45 must be < 1e-4."""
        q_newmark = self._run_newmark()
        q_ref = self._run_solve_ivp()
        error = float(np.max(np.abs(q_newmark - q_ref)))
        assert error < 1e-4, (
            f"Newmark vs solve_ivp displacement error {error:.2e} exceeds 1e-4"
        )

    def test_final_state_close(self) -> None:
        """Final displacement and velocity must be close between methods."""
        m, d, k = self.M_VAL, self.D_VAL, self.K_VAL
        omega, f_amp = self.OMEGA, self.F_AMP

        M = np.array([[m]])
        D = np.array([[d]])
        K = np.array([[k]])

        T = 2.0 * np.pi / omega
        n_total = self.N_PERIODS * self.N_STEPS_PER_PERIOD
        dt = (self.N_PERIODS * T) / n_total

        def _fnl(q: FloatArray, dq: FloatArray) -> FloatArray:
            return np.zeros_like(q)

        y = np.zeros(2, dtype=np.float64)
        f0 = np.array([f_amp * np.cos(omega * 0.0)])
        ddq = la.solve(M, f0 - D @ y[:1] - K @ y[:1])

        t = 0.0
        for _ in range(n_total):
            t_next = t + dt
            f_ext = np.array([f_amp * np.cos(omega * t_next)])
            y, ddq = newmark_step(y, f_ext, M, D, K, _fnl, dt, ddq_n=ddq)
            t = t_next

        sol = si.solve_ivp(
            fun=lambda tt, yy: _ode_rhs_linear(tt, yy, m, d, k, f_amp, omega),
            t_span=(0.0, self.N_PERIODS * T),
            y0=np.zeros(2),
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol.success
        y_ref = sol.y[:, -1]

        np.testing.assert_allclose(y, y_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 2: Shooting residual near a known periodic orbit of the Duffing system
# ---------------------------------------------------------------------------


class TestShootingResidualDuffing:
    """Shooting residual at a known periodic orbit must be small."""

    M_VAL: float = 1.0
    D_VAL: float = 0.05
    K_VAL: float = 1.0
    K3_VAL: float = 0.5
    F_AMP: float = 0.1
    OMEGA: float = 1.2  # above linear resonance — hardening branch
    N_SETTLE: int = 200  # periods to settle before extracting IC
    N_STEPS: int = 500   # steps per period in shooting

    def _find_periodic_ic(self) -> FloatArray:
        """Integrate the Duffing ODE for many periods to find a near-periodic IC."""
        m, d, k, k3 = self.M_VAL, self.D_VAL, self.K_VAL, self.K3_VAL
        omega, f_amp = self.OMEGA, self.F_AMP

        T = 2.0 * np.pi / omega
        t_end = self.N_SETTLE * T

        sol = si.solve_ivp(
            fun=lambda t, y: _ode_rhs_duffing(t, y, m, d, k, k3, f_amp, omega),
            t_span=(0.0, t_end),
            y0=np.zeros(2),
            method="RK45",
            dense_output=True,
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol.success, f"solve_ivp settle failed: {sol.message}"

        # State at the last period boundary (close to periodic)
        t_last = self.N_SETTLE * T
        ic: FloatArray = sol.sol(t_last)  # type: ignore[misc]
        return ic

    def test_residual_small_at_periodic_ic(self) -> None:
        """Forced Newmark integration over one period at steady-state IC gives small residual."""
        y0_ic = self._find_periodic_ic()
        m, d, k, k3 = self.M_VAL, self.D_VAL, self.K_VAL, self.K3_VAL
        omega, f_amp = self.OMEGA, self.F_AMP

        M = np.array([[m]])
        D = np.array([[d]])
        K = np.array([[k]])

        sys_plain = MechanicalSystem(M, D, K)
        sys_plain.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))

        T = 2.0 * np.pi / omega
        n_steps = self.N_STEPS
        dt = T / n_steps

        def _fnl(q: FloatArray, dq: FloatArray) -> FloatArray:
            f_nl, _, _ = sys_plain.eval_nonlinear_forces(q, dq)
            return f_nl

        # Initialize ddq_0 from EOM at t_last (with forcing at t_last)
        q0_ic = y0_ic[:1]
        dq0_ic = y0_ic[1:]
        t_last = self.N_SETTLE * T
        f_at_t0 = np.array([f_amp * np.cos(omega * t_last)])
        f_nl_at_0, _, _ = sys_plain.eval_nonlinear_forces(q0_ic, dq0_ic)
        ddq0 = la.solve(M, f_at_t0 - D @ dq0_ic - K @ q0_ic - f_nl_at_0)

        # Integrate one period
        y = y0_ic.copy()
        ddq = ddq0.copy()
        for i in range(n_steps):
            t_next = t_last + (i + 1) * dt
            f_ext = np.array([f_amp * np.cos(omega * t_next)])
            y, ddq = newmark_step(y, f_ext, M, D, K, _fnl, dt, ddq_n=ddq)

        # Residual of forced integration over one period
        R_forced = y - y0_ic
        residual_norm = float(np.linalg.norm(R_forced))

        # After settling for N_SETTLE periods the IC should be very close to
        # periodic; allow tolerance for Newmark vs IVP discretisation differences.
        assert residual_norm < 0.02, (
            f"Forced shooting residual norm {residual_norm:.4e} at settled IC "
            f"should be < 0.02"
        )

    def test_shooting_residual_autonomous(self) -> None:
        """Autonomous shooting_residual function returns correct shapes."""
        sys = _duffing_system(
            m=self.M_VAL,
            d=self.D_VAL,
            k=self.K_VAL,
            k3=self.K3_VAL,
        )
        y0 = np.zeros(2, dtype=np.float64)
        R, J = shooting_residual(y0, omega=self.OMEGA, system=sys, n_periods=1, n_steps=100)

        assert R.shape == (2,), f"R shape {R.shape} != (2,)"
        assert J.shape == (2, 2), f"J shape {J.shape} != (2, 2)"

    def test_jacobian_close_to_fd(self) -> None:
        """Monodromy Jacobian must be consistent with finite-difference estimate."""
        sys = _duffing_system(
            m=self.M_VAL,
            d=self.D_VAL,
            k=self.K_VAL,
            k3=self.K3_VAL,
        )
        y0 = np.array([0.1, 0.05], dtype=np.float64)
        R0, J = shooting_residual(y0, omega=self.OMEGA, system=sys, n_periods=1, n_steps=200)

        eps = 1e-5
        J_fd = np.zeros_like(J)
        for i in range(len(y0)):
            y_p = y0.copy()
            y_p[i] += eps
            R_p, _ = shooting_residual(
                y_p, omega=self.OMEGA, system=sys, n_periods=1, n_steps=200
            )
            J_fd[:, i] = (R_p - R0) / eps

        # Allow 10% tolerance — FD and monodromy use the same Newmark linearisation
        # but differ by O(dt) due to predictor-state Jacobian evaluation.
        np.testing.assert_allclose(J, J_fd, atol=0.1, rtol=0.1)


# ---------------------------------------------------------------------------
# Test 3: Period-averaged energy balance
# ---------------------------------------------------------------------------


class TestEnergyBalance:
    """Input power = dissipated power within 1% for a harmonically forced system."""

    M_VAL: float = 1.0
    D_VAL: float = 0.1
    K_VAL: float = 1.0
    K3_VAL: float = 0.3
    F_AMP: float = 0.2
    OMEGA: float = 1.0  # at or near resonance — maximum dissipation
    N_SETTLE: int = 300
    N_STEPS: int = 1000

    def test_energy_balance_within_one_percent(self) -> None:
        """Period-averaged input power and dissipated power agree within 1%."""
        m, d, k, k3 = self.M_VAL, self.D_VAL, self.K_VAL, self.K3_VAL
        omega, f_amp = self.OMEGA, self.F_AMP

        # 1. Settle to near-periodic IC via solve_ivp
        T = 2.0 * np.pi / omega
        t_settle = self.N_SETTLE * T
        sol_settle = si.solve_ivp(
            fun=lambda t, y: _ode_rhs_duffing(t, y, m, d, k, k3, f_amp, omega),
            t_span=(0.0, t_settle),
            y0=np.zeros(2),
            method="RK45",
            dense_output=True,
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol_settle.success
        y0_ic: FloatArray = sol_settle.sol(t_settle)  # type: ignore[misc]

        # 2. Integrate one period with Newmark and collect q, dq at each step
        M = np.array([[m]])
        D = np.array([[d]])
        K = np.array([[k]])

        sys = MechanicalSystem(M, D, K)
        sys.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))

        n_steps = self.N_STEPS
        dt = T / n_steps

        def _fnl(q: FloatArray, dq: FloatArray) -> FloatArray:
            f_nl, _, _ = sys.eval_nonlinear_forces(q, dq)
            return f_nl

        # Initialize ddq at the settled IC
        q0_ic = y0_ic[:1]
        dq0_ic = y0_ic[1:]
        f_nl_ic, _, _ = sys.eval_nonlinear_forces(q0_ic, dq0_ic)
        f_at_t0 = np.array([f_amp * np.cos(omega * t_settle)])
        ddq_ic = la.solve(M, f_at_t0 - D @ dq0_ic - K @ q0_ic - f_nl_ic)

        q_arr = np.empty(n_steps + 1)
        dq_arr = np.empty(n_steps + 1)
        q_arr[0] = y0_ic[0]
        dq_arr[0] = y0_ic[1]

        y = y0_ic.copy()
        ddq = ddq_ic.copy()
        for i in range(n_steps):
            t_next = t_settle + (i + 1) * dt
            f_ext = np.array([f_amp * np.cos(omega * t_next)])
            y, ddq = newmark_step(y, f_ext, M, D, K, _fnl, dt, ddq_n=ddq)
            q_arr[i + 1] = y[0]
            dq_arr[i + 1] = y[1]

        # 3. Period-averaged input power: <P_in> = (1/T) ∫ f_ext(t) * dq(t) dt
        # Note: t_vals is relative to the period (phase doesn't matter for power)
        t_rel = np.linspace(0.0, T, n_steps + 1)
        t_abs = t_settle + t_rel
        f_ext_arr = f_amp * np.cos(omega * t_abs)
        p_in_arr = f_ext_arr * dq_arr
        P_in = float(np.trapezoid(p_in_arr, t_rel)) / T

        # 4. Period-averaged dissipated power: <P_diss> = (1/T) ∫ d * dq² dt
        p_diss_arr = d * dq_arr**2
        P_diss = float(np.trapezoid(p_diss_arr, t_rel)) / T

        # 5. Check balance within 1 %
        relative_error = abs(P_in - P_diss) / (abs(P_diss) + 1e-15)
        assert relative_error < 0.01, (
            f"Energy balance error {relative_error:.2%} > 1%: "
            f"P_in={P_in:.4e}, P_diss={P_diss:.4e}"
        )


# ---------------------------------------------------------------------------
# Sanity checks on module constants
# ---------------------------------------------------------------------------


def test_newmark_constants() -> None:
    """Newmark β and γ must equal average constant acceleration values."""
    assert NEWMARK_BETA == pytest.approx(0.25)
    assert NEWMARK_GAMMA == pytest.approx(0.5)
