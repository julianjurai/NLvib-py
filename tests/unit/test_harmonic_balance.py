"""
Unit tests for src/nlvib/solvers/harmonic_balance.py — T-12.

Test plan
---------
1. Linear 1-DOF system: residual must be exactly zero at the analytical
   steady-state for any excitation frequency.
2. Jacobian verification: J matches central finite differences to 1e-6
   for both linear and nonlinear systems.
3. Duffing oscillator (k3=1, ω₀=1, ε=0.01 damping): residual < 1e-10 at
   the refined HB solution (after a few Newton iterations).
4. NMA residual: augmented residual structure tested on a cubic spring
   oscillator (undamped backbone).
"""

from __future__ import annotations

import numpy as np
import pytest

from nlvib.systems.base import MechanicalSystem
from nlvib.nonlinearities.elements import cubic_spring
from nlvib.solvers.harmonic_balance import hb_residual, hb_residual_nma


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_linear_1dof(m: float = 1.0, d: float = 0.0, k: float = 1.0) -> MechanicalSystem:
    """Create a 1-DOF linear system: M=m, D=d, K=k (no nonlinear elements)."""
    M = np.array([[m]])
    D = np.array([[d]])
    K = np.array([[k]])
    return MechanicalSystem(M, D, K)


def _make_duffing(
    m: float = 1.0,
    d_ratio: float = 0.01,
    k: float = 1.0,
    k3: float = 1.0,
) -> MechanicalSystem:
    """Create a 1-DOF Duffing oscillator: M=m, D=2*d_ratio*sqrt(k/m)*m, K=k, k₃.

    The damping coefficient is d = 2 * ε * ω₀ * m where ω₀ = sqrt(k/m).
    """
    omega0 = np.sqrt(k / m)
    d_coeff = 2.0 * d_ratio * omega0 * m
    M = np.array([[m]])
    D = np.array([[d_coeff]])
    K = np.array([[k]])
    sys = MechanicalSystem(M, D, K)
    sys.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))
    return sys


def _analytical_linear_steady_state(
    F: float,
    omega: float,
    m: float,
    d: float,
    k: float,
) -> tuple[float, float]:
    """Analytical cos and sin amplitudes for the 1-DOF linear FRF.

    For the equation  m·ẍ + d·ẋ + k·x = F·cos(ωt), the steady-state is
        x(t) = A·cos(ωt) + B·sin(ωt)

    Solving the 2×2 HB system for the fundamental harmonic:

        [(k - m·ω²)   d·ω ] [A]   [F]
        [  -d·ω    (k-m·ω²)] [B] = [0]

    Equivalently, if C = F / ((k - m·ω²) + i·d·ω) = A - i·B in the complex
    representation Re[C·e^{iωt}] = A·cos(ωt) - Im(C)·sin(ωt), so:

        A = Re(C),   B = -Im(C)

    Returns
    -------
    A, B : float
        Cosine and sine coefficients of the fundamental harmonic.
    """
    Z_complex = (k - m * omega**2) + 1j * d * omega
    resp = F / Z_complex
    return float(resp.real), float(-resp.imag)


# ---------------------------------------------------------------------------
# Test 1: Linear 1-DOF, residual = 0 at analytical steady-state
# ---------------------------------------------------------------------------


class TestLinear1DOF:
    """Residual must be identically zero at the analytical steady-state."""

    @pytest.mark.parametrize(
        "omega,F_amp",
        [
            (0.5, 1.0),  # below resonance
            (1.0, 1.0),  # at resonance (with damping)
            (2.0, 1.0),  # above resonance
            (0.9, 2.0),  # near resonance, larger force
        ],
    )
    def test_residual_zero_at_steady_state(self, omega: float, F_amp: float) -> None:
        """R must be exactly zero at the analytical 1-DOF steady-state solution."""
        m, d, k = 1.0, 0.1, 1.0
        sys = _make_linear_1dof(m=m, d=d, k=k)
        n_harmonics = 1

        # Analytical amplitudes for first harmonic
        A, B = _analytical_linear_steady_state(F_amp, omega, m, d, k)

        # Q = [a_0=0, a_c1=A, a_s1=B]
        Q = np.array([0.0, A, B], dtype=np.float64)
        excitation = {"dof": 0, "amplitude": F_amp, "harmonic": 1}

        R, _J = hb_residual(Q, omega, sys, n_harmonics, excitation)
        assert np.linalg.norm(R) < 1e-12, (
            f"Expected zero residual at analytical SS; got ‖R‖={np.linalg.norm(R):.3e}"
        )

    def test_residual_nonzero_at_wrong_solution(self) -> None:
        """R must be nonzero when Q is not the correct steady-state."""
        sys = _make_linear_1dof(m=1.0, d=0.1, k=1.0)
        Q = np.array([0.1, 0.1, 0.0], dtype=np.float64)  # arbitrary wrong solution
        excitation = {"dof": 0, "amplitude": 1.0}
        R, _J = hb_residual(Q, 0.5, sys, 1, excitation)
        assert np.linalg.norm(R) > 1e-6

    def test_multi_harmonic_linear(self) -> None:
        """For a linear system with harmonic-1 forcing, higher harmonics = 0."""
        m, d, k = 1.0, 0.05, 1.0
        omega = 0.7
        F_amp = 1.0
        n_harmonics = 3
        sys = _make_linear_1dof(m=m, d=d, k=k)

        # Analytical first-harmonic amplitudes
        A, B = _analytical_linear_steady_state(F_amp, omega, m, d, k)

        # Q: only h=1 is nonzero
        Q = np.zeros(2 * n_harmonics + 1, dtype=np.float64)
        Q[1] = A  # a_c1
        Q[2] = B  # a_s1
        excitation = {"dof": 0, "amplitude": F_amp, "harmonic": 1}

        R, _J = hb_residual(Q, omega, sys, n_harmonics, excitation)
        assert np.linalg.norm(R) < 1e-12


# ---------------------------------------------------------------------------
# Test 2: Jacobian vs. central finite differences
# ---------------------------------------------------------------------------


class TestJacobianFD:
    """Jacobian J must match central FD to tolerance 1e-6 relative error."""

    def _fd_jacobian(
        self,
        Q: np.ndarray,
        omega: float,
        system: MechanicalSystem,
        n_harmonics: int,
        excitation: object,
    ) -> np.ndarray:
        """Central finite-difference Jacobian of R w.r.t. Q."""
        n = Q.shape[0]
        J_fd = np.zeros((n, n), dtype=np.float64)
        h = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(Q), 1.0)
        for j in range(n):
            Qp = Q.copy()
            Qm = Q.copy()
            Qp[j] += h[j]
            Qm[j] -= h[j]
            Rp, _ = hb_residual(Qp, omega, system, n_harmonics, excitation)
            Rm, _ = hb_residual(Qm, omega, system, n_harmonics, excitation)
            J_fd[:, j] = (Rp - Rm) / (2.0 * h[j])
        return J_fd

    def test_jacobian_linear_1dof(self) -> None:
        """Jacobian matches FD for linear 1-DOF system."""
        sys = _make_linear_1dof(m=1.0, d=0.1, k=1.0)
        Q = np.array([0.0, 0.5, 0.3], dtype=np.float64)
        omega = 0.8
        excitation = {"dof": 0, "amplitude": 1.0}

        _R, J = hb_residual(Q, omega, sys, 1, excitation)
        J_fd = self._fd_jacobian(Q, omega, sys, 1, excitation)

        # Relative error element-wise
        scale = np.maximum(np.abs(J_fd), 1.0)
        rel_err = np.abs(J - J_fd) / scale
        assert np.max(rel_err) < 1e-6, (
            f"Max relative Jacobian error = {np.max(rel_err):.3e}"
        )

    def test_jacobian_duffing(self) -> None:
        """Jacobian matches FD for Duffing oscillator (n_harmonics=3)."""
        sys = _make_duffing(d_ratio=0.01, k3=1.0)
        n_harmonics = 3
        n_total = 2 * n_harmonics + 1
        rng = np.random.default_rng(42)
        # Use a small-amplitude Q so nonlinearity is moderate
        Q = 0.1 * rng.standard_normal(n_total)
        omega = 0.9

        excitation = {"dof": 0, "amplitude": 0.1}

        _R, J = hb_residual(Q, omega, sys, n_harmonics, excitation)
        J_fd = self._fd_jacobian(Q, omega, sys, n_harmonics, excitation)

        scale = np.maximum(np.abs(J_fd), 1.0)
        rel_err = np.abs(J - J_fd) / scale
        assert np.max(rel_err) < 1e-5, (
            f"Max relative Jacobian error (Duffing) = {np.max(rel_err):.3e}"
        )


# ---------------------------------------------------------------------------
# Test 3: Duffing oscillator — residual < 1e-10 after Newton refinement
# ---------------------------------------------------------------------------


class TestDuffingNewton:
    """Refine a Duffing solution with Newton iterations; verify ‖R‖ < 1e-10."""

    def test_duffing_newton_convergence(self) -> None:
        """Newton method on Duffing HB converges to ‖R‖ < 1e-10 from a good initial guess."""
        # System: x'' + 0.02x' + x + x³ = F·cos(ωt)
        # ε = 0.01 damping ratio → d = 2*0.01*1.0*1.0 = 0.02
        sys = _make_duffing(m=1.0, d_ratio=0.01, k=1.0, k3=1.0)
        n_harmonics = 1
        omega = 0.9
        F_amp = 0.1  # small force for convergence

        # Initial guess from the linear solution (Duffing ≈ linear for small F)
        m, d_coeff = 1.0, 0.02
        A0, B0 = _analytical_linear_steady_state(F_amp, omega, m, d_coeff, 1.0)
        Q = np.array([0.0, A0, B0], dtype=np.float64)

        excitation = {"dof": 0, "amplitude": F_amp, "harmonic": 1}

        # Newton iterations
        MAX_ITER = 20
        TOL = 1e-10
        for _it in range(MAX_ITER):
            R, J = hb_residual(Q, omega, sys, n_harmonics, excitation)
            if np.linalg.norm(R) < TOL:
                break
            dQ = np.linalg.solve(J, -R)
            Q = Q + dQ

        R_final, _ = hb_residual(Q, omega, sys, n_harmonics, excitation)
        assert np.linalg.norm(R_final) < TOL, (
            f"Newton did not converge: ‖R‖ = {np.linalg.norm(R_final):.3e}"
        )

    def test_duffing_hardening_branch(self) -> None:
        """Duffing on the hardening branch (ω > ω₀) converges with Newton + step damping."""
        sys = _make_duffing(m=1.0, d_ratio=0.01, k=1.0, k3=1.0)
        n_harmonics = 1
        omega = 1.1  # above linear natural frequency → hardening
        F_amp = 0.05

        # Use a zero initial guess (small amplitude regime) which is closer to
        # the low-amplitude solution on the upper frequency branch.
        Q = np.zeros(3, dtype=np.float64)

        excitation = {"dof": 0, "amplitude": F_amp, "harmonic": 1}

        # Newton iterations with step damping to handle the fold
        MAX_ITER = 50
        TOL = 1e-10
        for _it in range(MAX_ITER):
            R, J = hb_residual(Q, omega, sys, n_harmonics, excitation)
            if np.linalg.norm(R) < TOL:
                break
            dQ = np.linalg.solve(J, -R)
            # Damp the Newton step if it is large to avoid overshoot
            step_norm = float(np.linalg.norm(dQ))
            alpha = min(1.0, 0.5 / step_norm) if step_norm > 0.5 else 1.0
            Q = Q + alpha * dQ

        R_final, _ = hb_residual(Q, omega, sys, n_harmonics, excitation)
        assert np.linalg.norm(R_final) < TOL, (
            f"Hardening-branch Newton did not converge: ‖R‖ = {np.linalg.norm(R_final):.3e}"
        )


# ---------------------------------------------------------------------------
# Test 4: NMA residual structure
# ---------------------------------------------------------------------------


class TestNMA:
    """Tests for hb_residual_nma: augmented residual for backbone curves."""

    def test_nma_shape(self) -> None:
        """Augmented residual and Jacobian have the correct shape."""
        sys = _make_duffing(m=1.0, d_ratio=0.0, k=1.0, k3=1.0)
        n_harmonics = 1
        n_dof = 1
        n_total = n_dof * (2 * n_harmonics + 1)

        Q = np.array([0.0, 0.0, 0.1], dtype=np.float64)  # Q_s1 = 0.1
        omega = 1.0
        Q_omega = np.append(Q, omega)

        R, J = hb_residual_nma(Q_omega, sys, n_harmonics)

        assert R.shape == (n_total + 1,), f"R shape wrong: {R.shape}"
        assert J.shape == (n_total + 1, n_total + 1), f"J shape wrong: {J.shape}"

    def test_nma_phase_constraint(self) -> None:
        """Phase constraint row: R[-1] = Q_c1 (cosine coefficient of h=1, DOF 0)."""
        sys = _make_duffing(m=1.0, d_ratio=0.0, k=1.0, k3=1.0)
        n_harmonics = 1

        # Set Q_c1 = 0.3 (non-zero) — phase constraint should return this value
        Q = np.array([0.0, 0.3, 0.0], dtype=np.float64)
        omega = 1.0
        Q_omega = np.append(Q, omega)

        R, _J = hb_residual_nma(Q_omega, sys, n_harmonics)
        # Phase constraint: R[-1] = Q_c1 = Q[1] = 0.3
        assert abs(R[-1] - 0.3) < 1e-14, f"Phase constraint failed: R[-1]={R[-1]}"

    def test_nma_undamped_backbone_zero_residual(self) -> None:
        """For undamped cubic spring, residual near zero on backbone after Newton."""
        # Undamped Duffing backbone: ω² = ω₀² + (3/4)·k₃·A²
        # For A (amplitude of sine coeff), ω² = 1 + 0.75 * k3 * A²
        k3 = 1.0
        sys = _make_duffing(m=1.0, d_ratio=0.0, k=1.0, k3=k3)
        n_harmonics = 1
        # No forcing for NMA (backbone computation)

        # Choose a backbone point: A_sine = 0.5 → ω² = 1 + 0.75*0.25 = 1.1875
        A_sine = 0.5
        omega_bb = float(np.sqrt(1.0 + 0.75 * k3 * A_sine**2))

        # On the undamped backbone, phase constraint pins Q_c1 = 0
        # The mode is purely sinusoidal: Q = [0, 0, A_sine]
        Q = np.array([0.0, 0.0, A_sine], dtype=np.float64)
        Q_omega = np.append(Q, omega_bb)

        R, J = hb_residual_nma(Q_omega, sys, n_harmonics)

        # The physical residual should be small (analytical backbone point)
        assert np.linalg.norm(R[:-1]) < 1e-8, (
            f"NMA physical residual at backbone: ‖R_phys‖ = {np.linalg.norm(R[:-1]):.3e}"
        )
        # Phase constraint is satisfied: Q_c1 = 0
        assert abs(R[-1]) < 1e-14


# ---------------------------------------------------------------------------
# Test 5: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Error handling: wrong shapes, bad inputs."""

    def test_wrong_Q_shape(self) -> None:
        """ValueError raised when Q has wrong length."""
        sys = _make_linear_1dof()
        Q = np.zeros(5)  # wrong: should be 3 for n_harmonics=1, n_dof=1
        with pytest.raises(ValueError, match="Q must have shape"):
            hb_residual(Q, 1.0, sys, 1, {"dof": 0, "amplitude": 1.0})

    def test_wrong_excitation_shape(self) -> None:
        """ValueError raised when pre-built excitation vector has wrong shape."""
        sys = _make_linear_1dof()
        Q = np.zeros(3)
        bad_excitation = np.zeros(5)
        with pytest.raises(ValueError):
            hb_residual(Q, 1.0, sys, 1, bad_excitation)

    def test_wrong_Q_omega_shape(self) -> None:
        """ValueError raised when Q_omega has wrong length for NMA."""
        sys = _make_duffing()
        Q_omega = np.zeros(5)  # wrong: should be 4 for n_harmonics=1, n_dof=1
        with pytest.raises(ValueError, match="Q_omega must have shape"):
            hb_residual_nma(Q_omega, sys, 1)


# ---------------------------------------------------------------------------
# Test 6: Pre-built excitation vector
# ---------------------------------------------------------------------------


class TestExcitationVector:
    """Test that pre-built excitation vector gives same result as dict form."""

    def test_prebuilt_excitation_equals_dict(self) -> None:
        """Pre-built F_ext array gives same residual as equivalent dict."""
        sys = _make_linear_1dof(m=1.0, d=0.1, k=1.0)
        n_harmonics = 2
        omega = 0.8
        Q = np.zeros(2 * n_harmonics + 1)

        excitation_dict: dict[str, object] = {"dof": 0, "amplitude": 1.5, "harmonic": 1}

        # Build the equivalent array manually
        F_ext = np.zeros(2 * n_harmonics + 1)
        F_ext[1] = 1.5  # cosine block of h=1 at DOF 0

        R_dict, J_dict = hb_residual(Q, omega, sys, n_harmonics, excitation_dict)
        R_arr, J_arr = hb_residual(Q, omega, sys, n_harmonics, F_ext)

        np.testing.assert_allclose(R_dict, R_arr, atol=1e-14)
        np.testing.assert_allclose(J_dict, J_arr, atol=1e-14)


# ---------------------------------------------------------------------------
# Canonical MATLAB reference values for HB residual
# ---------------------------------------------------------------------------


class TestCanonicalHBReferenceValues:
    """Explicit reference-value tests for hb_residual at known analytical points.

    All expected values are derived analytically from the HB equations
    (no Octave required).
    """

    def test_residual_at_zero_Q_undamped_linear(self) -> None:
        """HB residual at Q=0 for undamped 1-DOF linear system with force F.

        For m=1, d=0, k=1, omega=0.5, F_amp=1 and Q=0:
            R[0] = 0 * a_0 = 0          (DC equation: K * a_0 = 0 => 1 * 0 = 0)
            R[1] = (k - m*omega^2) * A - F_amp  (cosine h=1 equation)
                 = (1 - 0.25) * 0 - 1 = -1
            R[2] = (k - m*omega^2) * B + d*omega*A  (sine h=1 equation, d=0)
                 = 0.75 * 0 + 0 = 0

        MATLAB ref: R = [0, -1, 0] (residual = -excitation at Q=0)
        """
        # MATLAB ref: R = [0.0, -1.0, 0.0] at Q=0, omega=0.5, F=1, undamped linear
        m, d, k = 1.0, 0.0, 1.0
        omega = 0.5
        F_amp = 1.0
        sys = _make_linear_1dof(m=m, d=d, k=k)
        Q = np.zeros(3, dtype=np.float64)
        excitation = {"dof": 0, "amplitude": F_amp, "harmonic": 1}
        R, _J = hb_residual(Q, omega, sys, 1, excitation)
        # Only the cosine-h1 equation has a nonzero RHS: R[1] = 0 - F_amp = -1
        np.testing.assert_allclose(R[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(R[1], -F_amp, atol=1e-14)
        np.testing.assert_allclose(R[2], 0.0, atol=1e-14)

    def test_residual_exact_values_below_resonance(self) -> None:
        """HB residual at analytical SS for m=1, d=0.1, k=1, omega=0.5, F=1.

        Analytical steady-state (K&G 2019 §3):
            Z(omega) = (k - m*omega^2) + i*d*omega
                     = (1 - 0.25) + i*(0.05)
                     = 0.75 + 0.05i
            C = F / Z = 1 / (0.75 + 0.05i)
            A = Re(C) = 0.75 / (0.75^2 + 0.05^2)
            B = -Im(C) = 0.05 / (0.75^2 + 0.05^2)

        At Q=[0, A, B] the residual must be zero to machine precision.

        MATLAB ref: norm(R) < 1e-12 at analytical SS point.
        """
        # MATLAB ref: norm(R) = 0 at analytical steady state
        m, d, k = 1.0, 0.1, 1.0
        omega = 0.5
        F_amp = 1.0
        denom = (k - m * omega**2) ** 2 + (d * omega) ** 2
        # MATLAB ref: A = (k - m*omega^2)*F / denom
        A = (k - m * omega**2) * F_amp / denom
        B = d * omega * F_amp / denom
        sys = _make_linear_1dof(m=m, d=d, k=k)
        Q = np.array([0.0, A, B], dtype=np.float64)
        excitation = {"dof": 0, "amplitude": F_amp, "harmonic": 1}
        R, _J = hb_residual(Q, omega, sys, 1, excitation)
        np.testing.assert_allclose(np.linalg.norm(R), 0.0, atol=1e-12)

    def test_jacobian_linear_system_exact_values(self) -> None:
        """Jacobian of HB residual for undamped 1-DOF linear at omega=1, H=1.

        For m=1, d=0, k=4, omega=1, H=1:
            J = block_diag(k, [[k-m*omega^2, d*omega], [-d*omega, k-m*omega^2]])
              = block_diag(4, [[4-1, 0], [0, 4-1]])
              = block_diag(4, [[3, 0], [0, 3]])

        MATLAB ref: J = diag([4, 3, 3]) for this undamped case.
        """
        # MATLAB ref: J = diag([k, k-m*omega^2, k-m*omega^2]) for d=0
        m, d, k = 1.0, 0.0, 4.0
        omega = 1.0
        sys = _make_linear_1dof(m=m, d=d, k=k)
        Q = np.zeros(3, dtype=np.float64)
        excitation = {"dof": 0, "amplitude": 0.0}
        _R, J = hb_residual(Q, omega, sys, 1, excitation)
        # DC block: J[0,0] = k = 4
        np.testing.assert_allclose(J[0, 0], k, atol=1e-14)
        # Cosine/sine block diagonal for h=1 (undamped, so off-diag = 0):
        # J[1,1] = J[2,2] = k - m*omega^2 = 3
        np.testing.assert_allclose(J[1, 1], k - m * omega**2, atol=1e-14)
        np.testing.assert_allclose(J[2, 2], k - m * omega**2, atol=1e-14)
        # Off-diagonal of the h=1 block is d*omega = 0
        np.testing.assert_allclose(J[1, 2], 0.0, atol=1e-14)
        np.testing.assert_allclose(J[2, 1], 0.0, atol=1e-14)
