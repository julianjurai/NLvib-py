"""
Unit tests for System_with_PolynomialStiffness (T-10).

Tests
-----
- 1-DOF cubic spring: f = k3 * q^3, verified at q = 1.5
- 2-DOF cross-coupling: f = c * q0 * q1^2, verified at q = [1.0, 2.0]
- Shape and type checks on system matrices
- Validation errors for inconsistent inputs

References
----------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  Appendix C, Table C.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlvib.systems.polynomial import System_with_PolynomialStiffness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1dof(m: float = 1.0, d: float = 0.0, k: float = 1e4) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return 1×1 dense M, D, K matrices."""
    return (
        np.array([[m]]),
        np.array([[d]]),
        np.array([[k]]),
    )


def _make_2dof(
    m1: float = 1.0,
    m2: float = 1.0,
    d1: float = 0.0,
    d2: float = 0.0,
    k1: float = 1e4,
    k2: float = 1e4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2×2 diagonal dense M, D, K matrices."""
    M = np.diag([m1, m2])
    D = np.diag([d1, d2])
    K = np.diag([k1, k2])
    return M, D, K


# ---------------------------------------------------------------------------
# 1-DOF cubic spring: f = k3 * q^3
# ---------------------------------------------------------------------------


class TestOneDOFCubic:
    """1-DOF Duffing-type cubic spring nonlinearity."""

    K3: float = 1e8
    Q_TEST: float = 1.5

    @pytest.fixture()
    def system(self) -> System_with_PolynomialStiffness:
        M, D, K = _make_1dof()
        return System_with_PolynomialStiffness(
            M=M,
            D=D,
            K=K,
            exponents=np.array([[3]], dtype=np.intp),
            coefficients=np.array([self.K3]),
        )

    def test_force_value(self, system: System_with_PolynomialStiffness) -> None:
        """Nonlinear force at q=1.5 must equal k3 * 1.5^3."""
        q = np.array([self.Q_TEST])
        dq = np.zeros(1)
        f, _, _ = system.eval_nonlinear_forces(q, dq)

        expected = self.K3 * self.Q_TEST**3
        assert f.shape == (1,), f"Expected shape (1,), got {f.shape}"
        assert float(f[0]) == pytest.approx(expected, rel=1e-12)

    def test_jacobian_value(self, system: System_with_PolynomialStiffness) -> None:
        """Displacement Jacobian at q=1.5 must equal 3*k3*q^2."""
        q = np.array([self.Q_TEST])
        dq = np.zeros(1)
        _, df_dq, df_ddq = system.eval_nonlinear_forces(q, dq)

        expected_jac = 3.0 * self.K3 * self.Q_TEST**2
        assert float(df_dq[0, 0]) == pytest.approx(expected_jac, rel=1e-12)
        assert float(df_ddq[0, 0]) == pytest.approx(0.0, abs=1e-15)

    def test_n_dof(self, system: System_with_PolynomialStiffness) -> None:
        """System must report 1 DOF."""
        assert system.n_dof == 1

    def test_one_nonlinear_element(self, system: System_with_PolynomialStiffness) -> None:
        """Exactly one nonlinear element must be registered."""
        assert len(system.nonlinear_elements) == 1

    def test_matrix_shapes(self, system: System_with_PolynomialStiffness) -> None:
        """M, D, K must be (1,1) sparse matrices."""
        assert system.M.shape == (1, 1)
        assert system.D.shape == (1, 1)
        assert system.K.shape == (1, 1)


# ---------------------------------------------------------------------------
# 2-DOF cross-coupling: f = c * q0^1 * q1^2
# ---------------------------------------------------------------------------


class TestTwoDOFCrossCoupling:
    """2-DOF cross-coupling polynomial nonlinearity."""

    C: float = 5e6

    @pytest.fixture()
    def system(self) -> System_with_PolynomialStiffness:
        M, D, K = _make_2dof()
        # Single monomial: c * q0^1 * q1^2
        return System_with_PolynomialStiffness(
            M=M,
            D=D,
            K=K,
            exponents=np.array([[1, 2]], dtype=np.intp),
            coefficients=np.array([self.C]),
        )

    def test_force_value(self, system: System_with_PolynomialStiffness) -> None:
        """Force at q=[1.0, 2.0] must equal c * 1.0 * 4.0."""
        q = np.array([1.0, 2.0])
        dq = np.zeros(2)
        f, _, _ = system.eval_nonlinear_forces(q, dq)

        expected = self.C * 1.0 * 4.0  # c * q0^1 * q1^2 = c * 1 * 4
        assert f.shape == (2,)
        # Force is placed at target_dof=0 (first nonzero Jacobian entry)
        assert float(f[0]) == pytest.approx(expected, rel=1e-12)

    def test_n_dof(self, system: System_with_PolynomialStiffness) -> None:
        """System must report 2 DOFs."""
        assert system.n_dof == 2

    def test_zero_velocity_force(self, system: System_with_PolynomialStiffness) -> None:
        """Polynomial stiffness must not produce velocity-dependent force."""
        q = np.array([1.0, 2.0])
        dq = np.array([3.0, 4.0])  # non-zero velocities
        f_with_vel, _, df_ddq = system.eval_nonlinear_forces(q, dq)
        f_zero_vel, _, _ = system.eval_nonlinear_forces(q, np.zeros(2))

        np.testing.assert_array_equal(f_with_vel, f_zero_vel)
        np.testing.assert_array_equal(df_ddq, np.zeros((2, 2)))

    def test_repr(self, system: System_with_PolynomialStiffness) -> None:
        """__repr__ must include class name and n_dof."""
        r = repr(system)
        assert "System_with_PolynomialStiffness" in r
        assert "n_dof=2" in r


# ---------------------------------------------------------------------------
# Multiple monomial terms
# ---------------------------------------------------------------------------


class TestMultipleTerms:
    """System with two monomials: k1*q0^3 + k2*q1^2."""

    K1: float = 1e7
    K2: float = 2e7

    @pytest.fixture()
    def system(self) -> System_with_PolynomialStiffness:
        M, D, K = _make_2dof()
        # Term 0: K1 * q0^3 * q1^0   (pure cubic in q0)
        # Term 1: K2 * q0^0 * q1^2   (pure quadratic in q1)
        return System_with_PolynomialStiffness(
            M=M,
            D=D,
            K=K,
            exponents=np.array([[3, 0], [0, 2]], dtype=np.intp),
            coefficients=np.array([self.K1, self.K2]),
        )

    def test_force_term1_at_q0_only(self, system: System_with_PolynomialStiffness) -> None:
        """At q=[q0, 0], only the cubic term in q0 contributes."""
        q0 = 2.0
        q = np.array([q0, 0.0])
        dq = np.zeros(2)
        f, _, _ = system.eval_nonlinear_forces(q, dq)

        # K1*q0^3*0^0 + K2*q0^0*0^2 = K1*q0^3 + 0
        expected = self.K1 * q0**3
        assert float(f[0]) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation for System_with_PolynomialStiffness."""

    def test_exponents_wrong_dof_count(self) -> None:
        """Exponents with wrong number of columns must raise ValueError."""
        M, D, K = _make_2dof()
        with pytest.raises(ValueError, match="2 DOFs"):
            System_with_PolynomialStiffness(
                M=M,
                D=D,
                K=K,
                exponents=np.array([[3]], dtype=np.intp),  # 1 col, system has 2 DOFs
                coefficients=np.array([1.0]),
            )

    def test_coefficients_wrong_length(self) -> None:
        """Coefficients of wrong length must raise ValueError."""
        M, D, K = _make_1dof()
        with pytest.raises(ValueError):
            System_with_PolynomialStiffness(
                M=M,
                D=D,
                K=K,
                exponents=np.array([[3]], dtype=np.intp),
                coefficients=np.array([1.0, 2.0]),  # 2 coefficients, 1 term
            )

    def test_exponents_1d_raises(self) -> None:
        """1-D exponents array must raise ValueError."""
        M, D, K = _make_1dof()
        with pytest.raises(ValueError, match="2-D"):
            System_with_PolynomialStiffness(
                M=M,
                D=D,
                K=K,
                exponents=np.array([3], dtype=np.intp),  # 1-D, not 2-D
                coefficients=np.array([1.0]),
            )
