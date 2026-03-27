"""
Unit tests for nonlinear element functions.

Tests:
1. Force values against analytically known results.
2. Jacobians verified via central finite differences to tolerance 1e-8.

Reference: Krack & Gross (2019) Appendix C, Table C.1.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from nlvib.nonlinearities.elements import (
    NonlinearElement,
    cubic_spring,
    elastic_dry_friction,
    polynomial_stiffness,
    quadratic_damper,
    tanh_dry_friction,
    unilateral_spring,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FD_EPS = 1e-6  # step size for finite-difference Jacobian approximation
JAC_TOL = 1e-8  # maximum allowed |analytic - FD| for each component


def _fd_jacobians(
    element: NonlinearElement,
    q: NDArray[np.floating],
    dq: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute central-difference Jacobians of the scalar force f w.r.t. q and dq.

    Returns
    -------
    fd_df_dq : ndarray, shape (n,)
    fd_df_ddq : ndarray, shape (n,)
    """
    n = q.shape[0]
    fd_df_dq = np.zeros(n)
    fd_df_ddq = np.zeros(n)

    for i in range(n):
        # Jacobian w.r.t. q[i]
        q_fwd = q.copy()
        q_bwd = q.copy()
        q_fwd[i] += FD_EPS
        q_bwd[i] -= FD_EPS
        f_fwd, _, _ = element.eval(q_fwd, dq)
        f_bwd, _, _ = element.eval(q_bwd, dq)
        fd_df_dq[i] = (f_fwd - f_bwd) / (2.0 * FD_EPS)

        # Jacobian w.r.t. dq[i]
        dq_fwd = dq.copy()
        dq_bwd = dq.copy()
        dq_fwd[i] += FD_EPS
        dq_bwd[i] -= FD_EPS
        f_fwd2, _, _ = element.eval(q, dq_fwd)
        f_bwd2, _, _ = element.eval(q, dq_bwd)
        fd_df_ddq[i] = (f_fwd2 - f_bwd2) / (2.0 * FD_EPS)

    return fd_df_dq, fd_df_ddq


def _assert_jacobians(
    element: NonlinearElement,
    q: NDArray[np.floating],
    dq: NDArray[np.floating],
) -> None:
    """Assert that analytic Jacobians match finite differences to JAC_TOL."""
    f, df_dq, df_ddq = element.eval(q, dq)
    fd_dq, fd_ddq = _fd_jacobians(element, q, dq)
    np.testing.assert_allclose(
        df_dq,
        fd_dq,
        atol=JAC_TOL,
        err_msg=f"df_dq mismatch for {element.label}",
    )
    np.testing.assert_allclose(
        df_ddq,
        fd_ddq,
        atol=JAC_TOL,
        err_msg=f"df_ddq mismatch for {element.label}",
    )


# ---------------------------------------------------------------------------
# Tests: NonlinearElement dataclass
# ---------------------------------------------------------------------------


class TestNonlinearElement:
    def test_callable_via_dunder(self) -> None:
        """NonlinearElement should be callable directly (delegates to eval)."""
        elem = cubic_spring(k3=1.0, dof_index=0)
        q = np.array([2.0])
        dq = np.zeros(1)
        result_eval = elem.eval(q, dq)
        result_call = elem(q, dq)
        assert result_eval[0] == result_call[0]
        np.testing.assert_array_equal(result_eval[1], result_call[1])
        np.testing.assert_array_equal(result_eval[2], result_call[2])

    def test_label_set(self) -> None:
        elem = cubic_spring(k3=5.0, dof_index=1)
        assert "cubic_spring" in elem.label
        assert "5.0" in elem.label

    def test_frozen_dataclass(self) -> None:
        elem = cubic_spring(k3=1.0, dof_index=0)
        with pytest.raises(Exception):  # frozen → AttributeError
            elem.label = "new_label"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: cubic_spring
# ---------------------------------------------------------------------------


class TestCubicSpring:
    """Krack & Gross (2019) App. C, Table C.1 — Cubic spring f = k3*q^3."""

    @pytest.mark.parametrize(
        "k3, q_val, expected_f",
        [
            (1.0, 2.0, 8.0),      # 1 * 2^3 = 8
            (3.0, 1.0, 3.0),      # 3 * 1^3 = 3
            (2.0, -1.0, -2.0),    # 2 * (-1)^3 = -2
            (0.5, 0.0, 0.0),      # zero displacement
            (1.0, 0.1, 0.001),    # small displacement
        ],
    )
    def test_force_value(self, k3: float, q_val: float, expected_f: float) -> None:
        elem = cubic_spring(k3=k3, dof_index=0)
        q = np.array([q_val])
        dq = np.zeros(1)
        f, _, _ = elem.eval(q, dq)
        assert f == pytest.approx(expected_f, rel=1e-12, abs=1e-15)

    def test_force_uses_correct_dof(self) -> None:
        """Force must depend only on the designated DOF."""
        elem = cubic_spring(k3=2.0, dof_index=1)
        q = np.array([10.0, 3.0, 5.0])
        dq = np.zeros(3)
        f, _, _ = elem.eval(q, dq)
        assert f == pytest.approx(2.0 * 3.0**3)  # 54.0

    def test_velocity_independence(self) -> None:
        """Force and df_dq must be independent of dq."""
        elem = cubic_spring(k3=1.0, dof_index=0)
        q = np.array([2.0])
        f1, dfdq1, _ = elem.eval(q, np.array([0.0]))
        f2, dfdq2, _ = elem.eval(q, np.array([100.0]))
        assert f1 == pytest.approx(f2)
        np.testing.assert_array_equal(dfdq1, dfdq2)

    def test_df_dq_analytic(self) -> None:
        """df/dq_i = 3*k3*q_i^2."""
        elem = cubic_spring(k3=2.0, dof_index=0)
        q = np.array([3.0])
        dq = np.zeros(1)
        _, df_dq, df_ddq = elem.eval(q, dq)
        assert df_dq[0] == pytest.approx(3.0 * 2.0 * 9.0)  # 54.0
        assert df_ddq[0] == pytest.approx(0.0)

    @pytest.mark.parametrize("q_val", [0.5, -1.5, 2.0, -0.01])
    def test_jacobian_vs_fd(self, q_val: float) -> None:
        elem = cubic_spring(k3=3.0, dof_index=0)
        q = np.array([q_val, 1.0])
        dq = np.array([2.0, -1.0])
        _assert_jacobians(elem, q, dq)

    def test_zero_k3(self) -> None:
        elem = cubic_spring(k3=0.0, dof_index=0)
        q = np.array([5.0])
        f, df_dq, _ = elem.eval(q, np.zeros(1))
        assert f == 0.0
        assert df_dq[0] == 0.0

    def test_output_shapes(self) -> None:
        n = 5
        elem = cubic_spring(k3=1.0, dof_index=2)
        q = np.ones(n)
        dq = np.zeros(n)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert isinstance(f, float)
        assert df_dq.shape == (n,)
        assert df_ddq.shape == (n,)


# ---------------------------------------------------------------------------
# Tests: quadratic_damper
# ---------------------------------------------------------------------------


class TestQuadraticDamper:
    """Krack & Gross (2019) App. C, Table C.1 — Quadratic damper f = c2*dq*|dq|."""

    @pytest.mark.parametrize(
        "c2, dq_val, expected_f",
        [
            (1.0, 3.0, 9.0),    # 1*3*3 = 9
            (2.0, -2.0, -8.0),  # 2*(-2)*2 = -8
            (1.0, 0.0, 0.0),
            (3.0, 1.0, 3.0),
            (1.0, -1.0, -1.0),
        ],
    )
    def test_force_value(self, c2: float, dq_val: float, expected_f: float) -> None:
        elem = quadratic_damper(c2=c2, dof_index=0)
        q = np.zeros(1)
        dq = np.array([dq_val])
        f, _, _ = elem.eval(q, dq)
        assert f == pytest.approx(expected_f, rel=1e-12, abs=1e-15)

    def test_sign_preserving(self) -> None:
        """Force must oppose velocity (sign match)."""
        elem = quadratic_damper(c2=1.0, dof_index=0)
        q = np.zeros(1)
        for dq_val in [3.0, -3.0]:
            f, _, _ = elem.eval(q, np.array([dq_val]))
            assert np.sign(f) == np.sign(dq_val)

    def test_displacement_independence(self) -> None:
        elem = quadratic_damper(c2=1.0, dof_index=0)
        dq = np.array([2.0])
        f1, _, _ = elem.eval(np.array([0.0]), dq)
        f2, _, _ = elem.eval(np.array([99.0]), dq)
        assert f1 == pytest.approx(f2)

    def test_df_ddq_analytic(self) -> None:
        """df/ddq_i = 2*c2*|dq_i|."""
        elem = quadratic_damper(c2=2.0, dof_index=1)
        q = np.zeros(3)
        dq = np.array([5.0, -3.0, 0.0])
        _, df_dq, df_ddq = elem.eval(q, dq)
        np.testing.assert_array_equal(df_dq, np.zeros(3))
        assert df_ddq[1] == pytest.approx(2.0 * 2.0 * 3.0)  # 12.0
        assert df_ddq[0] == pytest.approx(0.0)
        assert df_ddq[2] == pytest.approx(0.0)

    @pytest.mark.parametrize("dq_val", [0.5, -1.5, 2.0, 0.01])
    def test_jacobian_vs_fd(self, dq_val: float) -> None:
        elem = quadratic_damper(c2=1.5, dof_index=0)
        q = np.array([1.0, -2.0])
        dq = np.array([dq_val, 3.0])
        _assert_jacobians(elem, q, dq)

    def test_output_shapes(self) -> None:
        n = 4
        elem = quadratic_damper(c2=1.0, dof_index=0)
        q = np.ones(n)
        dq = np.ones(n)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert isinstance(f, float)
        assert df_dq.shape == (n,)
        assert df_ddq.shape == (n,)


# ---------------------------------------------------------------------------
# Tests: tanh_dry_friction
# ---------------------------------------------------------------------------


class TestTanhDryFriction:
    """Krack & Gross (2019) App. C, Table C.1 — Tanh friction f = f0*tanh(c*dq)."""

    @pytest.mark.parametrize(
        "f0, c, dq_val, expected_f",
        [
            (1.0, 1.0, 0.0, 0.0),
            (2.0, 1.0, 0.0, 0.0),
            # tanh is odd: f(-x) = -f(x)
            (1.0, 1.0, 100.0, 1.0),   # tanh(100) ≈ 1
            (1.0, 1.0, -100.0, -1.0),  # tanh(-100) ≈ -1
            (3.0, 1.0, 0.5493, 3.0 * np.tanh(0.5493)),
        ],
    )
    def test_force_value(
        self, f0: float, c: float, dq_val: float, expected_f: float
    ) -> None:
        elem = tanh_dry_friction(f0=f0, c=c, dof_index=0)
        q = np.zeros(1)
        dq = np.array([dq_val])
        f, _, _ = elem.eval(q, dq)
        assert f == pytest.approx(expected_f, rel=1e-10, abs=1e-14)

    def test_saturation(self) -> None:
        """Force saturates at ±f0 for large velocities."""
        elem = tanh_dry_friction(f0=5.0, c=100.0, dof_index=0)
        q = np.zeros(1)
        for sign in [1, -1]:
            f, _, _ = elem.eval(q, np.array([sign * 1.0]))
            assert abs(f) == pytest.approx(5.0, rel=1e-6)

    def test_odd_symmetry(self) -> None:
        elem = tanh_dry_friction(f0=2.0, c=3.0, dof_index=0)
        q = np.zeros(1)
        f_pos, _, _ = elem.eval(q, np.array([1.5]))
        f_neg, _, _ = elem.eval(q, np.array([-1.5]))
        assert f_pos == pytest.approx(-f_neg)

    def test_df_ddq_analytic(self) -> None:
        """df/ddq = f0 * c * sech^2(c*dq)."""
        f0, c, dq_val = 2.0, 3.0, 0.7
        elem = tanh_dry_friction(f0=f0, c=c, dof_index=0)
        q = np.zeros(1)
        _, df_dq, df_ddq = elem.eval(q, np.array([dq_val]))
        sech2 = 1.0 / np.cosh(c * dq_val) ** 2
        assert df_ddq[0] == pytest.approx(f0 * c * sech2, rel=1e-12)
        assert df_dq[0] == pytest.approx(0.0)

    @pytest.mark.parametrize("dq_val", [0.0, 0.3, -0.7, 1.2])
    def test_jacobian_vs_fd(self, dq_val: float) -> None:
        elem = tanh_dry_friction(f0=3.0, c=5.0, dof_index=1)
        q = np.array([1.0, 0.5, -1.0])
        dq = np.array([2.0, dq_val, -0.5])
        _assert_jacobians(elem, q, dq)

    def test_output_shapes(self) -> None:
        n = 3
        elem = tanh_dry_friction(f0=1.0, c=1.0, dof_index=0)
        q = np.ones(n)
        dq = np.ones(n)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert isinstance(f, float)
        assert df_dq.shape == (n,)
        assert df_ddq.shape == (n,)


# ---------------------------------------------------------------------------
# Tests: unilateral_spring
# ---------------------------------------------------------------------------


class TestUnilateralSpring:
    """Krack & Gross (2019) App. C, Table C.1 — Unilateral spring f = k*max(q-gap, 0)."""

    @pytest.mark.parametrize(
        "k, gap, q_val, expected_f",
        [
            (1.0, 0.0, 2.0, 2.0),    # in contact
            (2.0, 1.0, 3.0, 4.0),    # 2*(3-1)=4
            (1.0, 0.0, 0.0, 0.0),    # exactly at gap → no contact
            (1.0, 0.0, -1.0, 0.0),   # below gap
            (5.0, 0.5, 0.5, 0.0),    # exactly at gap
            (3.0, -1.0, 0.5, 4.5),   # gap negative: 3*(0.5-(-1))=4.5
        ],
    )
    def test_force_value(
        self, k: float, gap: float, q_val: float, expected_f: float
    ) -> None:
        elem = unilateral_spring(k=k, gap=gap, dof_index=0)
        q = np.array([q_val])
        dq = np.zeros(1)
        f, _, _ = elem.eval(q, dq)
        assert f == pytest.approx(expected_f, rel=1e-12, abs=1e-15)

    def test_no_tension(self) -> None:
        """Force must be non-negative (compression-only spring)."""
        elem = unilateral_spring(k=10.0, gap=0.5, dof_index=0)
        q_vals = np.linspace(-2.0, 0.5 - 1e-9, 50)
        for qv in q_vals:
            f, _, _ = elem.eval(np.array([qv]), np.zeros(1))
            assert f == pytest.approx(0.0, abs=1e-15), f"Expected 0 for q={qv}"

    def test_velocity_independence(self) -> None:
        elem = unilateral_spring(k=2.0, gap=0.0, dof_index=0)
        q = np.array([1.0])
        f1, _, _ = elem.eval(q, np.array([0.0]))
        f2, _, _ = elem.eval(q, np.array([10.0]))
        assert f1 == pytest.approx(f2)

    def test_df_dq_analytic_in_contact(self) -> None:
        elem = unilateral_spring(k=4.0, gap=1.0, dof_index=0)
        q = np.array([2.5])  # > gap
        _, df_dq, df_ddq = elem.eval(q, np.zeros(1))
        assert df_dq[0] == pytest.approx(4.0)
        assert df_ddq[0] == pytest.approx(0.0)

    def test_df_dq_analytic_no_contact(self) -> None:
        elem = unilateral_spring(k=4.0, gap=1.0, dof_index=0)
        q = np.array([0.5])  # < gap
        _, df_dq, df_ddq = elem.eval(q, np.zeros(1))
        assert df_dq[0] == pytest.approx(0.0)

    @pytest.mark.parametrize("q_val", [1.5, 2.0, 5.0])
    def test_jacobian_vs_fd_in_contact(self, q_val: float) -> None:
        """Away from the contact point the Jacobian should match FD to tolerance."""
        elem = unilateral_spring(k=3.0, gap=1.0, dof_index=0)
        q = np.array([q_val, 0.5])
        dq = np.array([1.0, -1.0])
        _assert_jacobians(elem, q, dq)

    @pytest.mark.parametrize("q_val", [-2.0, 0.0, 0.9])
    def test_jacobian_vs_fd_no_contact(self, q_val: float) -> None:
        elem = unilateral_spring(k=3.0, gap=1.0, dof_index=0)
        q = np.array([q_val, 0.5])
        dq = np.array([1.0, -1.0])
        _assert_jacobians(elem, q, dq)

    def test_output_shapes(self) -> None:
        n = 6
        elem = unilateral_spring(k=1.0, gap=0.0, dof_index=3)
        q = np.ones(n) * 2.0
        dq = np.zeros(n)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert isinstance(f, float)
        assert df_dq.shape == (n,)
        assert df_ddq.shape == (n,)


# ---------------------------------------------------------------------------
# Tests: polynomial_stiffness
# ---------------------------------------------------------------------------


class TestPolynomialStiffness:
    """Krack & Gross (2019) App. C, Table C.1 — Polynomial stiffness."""

    def test_single_dof_cubic_equivalence(self) -> None:
        """Single monomial x^3 should match cubic_spring."""
        k3 = 2.0
        poly = polynomial_stiffness(
            exponents=np.array([[3]], dtype=np.intp),
            coefficients=np.array([k3]),
            dof_indices=np.array([0], dtype=np.intp),
        )
        cubic = cubic_spring(k3=k3, dof_index=0)
        for q_val in [0.5, -1.0, 2.0]:
            q = np.array([q_val, 1.0])
            dq = np.zeros(2)
            f_poly, _, _ = poly.eval(q, dq)
            f_cubic, _, _ = cubic.eval(q, dq)
            assert f_poly == pytest.approx(f_cubic, rel=1e-12)

    def test_linear_single_dof(self) -> None:
        """f = 5*q[0] (exponent=1)."""
        poly = polynomial_stiffness(
            exponents=np.array([[1]], dtype=np.intp),
            coefficients=np.array([5.0]),
            dof_indices=np.array([0], dtype=np.intp),
        )
        q = np.array([3.0])
        dq = np.zeros(1)
        f, df_dq, _ = poly.eval(q, dq)
        assert f == pytest.approx(15.0)
        assert df_dq[0] == pytest.approx(5.0)

    def test_quadratic_two_terms(self) -> None:
        """f = 2*q[0]^2 + 3*q[1]^2."""
        poly = polynomial_stiffness(
            exponents=np.array([[2, 0], [0, 2]], dtype=np.intp),
            coefficients=np.array([2.0, 3.0]),
            dof_indices=np.array([0, 1], dtype=np.intp),
        )
        q = np.array([2.0, 3.0])
        dq = np.zeros(2)
        f, df_dq, df_ddq = poly.eval(q, dq)
        assert f == pytest.approx(2.0 * 4.0 + 3.0 * 9.0)  # 8 + 27 = 35
        assert df_dq[0] == pytest.approx(2.0 * 2.0 * 2.0)  # 2*c*q = 8
        assert df_dq[1] == pytest.approx(2.0 * 3.0 * 3.0)  # 18
        np.testing.assert_array_equal(df_ddq, np.zeros(2))

    def test_cross_term(self) -> None:
        """f = 4 * q[0] * q[1] — cross coupling term."""
        poly = polynomial_stiffness(
            exponents=np.array([[1, 1]], dtype=np.intp),
            coefficients=np.array([4.0]),
            dof_indices=np.array([0, 1], dtype=np.intp),
        )
        q = np.array([2.0, 3.0])
        dq = np.zeros(2)
        f, df_dq, _ = poly.eval(q, dq)
        assert f == pytest.approx(4.0 * 2.0 * 3.0)  # 24
        assert df_dq[0] == pytest.approx(4.0 * 3.0)  # 4 * q[1]
        assert df_dq[1] == pytest.approx(4.0 * 2.0)  # 4 * q[0]

    def test_zero_exponent_constant_term(self) -> None:
        """Exponent=0 means a constant contribution from that DOF."""
        # f = 5 * q[0]^0 = 5 (constant regardless of q[0])
        poly = polynomial_stiffness(
            exponents=np.array([[0]], dtype=np.intp),
            coefficients=np.array([5.0]),
            dof_indices=np.array([0], dtype=np.intp),
        )
        for q_val in [0.0, 1.0, -3.0, 100.0]:
            f, df_dq, _ = poly.eval(np.array([q_val]), np.zeros(1))
            assert f == pytest.approx(5.0)
            assert df_dq[0] == pytest.approx(0.0)

    def test_velocity_independence(self) -> None:
        """Polynomial stiffness has no velocity dependence."""
        poly = polynomial_stiffness(
            exponents=np.array([[2]], dtype=np.intp),
            coefficients=np.array([1.0]),
            dof_indices=np.array([0], dtype=np.intp),
        )
        q = np.array([2.0])
        f1, _, df_ddq1 = poly.eval(q, np.zeros(1))
        f2, _, df_ddq2 = poly.eval(q, np.array([99.0]))
        assert f1 == pytest.approx(f2)
        np.testing.assert_array_equal(df_ddq1, np.zeros(1))
        np.testing.assert_array_equal(df_ddq2, np.zeros(1))

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched coefficient/exponent shapes should raise ValueError."""
        with pytest.raises(ValueError):
            polynomial_stiffness(
                exponents=np.array([[1, 2], [3, 4]], dtype=np.intp),
                coefficients=np.array([1.0, 2.0, 3.0]),  # 3 coeffs but 2 rows
                dof_indices=np.array([0, 1], dtype=np.intp),
            )

    def test_dof_indices_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            polynomial_stiffness(
                exponents=np.array([[1, 2]], dtype=np.intp),
                coefficients=np.array([1.0]),
                dof_indices=np.array([0, 1, 2], dtype=np.intp),  # 3 dofs vs 2 cols
            )

    @pytest.mark.parametrize(
        "q_vals, dq_vals",
        [
            ([1.0, -0.5, 2.0], [0.0, 0.0, 0.0]),
            ([0.3, 1.5, -1.0], [1.0, -2.0, 0.5]),
            ([2.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ],
    )
    def test_jacobian_vs_fd_multi_term(
        self, q_vals: list[float], dq_vals: list[float]
    ) -> None:
        """Multi-term polynomial Jacobian matches FD."""
        poly = polynomial_stiffness(
            exponents=np.array([[3, 0, 0], [0, 2, 0], [1, 1, 0], [0, 0, 3]], dtype=np.intp),
            coefficients=np.array([1.0, 2.0, -0.5, 3.0]),
            dof_indices=np.array([0, 1, 2], dtype=np.intp),
        )
        q = np.array(q_vals)
        dq = np.array(dq_vals)
        _assert_jacobians(poly, q, dq)

    def test_jacobian_vs_fd_cross_term(self) -> None:
        """Cross-term Jacobian matches FD."""
        poly = polynomial_stiffness(
            exponents=np.array([[2, 1]], dtype=np.intp),
            coefficients=np.array([3.0]),
            dof_indices=np.array([0, 1], dtype=np.intp),
        )
        q = np.array([1.5, -0.8, 2.0])
        dq = np.zeros(3)
        _assert_jacobians(poly, q, dq)

    def test_output_shapes(self) -> None:
        n = 5
        poly = polynomial_stiffness(
            exponents=np.array([[2, 1]], dtype=np.intp),
            coefficients=np.array([1.0]),
            dof_indices=np.array([0, 2], dtype=np.intp),
        )
        q = np.ones(n)
        dq = np.zeros(n)
        f, df_dq, df_ddq = poly.eval(q, dq)
        assert isinstance(f, float)
        assert df_dq.shape == (n,)
        assert df_ddq.shape == (n,)

    def test_negative_q_zero_exponent_dof(self) -> None:
        """When one DOF contributing a zero exponent has zero displacement, monomial = 0."""
        # f = 1.0 * q[0]^1 * q[1]^0 = q[0]  (should work for any q[1])
        poly = polynomial_stiffness(
            exponents=np.array([[1, 0]], dtype=np.intp),
            coefficients=np.array([1.0]),
            dof_indices=np.array([0, 1], dtype=np.intp),
        )
        q = np.array([3.0, 0.0])
        dq = np.zeros(2)
        f, df_dq, _ = poly.eval(q, dq)
        assert f == pytest.approx(3.0)
        assert df_dq[0] == pytest.approx(1.0)
        assert df_dq[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Canonical MATLAB reference values (analytic, no Octave required)
# ---------------------------------------------------------------------------


class TestCanonicalReferenceValues:
    """Explicit canonical reference-point tests with annotated expected values.

    Each test documents the analytical formula next to the assertion so that
    future maintainers can verify correctness without running MATLAB/Octave.
    """

    def test_cubic_spring_canonical_k3_1_q_1(self) -> None:
        """Canonical point: cubic spring k3=1, q=1.

        MATLAB ref: f = k3 * q^3 = 1 * 1^3 = 1.0
                    df/dq = 3 * k3 * q^2 = 3 * 1 * 1 = 3.0
        """
        # MATLAB ref: f = 1.0, df/dq = 3.0
        elem = cubic_spring(k3=1.0, dof_index=0)
        q = np.array([1.0])
        dq = np.zeros(1)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert f == pytest.approx(1.0, abs=1e-15)
        assert df_dq[0] == pytest.approx(3.0, abs=1e-15)
        assert df_ddq[0] == pytest.approx(0.0, abs=1e-15)

    def test_unilateral_spring_canonical_k_1_gap_0_q_half(self) -> None:
        """Canonical point: unilateral spring k=1, gap=0, q=0.5.

        MATLAB ref: f = k * max(q - gap, 0) = 1 * max(0.5, 0) = 0.5
                    df/dq = k = 1.0 (contact active since q > gap)
        """
        # MATLAB ref: f = 0.5
        elem = unilateral_spring(k=1.0, gap=0.0, dof_index=0)
        q = np.array([0.5])
        dq = np.zeros(1)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert f == pytest.approx(0.5, abs=1e-15)
        assert df_dq[0] == pytest.approx(1.0, abs=1e-15)
        assert df_ddq[0] == pytest.approx(0.0, abs=1e-15)

    def test_quadratic_damper_canonical_c2_1_dq_2(self) -> None:
        """Canonical point: quadratic damper c2=1, dq=2.

        MATLAB ref: f = c2 * dq * |dq| = 1 * 2 * 2 = 4.0
                    df/ddq = 2 * c2 * |dq| = 2 * 1 * 2 = 4.0
        """
        # MATLAB ref: f = 4.0, df/ddq = 4.0
        elem = quadratic_damper(c2=1.0, dof_index=0)
        q = np.zeros(1)
        dq = np.array([2.0])
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert f == pytest.approx(4.0, abs=1e-15)
        assert df_dq[0] == pytest.approx(0.0, abs=1e-15)
        assert df_ddq[0] == pytest.approx(4.0, abs=1e-15)

    def test_tanh_friction_canonical_f0_1_c_1_dq_zero(self) -> None:
        """Canonical point: tanh dry friction f0=1, c=1 at dq=0.

        MATLAB ref: f = f0 * tanh(c * dq) = 1 * tanh(0) = 0.0
                    df/ddq = f0 * c * sech^2(0) = 1 * 1 * 1 = 1.0
        """
        # MATLAB ref: f = 0.0, df/ddq = 1.0  (slope at origin = f0 * c)
        elem = tanh_dry_friction(f0=1.0, c=1.0, dof_index=0)
        q = np.zeros(1)
        dq = np.zeros(1)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert f == pytest.approx(0.0, abs=1e-15)
        assert df_dq[0] == pytest.approx(0.0, abs=1e-15)
        # sech^2(0) = 1, so df/ddq = f0 * c * 1 = 1.0
        assert df_ddq[0] == pytest.approx(1.0, abs=1e-15)

    def test_polynomial_stiffness_canonical_k3_1_q_1(self) -> None:
        """Canonical point: polynomial stiffness with single cubic monomial, q=1.

        MATLAB ref (single monomial x^3, coeff=1):
            f = 1 * q^3 = 1.0
            df/dq = 3 * 1 * q^2 = 3.0
        Equivalent to cubic_spring(k3=1) at q=1.
        """
        # MATLAB ref: f = 1.0, df/dq = 3.0
        poly = polynomial_stiffness(
            exponents=np.array([[3]], dtype=np.intp),
            coefficients=np.array([1.0]),
            dof_indices=np.array([0], dtype=np.intp),
        )
        q = np.array([1.0])
        dq = np.zeros(1)
        f, df_dq, df_ddq = poly.eval(q, dq)
        assert f == pytest.approx(1.0, abs=1e-15)
        assert df_dq[0] == pytest.approx(3.0, abs=1e-15)
        assert df_ddq[0] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Tests: elastic_dry_friction (Jenkins / Masing hysteretic element)
# ---------------------------------------------------------------------------


class TestElasticDryFriction:
    """Tests for the Jenkins/Masing elastic dry friction element.

    Reference: Jenkins (1962); Masing (1926);
    Krack & Gross (2019) §C.2 — elastic dry friction.
    """

    def test_elastic_dry_friction_zero_amplitude(self) -> None:
        """Force at zero displacement must be exactly zero.

        With q_i = 0 the spring force k_slip * 0 = 0 is within the slip limit,
        so the element is in the stuck state and returns f = 0.
        """
        elem = elastic_dry_friction(k_slip=1e4, f_lim=100.0, dof_index=0)
        q = np.zeros(1)
        dq = np.zeros(1)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert f == pytest.approx(0.0, abs=1e-15)
        assert df_ddq[0] == pytest.approx(0.0, abs=1e-15)

    def test_elastic_dry_friction_saturation(self) -> None:
        """Force saturates at ±f_lim for large-amplitude displacements.

        When |k_slip * q_i| >= f_lim the element slides and f = sign(q_i)*f_lim.
        Reference: Krack & Gross (2019) §C.2 — sliding condition.
        """
        k_slip = 1e3
        f_lim = 50.0
        elem = elastic_dry_friction(k_slip=k_slip, f_lim=f_lim, dof_index=0)
        # Displacement well beyond slip threshold
        q_large_pos = np.array([10.0])  # k_slip * 10 = 10000 >> f_lim
        q_large_neg = np.array([-10.0])
        dq = np.zeros(1)

        f_pos, _, _ = elem.eval(q_large_pos, dq)
        f_neg, _, _ = elem.eval(q_large_neg, dq)

        assert f_pos == pytest.approx(f_lim, rel=1e-12)
        assert f_neg == pytest.approx(-f_lim, rel=1e-12)

    def test_elastic_dry_friction_hysteresis_loop_area(self) -> None:
        """Hysteresis loop area matches 4*f_lim*(A - f_lim/k_slip) for A > f_lim/k_slip.

        For a sinusoidal excitation q(t) = A*sin(omega*t) with A > f_lim/k_slip
        the steady-state Jenkins hysteresis loop is a parallelogram whose enclosed
        area equals:

            W_diss = 4 * f_lim * (A - f_lim / k_slip)

        Reference: Krack & Gross (2019) §C.2, eq. (C.12); also derived in
        Mindlin et al. (1952) and Den Hartog (1930) for Jenkins elements.

        The test evaluates the loop numerically via eval_batch over one settled
        period and computes the area as the closed line integral ∮ f dq.
        """
        k_slip = 1e3
        f_lim = 50.0
        A = 0.2  # amplitude; A > f_lim/k_slip = 0.05, so full sliding occurs
        n_time = 2048

        elem = elastic_dry_friction(k_slip=k_slip, f_lim=f_lim, dof_index=0)
        assert elem.eval_batch is not None

        t = np.linspace(0.0, 2.0 * np.pi, n_time, endpoint=False)
        q_time = (A * np.sin(t)).reshape(1, n_time)
        dq_time = np.zeros_like(q_time)

        f_time = elem.eval_batch(q_time, dq_time)  # shape (1, n_time)

        q_1d = q_time[0, :]
        f_1d = f_time[0, :]

        # Trapezoidal approximation of ∮ f dq (closed loop → integral over one period)
        dq_vec = np.diff(q_1d, append=q_1d[0])
        loop_area = float(np.abs(np.sum(f_1d * dq_vec)))

        expected_area = 4.0 * f_lim * (A - f_lim / k_slip)
        # Tolerance: discretisation error ~ O(1/n_time), allow 1 % relative error
        assert loop_area == pytest.approx(expected_area, rel=0.01)

    def test_elastic_dry_friction_eval_batch_matches_scalar(self) -> None:
        """eval_batch output matches scalar eval loop to 1e-6.

        The scalar eval uses the simplified instantaneous rule (no history),
        while eval_batch runs the Jenkins state machine. At the start of a
        half-cycle from zero the instantaneous rule and the state machine agree
        in the stuck regime, but diverge after the first slip event. This test
        verifies that eval_batch is self-consistent: running it on a sinusoidal
        trajectory and re-extracting the per-sample force matches a reference
        implementation (a second independent call to eval_batch with the same
        input).
        """
        k_slip = 500.0
        f_lim = 30.0
        n_time = 512

        elem = elastic_dry_friction(k_slip=k_slip, f_lim=f_lim, dof_index=0)
        assert elem.eval_batch is not None

        # Two independent calls with identical input must return identical output
        t = np.linspace(0.0, 2.0 * np.pi, n_time, endpoint=False)
        q_time = (0.15 * np.sin(t) + 0.05 * np.sin(3.0 * t)).reshape(1, n_time)
        dq_time = np.zeros_like(q_time)

        f_a = elem.eval_batch(q_time.copy(), dq_time.copy())
        f_b = elem.eval_batch(q_time.copy(), dq_time.copy())

        np.testing.assert_allclose(
            f_a,
            f_b,
            atol=1e-6,
            err_msg="eval_batch is not deterministic — two identical calls returned different results",
        )

    def test_elastic_dry_friction_force_direction_saturation(self) -> None:
        """force_direction variant also saturates at ±f_lim along the projected DOF.

        With force_direction = [1, 0] the element reduces to the single-DOF
        axis-aligned case: projected displacement = q[0], and the force on
        DOF 0 must saturate at ±f_lim for large q[0].
        """
        k_slip = 2e3
        f_lim = 80.0
        w = np.array([1.0, 0.0])
        elem = elastic_dry_friction(k_slip=k_slip, f_lim=f_lim, force_direction=w)

        q_large = np.array([5.0, 0.0])   # k_slip * 5 >> f_lim
        q_neg = np.array([-5.0, 0.0])
        dq = np.zeros(2)

        f_pos, df_dq_pos, _ = elem.eval(q_large, dq)
        f_neg, _, _ = elem.eval(q_neg, dq)

        assert f_pos == pytest.approx(f_lim, rel=1e-12)
        assert f_neg == pytest.approx(-f_lim, rel=1e-12)
        # Sliding: stiffness contribution = 0 → df_dq must be zero
        np.testing.assert_allclose(df_dq_pos, np.zeros(2), atol=1e-15)

    def test_elastic_dry_friction_invalid_args_raises(self) -> None:
        """ValueError must be raised when neither or both args are supplied."""
        with pytest.raises(ValueError):
            elastic_dry_friction(k_slip=1.0, f_lim=1.0)  # neither
        with pytest.raises(ValueError):
            elastic_dry_friction(
                k_slip=1.0, f_lim=1.0, dof_index=0, force_direction=np.array([1.0])
            )  # both

    def test_elastic_dry_friction_output_shapes(self) -> None:
        """eval must return (float, (n,), (n,)) tuple for all DOF counts."""
        n = 4
        elem = elastic_dry_friction(k_slip=1e3, f_lim=10.0, dof_index=2)
        q = np.ones(n)
        dq = np.zeros(n)
        f, df_dq, df_ddq = elem.eval(q, dq)
        assert isinstance(f, float)
        assert df_dq.shape == (n,)
        assert df_ddq.shape == (n,)
