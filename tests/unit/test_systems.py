"""
Unit tests for :mod:`nlvib.systems.base`.

Tests cover:
- ``n_dof`` property
- ``add_nonlinear_element`` registration
- ``eval_nonlinear_forces`` assembly from multiple elements at different DOFs
- Jacobian assembly consistency with finite differences
- Input validation (bad shapes, bad matrix sizes)
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from nlvib.nonlinearities.elements import (
    cubic_spring,
    quadratic_damper,
    tanh_dry_friction,
    unilateral_spring,
)
from nlvib.systems.base import MechanicalSystem
from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.systems.oscillators import SingleMassOscillator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(n: int) -> MechanicalSystem:
    """Create a diagonal n-DOF system with unit matrices."""
    M = np.eye(n)
    D = 0.1 * np.eye(n)
    K = 4.0 * np.eye(n)
    return MechanicalSystem(M, D, K)


def _fd_jacobian_q(
    system: MechanicalSystem, q: np.ndarray, dq: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Finite-difference Jacobian df/dq of the assembled force vector."""
    n = system.n_dof
    J = np.zeros((n, n))
    f0, _, _ = system.eval_nonlinear_forces(q, dq)
    for j in range(n):
        q_plus = q.copy()
        q_plus[j] += eps
        f_plus, _, _ = system.eval_nonlinear_forces(q_plus, dq)
        J[:, j] = (f_plus - f0) / eps
    return J


def _fd_jacobian_dq(
    system: MechanicalSystem, q: np.ndarray, dq: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Finite-difference Jacobian df/ddq of the assembled force vector."""
    n = system.n_dof
    J = np.zeros((n, n))
    f0, _, _ = system.eval_nonlinear_forces(q, dq)
    for j in range(n):
        dq_plus = dq.copy()
        dq_plus[j] += eps
        f_plus, _, _ = system.eval_nonlinear_forces(q, dq_plus)
        J[:, j] = (f_plus - f0) / eps
    return J


# ---------------------------------------------------------------------------
# n_dof property
# ---------------------------------------------------------------------------


class TestNDof:
    def test_ndof_1(self) -> None:
        sys = _make_system(1)
        assert sys.n_dof == 1

    def test_ndof_3(self) -> None:
        sys = _make_system(3)
        assert sys.n_dof == 3

    def test_ndof_from_sparse_input(self) -> None:
        n = 5
        M = sp.eye(n, format="csr")
        D = sp.diags([0.2] * n, format="csr")
        K = sp.diags([3.0] * n, format="csr")
        sys = MechanicalSystem(M, D, K)
        assert sys.n_dof == n

    def test_ndof_large(self) -> None:
        n = 20
        sys = _make_system(n)
        assert sys.n_dof == n


# ---------------------------------------------------------------------------
# Matrix storage
# ---------------------------------------------------------------------------


class TestMatrixStorage:
    def test_matrices_are_csr(self) -> None:
        sys = _make_system(4)
        assert sp.issparse(sys.M)
        assert sp.issparse(sys.D)
        assert sp.issparse(sys.K)
        assert isinstance(sys.M, sp.csr_matrix)
        assert isinstance(sys.D, sp.csr_matrix)
        assert isinstance(sys.K, sp.csr_matrix)

    def test_dense_input_converted(self) -> None:
        """Dense NumPy arrays must be converted to CSR."""
        n = 3
        sys = _make_system(n)
        assert_allclose(sys.M.toarray(), np.eye(n))
        assert_allclose(sys.D.toarray(), 0.1 * np.eye(n))
        assert_allclose(sys.K.toarray(), 4.0 * np.eye(n))

    def test_sparse_input_preserved(self) -> None:
        n = 5
        M = sp.coo_matrix(np.eye(n))
        D = sp.lil_matrix(np.eye(n) * 0.05)
        K = sp.csr_matrix(np.eye(n) * 10.0)
        sys = MechanicalSystem(M, D, K)
        assert isinstance(sys.M, sp.csr_matrix)
        assert_allclose(sys.K.toarray(), np.eye(n) * 10.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_nonsquare_M_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            MechanicalSystem(
                np.ones((2, 3)), np.eye(3), np.eye(3)
            )

    def test_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="size"):
            MechanicalSystem(np.eye(2), np.eye(3), np.eye(2))

    def test_wrong_q_shape_raises(self) -> None:
        sys = _make_system(3)
        with pytest.raises(ValueError, match="shape"):
            sys.eval_nonlinear_forces(np.zeros(2), np.zeros(3))

    def test_wrong_dq_shape_raises(self) -> None:
        sys = _make_system(3)
        with pytest.raises(ValueError, match="shape"):
            sys.eval_nonlinear_forces(np.zeros(3), np.zeros(5))


# ---------------------------------------------------------------------------
# eval_nonlinear_forces — basic assembly
# ---------------------------------------------------------------------------


class TestEvalNonlinearForcesBasic:
    def test_no_elements_returns_zeros(self) -> None:
        sys = _make_system(4)
        q = np.array([1.0, -0.5, 0.2, 0.0])
        dq = np.zeros(4)
        f, df_dq, df_ddq = sys.eval_nonlinear_forces(q, dq)
        assert_allclose(f, np.zeros(4))
        assert_allclose(df_dq, np.zeros((4, 4)))
        assert_allclose(df_ddq, np.zeros((4, 4)))

    def test_single_cubic_spring_at_dof0(self) -> None:
        """Cubic spring k3=2 at DOF 0: f[0] = 2*q[0]^3."""
        sys = _make_system(3)
        sys.add_nonlinear_element(cubic_spring(k3=2.0, dof_index=0))
        q = np.array([3.0, 0.0, 0.0])
        dq = np.zeros(3)
        f, df_dq, _ = sys.eval_nonlinear_forces(q, dq)
        assert_allclose(f[0], 2.0 * 3.0**3)
        assert_allclose(f[1:], 0.0)
        # Jacobian: df[0]/dq[0] = 3*k3*q[0]^2 = 3*2*9 = 54
        assert_allclose(df_dq[0, 0], 3.0 * 2.0 * 9.0)
        assert_allclose(df_dq[1:, :], 0.0)

    def test_single_cubic_spring_at_dof2(self) -> None:
        """Cubic spring at last DOF; other DOF forces must be zero."""
        sys = _make_system(4)
        sys.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=3))
        q = np.array([0.0, 0.0, 0.0, 2.0])
        dq = np.zeros(4)
        f, df_dq, _ = sys.eval_nonlinear_forces(q, dq)
        assert_allclose(f[3], 8.0)       # 1.0 * 2^3
        assert_allclose(f[:3], 0.0)
        assert_allclose(df_dq[3, 3], 12.0)  # 3*1*4

    def test_quadratic_damper_only_in_velocity_jacobian(self) -> None:
        sys = _make_system(2)
        sys.add_nonlinear_element(quadratic_damper(c2=3.0, dof_index=1))
        q = np.zeros(2)
        dq = np.array([0.0, 2.0])
        f, df_dq, df_ddq = sys.eval_nonlinear_forces(q, dq)
        assert_allclose(f[1], 3.0 * 2.0 * 2.0)   # c2 * dq * |dq|
        assert_allclose(df_dq, np.zeros((2, 2)))
        assert_allclose(df_ddq[1, 1], 2.0 * 3.0 * 2.0)  # 2*c2*|dq|


# ---------------------------------------------------------------------------
# eval_nonlinear_forces — multi-element assembly
# ---------------------------------------------------------------------------


class TestMultiElementAssembly:
    def test_two_cubic_springs_different_dofs(self) -> None:
        """Two cubic springs at DOF 0 and DOF 2 must not cross-contaminate."""
        sys = _make_system(3)
        sys.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=0))
        sys.add_nonlinear_element(cubic_spring(k3=5.0, dof_index=2))

        q = np.array([2.0, 0.5, 1.0])
        dq = np.zeros(3)
        f, df_dq, _ = sys.eval_nonlinear_forces(q, dq)

        assert_allclose(f[0], 1.0 * 2.0**3)
        assert_allclose(f[1], 0.0)
        assert_allclose(f[2], 5.0 * 1.0**3)

        assert_allclose(df_dq[0, 0], 3.0 * 1.0 * 4.0)
        assert_allclose(df_dq[2, 2], 3.0 * 5.0 * 1.0)
        # No cross-DOF contamination
        assert_allclose(df_dq[0, 2], 0.0)
        assert_allclose(df_dq[2, 0], 0.0)
        assert_allclose(df_dq[1, :], 0.0)

    def test_cubic_spring_and_tanh_friction_combined(self) -> None:
        """Cubic spring (displacement) + tanh friction (velocity) at different DOFs."""
        sys = _make_system(4)
        sys.add_nonlinear_element(cubic_spring(k3=2.0, dof_index=1))
        sys.add_nonlinear_element(tanh_dry_friction(f0=5.0, c=1.0, dof_index=3))

        q = np.array([0.0, 1.5, 0.0, 0.0])
        dq = np.array([0.0, 0.0, 0.0, 0.5])
        f, df_dq, df_ddq = sys.eval_nonlinear_forces(q, dq)

        expected_f1 = 2.0 * 1.5**3
        expected_f3 = 5.0 * np.tanh(1.0 * 0.5)

        assert_allclose(f[1], expected_f1, rtol=1e-12)
        assert_allclose(f[3], expected_f3, rtol=1e-12)
        assert_allclose(f[0], 0.0)
        assert_allclose(f[2], 0.0)

        # Displacement Jacobian: only from cubic spring at DOF 1
        assert_allclose(df_dq[1, 1], 3.0 * 2.0 * 1.5**2, rtol=1e-12)
        assert_allclose(df_dq[3, :], 0.0)

        # Velocity Jacobian: only from tanh at DOF 3
        sech2 = 1.0 - np.tanh(0.5)**2
        assert_allclose(df_ddq[3, 3], 5.0 * 1.0 * sech2, rtol=1e-12)
        assert_allclose(df_ddq[1, :], 0.0)

    def test_unilateral_spring_out_of_contact(self) -> None:
        """Unilateral spring below gap: force and Jacobian must be zero."""
        sys = _make_system(2)
        sys.add_nonlinear_element(unilateral_spring(k=100.0, gap=0.5, dof_index=0))
        q = np.array([0.3, 0.0])   # below gap
        dq = np.zeros(2)
        f, df_dq, df_ddq = sys.eval_nonlinear_forces(q, dq)
        assert_allclose(f, 0.0)
        assert_allclose(df_dq, 0.0)

    def test_unilateral_spring_in_contact(self) -> None:
        """Unilateral spring above gap: force and Jacobian active."""
        sys = _make_system(2)
        sys.add_nonlinear_element(unilateral_spring(k=100.0, gap=0.5, dof_index=0))
        q = np.array([0.8, 0.0])   # 0.3 m penetration
        dq = np.zeros(2)
        f, df_dq, _ = sys.eval_nonlinear_forces(q, dq)
        assert_allclose(f[0], 100.0 * 0.3, rtol=1e-12)
        assert_allclose(df_dq[0, 0], 100.0)


# ---------------------------------------------------------------------------
# Jacobian finite-difference consistency
# ---------------------------------------------------------------------------


class TestJacobianConsistency:
    def test_df_dq_vs_fd_cubic_spring(self) -> None:
        """Analytical df/dq must match FD to 1e-5 tolerance."""
        sys = _make_system(3)
        sys.add_nonlinear_element(cubic_spring(k3=3.0, dof_index=0))
        sys.add_nonlinear_element(cubic_spring(k3=0.5, dof_index=2))

        q = np.array([0.7, -1.2, 0.4])
        dq = np.zeros(3)

        _, df_dq_analytical, _ = sys.eval_nonlinear_forces(q, dq)
        df_dq_fd = _fd_jacobian_q(sys, q, dq)

        assert_allclose(df_dq_analytical, df_dq_fd, atol=1e-5)

    def test_df_ddq_vs_fd_quadratic_damper(self) -> None:
        """Analytical df/ddq must match FD to 1e-5 tolerance."""
        sys = _make_system(2)
        sys.add_nonlinear_element(quadratic_damper(c2=2.0, dof_index=0))
        sys.add_nonlinear_element(quadratic_damper(c2=4.0, dof_index=1))

        q = np.zeros(2)
        dq = np.array([1.5, -0.8])

        _, _, df_ddq_analytical = sys.eval_nonlinear_forces(q, dq)
        df_ddq_fd = _fd_jacobian_dq(sys, q, dq)

        assert_allclose(df_ddq_analytical, df_ddq_fd, atol=1e-5)

    def test_df_dq_vs_fd_mixed_elements(self) -> None:
        """Multi-element system: assembled df/dq matches FD."""
        sys = _make_system(4)
        sys.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=0))
        sys.add_nonlinear_element(unilateral_spring(k=50.0, gap=0.1, dof_index=2))
        sys.add_nonlinear_element(cubic_spring(k3=2.0, dof_index=3))

        q = np.array([0.5, 0.0, 0.5, -1.0])
        dq = np.zeros(4)

        _, df_dq_analytical, _ = sys.eval_nonlinear_forces(q, dq)
        df_dq_fd = _fd_jacobian_q(sys, q, dq)

        assert_allclose(df_dq_analytical, df_dq_fd, atol=1e-5)

    def test_df_ddq_vs_fd_tanh_friction(self) -> None:
        """Tanh dry-friction df/ddq matches FD across multiple DOFs.

        First-order FD with eps=1e-6 has O(eps) truncation error plus
        O(eps * f'' * eps) ≈ O(1e-5) for tanh.  Tolerance is 5e-5 to
        accommodate this without a tighter (slower) central-difference check.
        """
        sys = _make_system(3)
        sys.add_nonlinear_element(tanh_dry_friction(f0=10.0, c=2.0, dof_index=0))
        sys.add_nonlinear_element(tanh_dry_friction(f0=3.0, c=5.0, dof_index=2))

        q = np.zeros(3)
        dq = np.array([0.3, 0.0, -0.7])

        _, _, df_ddq_analytical = sys.eval_nonlinear_forces(q, dq)
        df_ddq_fd = _fd_jacobian_dq(sys, q, dq)

        assert_allclose(df_ddq_analytical, df_ddq_fd, atol=5e-5)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_no_elements(self) -> None:
        sys = _make_system(3)
        r = repr(sys)
        assert "n_dof=3" in r
        assert "n_elements=0" in r

    def test_repr_with_elements(self) -> None:
        sys = _make_system(2)
        sys.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=0))
        r = repr(sys)
        assert "n_elements=1" in r


# ---------------------------------------------------------------------------
# SingleMassOscillator
# ---------------------------------------------------------------------------


class TestSingleMassOscillator:
    """Tests for :class:`~nlvib.systems.oscillators.SingleMassOscillator`."""

    # --- subclass relationship ---

    def test_is_mechanical_system_subclass(self) -> None:
        """SingleMassOscillator must be a MechanicalSystem subclass (T-06)."""
        assert issubclass(SingleMassOscillator, MechanicalSystem)

    def test_instance_is_mechanical_system(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.05, k=4.0)
        assert isinstance(smo, MechanicalSystem)

    # --- n_dof ---

    def test_single_ndof_is_one(self) -> None:
        smo = SingleMassOscillator(m=2.0, d=0.1, k=10.0)
        assert smo.n_dof == 1

    # --- matrix shapes ---

    def test_M_shape_is_1x1(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        assert smo.M.shape == (1, 1)

    def test_D_shape_is_1x1(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        assert smo.D.shape == (1, 1)

    def test_K_shape_is_1x1(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        assert smo.K.shape == (1, 1)

    # --- matrix values ---

    def test_M_value(self) -> None:
        smo = SingleMassOscillator(m=3.5, d=0.2, k=8.0)
        assert_allclose(smo.M.toarray(), np.array([[3.5]]))

    def test_D_value(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.07, k=4.0)
        assert_allclose(smo.D.toarray(), np.array([[0.07]]))

    def test_K_value(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.0, k=9.81)
        assert_allclose(smo.K.toarray(), np.array([[9.81]]))

    # --- sparse type ---

    def test_matrices_are_csr(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        assert sp.issparse(smo.M) and isinstance(smo.M, sp.csr_matrix)
        assert sp.issparse(smo.D) and isinstance(smo.D, sp.csr_matrix)
        assert sp.issparse(smo.K) and isinstance(smo.K, sp.csr_matrix)

    # --- scalar parameter properties ---

    def test_mass_property(self) -> None:
        smo = SingleMassOscillator(m=2.0, d=0.1, k=5.0)
        assert smo.mass == pytest.approx(2.0)

    def test_damping_property(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.03, k=1.0)
        assert smo.damping == pytest.approx(0.03)

    def test_stiffness_property(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.0, k=7.5)
        assert smo.stiffness == pytest.approx(7.5)

    # --- validation ---

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            SingleMassOscillator(m=0.0, d=0.0, k=1.0)

    def test_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            SingleMassOscillator(m=-1.0, d=0.0, k=1.0)

    def test_negative_damping_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SingleMassOscillator(m=1.0, d=-0.1, k=1.0)

    def test_negative_stiffness_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SingleMassOscillator(m=1.0, d=0.0, k=-1.0)

    def test_zero_damping_allowed(self) -> None:
        """d=0 is physically valid (undamped system)."""
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        assert smo.damping == 0.0

    def test_zero_stiffness_allowed(self) -> None:
        """k=0 is physically valid (free mass)."""
        smo = SingleMassOscillator(m=1.0, d=0.0, k=0.0)
        assert smo.stiffness == 0.0

    # --- Duffing oscillator: eval_nonlinear_forces with cubic spring ---
    # The Duffing oscillator (K&G §5.1) has:
    #   f_nl = k3 * q^3
    # so eval_nonlinear_forces must return f[0] = k3 * q[0]^3.

    def test_single_duffing_force_value(self) -> None:
        """Cubic spring k3=0.5 at q=2.0 gives f_nl = 0.5 * 8 = 4.0 (K&G §5.1)."""
        k3 = 0.5
        q_val = 2.0
        smo = SingleMassOscillator(m=1.0, d=0.02, k=1.0)
        smo.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))
        q = np.array([q_val])
        dq = np.array([0.0])
        f, _, _ = smo.eval_nonlinear_forces(q, dq)
        expected = k3 * q_val**3
        assert_allclose(f[0], expected, rtol=1e-12)

    def test_duffing_force_multiple_displacements(self) -> None:
        """f_nl = k3 * q^3 holds for several test displacements (K&G §5.1)."""
        k3 = 1.0
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        smo.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))
        dq = np.array([0.0])
        for q_val in (-2.0, -0.5, 0.0, 0.5, 1.0, 3.0):
            q = np.array([q_val])
            f, _, _ = smo.eval_nonlinear_forces(q, dq)
            assert_allclose(f[0], k3 * q_val**3, rtol=1e-12,
                            err_msg=f"Failed at q={q_val}")

    def test_duffing_jacobian(self) -> None:
        """df_nl/dq = 3*k3*q^2 (analytic Jacobian of cubic spring, K&G §5.1)."""
        k3 = 2.0
        q_val = 1.5
        smo = SingleMassOscillator(m=1.0, d=0.0, k=1.0)
        smo.add_nonlinear_element(cubic_spring(k3=k3, dof_index=0))
        q = np.array([q_val])
        dq = np.array([0.0])
        _, df_dq, _ = smo.eval_nonlinear_forces(q, dq)
        expected_jac = 3.0 * k3 * q_val**2
        assert_allclose(df_dq[0, 0], expected_jac, rtol=1e-12)

    def test_no_nonlinear_forces_without_element(self) -> None:
        """Without any nonlinear elements, eval_nonlinear_forces returns zeros."""
        smo = SingleMassOscillator(m=1.0, d=0.05, k=4.0)
        q = np.array([3.0])
        dq = np.array([1.0])
        f, df_dq, df_ddq = smo.eval_nonlinear_forces(q, dq)
        assert_allclose(f, np.array([0.0]))
        assert_allclose(df_dq, np.zeros((1, 1)))
        assert_allclose(df_ddq, np.zeros((1, 1)))

    # --- repr ---

    def test_repr_contains_parameters(self) -> None:
        smo = SingleMassOscillator(m=1.0, d=0.05, k=4.0)
        r = repr(smo)
        assert "SingleMassOscillator" in r
        assert "1.0" in r
        assert "0.05" in r
        assert "4.0" in r


# ---------------------------------------------------------------------------
# ChainOfOscillators (T-07)
# ---------------------------------------------------------------------------


class TestChainOfOscillators:
    """Tests for :class:`~nlvib.systems.oscillators.ChainOfOscillators`."""

    # --- subclass relationship ---

    def test_is_mechanical_system_subclass(self) -> None:
        """ChainOfOscillators must be a MechanicalSystem subclass (T-07)."""
        assert issubclass(ChainOfOscillators, MechanicalSystem)

    def test_instance_is_mechanical_system(self) -> None:
        """Instance must pass isinstance check."""
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        assert isinstance(chain, MechanicalSystem)

    # --- n_dof ---

    def test_n_dof_two(self) -> None:
        chain = ChainOfOscillators(
            masses=[1.0, 2.0],
            stiffnesses=[0.5, 1.0, 0.5],
            dampings=[0.0, 0.0, 0.0],
        )
        assert chain.n_dof == 2

    def test_n_dof_five(self) -> None:
        chain = ChainOfOscillators(
            masses=[1.0] * 5,
            stiffnesses=[1.0] * 6,
            dampings=[0.0] * 6,
        )
        assert chain.n_dof == 5

    # --- 2-DOF stiffness matrix (analytical reference) ---
    #
    # Convention: k = [k0, k1, k2]
    #   Ground --[k0]-- m0 --[k1]-- m1 --[k2]-- Ground
    #
    # K = [[k0+k1,  -k1 ],
    #      [ -k1,  k1+k2]]

    def test_chain_2dof_K_matrix_all_springs(self) -> None:
        """2-DOF: full K matrix with left/inter/right boundary springs."""
        k0, k1, k2 = 3.0, 2.0, 1.0
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[k0, k1, k2],
            dampings=[0.0, 0.0, 0.0],
        )
        K = chain.K.toarray()
        expected = np.array([
            [k0 + k1, -k1],
            [-k1,     k1 + k2],
        ])
        assert_allclose(K, expected, atol=1e-15)

    def test_chain_2dof_K_matrix_no_right_boundary(self) -> None:
        """2-DOF: right boundary spring = 0 (free right end)."""
        k0, k1 = 1.0, 0.5
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[k0, k1, 0.0],
            dampings=[0.0, 0.0, 0.0],
        )
        K = chain.K.toarray()
        expected = np.array([
            [k0 + k1, -k1],
            [-k1,     k1],
        ])
        assert_allclose(K, expected, atol=1e-15)

    def test_chain_2dof_K_matrix_no_left_boundary(self) -> None:
        """2-DOF: left boundary spring = 0 (free left end)."""
        k1, k2 = 2.0, 1.5
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[0.0, k1, k2],
            dampings=[0.0, 0.0, 0.0],
        )
        K = chain.K.toarray()
        expected = np.array([
            [k1,      -k1],
            [-k1,     k1 + k2],
        ])
        assert_allclose(K, expected, atol=1e-15)

    def test_chain_2dof_K_symmetry(self) -> None:
        """K matrix must be symmetric."""
        chain = ChainOfOscillators(
            masses=[2.0, 3.0],
            stiffnesses=[4.0, 5.0, 6.0],
            dampings=[0.0, 0.0, 0.0],
        )
        K = chain.K.toarray()
        assert_allclose(K, K.T, atol=1e-15)

    # --- 5-DOF structure tests ---

    def test_chain_5dof_K_shape(self) -> None:
        """5-DOF: K matrix must have shape (5, 5)."""
        chain = ChainOfOscillators(
            masses=[1.0] * 5,
            stiffnesses=[1.0] * 6,
            dampings=[0.0] * 6,
        )
        assert chain.K.shape == (5, 5)

    def test_chain_5dof_K_diagonal_values(self) -> None:
        """5-DOF uniform chain: all diagonal entries = 2k (interior masses)
        and k+k_bnd (boundary masses).

        k = [1, 1, 1, 1, 1, 1]  →  K_ii = k[i] + k[i+1] = 2 for all i.
        """
        k_val = 1.0
        chain = ChainOfOscillators(
            masses=[1.0] * 5,
            stiffnesses=[k_val] * 6,
            dampings=[0.0] * 6,
        )
        K = chain.K.toarray()
        expected_diag = np.full(5, 2.0 * k_val)
        assert_allclose(np.diag(K), expected_diag, atol=1e-15)

    def test_chain_5dof_K_offdiagonal_values(self) -> None:
        """5-DOF uniform chain: all off-diagonal entries = -k."""
        k_val = 1.0
        chain = ChainOfOscillators(
            masses=[1.0] * 5,
            stiffnesses=[k_val] * 6,
            dampings=[0.0] * 6,
        )
        K = chain.K.toarray()
        # Super- and sub-diagonal: -k[i+1] = -1 for uniform chain
        expected_offdiag = np.full(4, -k_val)
        assert_allclose(np.diag(K, 1), expected_offdiag, atol=1e-15)
        assert_allclose(np.diag(K, -1), expected_offdiag, atol=1e-15)

    def test_chain_5dof_K_zero_elsewhere(self) -> None:
        """5-DOF K matrix must have zeros everywhere outside the tridiagonal."""
        chain = ChainOfOscillators(
            masses=[1.0] * 5,
            stiffnesses=[1.0] * 6,
            dampings=[0.0] * 6,
        )
        K = chain.K.toarray()
        for i in range(5):
            for j in range(5):
                if abs(i - j) > 1:
                    assert K[i, j] == 0.0, (
                        f"K[{i},{j}] = {K[i, j]} should be zero."
                    )

    def test_chain_5dof_K_symmetry(self) -> None:
        """5-DOF: K must be symmetric."""
        chain = ChainOfOscillators(
            masses=[1.0, 2.0, 3.0, 2.0, 1.0],
            stiffnesses=[0.5, 1.0, 1.5, 2.0, 1.0, 0.5],
            dampings=[0.0] * 6,
        )
        K = chain.K.toarray()
        assert_allclose(K, K.T, atol=1e-15)

    # --- Mass matrix ---

    def test_chain_2dof_M_matrix(self) -> None:
        """Mass matrix must be diagonal with correct values."""
        m1, m2 = 2.0, 3.0
        chain = ChainOfOscillators(
            masses=[m1, m2],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        M = chain.M.toarray()
        expected = np.diag([m1, m2])
        assert_allclose(M, expected, atol=1e-15)

    # --- Damping matrix ---

    def test_chain_2dof_D_matrix(self) -> None:
        """Damping matrix follows same tridiagonal assembly as K."""
        d0, d1, d2 = 0.1, 0.2, 0.3
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[0.0, 0.0, 0.0],
            dampings=[d0, d1, d2],
        )
        D = chain.D.toarray()
        expected = np.array([
            [d0 + d1, -d1],
            [-d1,     d1 + d2],
        ])
        assert_allclose(D, expected, atol=1e-15)

    # --- sparse format ---

    def test_matrices_are_csr(self) -> None:
        """M, D, K must all be scipy.sparse.csr_matrix instances."""
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.1, 0.1, 0.1],
        )
        assert isinstance(chain.M, sp.csr_matrix)
        assert isinstance(chain.D, sp.csr_matrix)
        assert isinstance(chain.K, sp.csr_matrix)

    # --- input validation ---

    def test_wrong_stiffnesses_length_raises(self) -> None:
        with pytest.raises(ValueError, match="stiffnesses"):
            ChainOfOscillators(
                masses=[1.0, 1.0],
                stiffnesses=[1.0, 1.0],  # should be length 3
                dampings=[0.0, 0.0, 0.0],
            )

    def test_wrong_dampings_length_raises(self) -> None:
        with pytest.raises(ValueError, match="dampings"):
            ChainOfOscillators(
                masses=[1.0, 1.0],
                stiffnesses=[1.0, 1.0, 1.0],
                dampings=[0.0, 0.0],  # should be length 3
            )

    def test_nonpositive_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ChainOfOscillators(
                masses=[1.0, 0.0],
                stiffnesses=[1.0, 1.0, 1.0],
                dampings=[0.0, 0.0, 0.0],
            )

    # --- accessors ---

    def test_masses_accessor_returns_copy(self) -> None:
        chain = ChainOfOscillators(
            masses=[1.0, 2.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        m = chain.masses
        m[0] = 999.0  # modifying the returned array must not affect the system
        assert chain.masses[0] == 1.0

    # --- repr ---

    def test_repr_chain(self) -> None:
        chain = ChainOfOscillators(
            masses=[1.0, 1.0, 1.0],
            stiffnesses=[1.0] * 4,
            dampings=[0.0] * 4,
        )
        r = repr(chain)
        assert "ChainOfOscillators" in r
        assert "n_dof=3" in r

    # --- canonical MATLAB reference values ---

    def test_chain_2dof_equal_masses_equal_springs_K_canonical(self) -> None:
        """Canonical 2-DOF chain: m=[1,1], k=[1,1,1] (equal left/inter/right springs).

        MATLAB ref (K&G 2019 §5, tridiagonal assembly with n+1=3 springs):
            K_ii = k[i] + k[i+1]:  K[0,0] = 1+1 = 2,  K[1,1] = 1+1 = 2
            K_ij = -k[i+1]:        K[0,1] = K[1,0] = -1

            => K = [[2, -1],
                    [-1,  2]]
        """
        # MATLAB ref: K = [[2, -1], [-1, 2]]
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        K = chain.K.toarray()
        expected_K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        assert_allclose(K, expected_K, atol=1e-15)

    def test_chain_2dof_equal_masses_equal_springs_M_canonical(self) -> None:
        """Canonical 2-DOF chain: m=[1,1] → M = diag([1, 1]) = I_2.

        MATLAB ref: M = [[1, 0], [0, 1]]
        """
        # MATLAB ref: M = [[1, 0], [0, 1]]
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        M = chain.M.toarray()
        expected_M = np.eye(2)
        assert_allclose(M, expected_M, atol=1e-15)

    def test_chain_2dof_equal_masses_equal_springs_D_zero(self) -> None:
        """Canonical 2-DOF chain with zero damping: D = 0.

        MATLAB ref: D = [[0, 0], [0, 0]]
        """
        # MATLAB ref: D = zeros(2,2)
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        D = chain.D.toarray()
        assert_allclose(D, np.zeros((2, 2)), atol=1e-15)

    def test_chain_2dof_natural_frequencies_canonical(self) -> None:
        """Natural frequencies of canonical 2-DOF equal-mass chain.

        For M = I, K = [[2,-1],[-1,2]], eigenvalue problem K*phi = lambda*M*phi:
            det(K - lambda*I) = (2-lambda)^2 - 1 = 0
            => lambda = 1  (omega_1 = 1)  and  lambda = 3  (omega_2 = sqrt(3))

        MATLAB ref: eig(K, M) = [1, 3]  =>  omega = [1.0, sqrt(3)]

        Since M=I for this canonical case, np.linalg.eigvalsh(K) gives the same result.
        """
        # MATLAB ref: natural frequencies omega = [1.0, sqrt(3) ≈ 1.7321]
        chain = ChainOfOscillators(
            masses=[1.0, 1.0],
            stiffnesses=[1.0, 1.0, 1.0],
            dampings=[0.0, 0.0, 0.0],
        )
        K = chain.K.toarray()
        # M = I for this canonical case, so eigvalsh(K) gives eigenvalues directly
        eigenvalues = np.linalg.eigvalsh(K)
        omega = np.sqrt(np.sort(eigenvalues))
        assert_allclose(omega[0], 1.0, atol=1e-12)
        assert_allclose(omega[1], np.sqrt(3.0), atol=1e-12)
