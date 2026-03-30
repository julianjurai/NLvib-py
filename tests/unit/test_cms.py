"""
Unit tests for CMS model reduction (T-11).

Tests both Craig-Bampton (fixed-interface) and Rubin (free-interface)
reductions applied to a 10-element clamped-free Euler-Bernoulli beam.

Acceptance criteria (from TASKS.md T-11):
- Reduced model has correct DOF count: n_boundary + n_internal_modes.
- First 3 eigenfrequencies of the reduced model match the full model
  within 1 %.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la
from numpy.testing import assert_allclose

from nlvib.systems.cms import craig_bampton, rubin
from nlvib.systems.fe_beam import FE_EulerBernoulliBeam

# ---------------------------------------------------------------------------
# Shared fixture: 10-element clamped-free steel beam
# ---------------------------------------------------------------------------

# Steel properties
_E = 2.0e11       # Young's modulus  [Pa]
_I_AREA = 1.0e-5  # Second moment of area  [m^4]
_RHO = 7_800.0    # Density  [kg/m^3]
_A = 5.0e-4       # Cross-sectional area  [m^2]
_L = 1.0          # Total length  [m]
_N_ELEM = 10

# Number of CMS modes to keep in tests
_N_INTERNAL_CB = 3   # Craig-Bampton fixed-interface modes
_N_MODES_RUBIN = 3   # Rubin free-interface modes retained

# How many full-model eigenfrequencies to compare
_N_COMPARE = 3

# 1 % relative tolerance on eigenfrequency comparison
_FREQ_RTOL = 0.01


@pytest.fixture(scope="module")
def beam() -> FE_EulerBernoulliBeam:
    """10-element clamped-free Euler-Bernoulli beam."""
    return FE_EulerBernoulliBeam(
        n_elements=_N_ELEM,
        L=_L,
        E=_E,
        I_area=_I_AREA,
        rho=_RHO,
        A=_A,
        bc="clamped-free",
    )


@pytest.fixture(scope="module")
def boundary_dofs(beam: FE_EulerBernoulliBeam) -> list[int]:
    """Boundary DOFs: w and theta of the free (last) node."""
    last_node = beam.n_beam_elements  # 0-indexed: node 10
    w_dof = beam.find_dof(last_node, "w")
    theta_dof = beam.find_dof(last_node, "theta")
    return [w_dof, theta_dof]


@pytest.fixture(scope="module")
def full_eigenfreqs(beam: FE_EulerBernoulliBeam) -> np.ndarray:
    """First _N_COMPARE eigenfrequencies of the full beam [rad/s]."""
    K_dense = np.asarray(beam.K.todense(), dtype=np.float64)
    M_dense = np.asarray(beam.M.todense(), dtype=np.float64)
    vals = la.eigh(K_dense, M_dense, eigvals_only=True,
                   subset_by_index=[0, _N_COMPARE - 1])
    return np.sqrt(np.maximum(vals, 0.0))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _eig_freqs(system: object, n: int) -> np.ndarray:  # type: ignore[type-arg]
    """Compute first n eigenfrequencies [rad/s] of a MechanicalSystem."""
    from nlvib.systems.base import MechanicalSystem
    assert isinstance(system, MechanicalSystem)
    K_d = np.asarray(system.K.todense(), dtype=np.float64)
    M_d = np.asarray(system.M.todense(), dtype=np.float64)
    vals = la.eigh(K_d, M_d, eigvals_only=True,
                   subset_by_index=[0, n - 1])
    return np.sqrt(np.maximum(vals, 0.0))


# ===========================================================================
# Craig-Bampton tests
# ===========================================================================

class TestCraigBampton:
    """Tests for craig_bampton()."""

    def test_reduced_dof_count(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Reduced model must have n_b + n_internal_modes DOFs."""
        n_b = len(boundary_dofs)
        reduced, _T = craig_bampton(beam, boundary_dofs, _N_INTERNAL_CB)
        expected_dof = n_b + _N_INTERNAL_CB
        assert reduced.n_dof == expected_dof, (
            f"Expected n_dof={expected_dof}, got {reduced.n_dof}"
        )

    def test_transformation_matrix_shape(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """T must have shape (n_full, n_b + n_internal_modes)."""
        reduced, T = craig_bampton(beam, boundary_dofs, _N_INTERNAL_CB)
        assert T.shape == (beam.n_dof, reduced.n_dof), (
            f"T.shape expected {(beam.n_dof, reduced.n_dof)}, got {T.shape}"
        )

    def test_transformation_boundary_identity(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Top-left block of T (boundary rows, boundary cols) must be identity."""
        n_b = len(boundary_dofs)
        _reduced, T = craig_bampton(beam, boundary_dofs, _N_INTERNAL_CB)
        T_bb = T[np.array(boundary_dofs, dtype=np.intp), :n_b]
        assert_allclose(T_bb, np.eye(n_b), atol=1e-12,
                        err_msg="T[b, :n_b] is not identity")

    def test_mass_matrix_symmetry(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Reduced mass matrix must be symmetric."""
        reduced, _T = craig_bampton(beam, boundary_dofs, _N_INTERNAL_CB)
        M_r = np.asarray(reduced.M.todense())
        assert_allclose(M_r, M_r.T, atol=1e-10,
                        err_msg="Reduced mass matrix is not symmetric")

    def test_stiffness_matrix_symmetry(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Reduced stiffness matrix must be symmetric."""
        reduced, _T = craig_bampton(beam, boundary_dofs, _N_INTERNAL_CB)
        K_r = np.asarray(reduced.K.todense())
        assert_allclose(K_r, K_r.T, atol=1e-10,
                        err_msg="Reduced stiffness matrix is not symmetric")

    def test_eigenfrequencies_match_full_model(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
        full_eigenfreqs: np.ndarray,
    ) -> None:
        """First _N_COMPARE eigenfrequencies must match full model within 1 %."""
        reduced, _T = craig_bampton(beam, boundary_dofs, _N_INTERNAL_CB)
        red_freqs = _eig_freqs(reduced, _N_COMPARE)

        rel_errors = np.abs(red_freqs - full_eigenfreqs) / full_eigenfreqs
        for k, (rf, ff, re) in enumerate(
            zip(red_freqs, full_eigenfreqs, rel_errors)
        ):
            assert re < _FREQ_RTOL, (
                f"Mode {k+1}: reduced={rf:.4f} rad/s, "
                f"full={ff:.4f} rad/s, rel_error={re:.2e} > {_FREQ_RTOL}"
            )

    def test_invalid_n_internal_modes_zero(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """n_internal_modes=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_internal_modes"):
            craig_bampton(beam, boundary_dofs, 0)

    def test_invalid_boundary_dofs_empty(
        self,
        beam: FE_EulerBernoulliBeam,
    ) -> None:
        """Empty boundary_dofs must raise ValueError."""
        with pytest.raises(ValueError):
            craig_bampton(beam, [], _N_INTERNAL_CB)

    def test_invalid_boundary_dofs_out_of_range(
        self,
        beam: FE_EulerBernoulliBeam,
    ) -> None:
        """Out-of-range boundary DOF must raise ValueError."""
        with pytest.raises(ValueError):
            craig_bampton(beam, [beam.n_dof + 1], _N_INTERNAL_CB)


# ===========================================================================
# Rubin tests
# ===========================================================================

class TestRubin:
    """Tests for rubin()."""

    def test_reduced_dof_count(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Reduced model must have n_modes + n_b DOFs."""
        n_b = len(boundary_dofs)
        reduced, _T = rubin(beam, boundary_dofs, _N_MODES_RUBIN)
        expected_dof = _N_MODES_RUBIN + n_b
        assert reduced.n_dof == expected_dof, (
            f"Expected n_dof={expected_dof}, got {reduced.n_dof}"
        )

    def test_transformation_matrix_shape(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """T must have shape (n_full, n_modes + n_b)."""
        reduced, T = rubin(beam, boundary_dofs, _N_MODES_RUBIN)
        assert T.shape == (beam.n_dof, reduced.n_dof), (
            f"T.shape expected {(beam.n_dof, reduced.n_dof)}, got {T.shape}"
        )

    def test_mass_matrix_symmetry(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Reduced mass matrix must be symmetric."""
        reduced, _T = rubin(beam, boundary_dofs, _N_MODES_RUBIN)
        M_r = np.asarray(reduced.M.todense())
        assert_allclose(M_r, M_r.T, atol=1e-10,
                        err_msg="Rubin reduced mass matrix is not symmetric")

    def test_stiffness_matrix_symmetry(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """Reduced stiffness matrix must be symmetric."""
        reduced, _T = rubin(beam, boundary_dofs, _N_MODES_RUBIN)
        K_r = np.asarray(reduced.K.todense())
        assert_allclose(K_r, K_r.T, atol=1e-10,
                        err_msg="Rubin reduced stiffness matrix is not symmetric")

    def test_eigenfrequencies_match_full_model(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
        full_eigenfreqs: np.ndarray,
    ) -> None:
        """First _N_COMPARE eigenfrequencies must match full model within 1 %."""
        reduced, _T = rubin(beam, boundary_dofs, _N_MODES_RUBIN)
        red_freqs = _eig_freqs(reduced, _N_COMPARE)

        rel_errors = np.abs(red_freqs - full_eigenfreqs) / full_eigenfreqs
        for k, (rf, ff, re) in enumerate(
            zip(red_freqs, full_eigenfreqs, rel_errors)
        ):
            assert re < _FREQ_RTOL, (
                f"Mode {k+1}: reduced={rf:.4f} rad/s, "
                f"full={ff:.4f} rad/s, rel_error={re:.2e} > {_FREQ_RTOL}"
            )

    def test_invalid_n_modes_zero(
        self,
        beam: FE_EulerBernoulliBeam,
        boundary_dofs: list[int],
    ) -> None:
        """n_modes=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_modes"):
            rubin(beam, boundary_dofs, 0)

    def test_invalid_boundary_dofs_empty(
        self,
        beam: FE_EulerBernoulliBeam,
    ) -> None:
        """Empty boundary_dofs must raise ValueError."""
        with pytest.raises(ValueError):
            rubin(beam, [], _N_MODES_RUBIN)
