"""
Unit tests for FE_EulerBernoulliBeam (T-08).

Validation strategy
-------------------
Analytical eigenfrequencies for a clamped-free (cantilever) Euler-Bernoulli
beam are given by

.. math::

    \\omega_n = (\\beta_n L)^2 \\sqrt{\\frac{EI}{\\rho A L^4}}

where the frequency parameters for the first three modes are

.. math::

    \\beta_1 L = 1.8751, \\quad \\beta_2 L = 4.6941, \\quad \\beta_3 L = 7.8548

Reference: Petyt (2010) *Introduction to Finite Element Vibration Analysis*,
Table 3.1 (also Leissa, 1969).

The FE model should reproduce these values within 1 % for n ≥ 10 elements
(standard convergence result for the Bernoulli-Euler element).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from nlvib.nonlinearities.elements import cubic_spring
from nlvib.systems.fe_beam import FE_EulerBernoulliBeam, build_beam_matrices

# ---------------------------------------------------------------------------
# Test-fixture parameters (steel-like beam)
# ---------------------------------------------------------------------------

# Material / geometry — chosen so that EI/rhoAL^4 is in a convenient range
_E = 210e9        # Young's modulus  [Pa]
_I = 1e-8         # Second moment    [m^4]
_RHO = 7800.0     # Density          [kg/m^3]
_A = 1e-4         # Cross-section    [m^2]
_L = 1.0          # Total length     [m]
_N_ELEM = 10      # Number of elements

# Analytical beta_n * L values for first 3 cantilever modes
_BETA_L = np.array([1.8751, 4.6941, 7.8548])

# Tolerance: first 3 FE eigenfrequencies within 1 % of analytical
_TOL_RELATIVE = 0.01

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analytical_omega(beta_L: np.ndarray, E: float, I_xx: float, rho: float,
                       A: float, L: float) -> np.ndarray:
    """Compute analytical cantilever eigenfrequencies [rad/s]."""
    return beta_L**2 * np.sqrt(E * I_xx / (rho * A * L**4))


def _fe_eigenfreqs(beam: FE_EulerBernoulliBeam, n_modes: int) -> np.ndarray:
    """Solve the free-vibration eigenvalue problem for *beam*.

    Uses ``scipy.linalg.eigh`` on the dense reduced matrices (safe for the
    small test sizes here).
    """
    K_dense = beam.K.toarray()
    M_dense = beam.M.toarray()
    eigvals = la.eigh(K_dense, M_dense, eigvals_only=True)
    # eigvals are omega^2; take positive values (eigh returns sorted ascending)
    pos_eigvals = eigvals[eigvals > 0.0]
    return np.sqrt(pos_eigvals[:n_modes])


# ---------------------------------------------------------------------------
# T-08 acceptance test: eigenfrequency accuracy
# ---------------------------------------------------------------------------


class TestCantileverEigenfrequencies:
    """Free-vibration eigenfrequency test for a clamped-free beam."""

    def setup_method(self) -> None:
        self.beam = FE_EulerBernoulliBeam(
            n_elements=_N_ELEM,
            L=_L,
            E=_E,
            I_area=_I,
            rho=_RHO,
            A=_A,
            bc="clamped-free",
        )
        self.omega_analytical = _analytical_omega(_BETA_L, _E, _I, _RHO, _A, _L)
        self.omega_fe = _fe_eigenfreqs(self.beam, n_modes=3)

    def test_first_three_modes_within_tolerance(self) -> None:
        """FE modes 1–3 are within 1 % of analytical cantilever values."""
        rel_errors = np.abs(self.omega_fe - self.omega_analytical) / self.omega_analytical
        np.testing.assert_array_less(
            rel_errors,
            _TOL_RELATIVE * np.ones(3),
            err_msg=(
                f"Relative eigenfrequency errors {rel_errors} exceed "
                f"tolerance {_TOL_RELATIVE}."
            ),
        )

    def test_mode_1_error(self) -> None:
        """First mode relative error < 1 %."""
        err = abs(self.omega_fe[0] - self.omega_analytical[0]) / self.omega_analytical[0]
        assert err < _TOL_RELATIVE, (
            f"Mode 1: FE={self.omega_fe[0]:.4f}, analytical={self.omega_analytical[0]:.4f}, "
            f"rel_error={err:.4e}"
        )

    def test_mode_2_error(self) -> None:
        """Second mode relative error < 1 %."""
        err = abs(self.omega_fe[1] - self.omega_analytical[1]) / self.omega_analytical[1]
        assert err < _TOL_RELATIVE, (
            f"Mode 2: FE={self.omega_fe[1]:.4f}, analytical={self.omega_analytical[1]:.4f}, "
            f"rel_error={err:.4e}"
        )

    def test_mode_3_error(self) -> None:
        """Third mode relative error < 1 %."""
        err = abs(self.omega_fe[2] - self.omega_analytical[2]) / self.omega_analytical[2]
        assert err < _TOL_RELATIVE, (
            f"Mode 3: FE={self.omega_fe[2]:.4f}, analytical={self.omega_analytical[2]:.4f}, "
            f"rel_error={err:.4e}"
        )


# ---------------------------------------------------------------------------
# Construction and shape tests
# ---------------------------------------------------------------------------


class TestBeamConstruction:
    """Tests for matrix shapes, DOF counts, and BC application."""

    def test_clamped_free_dof_count(self) -> None:
        """Clamped-free beam: n_dof = 2*(n_elements+1) - 2 (remove 2 at left)."""
        n = 10
        beam = FE_EulerBernoulliBeam(n, _L, _E, _I, _RHO, _A, "clamped-free")
        expected_dof = 2 * (n + 1) - 2  # 2 constrained DOFs at node 0
        assert beam.n_dof == expected_dof

    def test_clamped_clamped_dof_count(self) -> None:
        """Clamped-clamped beam: 4 constrained DOFs total."""
        n = 10
        beam = FE_EulerBernoulliBeam(n, _L, _E, _I, _RHO, _A, "clamped-clamped")
        expected_dof = 2 * (n + 1) - 4
        assert beam.n_dof == expected_dof

    def test_free_free_dof_count(self) -> None:
        """Free-free beam: all DOFs retained."""
        n = 6
        beam = FE_EulerBernoulliBeam(n, _L, _E, _I, _RHO, _A, "free-free")
        expected_dof = 2 * (n + 1)
        assert beam.n_dof == expected_dof

    def test_matrix_shapes_are_square(self) -> None:
        """M and K are square with shape (n_dof, n_dof)."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        n = beam.n_dof
        assert beam.M.shape == (n, n)
        assert beam.K.shape == (n, n)

    def test_matrices_are_csr(self) -> None:
        """M and K are returned as scipy.sparse.csr_matrix."""
        import scipy.sparse as sp

        beam = FE_EulerBernoulliBeam(4, _L, _E, _I, _RHO, _A, "clamped-free")
        assert sp.issparse(beam.M)
        assert sp.issparse(beam.K)

    def test_mass_matrix_positive_definite(self) -> None:
        """Reduced mass matrix eigenvalues are all positive."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        eigvals = la.eigvalsh(beam.M.toarray())
        assert np.all(eigvals > 0.0), f"Mass matrix not positive definite: {eigvals}"

    def test_stiffness_matrix_positive_definite(self) -> None:
        """Reduced stiffness matrix eigenvalues are all positive (clamped-free)."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        eigvals = la.eigvalsh(beam.K.toarray())
        assert np.all(eigvals > 0.0), f"Stiffness matrix not positive definite: {eigvals}"

    def test_stiffness_matrix_symmetric(self) -> None:
        """Stiffness matrix is symmetric."""
        beam = FE_EulerBernoulliBeam(4, _L, _E, _I, _RHO, _A, "clamped-clamped")
        K = beam.K.toarray()
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_mass_matrix_symmetric(self) -> None:
        """Mass matrix is symmetric."""
        beam = FE_EulerBernoulliBeam(4, _L, _E, _I, _RHO, _A, "clamped-clamped")
        M = beam.M.toarray()
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_invalid_bc_raises(self) -> None:
        """Unsupported BC string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported boundary condition"):
            FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "pinned-free")

    def test_zero_elements_raises(self) -> None:
        """n_elements=0 raises ValueError."""
        with pytest.raises(ValueError):
            FE_EulerBernoulliBeam(0, _L, _E, _I, _RHO, _A, "clamped-free")


# ---------------------------------------------------------------------------
# find_dof tests
# ---------------------------------------------------------------------------


class TestFindDof:
    """Tests for the find_dof() DOF-lookup method."""

    def setup_method(self) -> None:
        self.beam = FE_EulerBernoulliBeam(
            5, _L, _E, _I, _RHO, _A, "clamped-free"
        )

    def test_tip_w_dof_is_valid(self) -> None:
        """Tip transverse displacement DOF exists in reduced system."""
        idx = self.beam.find_dof(5, "w")
        assert 0 <= idx < self.beam.n_dof

    def test_tip_theta_dof_is_valid(self) -> None:
        """Tip rotation DOF exists in reduced system."""
        idx = self.beam.find_dof(5, "theta")
        assert 0 <= idx < self.beam.n_dof

    def test_root_w_constrained_raises(self) -> None:
        """Root w DOF is constrained in clamped-free beam — raises ValueError."""
        with pytest.raises(ValueError, match="constrained"):
            self.beam.find_dof(0, "w")

    def test_root_theta_constrained_raises(self) -> None:
        """Root theta DOF is constrained in clamped-free beam — raises ValueError."""
        with pytest.raises(ValueError, match="constrained"):
            self.beam.find_dof(0, "theta")

    def test_invalid_dof_type_raises(self) -> None:
        """Unknown dof_type raises ValueError."""
        with pytest.raises(ValueError, match="dof_type must be"):
            self.beam.find_dof(3, "v")

    def test_out_of_range_node_raises(self) -> None:
        """Out-of-range node index raises ValueError."""
        with pytest.raises(ValueError, match="node_index must be"):
            self.beam.find_dof(100, "w")

    def test_unique_dof_indices(self) -> None:
        """All free-node DOF indices are distinct."""
        n_nodes = 6  # 5 elements → 6 nodes; node 0 is clamped
        indices = []
        for node in range(1, n_nodes):
            for dof in ("w", "theta"):
                indices.append(self.beam.find_dof(node, dof))
        assert len(set(indices)) == len(indices), "Duplicate DOF indices returned."


# ---------------------------------------------------------------------------
# add_forcing tests
# ---------------------------------------------------------------------------


class TestAddForcing:
    """Tests for the add_forcing() method."""

    def test_forcing_registered(self) -> None:
        """add_forcing appends to the forcing list."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        beam.add_forcing(5, "w", amplitude=100.0)
        assert len(beam.forcing) == 1
        dof_idx, amp = beam.forcing[0]
        assert amp == pytest.approx(100.0)
        assert 0 <= dof_idx < beam.n_dof

    def test_multiple_forcing_locations(self) -> None:
        """Multiple forcing locations are all stored."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        beam.add_forcing(3, "w", amplitude=50.0)
        beam.add_forcing(5, "theta", amplitude=10.0)
        assert len(beam.forcing) == 2

    def test_forcing_on_constrained_dof_raises(self) -> None:
        """Forcing on a constrained DOF raises ValueError."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        with pytest.raises(ValueError):
            beam.add_forcing(0, "w", amplitude=1.0)


# ---------------------------------------------------------------------------
# add_nonlinear_attachment tests
# ---------------------------------------------------------------------------


class TestAddNonlinearAttachment:
    """Tests for the add_nonlinear_attachment() method."""

    def test_nonlinear_element_registered(self) -> None:
        """add_nonlinear_attachment registers element in base class registry."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        tip_dof = beam.find_dof(5, "w")
        spring = cubic_spring(1e6, tip_dof)
        beam.add_nonlinear_attachment(5, "w", spring)
        assert len(beam.nonlinear_elements) == 1

    def test_attachment_on_constrained_raises(self) -> None:
        """Attaching to a constrained DOF raises ValueError."""
        beam = FE_EulerBernoulliBeam(5, _L, _E, _I, _RHO, _A, "clamped-free")
        tip_dof = beam.find_dof(5, "w")
        spring = cubic_spring(1e6, tip_dof)
        with pytest.raises(ValueError):
            beam.add_nonlinear_attachment(0, "w", spring)


# ---------------------------------------------------------------------------
# build_beam_matrices utility function
# ---------------------------------------------------------------------------


class TestBuildBeamMatrices:
    """Tests for the build_beam_matrices() convenience function."""

    def test_shapes(self) -> None:
        """Full matrices have shape (2*(n+1), 2*(n+1))."""
        n = 4
        K, M = build_beam_matrices(n, _L, _E, _I, _RHO, _A)
        expected = 2 * (n + 1)
        assert K.shape == (expected, expected)
        assert M.shape == (expected, expected)

    def test_returns_sparse(self) -> None:
        """build_beam_matrices returns sparse matrices."""
        import scipy.sparse as sp

        K, M = build_beam_matrices(4, _L, _E, _I, _RHO, _A)
        assert sp.issparse(K)
        assert sp.issparse(M)

    def test_symmetric(self) -> None:
        """Full K and M are symmetric."""
        K, M = build_beam_matrices(4, _L, _E, _I, _RHO, _A)
        Kd = K.toarray()
        Md = M.toarray()
        np.testing.assert_allclose(Kd, Kd.T, atol=1e-12)
        np.testing.assert_allclose(Md, Md.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Repr test
# ---------------------------------------------------------------------------


def test_repr() -> None:
    """__repr__ includes n_elements, L, bc, and n_dof."""
    beam = FE_EulerBernoulliBeam(5, 2.0, _E, _I, _RHO, _A, "clamped-free")
    r = repr(beam)
    assert "FE_EulerBernoulliBeam" in r
    assert "n_elements=5" in r
    assert "clamped-free" in r
