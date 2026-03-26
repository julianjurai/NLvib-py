"""
Unit tests for :class:`~nlvib.systems.fe_rod.FE_ElasticRod`.

Eigenfrequency acceptance criterion
-------------------------------------
For a uniform clamped-free axial rod the exact analytical natural frequencies
are (Cook et al., §2; Rao, *Mechanical Vibrations*, §8):

.. math::

    \\omega_n = \\frac{(2n-1)\\pi}{2} \\cdot \\frac{c}{L},
    \\qquad c = \\sqrt{E/\\rho}, \\quad n = 1, 2, 3, \\ldots

The FEM eigenfrequencies are compared against these analytical values.
The spec requires the first 3 values to be within 1 % for n_elements ≥ 5.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from nlvib.systems.fe_rod import FE_ElasticRod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eigen_frequencies(rod: FE_ElasticRod) -> np.ndarray:  # type: ignore[type-arg]
    """Solve the undamped free-vibration problem and return sorted ω values."""
    K = rod.K.toarray()
    M = rod.M.toarray()
    # Use scipy.linalg.eigh for symmetric positive-(semi)definite generalised
    # eigenvalue problem K x = λ M x  →  λ = ω²
    eigenvalues, _ = la.eigh(K, M)
    # eigh returns eigenvalues in ascending order; guard against numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return np.sqrt(eigenvalues)


def _analytical_omega(n: int, E: float, rho: float, L: float) -> float:
    """Return the n-th exact angular eigenfrequency of a clamped-free rod."""
    c = np.sqrt(E / rho)
    return (2 * n - 1) * np.pi / 2.0 * c / L


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Material / geometric parameters for a generic steel-like rod.
# Construction/validation tests use a coarse 5-element mesh.
# Eigenfrequency accuracy tests use a 20-element mesh: standard bar-element
# theory requires roughly n_elements ≥ 2k elements to achieve <1 % error on
# mode k (consistent mass matrix, clamped-free rod).  With n=5 elements,
# mode-1 error is ~0.4 % (< 1 %) but mode-2 error is ~3.7 % and mode-3 is
# ~10 %, so only n ≥ 20 satisfies <1 % for all of modes 1–3.  The spec
# requirement "n ≥ 5 elements" is a minimum for the model to be valid; the
# 1 % tolerance on modes 1–3 is met here with n = 20.
_N_ELEM = 5        # used for construction/validation tests
_N_ELEM_EIG = 20   # used for eigenfrequency accuracy tests
_L = 1.0           # m
_E = 210e9         # Pa
_A = 1e-4          # m²
_RHO = 7800.0      # kg/m³


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_n_dof_clamped_free(self) -> None:
        """Clamped-free: n_elements+1 nodes − 1 constrained = n_elements DOFs."""
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        assert rod.n_dof == _N_ELEM

    def test_n_dof_clamped_clamped(self) -> None:
        """Clamped-clamped: n_elements+1 nodes − 2 constrained = n_elements−1 DOFs."""
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-clamped")
        assert rod.n_dof == _N_ELEM - 1

    def test_n_dof_free_free(self) -> None:
        """Free-free: all nodes free → n_elements+1 DOFs."""
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "free-free")
        assert rod.n_dof == _N_ELEM + 1

    def test_n_dof_free_clamped(self) -> None:
        """Free-clamped: n_elements+1 nodes − 1 constrained = n_elements DOFs."""
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "free-clamped")
        assert rod.n_dof == _N_ELEM

    def test_matrix_shapes(self) -> None:
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        n = rod.n_dof
        assert rod.M.shape == (n, n)
        assert rod.K.shape == (n, n)
        assert rod.D.shape == (n, n)

    def test_matrices_are_sparse(self) -> None:
        import scipy.sparse as sp
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        assert sp.issparse(rod.M)
        assert sp.issparse(rod.K)

    def test_mass_matrix_is_symmetric_positive_definite(self) -> None:
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        M = rod.M.toarray()
        assert np.allclose(M, M.T, atol=1e-12), "M is not symmetric"
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"M has non-positive eigenvalues: {eigvals}"

    def test_stiffness_matrix_is_symmetric_positive_definite(self) -> None:
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        K = rod.K.toarray()
        assert np.allclose(K, K.T, atol=1e-12), "K is not symmetric"
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0), f"K has non-positive eigenvalues: {eigvals}"

    def test_total_mass(self) -> None:
        """Sum of all entries of consistent M equals total rod mass × 2 (row sums = ρAL)."""
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "free-free")
        M = rod.M.toarray()
        expected_total_mass = _RHO * _A * _L
        # For a consistent mass matrix the row sums equal the lumped mass per node;
        # the total is ρAL.
        assert abs(M.sum() - expected_total_mass) / expected_total_mass < 1e-10

    def test_damping_matrix_is_zero(self) -> None:
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        assert rod.D.nnz == 0

    def test_metadata_attributes(self) -> None:
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        assert rod.n_elements == _N_ELEM
        assert rod.L == _L
        assert rod.E == _E
        assert rod.A == _A
        assert rod.rho == _RHO
        assert rod.bc == "clamped-free"
        assert np.isclose(rod.Le, _L / _N_ELEM)

    def test_repr(self) -> None:
        rod = FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "clamped-free")
        r = repr(rod)
        assert "FE_ElasticRod" in r
        assert "clamped-free" in r


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_n_elements(self) -> None:
        with pytest.raises(ValueError, match="n_elements"):
            FE_ElasticRod(0, _L, _E, _A, _RHO, "clamped-free")

    def test_invalid_L(self) -> None:
        with pytest.raises(ValueError, match="L"):
            FE_ElasticRod(_N_ELEM, 0.0, _E, _A, _RHO, "clamped-free")

    def test_invalid_E(self) -> None:
        with pytest.raises(ValueError, match="E"):
            FE_ElasticRod(_N_ELEM, _L, -1.0, _A, _RHO, "clamped-free")

    def test_invalid_A(self) -> None:
        with pytest.raises(ValueError, match="A"):
            FE_ElasticRod(_N_ELEM, _L, _E, 0.0, _RHO, "clamped-free")

    def test_invalid_rho(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            FE_ElasticRod(_N_ELEM, _L, _E, _A, 0.0, "clamped-free")

    def test_invalid_bc(self) -> None:
        with pytest.raises(ValueError, match="bc"):
            FE_ElasticRod(_N_ELEM, _L, _E, _A, _RHO, "pinned-free")


# ---------------------------------------------------------------------------
# Eigenfrequency accuracy tests  (primary acceptance criterion)
# ---------------------------------------------------------------------------

class TestEigenfrequencies:
    """First 3 eigenfrequencies of clamped-free rod vs. analytical solution.

    Analytical formula (Cook et al., §2):
        ω_n = (2n−1) π/2 · √(E/ρ) / L,  n = 1, 2, 3, …

    Tolerance: each FEM frequency within 1 % of analytical (spec requirement).
    Uses n_elements = 20 (satisfies n ≥ 5 spec; needed for <1 % on modes 2–3).
    """

    def setup_method(self) -> None:
        self.rod = FE_ElasticRod(_N_ELEM_EIG, _L, _E, _A, _RHO, "clamped-free")
        self.fem_omegas = _eigen_frequencies(self.rod)

    @pytest.mark.parametrize("mode_n", [1, 2, 3])
    def test_eigenfrequency_within_1_percent(self, mode_n: int) -> None:
        """FEM ω_n must be within 1 % of analytical value."""
        omega_analytical = _analytical_omega(mode_n, _E, _RHO, _L)
        omega_fem = self.fem_omegas[mode_n - 1]  # 0-indexed
        rel_error = abs(omega_fem - omega_analytical) / omega_analytical
        assert rel_error < 0.01, (
            f"Mode {mode_n}: FEM ω = {omega_fem:.4f} rad/s, "
            f"analytical ω = {omega_analytical:.4f} rad/s, "
            f"relative error = {rel_error:.4%} (> 1 %)"
        )

    def test_all_eigenfrequencies_positive(self) -> None:
        """All computed eigenfrequencies must be positive (clamped-free has no rigid modes)."""
        assert np.all(self.fem_omegas > 0.0)

    def test_eigenfrequencies_ascending(self) -> None:
        """Eigenfrequencies must be in ascending order."""
        assert np.all(np.diff(self.fem_omegas) >= 0.0)

    def test_eigenfrequency_errors_report(self) -> None:
        """Print eigenfrequency table for human inspection (never fails)."""
        print(f"\nEigenfrequency comparison (clamped-free rod, n_elements={_N_ELEM_EIG}):")
        print(f"{'Mode':>5}  {'Analytical (rad/s)':>20}  {'FEM (rad/s)':>15}  {'Rel error':>10}")
        for n in range(1, 4):
            omega_a = _analytical_omega(n, _E, _RHO, _L)
            omega_f = self.fem_omegas[n - 1]
            rel_err = abs(omega_f - omega_a) / omega_a
            print(f"{n:>5}  {omega_a:>20.4f}  {omega_f:>15.4f}  {rel_err:>10.4%}")


# ---------------------------------------------------------------------------
# Convergence test: finer mesh → smaller error
# ---------------------------------------------------------------------------

class TestMeshConvergence:
    """Error decreases as mesh is refined."""

    def _rel_error_mode1(self, n_elem: int) -> float:
        rod = FE_ElasticRod(n_elem, _L, _E, _A, _RHO, "clamped-free")
        omega_fem = _eigen_frequencies(rod)[0]
        omega_a = _analytical_omega(1, _E, _RHO, _L)
        return float(abs(omega_fem - omega_a) / omega_a)

    def test_error_decreases_with_refinement(self) -> None:
        err_coarse = self._rel_error_mode1(5)
        err_fine = self._rel_error_mode1(20)
        assert err_fine < err_coarse, (
            f"Finer mesh should give smaller error: "
            f"n=5 err={err_coarse:.4%}, n=20 err={err_fine:.4%}"
        )
