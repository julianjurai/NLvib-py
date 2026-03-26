"""
Component Mode Synthesis (CMS) model reduction for NLvib.

Provides two Craig-Bampton and Rubin (free-interface) reduction methods that
take a full-order :class:`~nlvib.systems.base.MechanicalSystem` and return a
reduced-order :class:`~nlvib.systems.base.MechanicalSystem` together with the
transformation matrix **T** that maps reduced DOFs to full DOFs.

Theory
------
Both methods partition the full DOF set into *boundary* (b) and *internal* (i)
subsets and build a Ritz basis for the internal DOFs:

Craig-Bampton (1968)
    Uses *constraint modes* (static response to unit boundary displacements)
    and *fixed-interface normal modes* (eigenmodes with boundary DOFs clamped).

Rubin (1975)
    Uses *free-interface normal modes* and *residual flexibility attachment
    modes* to correct for the truncated high-frequency content.

Equation references
-------------------
- Craig, R. R. & Bampton, M. C. C. (1968).  Coupling of substructures for
  dynamic analysis.  *AIAA Journal*, 6(7), 1313-1319.
- Rubin, S. (1975).  Improved component-mode representation for structural
  dynamic analysis.  *AIAA Journal*, 13(8), 995-1006.
- de Klerk, D., Rixen, D. J. & Voormeeren, S. N. (2008).  General framework
  for dynamic substructuring.  *AIAA Journal*, 46(5), 1169-1181.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from nlvib.systems.base import MechanicalSystem

__all__ = ["craig_bampton", "rubin"]

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _partition_indices(
    n_dof: int,
    boundary_dofs: list[int] | NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split DOF indices into boundary (b) and internal (i) sets.

    Parameters
    ----------
    n_dof:
        Total number of DOFs in the full-order system.
    boundary_dofs:
        Zero-based indices of the boundary (interface) DOFs.

    Returns
    -------
    b_idx : ndarray of int, shape (n_b,)
        Sorted boundary DOF indices.
    i_idx : ndarray of int, shape (n_i,)
        Sorted internal DOF indices.

    Raises
    ------
    ValueError
        If any boundary DOF is out of range or there are duplicate indices.
    """
    b_arr = np.asarray(boundary_dofs, dtype=np.intp)

    if b_arr.ndim != 1:
        raise ValueError("boundary_dofs must be a 1-D sequence.")

    if b_arr.size == 0:
        raise ValueError("boundary_dofs must contain at least one DOF index.")

    if np.any(b_arr < 0) or np.any(b_arr >= n_dof):
        raise ValueError(
            f"All boundary_dofs must be in [0, {n_dof - 1}]; "
            f"got min={b_arr.min()}, max={b_arr.max()}."
        )

    if b_arr.size != np.unique(b_arr).size:
        raise ValueError("boundary_dofs must not contain duplicate indices.")

    b_idx = np.sort(b_arr)
    all_dofs = np.arange(n_dof, dtype=np.intp)
    i_idx: NDArray[np.intp] = np.setdiff1d(all_dofs, b_idx).astype(np.intp)
    return b_idx, i_idx


def _extract_submatrices(
    A: csr_matrix,
    row_idx: NDArray[np.intp],
    col_idx: NDArray[np.intp],
) -> FloatArray:
    """Extract a dense submatrix A[row_idx, :][:, col_idx].

    Parameters
    ----------
    A:
        Source sparse matrix.
    row_idx:
        Row indices to extract.
    col_idx:
        Column indices to extract.

    Returns
    -------
    ndarray, shape (len(row_idx), len(col_idx))
        Dense submatrix.
    """
    return np.asarray(A[np.ix_(row_idx, col_idx)].todense(), dtype=np.float64)


# ---------------------------------------------------------------------------
# Craig-Bampton reduction
# ---------------------------------------------------------------------------


def craig_bampton(
    system: MechanicalSystem,
    boundary_dofs: list[int] | NDArray[np.intp],
    n_internal_modes: int,
) -> tuple[MechanicalSystem, FloatArray]:
    """Craig-Bampton (fixed-interface) component mode synthesis reduction.

    Partitions the full-order system's DOFs into *boundary* (b) and
    *internal* (i) subsets and builds the Craig-Bampton Ritz basis:

    .. math::

        T = \\begin{bmatrix} I_b & 0 \\\\ \\Phi_c & \\Phi_n \\end{bmatrix}

    where

    - :math:`\\Phi_c = -K_{ii}^{-1} K_{ib}` are the **constraint modes**
      (static response of internal DOFs to unit boundary displacements,
      Craig & Bampton 1968, eq. 4);
    - :math:`\\Phi_n` are the first ``n_internal_modes`` **fixed-interface
      normal modes** (eigenvectors of :math:`K_{ii} - \\omega^2 M_{ii}` with
      boundary DOFs clamped, Craig & Bampton 1968, eq. 5).

    The reduced-order matrices are

    .. math::

        M_r = T^\\top M T, \\quad K_r = T^\\top K T.

    The reduced system has :math:`n_b + n_{\\text{int}}` DOFs, where
    :math:`n_b` = len(boundary_dofs) and :math:`n_{\\text{int}}` =
    ``n_internal_modes``.

    Parameters
    ----------
    system:
        Full-order :class:`~nlvib.systems.base.MechanicalSystem`.
    boundary_dofs:
        Zero-based DOF indices designating the boundary (interface) DOFs.
        Must be a non-empty subset of ``[0, system.n_dof)``.
    n_internal_modes:
        Number of fixed-interface normal modes to retain
        (≥ 1, ≤ ``system.n_dof - len(boundary_dofs)``).

    Returns
    -------
    reduced : MechanicalSystem
        Reduced-order system with ``len(boundary_dofs) + n_internal_modes``
        DOFs.  Damping matrix is zero (pass-through damping is not yet
        implemented for the reduced basis).
    T : ndarray, shape (n_full, n_b + n_int)
        Transformation matrix such that ``q_full ≈ T @ q_reduced``.

    Raises
    ------
    ValueError
        If *boundary_dofs* is invalid or *n_internal_modes* is out of range.

    References
    ----------
    Craig, R. R. & Bampton, M. C. C. (1968). Coupling of substructures for
    dynamic analysis. *AIAA Journal*, 6(7), 1313–1319.
    """
    n = system.n_dof
    b_idx, i_idx = _partition_indices(n, boundary_dofs)

    n_b = b_idx.size
    n_i = i_idx.size

    if n_internal_modes < 1:
        raise ValueError(
            f"n_internal_modes must be ≥ 1; got {n_internal_modes}."
        )
    if n_internal_modes > n_i:
        raise ValueError(
            f"n_internal_modes ({n_internal_modes}) exceeds the number of "
            f"internal DOFs ({n_i})."
        )

    K = system.K
    M = system.M

    # ---- Submatrices -------------------------------------------------------
    K_ii = _extract_submatrices(K, i_idx, i_idx)  # (n_i, n_i)
    K_ib = _extract_submatrices(K, i_idx, b_idx)  # (n_i, n_b)
    M_ii = _extract_submatrices(M, i_idx, i_idx)  # (n_i, n_i)

    # ---- Constraint modes --------------------------------------------------
    # Phi_c = -K_ii^{-1} K_ib  (Craig & Bampton 1968, eq. 4)
    # Use sparse solver if K_ii is large enough.
    K_ii_csr = csr_matrix(K_ii)
    Phi_c: FloatArray = spla.spsolve(K_ii_csr, -K_ib)  # (n_i, n_b)
    if Phi_c.ndim == 1:
        # spsolve returns 1-D when K_ib has a single column; reshape.
        Phi_c = Phi_c.reshape(n_i, 1)

    # ---- Fixed-interface normal modes --------------------------------------
    # Solve: K_ii Phi_n = omega^2 M_ii Phi_n  (Craig & Bampton 1968, eq. 5)
    # Use scipy.linalg.eigh (symmetric positive definite pair).
    eigenvalues, eigenvectors = la.eigh(
        K_ii, M_ii, subset_by_index=[0, n_internal_modes - 1]
    )
    Phi_n: FloatArray = eigenvectors  # (n_i, n_internal_modes)

    # ---- Build transformation matrix T ------------------------------------
    # T has shape (n_full, n_b + n_internal_modes).
    # DOF ordering in reduced system: boundary DOFs first, then modal coords.
    n_r = n_b + n_internal_modes
    T = np.zeros((n, n_r), dtype=np.float64)

    # Boundary block: identity for boundary DOFs
    T[b_idx, :n_b] = np.eye(n_b)

    # Internal block: constraint modes (columns 0..n_b-1) + normal modes
    T[i_idx, :n_b] = Phi_c
    T[i_idx, n_b:] = Phi_n

    # ---- Reduced matrices --------------------------------------------------
    # M_r = T^T M T,  K_r = T^T K T
    # Convert full M, K to dense for the triple product (size is manageable
    # for typical CMS substructures; the *reduced* system is small).
    M_dense = np.asarray(M.todense(), dtype=np.float64)
    K_dense = np.asarray(K.todense(), dtype=np.float64)

    # T^T (n_r, n) @ M_dense (n, n) @ T (n, n_r) -- two matrix multiplies
    MT = M_dense @ T           # (n, n_r)
    M_r: FloatArray = T.T @ MT  # (n_r, n_r)

    KT = K_dense @ T           # (n, n_r)
    K_r: FloatArray = T.T @ KT  # (n_r, n_r)

    # Symmetrise (remove floating-point asymmetry)
    M_r = 0.5 * (M_r + M_r.T)
    K_r = 0.5 * (K_r + K_r.T)

    # Zero damping in reduced model
    D_r = np.zeros((n_r, n_r), dtype=np.float64)

    reduced = MechanicalSystem(M_r, D_r, K_r)
    return reduced, T


# ---------------------------------------------------------------------------
# Rubin (free-interface) reduction
# ---------------------------------------------------------------------------


def rubin(
    system: MechanicalSystem,
    boundary_dofs: list[int] | NDArray[np.intp],
    n_modes: int,
) -> tuple[MechanicalSystem, FloatArray]:
    """Rubin (free-interface) component mode synthesis reduction.

    Builds a reduced basis from:

    1. **Free-interface normal modes** :math:`\\Phi_f` — the first ``n_modes``
       eigenvectors of the undamped free vibration problem
       :math:`(K - \\omega^2 M)\\phi = 0`.
    2. **Residual flexibility attachment modes** :math:`G_r` — static
       correction for the influence of boundary forces, corrected to remove
       the contribution of the retained modes (Rubin 1975, eq. 14):

    .. math::

        G_r = F_s - \\Phi_f \\Lambda_f^{-1} \\Phi_f^\\top

    where :math:`F_s = K^+` is the pseudo-inverse of *K* evaluated at the
    boundary DOFs, and :math:`\\Lambda_f = \\mathrm{diag}(\\omega_f^2)`.

    The Ritz basis is

    .. math::

        \\Psi = \\begin{bmatrix} \\Phi_f & G_r \\end{bmatrix}

    and the reduced matrices are obtained by the Galerkin projection

    .. math::

        M_r = \\Psi^\\top M \\Psi, \\quad K_r = \\Psi^\\top K \\Psi.

    The reduced system has :math:`n_{\\text{modes}} + n_b` DOFs.

    .. note::

        When *K* is singular (e.g. free-free system with rigid-body modes)
        the pseudo-inverse is used via ``scipy.linalg.lstsq``.  For
        well-constrained systems (positive definite *K*) the exact solve is
        used.

    Parameters
    ----------
    system:
        Full-order :class:`~nlvib.systems.base.MechanicalSystem`.  Must be
        undamped (or damping is ignored in the reduction).
    boundary_dofs:
        Zero-based DOF indices of the boundary (interface) DOFs.
    n_modes:
        Number of free-interface normal modes to retain.

    Returns
    -------
    reduced : MechanicalSystem
        Reduced-order system with ``n_modes + len(boundary_dofs)`` DOFs.
        Damping matrix is zero.
    T : ndarray, shape (n_full, n_modes + n_b)
        Transformation (Ritz) matrix.

    Raises
    ------
    ValueError
        If *boundary_dofs* is invalid or *n_modes* is out of range.

    References
    ----------
    Rubin, S. (1975).  Improved component-mode representation for structural
    dynamic analysis.  *AIAA Journal*, 13(8), 995–1006.
    """
    n = system.n_dof
    b_idx, _ = _partition_indices(n, boundary_dofs)

    n_b = b_idx.size

    if n_modes < 1:
        raise ValueError(f"n_modes must be ≥ 1; got {n_modes}.")
    if n_modes > n:
        raise ValueError(
            f"n_modes ({n_modes}) must not exceed system n_dof ({n})."
        )

    K = system.K
    M = system.M

    K_dense = np.asarray(K.todense(), dtype=np.float64)
    M_dense = np.asarray(M.todense(), dtype=np.float64)

    # ---- Free-interface normal modes (Rubin 1975, eq. 3) ------------------
    # Solve the full undamped eigenproblem.
    eigenvalues, eigenvectors = la.eigh(
        K_dense, M_dense, subset_by_index=[0, n_modes - 1]
    )
    omega_sq: FloatArray = eigenvalues          # (n_modes,)
    Phi_f: FloatArray = eigenvectors            # (n, n_modes)

    # ---- Residual flexibility attachment modes (Rubin 1975, eq. 14) -------
    # F_s = K^{+} evaluated at boundary columns, i.e. F_s[:, b_idx].
    # For positive-definite K use the direct solve; for singular K (rigid-body
    # modes with omega_sq ~ 0) fall back to least-squares pseudo-inverse.
    # We need the columns of K^+ corresponding to boundary DOFs.
    # Solve K x = e_j for each boundary DOF j.
    e_boundary = np.zeros((n, n_b), dtype=np.float64)
    for col_idx_local, j in enumerate(b_idx):
        e_boundary[j, col_idx_local] = 1.0

    # Detect near-singularity: any eigenvalue close to zero indicates
    # rigid-body modes are present.
    _RIGID_BODY_TOL: float = 1e-6 * (np.abs(omega_sq).max() if omega_sq.size else 1.0)
    has_rigid_body = bool(np.any(np.abs(omega_sq) < _RIGID_BODY_TOL))

    if has_rigid_body:
        # Pseudo-inverse via lstsq (handles rank-deficient K)
        Fs_b, _res, _rank, _sv = la.lstsq(K_dense, e_boundary)
    else:
        # K is positive definite — use LU factorisation (via solve)
        Fs_b = la.solve(K_dense, e_boundary)  # (n, n_b)

    # Residual flexibility: remove retained-mode contribution
    # G_r = Fs_b - Phi_f diag(1/omega_sq) Phi_f^T e_boundary
    # = Fs_b - Phi_f (Phi_f^T e_boundary / omega_sq[:, None])
    # Avoid division by zero for near-zero eigenvalues (rigid-body modes).
    safe_omega_sq = np.where(np.abs(omega_sq) < _RIGID_BODY_TOL, np.inf, omega_sq)
    PhiT_eb = Phi_f.T @ e_boundary      # (n_modes, n_b)
    G_r: FloatArray = Fs_b - Phi_f @ (PhiT_eb / safe_omega_sq[:, np.newaxis])

    # ---- Assemble Ritz basis T = [Phi_f | G_r] ----------------------------
    T = np.concatenate([Phi_f, G_r], axis=1)  # (n, n_modes + n_b)

    # ---- Reduced matrices --------------------------------------------------
    MT = M_dense @ T
    M_r: FloatArray = T.T @ MT
    KT = K_dense @ T
    K_r: FloatArray = T.T @ KT

    # Symmetrise
    M_r = 0.5 * (M_r + M_r.T)
    K_r = 0.5 * (K_r + K_r.T)

    D_r = np.zeros_like(M_r)

    reduced = MechanicalSystem(M_r, D_r, K_r)
    return reduced, T
