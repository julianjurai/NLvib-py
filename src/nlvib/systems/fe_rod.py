"""
Finite-element model of a uniform elastic (axial) rod.

Each element is a 2-node bar element with one axial displacement DOF per
node.  The assembly follows the standard direct-stiffness method described in
most FEM textbooks (see e.g. Cook et al., *Concepts and Applications of
Finite Element Analysis*, 4th ed., §2).

The local element matrices are

.. math::

    K_e = \\frac{EA}{L_e}
          \\begin{bmatrix} 1 & -1 \\\\ -1 & 1 \\end{bmatrix}

.. math::

    M_e = \\frac{\\rho A L_e}{6}
          \\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \\end{bmatrix}

where :math:`L_e = L / n_{\\text{elements}}` is the element length.

Boundary conditions are imposed by removing the rows and columns that
correspond to constrained DOFs (Dirichlet elimination).

References
----------
Cook, R.D., Malkus, D.S., Plesha, M.E. & Witt, R.J. (2002). *Concepts and
Applications of Finite Element Analysis*, 4th ed. Wiley.

Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  ISBN 978-3-030-14022-9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from nlvib.systems.base import MechanicalSystem

__all__ = ["FE_ElasticRod"]

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Supported boundary condition strings → sets of constrained node indices
# ---------------------------------------------------------------------------
# Nodes are numbered 0 .. n_elements (inclusive).  Node 0 is the left end,
# node n_elements is the right end.

_BC_CONSTRAINED_NODES: dict[str, tuple[int, ...]] = {
    "clamped-free": (0,),
    "free-clamped": (-1,),  # -1 resolved later to n_elements
    "clamped-clamped": (0, -1),
    "free-free": (),
}


def _element_stiffness(E: float, A: float, Le: float) -> FloatArray:
    """Return the 2×2 local bar-element stiffness matrix.

    .. math::

        K_e = \\frac{EA}{L_e}
              \\begin{bmatrix} 1 & -1 \\\\ -1 & 1 \\end{bmatrix}

    Parameters
    ----------
    E:
        Young's modulus [Pa].
    A:
        Cross-sectional area [m²].
    Le:
        Element length [m].

    Returns
    -------
    ndarray of shape (2, 2)
    """
    c = E * A / Le
    return c * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)


def _element_mass(rho: float, A: float, Le: float) -> FloatArray:
    """Return the 2×2 local bar-element consistent mass matrix.

    .. math::

        M_e = \\frac{\\rho A L_e}{6}
              \\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \\end{bmatrix}

    Parameters
    ----------
    rho:
        Material density [kg/m³].
    A:
        Cross-sectional area [m²].
    Le:
        Element length [m].

    Returns
    -------
    ndarray of shape (2, 2)
    """
    c = rho * A * Le / 6.0
    return c * np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)


def _assemble_global(
    n_elements: int,
    Ke: FloatArray,
    Me: FloatArray,
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble global stiffness and mass matrices from identical element matrices.

    Uses COO accumulation for efficient sparse assembly.

    Parameters
    ----------
    n_elements:
        Number of bar elements.
    Ke:
        2×2 local stiffness matrix (same for every element because the rod is
        uniform).
    Me:
        2×2 local mass matrix (same for every element).

    Returns
    -------
    K_global, M_global : csr_matrix
        Global matrices of size ``(n_nodes, n_nodes)`` where
        ``n_nodes = n_elements + 1``.
    """
    n_nodes = n_elements + 1
    # Number of scalar entries per element (2×2 = 4), total entries in COO
    n_entries = n_elements * 4

    rows = np.empty(n_entries, dtype=np.intp)
    cols = np.empty(n_entries, dtype=np.intp)
    k_vals = np.empty(n_entries, dtype=np.float64)
    m_vals = np.empty(n_entries, dtype=np.float64)

    for e in range(n_elements):
        # Global node indices for element e
        node_ids = np.array([e, e + 1], dtype=np.intp)
        # Flatten local 2×2 → 4 entries
        start = e * 4
        for local_i in range(2):
            for local_j in range(2):
                idx = start + local_i * 2 + local_j
                rows[idx] = node_ids[local_i]
                cols[idx] = node_ids[local_j]
                k_vals[idx] = Ke[local_i, local_j]
                m_vals[idx] = Me[local_i, local_j]

    K_global: csr_matrix = csr_matrix(
        (k_vals, (rows, cols)), shape=(n_nodes, n_nodes)
    )
    M_global: csr_matrix = csr_matrix(
        (m_vals, (rows, cols)), shape=(n_nodes, n_nodes)
    )
    return K_global, M_global


def _apply_bc(
    K: csr_matrix,
    M: csr_matrix,
    constrained_dofs: list[int],
) -> tuple[csr_matrix, csr_matrix]:
    """Remove constrained DOFs from global matrices (Dirichlet elimination).

    Parameters
    ----------
    K:
        Global stiffness matrix before BC application.
    M:
        Global mass matrix before BC application.
    constrained_dofs:
        List of DOF (node) indices to remove.

    Returns
    -------
    K_red, M_red : csr_matrix
        Reduced matrices after DOF elimination.
    """
    n = K.shape[0]
    free_dofs = [i for i in range(n) if i not in constrained_dofs]
    free_arr = np.array(free_dofs, dtype=np.intp)

    # Index into CSR arrays using fancy indexing via tocsc/toarray slicing.
    # scipy supports row/col fancy indexing on lil_matrix; use it here.
    K_lil = K.tolil()
    M_lil = M.tolil()

    K_red: csr_matrix = csr_matrix(K_lil[free_arr, :][:, free_arr])
    M_red: csr_matrix = csr_matrix(M_lil[free_arr, :][:, free_arr])
    return K_red, M_red


class FE_ElasticRod(MechanicalSystem):
    """Finite-element model of a uniform elastic (axial) rod.

    Assembles global mass and stiffness matrices from 2-node bar elements and
    applies boundary conditions by eliminating constrained DOFs.  The rod has
    one axial displacement DOF per node, giving ``n_elements + 1`` nodes and
    ``n_elements + 1 - n_constrained`` free DOFs.

    The equation of motion for the unconstrained system is

    .. math::

        M \\ddot{u} + K u = f_{\\text{ext}}

    where **u** is the vector of free-DOF axial displacements.

    Parameters
    ----------
    n_elements:
        Number of bar elements (must be ≥ 1).
    L:
        Total rod length [m].
    E:
        Young's modulus [Pa].
    A:
        Cross-sectional area [m²].
    rho:
        Material density [kg/m³].
    bc:
        Boundary condition string.  Supported values:

        ``"clamped-free"``
            Left end fixed, right end free.
        ``"free-clamped"``
            Left end free, right end fixed.
        ``"clamped-clamped"``
            Both ends fixed.
        ``"free-free"``
            Both ends free (no rigid-body constraint).

    Raises
    ------
    ValueError
        If *n_elements* < 1, any material parameter ≤ 0, or *bc* is not a
        recognised string.

    Notes
    -----
    The zero damping matrix D is initialised as a sparse zero matrix of the
    same size as M and K (no structural damping model at this level; add via
    modal damping or Rayleigh proportional damping as needed).

    Examples
    --------
    Create a clamped-free steel rod with 5 elements:

    >>> rod = FE_ElasticRod(
    ...     n_elements=5,
    ...     L=1.0,
    ...     E=210e9,
    ...     A=1e-4,
    ...     rho=7800.0,
    ...     bc="clamped-free",
    ... )
    >>> rod.n_dof  # 5 elements → 6 nodes → 1 clamped → 5 free DOFs
    5
    """

    def __init__(
        self,
        n_elements: int,
        L: float,
        E: float,
        A: float,
        rho: float,
        bc: str,
    ) -> None:
        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        if n_elements < 1:
            raise ValueError(f"n_elements must be >= 1; got {n_elements}.")
        if L <= 0.0:
            raise ValueError(f"L must be > 0; got {L}.")
        if E <= 0.0:
            raise ValueError(f"E must be > 0; got {E}.")
        if A <= 0.0:
            raise ValueError(f"A must be > 0; got {A}.")
        if rho <= 0.0:
            raise ValueError(f"rho must be > 0; got {rho}.")
        bc_lower = bc.lower()
        if bc_lower not in _BC_CONSTRAINED_NODES:
            supported = ", ".join(f'"{k}"' for k in _BC_CONSTRAINED_NODES)
            raise ValueError(
                f'Unsupported bc "{bc}". Supported: {supported}.'
            )

        # ------------------------------------------------------------------
        # Element geometry and material
        # ------------------------------------------------------------------
        Le = L / n_elements

        # ------------------------------------------------------------------
        # Local element matrices (uniform rod → same for every element)
        # ------------------------------------------------------------------
        Ke = _element_stiffness(E, A, Le)
        Me = _element_mass(rho, A, Le)

        # ------------------------------------------------------------------
        # Global assembly
        # ------------------------------------------------------------------
        K_global, M_global = _assemble_global(n_elements, Ke, Me)

        # ------------------------------------------------------------------
        # Resolve constrained DOF indices (−1 → last node)
        # ------------------------------------------------------------------
        n_nodes = n_elements + 1
        raw_constrained = _BC_CONSTRAINED_NODES[bc_lower]
        constrained_dofs: list[int] = [
            int(idx) % n_nodes for idx in raw_constrained
        ]

        # ------------------------------------------------------------------
        # Apply boundary conditions
        # ------------------------------------------------------------------
        K_red, M_red = _apply_bc(K_global, M_global, constrained_dofs)

        # Zero damping matrix
        n_free = K_red.shape[0]
        D_zero: csr_matrix = csr_matrix((n_free, n_free), dtype=np.float64)

        # ------------------------------------------------------------------
        # Initialise base class
        # ------------------------------------------------------------------
        super().__init__(M_red, D_zero, K_red)

        # ------------------------------------------------------------------
        # Store metadata as public read-only attributes
        # ------------------------------------------------------------------
        self.n_elements: int = n_elements
        """Number of bar elements."""

        self.L: float = L
        """Total rod length [m]."""

        self.E: float = E
        """Young's modulus [Pa]."""

        self.A: float = A
        """Cross-sectional area [m²]."""

        self.rho: float = rho
        """Material density [kg/m³]."""

        self.bc: str = bc_lower
        """Boundary condition string (lower-cased)."""

        self.Le: float = Le
        """Element length [m]."""

        self.constrained_dofs: list[int] = constrained_dofs
        """Node indices that were eliminated by the boundary condition."""

    def __repr__(self) -> str:
        return (
            f"FE_ElasticRod("
            f"n_elements={self.n_elements}, "
            f"L={self.L}, "
            f"E={self.E}, "
            f"A={self.A}, "
            f"rho={self.rho}, "
            f"bc={self.bc!r}, "
            f"n_dof={self.n_dof})"
        )
