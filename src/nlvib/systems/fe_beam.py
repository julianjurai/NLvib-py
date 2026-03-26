"""
Finite-element Euler-Bernoulli beam for NLvib.

Implements :class:`FE_EulerBernoulliBeam`, a Bernoulli-Euler beam model
assembled from standard two-node beam elements (4 DOF/element: transverse
displacement *w* and rotation *θ* at each end).

The global free-vibration eigenvalue problem is

.. math::

    K \\, \\phi = \\omega^2 M \\, \\phi

where *K* and *M* are the reduced (boundary-condition applied) stiffness and
mass matrices assembled from element contributions.

Equation references
-------------------
- Local element stiffness matrix  : Krack & Gross (2019) §5; Petyt (2010) §4
- Local element mass matrix       : consistent mass, Petyt (2010) §4
- Assembly + BC reduction         : standard FEM textbook procedure
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from nlvib.nonlinearities.elements import NonlinearElement
from nlvib.systems.base import MechanicalSystem

__all__ = ["FE_EulerBernoulliBeam"]

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Named constants (no magic numbers)
# ---------------------------------------------------------------------------

# DOF labels per node
_DOF_W: Literal["w"] = "w"        # transverse displacement
_DOF_THETA: Literal["theta"] = "theta"  # rotation

# Number of DOF per node / per element
_N_DOF_PER_NODE: int = 2
_N_NODES_PER_ELEMENT: int = 2
_N_DOF_PER_ELEMENT: int = _N_DOF_PER_NODE * _N_NODES_PER_ELEMENT  # = 4

# Consistent mass matrix scalar multiplier numerator
_MASS_SCALE: float = 420.0

# Boundary-condition string literals
_BC_CLAMPED_FREE: str = "clamped-free"
_BC_CLAMPED_CLAMPED: str = "clamped-clamped"
_BC_FREE_FREE: str = "free-free"

_SUPPORTED_BCS: frozenset[str] = frozenset(
    [_BC_CLAMPED_FREE, _BC_CLAMPED_CLAMPED, _BC_FREE_FREE]
)

# ---------------------------------------------------------------------------
# Local element matrices (pure functions, no side effects)
# ---------------------------------------------------------------------------


def _local_stiffness(EI: float, Le: float) -> FloatArray:
    """Build the 4×4 Euler-Bernoulli element stiffness matrix.

    The matrix in natural ordering ``[w_1, θ_1, w_2, θ_2]`` is

    .. math::

        K_e = \\frac{EI}{L_e^3}
        \\begin{bmatrix}
          12 &  6L_e & -12 &  6L_e \\\\
          6L_e &  4L_e^2 & -6L_e &  2L_e^2 \\\\
         -12 & -6L_e &  12 & -6L_e \\\\
          6L_e &  2L_e^2 & -6L_e &  4L_e^2
        \\end{bmatrix}

    Reference: Krack & Gross (2019) §5 / Petyt (2010) eq. 4.24.

    Parameters
    ----------
    EI:
        Flexural rigidity [N·m²].
    Le:
        Element length [m].

    Returns
    -------
    ndarray, shape (4, 4)
        Local element stiffness matrix.
    """
    c = EI / Le**3
    L = Le
    return c * np.array(
        [
            [12.0,   6.0 * L, -12.0,   6.0 * L],
            [6.0 * L,  4.0 * L**2, -6.0 * L,  2.0 * L**2],
            [-12.0,  -6.0 * L,  12.0,  -6.0 * L],
            [6.0 * L,  2.0 * L**2, -6.0 * L,  4.0 * L**2],
        ],
        dtype=np.float64,
    )


def _local_mass(rho: float, A: float, Le: float) -> FloatArray:
    """Build the 4×4 Euler-Bernoulli consistent element mass matrix.

    The matrix in natural ordering ``[w_1, θ_1, w_2, θ_2]`` is

    .. math::

        M_e = \\frac{\\rho A L_e}{420}
        \\begin{bmatrix}
          156  &  22L_e &  54   & -13L_e \\\\
          22L_e &  4L_e^2  &  13L_e  & -3L_e^2 \\\\
          54   &  13L_e &  156  & -22L_e \\\\
         -13L_e & -3L_e^2  & -22L_e &   4L_e^2
        \\end{bmatrix}

    Reference: Krack & Gross (2019) §5 / Petyt (2010) eq. 4.26.

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
    ndarray, shape (4, 4)
        Local element mass matrix.
    """
    c = rho * A * Le / _MASS_SCALE
    L = Le
    return c * np.array(
        [
            [156.0,   22.0 * L,   54.0,  -13.0 * L],
            [22.0 * L,   4.0 * L**2,  13.0 * L,  -3.0 * L**2],
            [54.0,   13.0 * L,  156.0,  -22.0 * L],
            [-13.0 * L,  -3.0 * L**2, -22.0 * L,   4.0 * L**2],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _assemble_global(
    n_elements: int,
    Le: float,
    EI: float,
    rho: float,
    A: float,
) -> tuple[FloatArray, FloatArray]:
    """Assemble unreduced global stiffness and mass matrices.

    Uses COO (triplet) accumulation for efficient sparse insertion, then
    returns dense arrays that the BC routine converts to sparse CSR.

    Parameters
    ----------
    n_elements:
        Number of beam elements.
    Le:
        Element length [m].
    EI:
        Flexural rigidity [N·m²].
    rho:
        Material density [kg/m³].
    A:
        Cross-sectional area [m²].

    Returns
    -------
    K_global : ndarray, shape (n_total_dof, n_total_dof)
        Global (unreduced) stiffness matrix.
    M_global : ndarray, shape (n_total_dof, n_total_dof)
        Global (unreduced) mass matrix.
    """
    n_nodes = n_elements + 1
    n_total = n_nodes * _N_DOF_PER_NODE

    Ke = _local_stiffness(EI, Le)
    Me = _local_mass(rho, A, Le)

    K_global = np.zeros((n_total, n_total), dtype=np.float64)
    M_global = np.zeros((n_total, n_total), dtype=np.float64)

    for e in range(n_elements):
        # Global DOF indices for element e (nodes e and e+1)
        # Node i has DOFs [2i, 2i+1] = [w_i, theta_i]
        dofs = np.array(
            [
                2 * e,
                2 * e + 1,
                2 * e + 2,
                2 * e + 3,
            ],
            dtype=np.intp,
        )
        # Scatter-add (vectorised outer index)
        ix = np.ix_(dofs, dofs)
        K_global[ix] += Ke
        M_global[ix] += Me

    return K_global, M_global


def _constrained_dofs(n_elements: int, bc: str) -> list[int]:
    """Return the list of global DOF indices that are constrained (zero) for *bc*.

    Global DOF ordering: node i → [2i (w), 2i+1 (θ)] for i = 0 … n_elements.

    Parameters
    ----------
    n_elements:
        Number of beam elements.
    bc:
        Boundary condition string (``"clamped-free"``, ``"clamped-clamped"``,
        or ``"free-free"``).

    Returns
    -------
    list of int
        Sorted list of zero-based DOF indices to eliminate.

    Raises
    ------
    ValueError
        If *bc* is not one of the supported strings.
    """
    if bc not in _SUPPORTED_BCS:
        raise ValueError(
            f"Unsupported boundary condition '{bc}'. "
            f"Choose from: {sorted(_SUPPORTED_BCS)}."
        )

    n_nodes = n_elements + 1
    last_node = n_nodes - 1
    constrained: list[int] = []

    if bc == _BC_CLAMPED_FREE:
        # Left end fully clamped: w=0, θ=0 at node 0
        constrained = [0, 1]
    elif bc == _BC_CLAMPED_CLAMPED:
        # Both ends fully clamped
        constrained = [0, 1, 2 * last_node, 2 * last_node + 1]
    elif bc == _BC_FREE_FREE:
        # No constraints — all DOFs retained
        constrained = []

    return sorted(constrained)


def _apply_bc(
    K_full: FloatArray,
    M_full: FloatArray,
    constrained: list[int],
) -> tuple[csr_matrix, csr_matrix, NDArray[np.intp]]:
    """Eliminate constrained DOFs from global matrices.

    Parameters
    ----------
    K_full:
        Full (unreduced) stiffness matrix.
    M_full:
        Full (unreduced) mass matrix.
    constrained:
        List of DOF indices to eliminate.

    Returns
    -------
    K_reduced : csr_matrix
        Reduced stiffness matrix.
    M_reduced : csr_matrix
        Reduced mass matrix.
    free_dofs : ndarray of int
        Indices of the free (retained) DOFs in the unreduced system.
    """
    n_total = K_full.shape[0]
    all_dofs = np.arange(n_total, dtype=np.intp)
    constrained_arr = np.array(constrained, dtype=np.intp)
    free_dofs: NDArray[np.intp] = np.setdiff1d(all_dofs, constrained_arr).astype(
        np.intp
    )

    ix = np.ix_(free_dofs, free_dofs)
    K_reduced = csr_matrix(K_full[ix])
    M_reduced = csr_matrix(M_full[ix])
    return K_reduced, M_reduced, free_dofs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FE_EulerBernoulliBeam(MechanicalSystem):
    """Finite-element Euler-Bernoulli beam model.

    Assembles global consistent mass and stiffness matrices from
    ``n_elements`` standard two-node Bernoulli-Euler beam elements and
    applies boundary conditions by removing constrained DOFs.

    Each node has two DOF: transverse displacement *w* and rotation *θ*.
    Global DOF ordering for node *i*: ``[2i, 2i+1] = [w_i, θ_i]``.

    After BC application the system matrices are stored in
    :class:`~scipy.sparse.csr_matrix` format and passed to the
    :class:`~nlvib.systems.base.MechanicalSystem` base class.

    Equation references
    -------------------
    - Element matrices : Krack & Gross (2019) §5; Petyt (2010) §4
    - BC application   : standard static condensation

    Parameters
    ----------
    n_elements:
        Number of beam elements (≥ 1).
    L:
        Total beam length [m].
    E:
        Young's modulus [Pa].
    I_area:
        Second moment of area [m⁴].
    rho:
        Material density [kg/m³].
    A:
        Cross-sectional area [m²].
    bc:
        Boundary condition string.  Supported values:

        - ``"clamped-free"``     — cantilever beam
        - ``"clamped-clamped"``  — both ends fixed
        - ``"free-free"``        — no constraints (rigid-body modes present)

    Raises
    ------
    ValueError
        If *n_elements* < 1 or *bc* is not supported.
    """

    def __init__(
        self,
        n_elements: int,
        L: float,
        E: float,
        I_area: float,
        rho: float,
        A: float,
        bc: str,
    ) -> None:
        if n_elements < 1:
            raise ValueError(f"n_elements must be ≥ 1; got {n_elements}.")

        self._n_elements = n_elements
        self._L_total = L
        self._E = E
        self._I = I_area
        self._rho = rho
        self._A = A
        self._bc = bc

        Le = L / n_elements
        EI = E * I_area
        self._Le = Le

        # Assemble full (unreduced) matrices
        K_full, M_full = _assemble_global(n_elements, Le, EI, rho, A)

        # Determine constrained DOFs and apply BCs
        constrained = _constrained_dofs(n_elements, bc)
        K_red, M_red, free_dofs = _apply_bc(K_full, M_full, constrained)
        self._free_dofs: NDArray[np.intp] = free_dofs

        # Zero damping by default
        n_free = K_red.shape[0]
        D_red = csr_matrix((n_free, n_free), dtype=np.float64)

        super().__init__(M_red, D_red, K_red)

        # Forcing registry: list of (free_dof_index, amplitude)
        self._forcing: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_beam_elements(self) -> int:
        """Number of beam elements."""
        return self._n_elements

    @property
    def L_total(self) -> float:
        """Total beam length [m]."""
        return self._L_total

    @property
    def element_length(self) -> float:
        """Length of a single beam element [m]."""
        return self._Le

    @property
    def free_dofs(self) -> NDArray[np.intp]:
        """Global (unreduced) DOF indices that are retained after BC application."""
        return self._free_dofs.copy()

    @property
    def bc(self) -> str:
        """Boundary condition string."""
        return self._bc

    # ------------------------------------------------------------------
    # DOF look-up
    # ------------------------------------------------------------------

    def find_dof(self, node_index: int, dof_type: str) -> int:
        """Return the reduced (post-BC) DOF index for a given node and DOF type.

        Maps a physical node/DOF pair to the index into the reduced system
        vectors ``q`` and the rows/columns of **M**, **K**.

        Parameters
        ----------
        node_index:
            Zero-based node index (0 = left end, ``n_elements`` = right end).
        dof_type:
            ``"w"`` for transverse displacement or ``"theta"`` for rotation.

        Returns
        -------
        int
            Zero-based index into the reduced DOF vector.

        Raises
        ------
        ValueError
            If *node_index* is out of range, *dof_type* is not ``"w"`` or
            ``"theta"``, or the requested DOF is constrained.
        """
        n_nodes = self._n_elements + 1
        if node_index < 0 or node_index >= n_nodes:
            raise ValueError(
                f"node_index must be in [0, {n_nodes - 1}]; got {node_index}."
            )
        if dof_type not in (_DOF_W, _DOF_THETA):
            raise ValueError(
                f"dof_type must be 'w' or 'theta'; got '{dof_type}'."
            )

        # Global (unreduced) DOF index
        dof_offset = 0 if dof_type == _DOF_W else 1
        global_dof = 2 * node_index + dof_offset

        # Map to reduced index
        hits = np.flatnonzero(self._free_dofs == global_dof)
        if hits.size == 0:
            raise ValueError(
                f"DOF (node={node_index}, dof_type='{dof_type}') is "
                f"constrained and not present in the reduced system."
            )
        return int(hits[0])

    # ------------------------------------------------------------------
    # Forcing
    # ------------------------------------------------------------------

    def add_forcing(
        self,
        node_index: int,
        dof_type: str,
        amplitude: float,
    ) -> None:
        """Register a harmonic forcing location on the beam.

        Stores the reduced DOF index and amplitude for use by the solver.
        Does not modify **M**, **D**, or **K**.

        Parameters
        ----------
        node_index:
            Zero-based node index.
        dof_type:
            ``"w"`` or ``"theta"``.
        amplitude:
            Force amplitude [N] (for *w*) or moment amplitude [N·m] (for *θ*).
        """
        reduced_dof = self.find_dof(node_index, dof_type)
        self._forcing.append((reduced_dof, float(amplitude)))

    @property
    def forcing(self) -> list[tuple[int, float]]:
        """Read-only list of ``(reduced_dof_index, amplitude)`` tuples."""
        return list(self._forcing)

    # ------------------------------------------------------------------
    # Nonlinear attachments
    # ------------------------------------------------------------------

    def add_nonlinear_attachment(
        self,
        node_index: int,
        dof_type: str,
        element: NonlinearElement,
    ) -> None:
        """Attach a nonlinear element at a node DOF.

        Delegates to :meth:`~nlvib.systems.base.MechanicalSystem.add_nonlinear_element`
        after verifying the node/DOF is valid and free.

        The element's ``eval(q, dq)`` receives the **full reduced state vectors**
        ``q`` and ``dq`` of length ``n_dof``; the element must therefore use
        the correct reduced DOF index internally.  Use :meth:`find_dof` to
        obtain that index before constructing the element.

        Parameters
        ----------
        node_index:
            Zero-based node index.
        dof_type:
            ``"w"`` or ``"theta"``.
        element:
            A :class:`~nlvib.nonlinearities.elements.NonlinearElement` whose
            internal DOF index references the **reduced** system.
        """
        # Validate node/DOF (raises ValueError if constrained)
        _ = self.find_dof(node_index, dof_type)
        self.add_nonlinear_element(element)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FE_EulerBernoulliBeam("
            f"n_elements={self._n_elements}, "
            f"L={self._L_total}, "
            f"bc='{self._bc}', "
            f"n_dof={self.n_dof})"
        )


# ---------------------------------------------------------------------------
# Module-level sparse assembly (alternative public API — not used internally)
# ---------------------------------------------------------------------------


def build_beam_matrices(
    n_elements: int,
    L: float,
    E: float,
    I_area: float,
    rho: float,
    A: float,
) -> tuple[csr_matrix, csr_matrix]:
    """Build the *unreduced* global mass and stiffness matrices for a beam.

    Convenience function that returns the full (pre-BC) matrices as
    :class:`~scipy.sparse.csr_matrix`.  Useful for inspection and testing
    without instantiating a full :class:`FE_EulerBernoulliBeam`.

    Parameters
    ----------
    n_elements:
        Number of beam elements.
    L:
        Total beam length [m].
    E:
        Young's modulus [Pa].
    I_area:
        Second moment of area [m⁴].
    rho:
        Material density [kg/m³].
    A:
        Cross-sectional area [m²].

    Returns
    -------
    K : csr_matrix
    M : csr_matrix
    """
    Le = L / n_elements
    EI = E * I_area
    K_full, M_full = _assemble_global(n_elements, Le, EI, rho, A)
    return csr_matrix(K_full), csr_matrix(M_full)


def _build_beam_matrices_sparse(
    n_elements: int,
    Le: float,
    EI: float,
    rho: float,
    A: float,
) -> tuple[sp.coo_matrix, sp.coo_matrix]:
    """Build global matrices using COO accumulation (internal, used for large n).

    This is an alternative to :func:`_assemble_global` that avoids a dense
    intermediate array.  Currently kept for reference; used when
    ``n_elements * _N_DOF_PER_ELEMENT`` exceeds a threshold in future
    optimisation.

    Parameters
    ----------
    n_elements:
        Number of beam elements.
    Le:
        Element length [m].
    EI:
        Flexural rigidity [N·m²].
    rho:
        Material density [kg/m³].
    A:
        Cross-sectional area [m²].

    Returns
    -------
    K_coo : coo_matrix
    M_coo : coo_matrix
    """
    n_nodes = n_elements + 1
    n_total = n_nodes * _N_DOF_PER_NODE
    nnz_per_elem = _N_DOF_PER_ELEMENT**2  # 16 entries per element

    Ke = _local_stiffness(EI, Le)
    Me = _local_mass(rho, A, Le)

    rows = np.empty(n_elements * nnz_per_elem, dtype=np.intp)
    cols = np.empty(n_elements * nnz_per_elem, dtype=np.intp)
    k_data = np.empty(n_elements * nnz_per_elem, dtype=np.float64)
    m_data = np.empty(n_elements * nnz_per_elem, dtype=np.float64)

    local_ij = np.array(
        [[i, j] for i in range(_N_DOF_PER_ELEMENT) for j in range(_N_DOF_PER_ELEMENT)],
        dtype=np.intp,
    )  # shape (16, 2)

    for e in range(n_elements):
        base = 2 * e
        start = e * nnz_per_elem
        end = start + nnz_per_elem
        rows[start:end] = base + local_ij[:, 0]
        cols[start:end] = base + local_ij[:, 1]
        k_data[start:end] = Ke.ravel()
        m_data[start:end] = Me.ravel()

    K_coo = sp.coo_matrix((k_data, (rows, cols)), shape=(n_total, n_total))
    M_coo = sp.coo_matrix((m_data, (rows, cols)), shape=(n_total, n_total))
    return K_coo, M_coo
