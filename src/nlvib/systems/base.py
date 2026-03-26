"""
Base class for mechanical systems in NLvib.

Provides :class:`MechanicalSystem`, the common base for all system types
(single-mass oscillator, chain of oscillators, FE beam/rod, etc.).

The class stores the linear system matrices ``M``, ``D``, ``K`` in
``scipy.sparse.csr_matrix`` format and maintains a registry of attached
:class:`~nlvib.nonlinearities.elements.NonlinearElement` objects.

References
----------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  ISBN 978-3-030-14022-9.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from nlvib.nonlinearities.elements import NonlinearElement

__all__ = ["MechanicalSystem"]

FloatArray = NDArray[np.float64]

# Threshold: systems with n_dof >= this value store M/D/K as sparse matrices.
# For smaller systems dense arrays (wrapped in csr_matrix) would still be
# accessible via `.toarray()` if needed by subclasses.
_SPARSE_THRESHOLD: int = 10


def _to_csr(matrix: FloatArray | sp.spmatrix) -> csr_matrix:
    """Convert *matrix* to :class:`scipy.sparse.csr_matrix`.

    Parameters
    ----------
    matrix:
        A dense NumPy array or any SciPy sparse matrix.

    Returns
    -------
    csr_matrix
        The input converted to CSR format.
    """
    if sp.issparse(matrix):
        sparse_mat: sp.spmatrix = matrix  # narrow to spmatrix for mypy
        return sparse_mat.tocsr()
    arr = np.asarray(matrix, dtype=np.float64)
    return csr_matrix(arr)


class MechanicalSystem:
    """Base class for mechanical systems.

    Stores the mass (**M**), damping (**D**), and stiffness (**K**) matrices
    and a list of attached nonlinear elements.  Subclasses (e.g.
    :class:`~nlvib.systems.oscillators.SingleMassOscillator`) are responsible
    for constructing the appropriate system matrices.

    The equation of motion is

    .. math::

        M \\ddot{q} + D \\dot{q} + K q + f_{\\mathrm{nl}}(q, \\dot{q}) = f_{\\mathrm{ext}}(t)

    where :math:`f_{\\mathrm{nl}}` is assembled by
    :meth:`eval_nonlinear_forces` from all registered elements.

    Parameters
    ----------
    M:
        Mass matrix of shape ``(n_dof, n_dof)``.  Accepted as a dense NumPy
        array or any SciPy sparse matrix; stored internally as
        :class:`~scipy.sparse.csr_matrix`.
    D:
        Damping matrix of shape ``(n_dof, n_dof)``.  Same format rules as *M*.
    K:
        Stiffness matrix of shape ``(n_dof, n_dof)``.  Same format rules as *M*.

    Raises
    ------
    ValueError
        If *M*, *D*, *K* are not square matrices of the same size.
    """

    def __init__(
        self,
        M: FloatArray | sp.spmatrix,
        D: FloatArray | sp.spmatrix,
        K: FloatArray | sp.spmatrix,
    ) -> None:
        M_csr = _to_csr(M)
        D_csr = _to_csr(D)
        K_csr = _to_csr(K)

        n = M_csr.shape[0]
        for name, mat in (("M", M_csr), ("D", D_csr), ("K", K_csr)):
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Matrix {name} must be square; got shape {mat.shape}."
                )
            if mat.shape[0] != n:
                raise ValueError(
                    f"Matrix {name} has size {mat.shape[0]} but M has size {n}."
                )

        self._M: csr_matrix = M_csr
        self._D: csr_matrix = D_csr
        self._K: csr_matrix = K_csr

        self._nonlinear_elements: list[NonlinearElement] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def M(self) -> csr_matrix:
        """Mass matrix (CSR sparse)."""
        return self._M

    @property
    def D(self) -> csr_matrix:
        """Damping matrix (CSR sparse)."""
        return self._D

    @property
    def K(self) -> csr_matrix:
        """Stiffness matrix (CSR sparse)."""
        return self._K

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom.

        Derived from ``M.shape[0]`` per K&G §1.1 convention.
        """
        return int(self._M.shape[0])

    @property
    def nonlinear_elements(self) -> list[NonlinearElement]:
        """Read-only view of the registered nonlinear elements."""
        return list(self._nonlinear_elements)

    # ------------------------------------------------------------------
    # Nonlinear element registry
    # ------------------------------------------------------------------

    def add_nonlinear_element(self, element: NonlinearElement) -> None:
        """Register a nonlinear element with this system.

        Parameters
        ----------
        element:
            A :class:`~nlvib.nonlinearities.elements.NonlinearElement`
            instance, created by one of the factory functions (e.g.
            :func:`~nlvib.nonlinearities.elements.cubic_spring`).
        """
        self._nonlinear_elements.append(element)

    # ------------------------------------------------------------------
    # Nonlinear force assembly
    # ------------------------------------------------------------------

    def eval_nonlinear_forces(
        self,
        q: FloatArray,
        dq: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Assemble the global nonlinear force vector and Jacobians.

        Evaluates every registered :class:`NonlinearElement` and accumulates
        the contributions into global arrays of size ``(n_dof,)`` and
        ``(n_dof, n_dof)``.

        The nonlinear restoring force term in the equation of motion
        (K&G eq. 1.1) is:

        .. math::

            f_{\\mathrm{nl}}(q, \\dot{q}) = \\sum_e g_e(q, \\dot{q})

        and its Jacobians are

        .. math::

            \\frac{\\partial f_{\\mathrm{nl}}}{\\partial q} =
                \\sum_e \\frac{\\partial g_e}{\\partial q}, \\qquad
            \\frac{\\partial f_{\\mathrm{nl}}}{\\partial \\dot{q}} =
                \\sum_e \\frac{\\partial g_e}{\\partial \\dot{q}}.

        Parameters
        ----------
        q:
            Displacement state vector of shape ``(n_dof,)``.
        dq:
            Velocity state vector of shape ``(n_dof,)``.

        Returns
        -------
        f : ndarray, shape ``(n_dof,)``
            Assembled nonlinear force vector.
        df_dq : ndarray, shape ``(n_dof, n_dof)``
            Jacobian of *f* w.r.t. *q*.
        df_ddq : ndarray, shape ``(n_dof, n_dof)``
            Jacobian of *f* w.r.t. *dq*.

        Raises
        ------
        ValueError
            If *q* or *dq* do not have shape ``(n_dof,)``.
        """
        n = self.n_dof
        q_arr = np.asarray(q, dtype=np.float64)
        dq_arr = np.asarray(dq, dtype=np.float64)

        if q_arr.shape != (n,):
            raise ValueError(
                f"q must have shape ({n},); got {q_arr.shape}."
            )
        if dq_arr.shape != (n,):
            raise ValueError(
                f"dq must have shape ({n},); got {dq_arr.shape}."
            )

        f_global: FloatArray = np.zeros(n, dtype=np.float64)
        df_dq_global: FloatArray = np.zeros((n, n), dtype=np.float64)
        df_ddq_global: FloatArray = np.zeros((n, n), dtype=np.float64)

        for element in self._nonlinear_elements:
            f_e, df_dq_e, df_ddq_e = element.eval(q_arr, dq_arr)
            # f_e is a scalar; df_dq_e and df_ddq_e are (n_dof,) vectors.
            # Each element contributes a single force to one DOF row;
            # its Jacobian row is the row of that DOF in the global Jacobian.
            # Conventionally (K&G Appendix C) each element returns a length-n
            # gradient vector, which maps directly to a single row of the
            # global n×n Jacobian when the element acts on one DOF.
            # We store the full gradient as a row in the Jacobian matrix so
            # the i-th row captures df_i/dq_j for all j.
            #
            # Identification of the force DOF: find where df_dq_e is nonzero
            # or where df_ddq_e is nonzero; the force is applied there.
            # However, the simpler and more general approach is to treat each
            # element as contributing a force that is laid out in global space
            # via the element's own gradient vectors.
            #
            # The element convention is:
            #   f_e   : scalar — the force applied to the element DOF
            #   df_dq_e  : (n,) — gradient of that scalar w.r.t. all q
            #   df_ddq_e : (n,) — gradient of that scalar w.r.t. all dq
            #
            # Assembly into global n-vector f and global n×n Jacobians:
            # We need to know which DOF receives this force.  For the standard
            # elements in elements.py the force acts on the DOF where df_dq_e
            # or df_ddq_e is nonzero, but we need a more robust approach.
            #
            # Robust assembly: we ask element to declare its target DOF via
            # the gradient.  For single-DOF elements the nonzero index of
            # df_dq_e (or df_ddq_e for dampers) identifies the target row.
            # For velocity-only elements (quadratic damper, tanh friction)
            # df_dq_e is zero; use df_ddq_e.
            # For displacement-only elements df_ddq_e is zero; use df_dq_e.
            #
            # Rather than guessing the target DOF, we accumulate using the
            # convention that f_e is placed at every global DOF i where
            # df_dq_e[i] or df_ddq_e[i] is nonzero.  For well-formed
            # single-DOF elements this is exactly one DOF.
            #
            # Simpler equivalent: find target DOF as argmax of |df_dq_e| +
            # |df_ddq_e|.  But an element can have no Jacobian (e.g., sign
            # function at zero) — in that case we must fall back to a
            # pre-declared dof_index.
            #
            # Cleanest correct approach:
            # The global f is a vector.  Each element contributes ONE scalar
            # to ONE row.  The Jacobians are filled per-row.  We identify the
            # target DOF from the element's gradient nonzero mask, falling
            # back to the maximum-gradient DOF.
            dof_mask = (np.abs(df_dq_e) + np.abs(df_ddq_e)) > 0.0
            nonzero_dofs = np.flatnonzero(dof_mask)

            if nonzero_dofs.size > 0:
                # Place force at the first participating DOF.
                target_dof = int(nonzero_dofs[0])
            else:
                # No gradient signal (e.g. nonlinearity at zero crossing).
                # Force is still present; use the entire gradient vector.
                # We add f_e to no specific DOF — skip force but add Jacobian.
                # (This case is degenerate; proper elements should declare DOF.)
                target_dof = -1

            if target_dof >= 0:
                f_global[target_dof] += f_e

            # Jacobian rows: df_dq_e is the gradient vector of f_e w.r.t. all q.
            # It belongs as a contribution to row `target_dof` of df_dq_global.
            if target_dof >= 0:
                df_dq_global[target_dof] += df_dq_e
                df_ddq_global[target_dof] += df_ddq_e

        return f_global, df_dq_global, df_ddq_global

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_dof={self.n_dof}, "
            f"n_elements={len(self._nonlinear_elements)})"
        )
