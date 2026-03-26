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

            # Determine which DOF row receives this element's scalar force.
            # Priority 1: explicit target_dof declared on the element (e.g. for
            # polynomial_stiffness spanning multiple DOFs).
            # Priority 2: infer from first nonzero entry in the gradient.
            if element.target_dof is not None:
                target_dof = element.target_dof
            else:
                dof_mask = (np.abs(df_dq_e) + np.abs(df_ddq_e)) > 0.0
                nonzero_dofs = np.flatnonzero(dof_mask)
                target_dof = int(nonzero_dofs[0]) if nonzero_dofs.size > 0 else -1

            if target_dof >= 0:
                f_global[target_dof] += f_e
                df_dq_global[target_dof] += df_dq_e
                df_ddq_global[target_dof] += df_ddq_e

        return f_global, df_dq_global, df_ddq_global

    def eval_nonlinear_forces_batch(
        self,
        q_time: FloatArray,
        dq_time: FloatArray,
    ) -> FloatArray:
        """Assemble the nonlinear force matrix over a batch of time samples.

        Uses vectorised ``eval_batch`` when available on each element,
        falling back to the scalar ``eval_nonlinear_forces`` loop for any
        element that does not supply one.

        Parameters
        ----------
        q_time:
            Displacement matrix of shape ``(n_dof, n_time)``.
        dq_time:
            Velocity matrix of shape ``(n_dof, n_time)``.

        Returns
        -------
        f_nl_time : ndarray, shape ``(n_dof, n_time)``
            Assembled nonlinear force matrix.
        """
        n = self.n_dof
        n_time = q_time.shape[1]
        f_nl_time: FloatArray = np.zeros((n, n_time), dtype=np.float64)

        scalar_elements = []
        for element in self._nonlinear_elements:
            if element.eval_batch is not None:
                f_nl_time += element.eval_batch(q_time, dq_time)
            else:
                scalar_elements.append(element)

        # Fall back to scalar loop for elements without eval_batch
        if scalar_elements:
            for t in range(n_time):
                for element in scalar_elements:
                    f_e, _df_dq, _df_ddq = element.eval(q_time[:, t], dq_time[:, t])
                    dof_mask = (np.abs(_df_dq) + np.abs(_df_ddq)) > 0.0
                    nonzero_dofs = np.flatnonzero(dof_mask)
                    if nonzero_dofs.size > 0:
                        f_nl_time[int(nonzero_dofs[0]), t] += f_e

        return f_nl_time

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_dof={self.n_dof}, "
            f"n_elements={len(self._nonlinear_elements)})"
        )
