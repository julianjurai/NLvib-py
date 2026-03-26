"""
System with polynomial stiffness nonlinearity for NLvib.

Provides :class:`System_with_PolynomialStiffness`, a multi-DOF mechanical
system whose nonlinear restoring force is expressed as a sum of polynomial
monomials in the displacements.

This corresponds to the MATLAB class ``System_with_PolynomialStiffnessNonlinearity``
from the NLvib toolbox.

References
----------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  ISBN 978-3-030-14022-9.  Appendix C, Table C.1.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from nlvib.nonlinearities.elements import polynomial_stiffness
from nlvib.systems.base import MechanicalSystem

__all__ = ["System_with_PolynomialStiffness"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


class System_with_PolynomialStiffness(MechanicalSystem):
    """Mechanical system with a polynomial stiffness nonlinearity.

    Models the equation of motion

    .. math::

        M \\ddot{q} + D \\dot{q} + K q + f_{\\mathrm{nl}}(q) = f_{\\mathrm{ext}}(t)

    where the nonlinear restoring force is the polynomial

    .. math::

        f_{\\mathrm{nl}}(q) =
            \\sum_{m=1}^{M} c_m \\prod_{j=1}^{n_{\\mathrm{dof}}} q_j^{e_{m,j}}

    with monomial exponents :math:`e_{m,j}` and scalar coefficients
    :math:`c_m`.  This corresponds to ``System_with_PolynomialStiffnessNonlinearity``
    in the MATLAB NLvib toolbox and Krack & Gross (2019) Appendix C, Table C.1.

    Parameters
    ----------
    M:
        Mass matrix of shape ``(n_dof, n_dof)``.  Accepted as a dense NumPy
        array or any SciPy sparse matrix.
    D:
        Damping matrix of shape ``(n_dof, n_dof)``.  Same format rules as *M*.
    K:
        Stiffness matrix of shape ``(n_dof, n_dof)``.  Same format rules as *M*.
    exponents:
        Integer array of shape ``(n_terms, n_dof)`` — each row is the exponent
        vector for one monomial term, with one column per DOF of the system.
    coefficients:
        Float array of shape ``(n_terms,)`` — scalar coefficient for each
        monomial term.

    Raises
    ------
    ValueError
        If *M*, *D*, *K* are not square matrices of the same size, or if
        *exponents* and *coefficients* have incompatible shapes, or if the
        number of columns in *exponents* does not equal ``n_dof``.

    Examples
    --------
    1-DOF cubic Duffing spring (k3 = 1e8):

    >>> import numpy as np
    >>> from nlvib.systems.polynomial import System_with_PolynomialStiffness
    >>> sys = System_with_PolynomialStiffness(
    ...     M=np.array([[1.0]]),
    ...     D=np.array([[0.0]]),
    ...     K=np.array([[1e4]]),
    ...     exponents=np.array([[3]]),
    ...     coefficients=np.array([1e8]),
    ... )
    >>> f, _, _ = sys.eval_nonlinear_forces(np.array([1.5]), np.zeros(1))
    >>> float(f[0])  # doctest: +ELLIPSIS
    337500000.0...
    """

    def __init__(
        self,
        M: FloatArray | sp.spmatrix,
        D: FloatArray | sp.spmatrix,
        K: FloatArray | sp.spmatrix,
        exponents: IntArray | NDArray[np.int_],
        coefficients: FloatArray | NDArray[np.float64],
    ) -> None:
        super().__init__(M, D, K)

        exponents_arr = np.asarray(exponents, dtype=np.intp)
        coefficients_arr = np.asarray(coefficients, dtype=np.float64)

        # Validate exponent dimensions vs. system size
        if exponents_arr.ndim != 2:
            raise ValueError(
                f"exponents must be a 2-D array of shape (n_terms, n_dof); "
                f"got shape {exponents_arr.shape}."
            )

        n_terms, n_exp_dofs = exponents_arr.shape

        if n_exp_dofs != self.n_dof:
            raise ValueError(
                f"exponents has {n_exp_dofs} columns but system has "
                f"{self.n_dof} DOFs — they must match."
            )

        if coefficients_arr.shape != (n_terms,):
            raise ValueError(
                f"coefficients must have shape ({n_terms},); "
                f"got {coefficients_arr.shape}."
            )

        # All DOFs participate: indices 0 … n_dof-1
        dof_indices: IntArray = np.arange(self.n_dof, dtype=np.intp)

        element = polynomial_stiffness(
            exponents=exponents_arr,
            coefficients=coefficients_arr,
            dof_indices=dof_indices,
        )
        self.add_nonlinear_element(element)

    def __repr__(self) -> str:
        return (
            f"System_with_PolynomialStiffness("
            f"n_dof={self.n_dof}, "
            f"n_elements={len(self.nonlinear_elements)})"
        )
