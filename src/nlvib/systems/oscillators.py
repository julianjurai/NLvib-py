"""
Concrete oscillator system classes for NLvib.

This module provides ready-to-use mechanical system classes built on top of
:class:`~nlvib.systems.base.MechanicalSystem`.  Each class constructs the
appropriate linear system matrices and exposes the standard nonlinear element
API inherited from the base class.

Classes
-------
SingleMassOscillator
    1-DOF Duffing-type oscillator: :math:`m\\ddot{q} + d\\dot{q} + kq + f_{\\mathrm{nl}} = f_{\\mathrm{ext}}`.
ChainOfOscillators
    n-DOF chain of masses coupled by springs and dampers (K&G §5).

References
----------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  ISBN 978-3-030-14022-9.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from nlvib.systems.base import MechanicalSystem

__all__ = ["SingleMassOscillator", "ChainOfOscillators"]

FloatArray = NDArray[np.float64]


class SingleMassOscillator(MechanicalSystem):
    """Single-degree-of-freedom linear oscillator (Duffing base system).

    Represents the linear part of the equation of motion

    .. math::

        m \\ddot{q} + d \\dot{q} + k q + f_{\\mathrm{nl}}(q, \\dot{q})
        = f_{\\mathrm{ext}}(t)

    as described in Krack & Gross (2019) §5.1.  Nonlinear terms (e.g. a
    cubic spring that yields the Duffing oscillator) are attached via
    :meth:`~nlvib.systems.base.MechanicalSystem.add_nonlinear_element`.

    System matrices are all 1×1 :class:`~scipy.sparse.csr_matrix` objects:

    - :attr:`M` = ``[[m]]``
    - :attr:`D` = ``[[d]]``
    - :attr:`K` = ``[[k]]``

    Parameters
    ----------
    m:
        Mass [kg].  Must be strictly positive (``m > 0``).
    d:
        Viscous damping coefficient [N·s/m].  Must be non-negative
        (``d >= 0``).
    k:
        Linear stiffness [N/m].  Must be non-negative (``k >= 0``).

    Raises
    ------
    ValueError
        If ``m <= 0``, ``d < 0``, or ``k < 0``.

    Examples
    --------
    Construct a Duffing oscillator with :math:`k_3 = 0.5`:

    >>> from nlvib.systems.oscillators import SingleMassOscillator
    >>> from nlvib.nonlinearities.elements import cubic_spring
    >>> smo = SingleMassOscillator(m=1.0, d=0.02, k=1.0)
    >>> smo.add_nonlinear_element(cubic_spring(k3=0.5, dof_index=0))
    >>> smo.n_dof
    1
    """

    def __init__(self, m: float, d: float, k: float) -> None:
        if m <= 0.0:
            raise ValueError(f"Mass m must be strictly positive; got m={m!r}.")
        if d < 0.0:
            raise ValueError(
                f"Damping coefficient d must be non-negative; got d={d!r}."
            )
        if k < 0.0:
            raise ValueError(
                f"Stiffness k must be non-negative; got k={k!r}."
            )

        M_arr = np.array([[m]], dtype=np.float64)
        D_arr = np.array([[d]], dtype=np.float64)
        K_arr = np.array([[k]], dtype=np.float64)

        super().__init__(
            sp.csr_matrix(M_arr),
            sp.csr_matrix(D_arr),
            sp.csr_matrix(K_arr),
        )

        # Store scalar parameters for convenient access and repr.
        self._m: float = float(m)
        self._d: float = float(d)
        self._k: float = float(k)

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    @property
    def mass(self) -> float:
        """Scalar mass parameter *m* [kg]."""
        return self._m

    @property
    def damping(self) -> float:
        """Scalar damping parameter *d* [N·s/m]."""
        return self._d

    @property
    def stiffness(self) -> float:
        """Scalar linear stiffness parameter *k* [N/m]."""
        return self._k

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SingleMassOscillator("
            f"m={self._m}, d={self._d}, k={self._k}, "
            f"n_elements={len(self._nonlinear_elements)})"
        )


# ---------------------------------------------------------------------------
# Private matrix builders (no Python loops — pure scipy.sparse.diags)
# ---------------------------------------------------------------------------


def _build_mass_matrix(m: FloatArray) -> sp.csr_matrix:
    """Build a diagonal mass matrix from an array of scalar masses.

    Parameters
    ----------
    m:
        1-D array of mass values, length *n*.

    Returns
    -------
    scipy.sparse.csr_matrix
        Diagonal sparse mass matrix of shape ``(n, n)``.
    """
    return sp.diags(m, offsets=0, shape=(m.size, m.size), format="csr")


def _build_tridiagonal_matrix(k: FloatArray) -> sp.csr_matrix:
    r"""Build a symmetric tridiagonal matrix from n+1 spring/damper coefficients.

    Given coefficients ``k[0], k[1], ..., k[n]`` (length *n*+1), the
    resulting *n*×*n* tridiagonal matrix has:

    .. math::

        A_{ii}     &= k_i + k_{i+1}, \quad 0 \le i \le n-1 \\
        A_{i,i+1}  &= A_{i+1,i} = -k_{i+1}, \quad 0 \le i \le n-2

    This corresponds to the chain-of-oscillators stiffness assembly
    in K&G (2019) §5.  The computation uses :func:`scipy.sparse.diags`
    exclusively — no Python loops.

    Parameters
    ----------
    k:
        1-D array of length *n*+1 (spring or damper coefficients).

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric tridiagonal sparse matrix of shape ``(n, n)``.
    """
    n = k.size - 1  # number of DOFs

    # Diagonal: k[i] + k[i+1]  for  i = 0, ..., n-1
    diag_main: FloatArray = k[:n] + k[1:]  # shape (n,)

    # Off-diagonals: -k[i+1]  for  i = 0, ..., n-2
    diag_off: FloatArray = -k[1:n]  # shape (n-1,)

    return sp.diags(
        [diag_off, diag_main, diag_off],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )


class ChainOfOscillators(MechanicalSystem):
    r"""Chain of *n* coupled mass–spring–damper oscillators.

    Models the system shown in K&G (2019) §5, which consists of *n* masses
    connected in series by springs and dampers, with optional boundary springs
    at each end.

    **Spring/damper index convention** (*n*+1 entries each):

    .. code-block::

        Ground --[k0]-- m0 --[k1]-- m1 -- ... --[k_{n-1}]-- m_{n-1} --[k_n]-- Ground

    * ``stiffnesses[0]``: spring between left ground and mass 0.
    * ``stiffnesses[i]`` (1 ≤ i ≤ n−1): inter-mass spring between mass i−1
      and mass i.
    * ``stiffnesses[n]``: spring between mass n−1 and right ground.

    A zero entry means the corresponding spring/damper is absent.

    **Stiffness matrix** (symmetric tridiagonal, K&G §5):

    .. math::

        K_{ii}     &= k_i + k_{i+1}, \quad 0 \le i \le n-1 \\
        K_{i,i+1}  &= K_{i+1,i} = -k_{i+1}, \quad 0 \le i \le n-2

    The same pattern applies to the damping matrix *D*.
    The mass matrix *M* is diagonal: :math:`M_{ii} = m_i`.

    Parameters
    ----------
    masses:
        Array-like of length *n* with mass values (kg).  All values must be
        strictly positive.
    stiffnesses:
        Array-like of length *n*+1 with stiffness values (N/m).
    dampings:
        Array-like of length *n*+1 with damping coefficients (N·s/m).

    Raises
    ------
    ValueError
        If ``len(stiffnesses)`` or ``len(dampings)`` is not ``len(masses)+1``.
    ValueError
        If any mass value is non-positive.

    Examples
    --------
    Two-DOF chain, left and inter-mass springs only:

    >>> sys = ChainOfOscillators(
    ...     masses=[1.0, 1.0],
    ...     stiffnesses=[1.0, 0.5, 0.0],
    ...     dampings=[0.0, 0.0, 0.0],
    ... )
    >>> sys.n_dof
    2
    >>> sys.K.toarray()
    array([[1.5, -0.5],
           [-0.5,  0.5]])
    """

    def __init__(
        self,
        masses: ArrayLike,
        stiffnesses: ArrayLike,
        dampings: ArrayLike,
    ) -> None:
        m_arr: FloatArray = np.asarray(masses, dtype=np.float64).ravel()
        k_arr: FloatArray = np.asarray(stiffnesses, dtype=np.float64).ravel()
        d_arr: FloatArray = np.asarray(dampings, dtype=np.float64).ravel()

        n = m_arr.size

        if k_arr.size != n + 1:
            raise ValueError(
                f"stiffnesses must have length n+1={n + 1}; got {k_arr.size}."
            )
        if d_arr.size != n + 1:
            raise ValueError(
                f"dampings must have length n+1={n + 1}; got {d_arr.size}."
            )
        if np.any(m_arr <= 0.0):
            raise ValueError("All mass values must be strictly positive.")

        M: sp.csr_matrix = _build_mass_matrix(m_arr)
        K: sp.csr_matrix = _build_tridiagonal_matrix(k_arr)
        D: sp.csr_matrix = _build_tridiagonal_matrix(d_arr)

        super().__init__(M, D, K)

        # Store parameter arrays for introspection.
        self._masses: FloatArray = m_arr
        self._stiffnesses: FloatArray = k_arr
        self._dampings: FloatArray = d_arr

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    @property
    def masses(self) -> FloatArray:
        """Array of mass values (kg), length *n*."""
        return self._masses.copy()

    @property
    def stiffnesses(self) -> FloatArray:
        """Array of stiffness values (N/m), length *n*+1."""
        return self._stiffnesses.copy()

    @property
    def dampings(self) -> FloatArray:
        """Array of damping values (N·s/m), length *n*+1."""
        return self._dampings.copy()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChainOfOscillators("
            f"n_dof={self.n_dof}, "
            f"n_elements={len(self._nonlinear_elements)})"
        )
