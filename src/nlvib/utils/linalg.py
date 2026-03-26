"""
Linear-algebra utilities for NLvib: variable scaling and arc-length metrics.

These support the arc-length continuation solver (Krack & Gross 2019, §4).

References
----------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*.
Springer.  §4.2 (variable scaling), §4.1 (continuation / arc-length step).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "dynamic_scaling",
    "arc_length",
]

_FloatArray = npt.NDArray[np.float64]


def dynamic_scaling(
    x: npt.ArrayLike,
    x_ref: npt.ArrayLike,
) -> _FloatArray:
    """Compute NLvib's dynamic (reference-based) scaling vector.

    In problems where the state vector ``x`` mixes quantities with very
    different magnitudes (e.g. displacements in μm and frequencies in kHz),
    the continuation step-size control and arc-length parametrisation can
    behave poorly without pre-scaling.  NLvib scales each component
    independently using a reference value (Krack & Gross 2019, §4.2):

        s_i = x_i / max(|x_ref_i|, ε)

    where ``ε = numpy.finfo(float).eps`` prevents division by zero.

    Parameters
    ----------
    x:
        State vector to be scaled, shape ``(n,)``.
    x_ref:
        Reference vector used to define the scale, shape ``(n,)``.  Typically
        the solution at the previous continuation point, or the initial guess.

    Returns
    -------
    NDArray[np.float64]
        Scaled vector ``s``, shape ``(n,)``, dimensionless.

    Notes
    -----
    Equation reference: Krack & Gross (2019) §4.2.

    The ``max(|x_ref_i|, ε)`` guard ensures that near-zero reference values do
    not blow up the scaled quantity.  This is equivalent to the MATLAB NLvib
    implementation which uses ``max(abs(x_ref), eps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``x_ref`` do not have the same shape.
    """
    x_arr: _FloatArray = np.asarray(x, dtype=np.float64)
    x_ref_arr: _FloatArray = np.asarray(x_ref, dtype=np.float64)

    if x_arr.shape != x_ref_arr.shape:
        raise ValueError(
            f"x and x_ref must have the same shape; "
            f"got {x_arr.shape} and {x_ref_arr.shape}."
        )

    eps = np.finfo(float).eps
    scale: _FloatArray = np.maximum(np.abs(x_ref_arr), eps)
    return x_arr / scale


def arc_length(
    x_prev: npt.ArrayLike,
    x_curr: npt.ArrayLike,
) -> float:
    """Euclidean arc-length (step size) between two continuation points.

    Computes the Euclidean distance between consecutive solution points on the
    continuation branch (Krack & Gross 2019, §4.1):

        Δs = ‖x_curr − x_prev‖₂

    Parameters
    ----------
    x_prev:
        State vector at the previous continuation point, shape ``(n,)``.
    x_curr:
        State vector at the current continuation point, shape ``(n,)``.

    Returns
    -------
    float
        Non-negative arc-length ``Δs ≥ 0``.

    Notes
    -----
    Equation reference: Krack & Gross (2019) §4.1, arc-length condition
    ``‖x_{k+1} − x_k‖ = Δs``.

    For continuation with mixed physical units, pre-scale ``x_prev`` and
    ``x_curr`` with :func:`dynamic_scaling` before calling this function.

    Raises
    ------
    ValueError
        If ``x_prev`` and ``x_curr`` do not have the same shape.
    """
    x_p: _FloatArray = np.asarray(x_prev, dtype=np.float64)
    x_c: _FloatArray = np.asarray(x_curr, dtype=np.float64)

    if x_p.shape != x_c.shape:
        raise ValueError(
            f"x_prev and x_curr must have the same shape; "
            f"got {x_p.shape} and {x_c.shape}."
        )

    return float(np.linalg.norm(x_c - x_p))
