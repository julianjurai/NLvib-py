"""
Fourier / AFT transforms for the NLvib harmonic balance method.

Convention
----------
The real-valued Fourier representation used throughout NLvib (Krack & Gross 2019)
stores coefficients in the following order per DOF:

    [a_0, a_1, b_1, a_2, b_2, ..., a_H, b_H]

where q(t) = a_0 + Σ_{h=1}^{H} (a_h cos(h Ω t) + b_h sin(h Ω t)).

FFT convention
--------------
numpy.fft uses the sign convention  X[k] = Σ x[n] exp(-2πi k n / N).
The inverse is                        x[n] = (1/N) Σ X[k] exp(+2πi k n / N).

For the cosine/sine decomposition this means:
    a_0  = real(X[0]) / N
    a_h  = 2 * real(X[h]) / N      h = 1, …, H
    b_h  = -2 * imag(X[h]) / N     h = 1, …, H

References
----------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration Problems*.
Springer.  §2.3 (AFT), §4.2 (scaling).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    "time_to_freq",
    "freq_to_time",
    "aft_transform",
]

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_TWO = 2  # factor relating two-sided to one-sided FFT amplitudes

# Convenience alias used for all public signatures
_FloatArray = npt.NDArray[np.float64]


def time_to_freq(
    q_time: npt.ArrayLike,
    n_harmonics: int,
) -> _FloatArray:
    """Convert a uniformly-sampled time signal to real-valued cosine-sine coefficients.

    The output vector follows the NLvib convention (Krack & Gross 2019, §2.3):

        Q = [a_0, a_1, b_1, a_2, b_2, ..., a_H, b_H]   (length 2H+1 per DOF)

    where

        q(t) = a_0 + Σ_{h=1}^{H} (a_h cos(h Ω t) + b_h sin(h Ω t))

    Parameters
    ----------
    q_time:
        Real-valued time-domain signal. Shape ``(n_dof, n_time)`` or ``(n_time,)``
        for a single DOF. The number of time samples ``n_time`` must be at least
        ``2 * n_harmonics + 1`` and should be a power of 2 for FFT efficiency.
    n_harmonics:
        Number of harmonics *H* to retain (not including the zero-th harmonic).

    Returns
    -------
    NDArray[np.float64]
        Real-valued coefficient array of shape ``(n_dof, 2*n_harmonics+1)`` or
        ``(2*n_harmonics+1,)`` for a single DOF.

    Notes
    -----
    Equation reference: Krack & Gross (2019) §2.3, eq. (2.3)-(2.4).
    Sign convention: a_h = 2·Re(X[h])/N,  b_h = -2·Im(X[h])/N.

    Raises
    ------
    ValueError
        If ``n_time < 2 * n_harmonics + 1`` or ``q_time`` has wrong dimensionality.
    """
    scalar_input = np.ndim(q_time) == 1
    q: _FloatArray = np.atleast_2d(np.asarray(q_time, dtype=np.float64))
    n_dof, n_time = q.shape

    if n_time < _TWO * n_harmonics + 1:
        raise ValueError(
            f"n_time ({n_time}) must be >= 2*n_harmonics+1 ({2*n_harmonics+1})."
        )

    # One-sided FFT; shape (n_dof, n_time//2 + 1)
    X = np.fft.rfft(q, n=n_time, axis=1)

    n_coeffs = _TWO * n_harmonics + 1
    Q: _FloatArray = np.empty((n_dof, n_coeffs), dtype=np.float64)

    # Zero-th harmonic (DC)
    Q[:, 0] = X[:, 0].real / n_time

    # Higher harmonics: cosine (a_h) and sine (b_h) coefficients
    h_idx = np.arange(1, n_harmonics + 1)            # shape (H,)
    Q[:, 1::2] = (_TWO / n_time) * X[:, h_idx].real  # a_h columns
    Q[:, 2::2] = (-_TWO / n_time) * X[:, h_idx].imag  # b_h columns

    result: _FloatArray = Q[0] if scalar_input else Q
    return result


def freq_to_time(
    Q_freq: npt.ArrayLike,
    n_time_samples: int,
) -> _FloatArray:
    """Reconstruct a time-domain signal from real-valued cosine-sine coefficients.

    This is the inverse of :func:`time_to_freq`.  The input coefficient layout
    follows the NLvib convention (Krack & Gross 2019, §2.3):

        Q = [a_0, a_1, b_1, a_2, b_2, ..., a_H, b_H]

    and the reconstructed signal is

        q(t_k) = a_0 + Σ_{h=1}^{H} (a_h cos(2π h k/N) + b_h sin(2π h k/N))

    where N = ``n_time_samples`` and k = 0, 1, …, N-1.

    Parameters
    ----------
    Q_freq:
        Real-valued coefficient array of shape ``(n_dof, 2*H+1)`` or ``(2*H+1,)``.
    n_time_samples:
        Number of time points *N* to reconstruct.

    Returns
    -------
    NDArray[np.float64]
        Time-domain array of shape ``(n_dof, n_time_samples)`` or
        ``(n_time_samples,)`` for a single DOF.

    Notes
    -----
    Equation reference: Krack & Gross (2019) §2.3, eq. (2.3).

    Raises
    ------
    ValueError
        If the coefficient array has an even length (must be 2H+1, odd).
    """
    scalar_input = np.ndim(Q_freq) == 1
    Q: _FloatArray = np.atleast_2d(np.asarray(Q_freq, dtype=np.float64))
    n_dof, n_coeffs = Q.shape

    if n_coeffs % 2 == 0:
        raise ValueError(
            f"Coefficient vector length must be odd (2H+1), got {n_coeffs}."
        )

    n_harmonics = (n_coeffs - 1) // 2

    # Build one-sided complex spectrum X of length n_time_samples//2 + 1
    n_rfft = n_time_samples // 2 + 1
    X = np.zeros((n_dof, n_rfft), dtype=np.complex128)

    # DC component: a_0 * N
    X[:, 0] = Q[:, 0] * n_time_samples

    # Harmonics — invert the forward relations:
    #   a_h = 2·Re(X[h])/N  →  Re(X[h]) = a_h * N / 2
    #   b_h = -2·Im(X[h])/N →  Im(X[h]) = -b_h * N / 2
    h_idx = np.arange(1, n_harmonics + 1)
    a_h = Q[:, 1::2]  # (n_dof, H)
    b_h = Q[:, 2::2]  # (n_dof, H)
    X[:, h_idx] = (n_time_samples / _TWO) * (a_h - 1j * b_h)

    q: _FloatArray = np.fft.irfft(X, n=n_time_samples, axis=1)

    result: _FloatArray = q[0] if scalar_input else q
    return result


def aft_transform(
    Q_freq: npt.ArrayLike,
    force_fn: Callable[[_FloatArray], _FloatArray],
    n_time: int,
) -> _FloatArray:
    """Full Alternating Frequency-Time (AFT) transform.

    Implements the three-step AFT pipeline (Krack & Gross 2019, §2.3):

    1. **Frequency → Time**: Map harmonic coefficients Q to the time domain.
    2. **Nonlinear evaluation**: Apply ``force_fn`` point-wise in time.
    3. **Time → Frequency**: Map the nonlinear force time series back to the
       frequency domain, retaining the same number of harmonics.

    Parameters
    ----------
    Q_freq:
        Real-valued harmonic coefficient array, shape ``(n_dof, 2*H+1)`` or
        ``(2*H+1,)``.  Layout: ``[a_0, a_1, b_1, …, a_H, b_H]`` per DOF.
    force_fn:
        Callable with signature ``force_fn(q_time) -> f_time``.
        Receives the time-domain displacement array of shape
        ``(n_dof, n_time)`` (or ``(n_time,)`` for a single DOF) and must
        return a force array of the **same shape**.  Must be purely vectorised —
        no internal Python loops over time steps.
    n_time:
        Number of time samples to use for the intermediate time-domain
        representation.  Should satisfy ``n_time >= 2*H+1``; a power of 2
        is recommended for FFT efficiency.

    Returns
    -------
    NDArray[np.float64]
        Harmonic coefficients of the nonlinear force, same shape as ``Q_freq``.

    Notes
    -----
    Equation reference: Krack & Gross (2019) §2.3, equations (2.14)-(2.16).

    The number of harmonics retained is inferred from ``Q_freq`` as
    ``H = (n_coeffs - 1) // 2``.

    Raises
    ------
    ValueError
        If ``n_time < 2*H+1`` or coefficient array has even length.
    """
    scalar_input = np.ndim(Q_freq) == 1
    Q: _FloatArray = np.atleast_2d(np.asarray(Q_freq, dtype=np.float64))
    _n_dof, n_coeffs = Q.shape

    if n_coeffs % 2 == 0:
        raise ValueError(
            f"Coefficient vector length must be odd (2H+1), got {n_coeffs}."
        )

    n_harmonics = (n_coeffs - 1) // 2

    if n_time < _TWO * n_harmonics + 1:
        raise ValueError(
            f"n_time ({n_time}) must be >= 2*H+1 = {2*n_harmonics+1}."
        )

    # Step 1 — Frequency → Time
    q_time = freq_to_time(Q, n_time)  # (n_dof, n_time)

    # Step 2 — Nonlinear evaluation in time domain
    # Step 3 — Time → Frequency
    if scalar_input:
        f_time = force_fn(q_time[0])  # pass (n_time,)
        return time_to_freq(f_time, n_harmonics)
    else:
        f_time = force_fn(q_time)     # (n_dof, n_time)
        return time_to_freq(f_time, n_harmonics)
