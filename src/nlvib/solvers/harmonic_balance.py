"""
Harmonic Balance (HB) residual and Alternating Frequency-Time (AFT) for NLvib.

Implements the HB residual for forced-response analysis (FRF) and for nonlinear
modal analysis (NMA, backbone curves).

Layout convention
-----------------
For ``n_dof`` degrees of freedom and ``H`` retained harmonics the real-valued
Fourier coefficient vector is laid out as:

    Q = [Q_0, Q_c1, Q_s1, Q_c2, Q_s2, ..., Q_cH, Q_sH]

where each block ``Q_h`` has length ``n_dof``, so the total length is
``n_dof * (2*H + 1)``.

Within each block ``Q_h`` the entries correspond to DOF 0, 1, ..., n_dof-1.

Equation references
-------------------
Krack, M. & Gross, J. (2019). *Harmonic Balance for Nonlinear Vibration
Problems*. Springer.  §2, §3, Appendix C.

The HB residual for harmonic index *h* is (K&G §2, eq. 2.2):

    R_h = Z_h * Q_h + F_nl_h - F_ext_h = 0

where the dynamic stiffness matrix at frequency *h·ω* is:

    Z_h = -(h·ω)²·M + K               (h = 0, cosine block)
    Z_h = -(h·ω)²·M + K  (cos)
          -(h·ω)·D        (off-diagonal coupling of cos↔sin blocks)

For the real-valued cos/sin representation the 2×2 harmonic block of Z_h for
h ≥ 1 acting on [Q_ch, Q_sh]ᵀ is (K&G §2.2):

    Z_h = [  -(hω)²M + K    hω·D  ]
          [  -hω·D          -(hω)²M + K  ]

(K&G Appendix C, eq. C.3 — sign: +hω·D in the upper-right because d/dt of
cos(hωt) = -hω sin(hωt) and D acts on velocity.)
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from nlvib.systems.base import MechanicalSystem
from nlvib.utils.transforms import freq_to_time, time_to_freq

__all__ = ["hb_residual", "hb_residual_nma"]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatArray = npt.NDArray[np.float64]

# Excitation specification type: dict with keys "dof" (int) and "amplitude"
# (float), or a pre-built coefficient vector of shape (n_dof*(2H+1),).
ExcitationType = Union[dict[str, Any], FloatArray]

# Number of time samples for the AFT intermediate representation.
# Must be >= 2*H+1; power of 2 is optimal for FFT performance.
# NLvib (MATLAB) default: n_time = 2^ceil(log2(2*H+3)) — next power of 2
# above the Nyquist minimum.  We use a fixed over-sampling factor of 8.
_AFT_OVERSAMPLING: int = 8

# Step size for the finite-difference Jacobian of the nonlinear force
# coefficients (used only when the system has nonlinear elements).
# Central differences with h = sqrt(ε_machine) ≈ 1.5e-8 gives ~1e-6 accuracy.
_FD_STEP: float = float(np.sqrt(np.finfo(float).eps))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _n_time_from_harmonics(n_harmonics: int) -> int:
    """Return the next power-of-2 number of time samples for the given H.

    Equation reference: Nyquist criterion requires n_time ≥ 2*H+1.
    We over-sample by ``_AFT_OVERSAMPLING`` and round to next power of 2 for
    FFT efficiency.
    """
    minimum = _AFT_OVERSAMPLING * (2 * n_harmonics + 1)
    return int(2 ** np.ceil(np.log2(minimum)))


def _block_indices(n_dof: int, n_harmonics: int) -> list[npt.NDArray[np.intp]]:
    """Return a list of index arrays for each harmonic block in Q.

    Returns
    -------
    list of length ``2*n_harmonics + 1``
        Entry 0: indices of the DC block (length n_dof).
        Entries 2h-1, 2h: cosine and sine block indices for harmonic h.
    """
    n_blocks = 2 * n_harmonics + 1
    return [
        np.arange(b * n_dof, (b + 1) * n_dof, dtype=np.intp)
        for b in range(n_blocks)
    ]


def _build_excitation_vector(
    excitation: ExcitationType,
    n_dof: int,
    n_harmonics: int,
) -> FloatArray:
    """Convert an excitation specification to the full F_ext coefficient vector.

    Parameters
    ----------
    excitation:
        Either:

        - A ``dict`` with keys ``"dof"`` (int, zero-based), ``"amplitude"``
          (float), and optionally ``"harmonic"`` (int, default 1).  The force
          is applied as a pure cosine at the specified harmonic.
        - A pre-built coefficient vector of shape ``(n_dof*(2*H+1),)`` — used
          as-is.

    n_dof:
        Number of degrees of freedom.
    n_harmonics:
        Number of retained harmonics H.

    Returns
    -------
    FloatArray, shape ``(n_dof * (2*H+1),)``
        Excitation coefficient vector F_ext.
    """
    n_total = n_dof * (2 * n_harmonics + 1)

    if isinstance(excitation, dict):
        dof = int(excitation["dof"])
        amplitude = float(excitation["amplitude"])
        harmonic = int(excitation.get("harmonic", 1))

        F_ext = np.zeros(n_total, dtype=np.float64)
        if harmonic == 0:
            # DC excitation: block 0, DOF `dof`
            F_ext[dof] = amplitude
        else:
            # Cosine block for harmonic h: block index = 2*h - 1
            cosine_block_start = (2 * harmonic - 1) * n_dof
            F_ext[cosine_block_start + dof] = amplitude
        return F_ext
    else:
        F_arr = np.asarray(excitation, dtype=np.float64)
        if F_arr.shape != (n_total,):
            raise ValueError(
                f"Pre-built excitation vector must have shape ({n_total},); "
                f"got {F_arr.shape}."
            )
        return F_arr.copy()


def _linear_residual(
    Q: FloatArray,
    omega: float,
    M: sp.csr_matrix,
    D: sp.csr_matrix,
    K: sp.csr_matrix,
    n_dof: int,
    n_harmonics: int,
) -> FloatArray:
    """Compute the linear part of the HB residual: Z * Q.

    For the real cosine/sine layout the dynamic stiffness acts block-by-block:

    - h = 0 (DC):  Z_0 * Q_0 = K * Q_0
    - h ≥ 1:       [Z_cc  Z_cs] [Q_ch]   [(-(hω)²M + K)  hω·D ] [Q_ch]
                   [Z_sc  Z_ss] [Q_sh] = [-hω·D        -(hω)²M+K] [Q_sh]

    Reference: Krack & Gross (2019) §2.2, eq. (2.7); Appendix C eq. (C.3).

    Parameters
    ----------
    Q:
        Fourier coefficient vector, shape ``(n_dof * (2*H+1),)``.
    omega:
        Fundamental excitation frequency.
    M, D, K:
        System matrices (CSR sparse).
    n_dof:
        Number of DOFs.
    n_harmonics:
        Number of harmonics H.

    Returns
    -------
    FloatArray, shape ``(n_dof * (2*H+1),)``
    """
    n_total = n_dof * (2 * n_harmonics + 1)
    R_lin = np.zeros(n_total, dtype=np.float64)

    M_dense: FloatArray = M.toarray()
    D_dense: FloatArray = D.toarray()
    K_dense: FloatArray = K.toarray()

    # DC block: h = 0, Z_0 = K
    idx0 = np.arange(n_dof, dtype=np.intp)
    R_lin[idx0] = K_dense @ Q[idx0]

    # Higher harmonic blocks: h = 1, ..., H
    for h in range(1, n_harmonics + 1):
        h_omega = h * omega
        # Indices of cosine and sine blocks
        ic = np.arange((2 * h - 1) * n_dof, 2 * h * n_dof, dtype=np.intp)
        is_ = np.arange(2 * h * n_dof, (2 * h + 1) * n_dof, dtype=np.intp)

        Q_c = Q[ic]
        Q_s = Q[is_]

        # K&G §2.2 eq. (2.7): 2x2 block structure
        A = -(h_omega**2) * M_dense + K_dense  # Z_cc = Z_ss
        B = h_omega * D_dense                   # Z_cs = -Z_sc

        R_lin[ic] = A @ Q_c + B @ Q_s
        R_lin[is_] = -B @ Q_c + A @ Q_s

    return R_lin


def _linear_jacobian(
    omega: float,
    M: sp.csr_matrix,
    D: sp.csr_matrix,
    K: sp.csr_matrix,
    n_dof: int,
    n_harmonics: int,
) -> FloatArray:
    """Build the analytical Jacobian of the linear HB residual dR_lin/dQ.

    The Jacobian is block-diagonal: each harmonic contributes one (or two for
    h≥1) ``n_dof × n_dof`` blocks.

    Reference: Krack & Gross (2019) §2.2, eq. (2.7); Appendix C eq. (C.3).

    Returns
    -------
    FloatArray, shape ``(n_dof*(2H+1), n_dof*(2H+1))``
    """
    n_total = n_dof * (2 * n_harmonics + 1)
    J_lin = np.zeros((n_total, n_total), dtype=np.float64)

    M_dense: FloatArray = M.toarray()
    D_dense: FloatArray = D.toarray()
    K_dense: FloatArray = K.toarray()

    # DC block: dR_0/dQ_0 = K
    idx0 = np.arange(n_dof, dtype=np.intp)
    J_lin[np.ix_(idx0, idx0)] = K_dense

    # Higher harmonic blocks
    for h in range(1, n_harmonics + 1):
        h_omega = h * omega
        ic = np.arange((2 * h - 1) * n_dof, 2 * h * n_dof, dtype=np.intp)
        is_ = np.arange(2 * h * n_dof, (2 * h + 1) * n_dof, dtype=np.intp)

        A = -(h_omega**2) * M_dense + K_dense
        B = h_omega * D_dense

        J_lin[np.ix_(ic, ic)] = A
        J_lin[np.ix_(ic, is_)] = B
        J_lin[np.ix_(is_, ic)] = -B
        J_lin[np.ix_(is_, is_)] = A

    return J_lin


def _build_nl_force_fn_with_vel(
    system: MechanicalSystem,
    Q_freq: FloatArray,
    omega: float,
    n_harmonics: int,
    n_time: int,
) -> "tuple[FloatArray, FloatArray]":
    """Evaluate nonlinear forces via AFT and return (F_nl_coeffs, dF_nl_dQ).

    Computes the nonlinear force Fourier coefficients F_nl and their Jacobian
    dF_nl/dQ using:
    1. The AFT pipeline for the force vector itself.
    2. Central finite differences in the frequency domain for the Jacobian.

    The velocity time series is derived analytically from the displacement
    coefficients: for each harmonic h,

        dq_h(t) = -h·ω·Q_sh·cos(h·ω·t) + h·ω·Q_ch·sin(h·ω·t)

    which in coefficient form means:
        dQ_{h,c} = +h·ω·Q_{h,s}
        dQ_{h,s} = -h·ω·Q_{h,c}

    Reference: Krack & Gross (2019) §2.1, derivative of Fourier series.

    Parameters
    ----------
    system:
        MechanicalSystem with nonlinear elements registered.
    Q_freq:
        Full Fourier coefficient vector, shape ``(n_dof*(2H+1),)``.
    omega:
        Fundamental excitation frequency.
    n_harmonics:
        Number of harmonics H.
    n_time:
        Number of time samples for AFT.

    Returns
    -------
    F_nl : FloatArray, shape ``(n_dof*(2H+1),)``
        Nonlinear force Fourier coefficients.
    dF_nl_dQ : FloatArray, shape ``(n_dof*(2H+1), n_dof*(2H+1))``
        Jacobian of F_nl w.r.t. Q, computed by central finite differences.
    """
    n_dof = system.n_dof
    n_total = n_dof * (2 * n_harmonics + 1)

    def _eval_fnl(Q_vec: FloatArray) -> FloatArray:
        """Evaluate F_nl Fourier coefficients for a given Q vector."""
        # Reshape Q into (n_dof, 2H+1) block layout for freq_to_time
        Q_mat = Q_vec.reshape(2 * n_harmonics + 1, n_dof).T  # (n_dof, 2H+1)

        # Displacement time series: shape (n_dof, n_time)
        q_time = freq_to_time(Q_mat, n_time)  # (n_dof, n_time)

        # Velocity time series from differentiation of Fourier series (K&G §2.1)
        # dq_mat[h] = velocity coefficients for harmonic h
        dQ_mat = np.zeros_like(Q_mat)  # (n_dof, 2H+1)
        # DC velocity coefficient = 0 (dQ_mat[:, 0] = 0 by default)
        for h in range(1, n_harmonics + 1):
            h_omega = h * omega
            # Cosine coeff index: 2h-1, Sine coeff index: 2h
            c_idx = 2 * h - 1
            s_idx = 2 * h
            # d/dt [a_h cos(hωt) + b_h sin(hωt)]
            # = -h·ω·a_h sin(hωt) + h·ω·b_h cos(hωt)
            # → new cosine coeff = h·ω·b_h, new sine coeff = -h·ω·a_h
            dQ_mat[:, c_idx] = h_omega * Q_mat[:, s_idx]   # +h·ω·b_h
            dQ_mat[:, s_idx] = -h_omega * Q_mat[:, c_idx]  # -h·ω·a_h

        dq_time = freq_to_time(dQ_mat, n_time)  # (n_dof, n_time)

        # Evaluate nonlinear forces at each time point (vectorised via eval_batch)
        f_nl_time = system.eval_nonlinear_forces_batch(q_time, dq_time)

        # Transform back to frequency domain
        F_nl_mat = time_to_freq(f_nl_time, n_harmonics)  # (n_dof, 2H+1)

        # Flatten: row-major from (n_dof, 2H+1) block layout
        # Q layout: [block0_dof0..n, block1_dof0..n, ...], i.e. column-major of Q_mat
        return F_nl_mat.T.ravel().copy()  # shape (n_total,)

    # Evaluate nominal nonlinear force
    F_nl = _eval_fnl(Q_freq)

    # Jacobian by central finite differences (K&G Appendix C)
    dF_nl_dQ = np.zeros((n_total, n_total), dtype=np.float64)
    h_fd = _FD_STEP * np.maximum(np.abs(Q_freq), 1.0)

    # Use vectorised column-by-column central differences
    for j in range(n_total):
        Q_plus = Q_freq.copy()
        Q_minus = Q_freq.copy()
        Q_plus[j] += h_fd[j]
        Q_minus[j] -= h_fd[j]
        dF_nl_dQ[:, j] = (_eval_fnl(Q_plus) - _eval_fnl(Q_minus)) / (2.0 * h_fd[j])

    return F_nl, dF_nl_dQ


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hb_residual(
    Q: FloatArray,
    omega: float,
    system: MechanicalSystem,
    n_harmonics: int,
    excitation: ExcitationType,
    n_time: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Harmonic Balance residual for forced-response analysis.

    Evaluates the HB residual vector **R** and its Jacobian **J** = ∂R/∂Q
    for a mechanical system subject to periodic external forcing at frequency
    *ω*.

    The residual is (Krack & Gross 2019 §2, eq. 2.2):

        R = Z(ω)·Q + F_nl(Q) - F_ext = 0

    where

    - **Z(ω)·Q** is the linear dynamic stiffness contribution (computed
      analytically),
    - **F_nl(Q)** are the nonlinear force Fourier coefficients obtained via
      the AFT (K&G §2.3 eqs. 2.14–2.16),
    - **F_ext** is the external forcing coefficient vector.

    The Jacobian **J** is:

        J = dZ/dQ + dF_nl/dQ = Z + dF_nl/dQ

    where *dZ/dQ = Z* (linear part, analytical) and *dF_nl/dQ* is computed by
    central finite differences in the frequency domain.

    Parameters
    ----------
    Q:
        Real-valued Fourier coefficient vector, shape
        ``(n_dof * (2*n_harmonics + 1),)``.
        Layout: ``[Q_0, Q_c1, Q_s1, ..., Q_cH, Q_sH]`` where each block has
        length ``n_dof``.
    omega:
        Fundamental excitation frequency [rad/s].
    system:
        :class:`~nlvib.systems.base.MechanicalSystem` instance with M, D, K
        and registered nonlinear elements.
    n_harmonics:
        Number of harmonics *H* to retain (not counting the DC component).
    excitation:
        External forcing specification.  Either a ``dict`` with keys
        ``"dof"`` (int), ``"amplitude"`` (float), and optionally
        ``"harmonic"`` (int, default 1 — first harmonic / fundamental), or
        a pre-built coefficient vector of shape ``(n_dof*(2H+1),)``.
    n_time:
        Number of time samples for the AFT intermediate representation.
        If *None*, defaults to the next power-of-2 above
        ``_AFT_OVERSAMPLING * (2*H+1)``.

    Returns
    -------
    R : FloatArray, shape ``(n_dof*(2H+1),)``
        Residual vector.
    J : FloatArray, shape ``(n_dof*(2H+1), n_dof*(2H+1))``
        Jacobian matrix ∂R/∂Q.

    Raises
    ------
    ValueError
        If *Q* has the wrong length for the given *system* and *n_harmonics*.

    Notes
    -----
    Equation references:

    - K&G §2.2 eq. (2.7): dynamic stiffness Z_h for each harmonic block.
    - K&G §2.3 eqs. (2.14)-(2.16): AFT for nonlinear forces.
    - K&G Appendix C eq. (C.3): cos/sin block structure of Z_h.
    """
    n_dof = system.n_dof
    n_total = n_dof * (2 * n_harmonics + 1)

    Q_arr = np.asarray(Q, dtype=np.float64)
    if Q_arr.shape != (n_total,):
        raise ValueError(
            f"Q must have shape ({n_total},) for n_dof={n_dof}, "
            f"n_harmonics={n_harmonics}; got {Q_arr.shape}."
        )

    if n_time is None:
        n_time = _n_time_from_harmonics(n_harmonics)

    # 1. Linear residual and Jacobian (analytical)
    R_lin = _linear_residual(
        Q_arr, omega, system.M, system.D, system.K, n_dof, n_harmonics
    )
    J_lin = _linear_jacobian(
        omega, system.M, system.D, system.K, n_dof, n_harmonics
    )

    # 2. External forcing vector
    F_ext = _build_excitation_vector(excitation, n_dof, n_harmonics)

    # 3. Nonlinear forces via AFT (if any nonlinear elements present)
    if system.nonlinear_elements:
        F_nl, dF_nl_dQ = _build_nl_force_fn_with_vel(
            system, Q_arr, omega, n_harmonics, n_time
        )
    else:
        F_nl = np.zeros(n_total, dtype=np.float64)
        dF_nl_dQ = np.zeros((n_total, n_total), dtype=np.float64)

    # 4. Assemble residual and Jacobian
    R: FloatArray = R_lin + F_nl - F_ext
    J: FloatArray = J_lin + dF_nl_dQ

    return R, J


def hb_residual_nma(
    Q_omega: FloatArray,
    system: MechanicalSystem,
    n_harmonics: int,
    n_time: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Harmonic Balance residual for Nonlinear Modal Analysis (NMA).

    Evaluates the HB residual for the autonomous (unforced) system augmented
    with a phase normalisation constraint.  This formulation is used for
    tracing backbone curves in the frequency–amplitude plane.

    The augmented residual is (Krack & Gross 2019 §3, eq. 3.7):

        R_nma = [Z(ω)·Q + F_nl(Q)]           ← n_dof*(2H+1) equations
                [φ(Q) - φ_ref         ]        ← 1 phase constraint

    The phase constraint

        φ(Q) = Q_{c1, ref_dof} = 0

    pins the cosine coefficient of the fundamental harmonic at the reference
    DOF (DOF 0) to zero, removing the arbitrary phase of the autonomous
    solution (K&G §3.1).

    The augmented state vector is:

        [Q; ω]   (length n_dof*(2H+1) + 1)

    Parameters
    ----------
    Q_omega:
        Augmented state vector ``[Q; omega]``, shape
        ``(n_dof*(2H+1) + 1,)``.  The last entry is the frequency ω
        (continuation parameter).
    system:
        :class:`~nlvib.systems.base.MechanicalSystem` with M, D, K and
        nonlinear elements.
    n_harmonics:
        Number of harmonics *H*.
    n_time:
        Number of time samples for AFT.  Defaults to next power-of-2 above
        ``_AFT_OVERSAMPLING * (2*H+1)``.

    Returns
    -------
    R : FloatArray, shape ``(n_dof*(2H+1) + 1,)``
        Augmented residual vector.
    J : FloatArray, shape ``(n_dof*(2H+1) + 1, n_dof*(2H+1) + 1)``
        Jacobian ∂R/∂[Q; ω].

    Raises
    ------
    ValueError
        If *Q_omega* has the wrong length.

    Notes
    -----
    Equation references:

    - K&G §3, eq. (3.7): augmented residual for NMA.
    - K&G §3.1: phase normalisation via pinning one Fourier coefficient.
    - K&G §2.2 eq. (2.7): dynamic stiffness blocks.
    - K&G §2.3 eqs. (2.14)-(2.16): AFT.
    """
    n_dof = system.n_dof
    n_total = n_dof * (2 * n_harmonics + 1)
    n_aug = n_total + 1  # augmented length including omega

    Q_omega_arr = np.asarray(Q_omega, dtype=np.float64)
    if Q_omega_arr.shape != (n_aug,):
        raise ValueError(
            f"Q_omega must have shape ({n_aug},) for n_dof={n_dof}, "
            f"n_harmonics={n_harmonics}; got {Q_omega_arr.shape}."
        )

    Q_arr = Q_omega_arr[:n_total]
    omega = float(Q_omega_arr[n_total])

    if n_time is None:
        n_time = _n_time_from_harmonics(n_harmonics)

    # 1. Linear residual and Jacobian w.r.t. Q (analytical)
    R_lin = _linear_residual(
        Q_arr, omega, system.M, system.D, system.K, n_dof, n_harmonics
    )
    J_lin_Q = _linear_jacobian(
        omega, system.M, system.D, system.K, n_dof, n_harmonics
    )

    # 2. Jacobian of R_lin w.r.t. omega: dZ(omega)/domega * Q
    # dZ_h/domega = -2h²ω·M (for both cos and sin blocks) and
    # coupling term: dB_h/domega = h·D (off-diagonal).
    # Computed by central FD in omega for robustness.
    h_fd_om = _FD_STEP * max(abs(omega), 1.0)
    R_lin_p = _linear_residual(
        Q_arr, omega + h_fd_om, system.M, system.D, system.K, n_dof, n_harmonics
    )
    R_lin_m = _linear_residual(
        Q_arr, omega - h_fd_om, system.M, system.D, system.K, n_dof, n_harmonics
    )
    dR_lin_domega: FloatArray = (R_lin_p - R_lin_m) / (2.0 * h_fd_om)

    # 3. Nonlinear forces (no external forcing for NMA)
    if system.nonlinear_elements:
        F_nl, dF_nl_dQ = _build_nl_force_fn_with_vel(
            system, Q_arr, omega, n_harmonics, n_time
        )
    else:
        F_nl = np.zeros(n_total, dtype=np.float64)
        dF_nl_dQ = np.zeros((n_total, n_total), dtype=np.float64)

    # dF_nl/domega: nonlinear forces depend on omega through the velocity
    # time series (dq_time depends on omega).  Compute by FD.
    if system.nonlinear_elements:
        F_nl_p, _ = _build_nl_force_fn_with_vel(
            system, Q_arr, omega + h_fd_om, n_harmonics, n_time
        )
        F_nl_m, _ = _build_nl_force_fn_with_vel(
            system, Q_arr, omega - h_fd_om, n_harmonics, n_time
        )
        dF_nl_domega: FloatArray = (F_nl_p - F_nl_m) / (2.0 * h_fd_om)
    else:
        dF_nl_domega = np.zeros(n_total, dtype=np.float64)

    # 4. Phase constraint: Q_c1 at DOF 0 = 0
    # Index of the cosine coefficient of harmonic 1 at DOF 0:
    # Block 1 starts at n_dof*1 (since block 0 is DC, block 1 is cos_h1)
    phase_idx = n_dof  # = (2*1 - 1)*n_dof + 0

    # 5. Assemble augmented residual
    R_phys: FloatArray = R_lin + F_nl  # (n_total,)
    R_phase = Q_arr[phase_idx]  # scalar phase constraint
    R: FloatArray = np.append(R_phys, R_phase)

    # 6. Assemble augmented Jacobian (n_aug × n_aug)
    J: FloatArray = np.zeros((n_aug, n_aug), dtype=np.float64)

    # Physical rows, Q columns
    J[:n_total, :n_total] = J_lin_Q + dF_nl_dQ

    # Physical rows, omega column
    J[:n_total, n_total] = dR_lin_domega + dF_nl_domega

    # Phase constraint row, Q columns: d(Q_arr[phase_idx])/dQ = e_{phase_idx}
    J[n_total, phase_idx] = 1.0

    # Phase constraint row, omega column: phase constraint is independent of omega
    # J[n_total, n_total] = 0.0  (already zero)

    return R, J
