"""
Example 01 — Duffing Oscillator
================================

Demonstrates frequency-response analysis of a 1-DOF Duffing oscillator
using both Harmonic Balance (HB) and Shooting continuation.

System parameters (Krack & Gross 2019, §5.1):
    m  = 1   kg
    d  = 0.02 N·s/m
    k  = 1   N/m  (linear stiffness)
    k3 = 0.5 N/m³ (cubic stiffness)
    F  = 0.1 N    (harmonic forcing at DOF 0, fundamental harmonic)

Frequency sweep: ω ∈ [0.5, 1.5] rad/s
Harmonics retained: H = 5
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from nlvib.nonlinearities.elements import cubic_spring
from nlvib.systems.oscillators import SingleMassOscillator
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.solvers.shooting import shooting_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions
from nlvib.visualization.plots import (
    plot_frf,
    plot_harmonic_content,
    plot_time_series,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------

MASS: float = 1.0
DAMPING: float = 0.02
STIFFNESS: float = 1.0
K3: float = 0.5
FORCE_AMPLITUDE: float = 0.1
OMEGA_MIN: float = 0.5
OMEGA_MAX: float = 1.5
N_HARMONICS: int = 5

# Excitation spec for hb_residual: cosine forcing at fundamental harmonic, DOF 0
EXCITATION: dict[str, object] = {"dof": 0, "amplitude": FORCE_AMPLITUDE, "harmonic": 1}

# ---------------------------------------------------------------------------
# Visualization adapter
# ---------------------------------------------------------------------------


@dataclass
class FRFResult:
    """Minimal adapter exposing .omega, .amplitude, .stability for plot_frf."""

    omega: npt.NDArray[np.float64]
    amplitude: npt.NDArray[np.float64]
    stability: npt.NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Helper: build system
# ---------------------------------------------------------------------------


def build_system() -> SingleMassOscillator:
    """Construct the Duffing oscillator system (K&G §5.1)."""
    sys = SingleMassOscillator(m=MASS, d=DAMPING, k=STIFFNESS)
    sys.add_nonlinear_element(cubic_spring(k3=K3, dof_index=0))
    return sys


# ---------------------------------------------------------------------------
# Helper: initial guess for HB at low frequency
# ---------------------------------------------------------------------------


def _hb_initial_guess(omega0: float, n_dof: int, n_harmonics: int) -> npt.NDArray[np.float64]:
    """Approximate initial Q for the linear system at omega0.

    Uses the linear FRF amplitude as the cosine coefficient of the fundamental
    harmonic; all other coefficients are zero.
    """
    n_total = n_dof * (2 * n_harmonics + 1)
    Q0 = np.zeros(n_total, dtype=np.float64)
    # Linear amplitude: F / |-(ω²m) + k + i·ω·d|
    denom = abs(-(omega0**2) * MASS + STIFFNESS + 1j * omega0 * DAMPING)
    amp = FORCE_AMPLITUDE / denom if denom > 1e-12 else 0.0
    # Cosine coefficient of harmonic 1 at DOF 0: index = n_dof * (2*1 - 1) + 0 = n_dof
    Q0[n_dof] = amp
    return Q0


# ---------------------------------------------------------------------------
# HB continuation
# ---------------------------------------------------------------------------


def run_hb_continuation(
    system: SingleMassOscillator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Run HB arc-length continuation from ω_min to ω_max.

    Returns
    -------
    omega_hb : (n_pts,) array of frequencies
    amp_hb   : (n_pts,) array of fundamental-harmonic amplitudes at DOF 0
    stab_hb  : (n_pts,) boolean stability array
    """
    n_dof = system.n_dof
    omega0 = OMEGA_MIN
    Q0 = _hb_initial_guess(omega0, n_dof, N_HARMONICS)

    def residual_fn(
        Q: npt.NDArray[np.float64], lam: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return hb_residual(Q, lam, system, N_HARMONICS, EXCITATION)

    opts = ContinuationOptions(
        ds_initial=0.02,
        ds_min=1e-5,
        ds_max=0.1,
        max_steps=1000,
        max_newton_iter=20,
        newton_tol=1e-10,
        adapt_step=True,
        lambda_min=OMEGA_MIN - 0.05,
        lambda_max=OMEGA_MAX + 0.05,
    )

    solver = ContinuationSolver()
    result = solver.run(residual_fn, Q0, omega0, opts)

    print(f"  HB continuation: {result.n_steps} steps, "
          f"converged={result.converged}, message='{result.message}'")

    # Extract omega (last column) and Q coefficients (all but last)
    omega_arr = result.solutions[:, -1]
    Q_arr = result.solutions[:, :-1]

    # Fundamental harmonic amplitude at DOF 0:
    #   sqrt(Q_c1[0]^2 + Q_s1[0]^2)
    # Q layout: [DC(n_dof), cos1(n_dof), sin1(n_dof), cos2(n_dof), sin2(n_dof), ...]
    # cos1 block: indices n_dof to 2*n_dof-1  → DOF 0 is index n_dof
    # sin1 block: indices 2*n_dof to 3*n_dof-1 → DOF 0 is index 2*n_dof
    q_c1 = Q_arr[:, n_dof]       # cosine coeff of H1 at DOF 0
    q_s1 = Q_arr[:, 2 * n_dof]   # sine coeff of H1 at DOF 0
    amp_arr = np.sqrt(q_c1**2 + q_s1**2)

    # stability: True = stable in the solver convention means unstable_flag=False
    # plot_frf draws solid for stable (stability==True in the FRFResult)
    # The solver sets stability=True when BETWEEN fold points (unstable segment).
    # So we invert: stable_for_plot = NOT solver_unstable_flag
    stab_arr = ~result.stability

    return omega_arr, amp_arr, stab_arr


# ---------------------------------------------------------------------------
# Shooting continuation
# ---------------------------------------------------------------------------


def run_shooting_continuation(
    system: SingleMassOscillator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Run shooting arc-length continuation from ω_min to ω_max.

    Returns
    -------
    omega_sh  : (n_pts,) frequencies
    amp_sh    : (n_pts,) amplitude at DOF 0
    stab_sh   : (n_pts,) stability boolean
    """
    omega0 = OMEGA_MIN

    # Initial state: use linear steady-state amplitude at omega0
    denom = abs(-(omega0**2) * MASS + STIFFNESS + 1j * omega0 * DAMPING)
    lin_amp = FORCE_AMPLITUDE / denom if denom > 1e-12 else 0.0
    # Steady-state: q(t) ≈ A·cos(ω₀t + φ).  At t=0 with φ=0: q=A, dq=0
    y0 = np.array([lin_amp, 0.0], dtype=np.float64)

    # External forcing: F·cos(ω·t) at DOF 0
    def f_ext_fn(t: float) -> npt.NDArray[np.float64]:
        return np.array([FORCE_AMPLITUDE * np.cos(omega0 * t)], dtype=np.float64)

    # We need a closure that captures the current omega for the shooting residual.
    # The continuation solver passes (y0, omega) so we wrap shooting_residual:
    def residual_fn(
        y: npt.NDArray[np.float64], lam: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # Build f_ext_fn with the current frequency lam
        def _f(t: float) -> npt.NDArray[np.float64]:
            return np.array([FORCE_AMPLITUDE * np.cos(lam * t)], dtype=np.float64)

        return shooting_residual(
            y,
            lam,
            system,
            n_periods=1,
            n_steps=128,
            f_ext_fn=_f,
        )

    opts = ContinuationOptions(
        ds_initial=0.02,
        ds_min=1e-5,
        ds_max=0.1,
        max_steps=800,
        max_newton_iter=25,
        newton_tol=1e-8,
        adapt_step=True,
        lambda_min=OMEGA_MIN - 0.05,
        lambda_max=OMEGA_MAX + 0.05,
    )

    solver = ContinuationSolver()
    result = solver.run(residual_fn, y0, omega0, opts)

    print(f"  Shooting continuation: {result.n_steps} steps, "
          f"converged={result.converged}, message='{result.message}'")

    # solutions[:, -1] = omega, solutions[:, 0] = q, solutions[:, 1] = dq
    omega_arr = result.solutions[:, -1]
    # Amplitude = absolute displacement at DOF 0 (max over a period ≈ |q_0| for
    # a near-cosine steady state)
    amp_arr = np.abs(result.solutions[:, 0])
    stab_arr = ~result.stability  # invert: solver True = unstable

    return omega_arr, amp_arr, stab_arr


# ---------------------------------------------------------------------------
# Extract Q at peak amplitude for harmonic content / time series
# ---------------------------------------------------------------------------


def _get_peak_hb_solution(
    system: SingleMassOscillator,
    omega_hb: npt.NDArray[np.float64],
    amp_hb: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64]]:
    """Return (omega_peak, Q_peak) from the HB continuation result."""
    peak_idx = int(np.argmax(amp_hb))
    omega_peak = float(omega_hb[peak_idx])
    # Re-run a single HB residual evaluation at peak to get clean Q solution.
    # Since the continuation already solved it, re-use the stored Q:
    # We need to re-run continuation — instead just use the stored solutions from
    # a second pass is wasteful. Use the stored peak Q directly.
    return omega_peak, None  # placeholder — see caller


def _reconstruct_time_series_from_Q(
    Q: npt.NDArray[np.float64],
    omega: float,
    n_harmonics: int,
    n_time: int = 256,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reconstruct time-domain displacement from HB Fourier coefficients.

    Parameters
    ----------
    Q:
        Full HB coefficient vector, shape (n_dof*(2H+1),).
    omega:
        Excitation frequency [rad/s].
    n_harmonics:
        Number of harmonics H.
    n_time:
        Number of time samples per period.

    Returns
    -------
    t, q, dq  — shape (n_time,) each (for n_dof=1)
    """
    T = 2.0 * np.pi / omega
    t = np.linspace(0.0, T, n_time, endpoint=False)

    # n_dof=1 case
    n_dof = 1
    q = np.zeros(n_time, dtype=np.float64)
    dq = np.zeros(n_time, dtype=np.float64)

    # DC component
    q += Q[0]  # Q_0 is index 0 for n_dof=1

    for h in range(1, n_harmonics + 1):
        h_omega = h * omega
        # cosine coeff index: (2h-1)*n_dof + 0 = 2h-1
        # sine coeff index:   2h*n_dof + 0    = 2h
        c_idx = (2 * h - 1) * n_dof  # = 2h-1
        s_idx = 2 * h * n_dof         # = 2h
        Q_c = Q[c_idx]
        Q_s = Q[s_idx]
        q  += Q_c * np.cos(h_omega * t) + Q_s * np.sin(h_omega * t)
        dq += -h_omega * Q_c * np.sin(h_omega * t) + h_omega * Q_s * np.cos(h_omega * t)

    return t, q, dq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Example 01 — Duffing Oscillator")
    print("=" * 60)

    # 1. Build system
    system = build_system()
    print(f"System: {system}")

    # 2. HB continuation
    print("\nRunning HB continuation ...")
    omega_hb, amp_hb, stab_hb = run_hb_continuation(system)

    # 3. Shooting continuation
    print("\nRunning Shooting continuation ...")
    omega_sh, amp_sh, stab_sh = run_shooting_continuation(system)

    # 4. Find peak amplitude and resonance frequency (from HB)
    peak_idx_hb = int(np.argmax(amp_hb))
    peak_amp_hb = float(amp_hb[peak_idx_hb])
    omega_res_hb = float(omega_hb[peak_idx_hb])

    # 5. Get HB solution at peak for harmonic content and time series
    #    Re-extract from the continuation solution by re-running hb continuation
    #    and capturing the stored Q at peak.  Since we only have (omega, amp) back,
    #    re-run a single Newton solve at omega_res to get Q.
    n_dof = system.n_dof
    n_total = n_dof * (2 * N_HARMONICS + 1)
    Q_peak_guess = _hb_initial_guess(omega_res_hb, n_dof, N_HARMONICS)
    # Refine with a few Newton steps
    Q_peak = Q_peak_guess.copy()
    for _ in range(30):
        R, J = hb_residual(Q_peak, omega_res_hb, system, N_HARMONICS, EXCITATION)
        if np.linalg.norm(R) < 1e-12:
            break
        try:
            dQ = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            break
        Q_peak = Q_peak + dQ

    # 6. Harmonic amplitudes at peak: sqrt(Q_ch^2 + Q_sh^2) for h=1..H
    harmonic_amps = np.zeros(N_HARMONICS, dtype=np.float64)
    for h in range(1, N_HARMONICS + 1):
        c_idx = (2 * h - 1) * n_dof  # cosine coeff for harmonic h, DOF 0
        s_idx = 2 * h * n_dof          # sine coeff for harmonic h, DOF 0
        harmonic_amps[h - 1] = np.sqrt(Q_peak[c_idx] ** 2 + Q_peak[s_idx] ** 2)

    # 7. Time series at peak
    t_ts, q_ts, dq_ts = _reconstruct_time_series_from_Q(Q_peak, omega_res_hb, N_HARMONICS)

    # -----------------------------------------------------------------------
    # 8. Plots
    # -----------------------------------------------------------------------

    # --- FRF (HB + shooting overlaid) ---
    hb_result = FRFResult(
        omega=omega_hb,
        amplitude=amp_hb,
        stability=stab_hb,
    )
    sh_result = FRFResult(
        omega=omega_sh,
        amplitude=amp_sh,
        stability=stab_sh,
    )

    # Plot HB first, then overlay shooting on the same axes
    fig_frf = plot_frf(hb_result)
    # Get the axes from the figure and overlay shooting
    ax_frf = fig_frf.axes[0]
    # Split shooting by stability for consistent style
    from nlvib.visualization.plots import _split_by_stability
    stable_added = False
    unstable_added = False
    for x_seg, y_seg, is_stable in _split_by_stability(
        omega_sh, amp_sh, stab_sh
    ):
        ls = "-" if is_stable else "--"
        lbl = None
        if is_stable and not stable_added:
            lbl = "shooting (stable)"
            stable_added = True
        elif not is_stable and not unstable_added:
            lbl = "shooting (unstable)"
            unstable_added = True
        ax_frf.plot(x_seg, y_seg, ls, color="tab:orange", label=lbl, alpha=0.8)

    # Relabel HB traces
    lines = ax_frf.get_lines()
    for line in lines:
        if line.get_label() == "stable":
            line.set_label("HB (stable)")
        elif line.get_label() == "unstable":
            line.set_label("HB (unstable)")

    ax_frf.set_title("Duffing Oscillator — FRF (HB + Shooting)")
    ax_frf.legend(fontsize=8)
    frf_path = os.path.join(_OUTPUT_DIR, "frf.png")
    fig_frf.savefig(frf_path, dpi=150, bbox_inches="tight")
    plt.close(fig_frf)
    print(f"\nSaved: {frf_path}")

    # --- Harmonic content at peak ---
    fig_hc = plot_harmonic_content(harmonic_amps, omega_res_hb)
    fig_hc.axes[0].set_title(
        f"Harmonic Content at Peak (ω = {omega_res_hb:.4f} rad/s)"
    )
    hc_path = os.path.join(_OUTPUT_DIR, "harmonic_content.png")
    fig_hc.savefig(hc_path, dpi=150, bbox_inches="tight")
    plt.close(fig_hc)
    print(f"Saved: {hc_path}")

    # --- Time series at peak ---
    fig_ts = plot_time_series(t_ts, q_ts, dq=dq_ts, dof=0)
    fig_ts.axes[0].set_title(
        f"Time Series at Peak (ω = {omega_res_hb:.4f} rad/s)"
    )
    ts_path = os.path.join(_OUTPUT_DIR, "time_series.png")
    fig_ts.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ts)
    print(f"Saved: {ts_path}")

    # -----------------------------------------------------------------------
    # 9. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Value':>15}")
    print("-" * 52)
    print(f"{'Peak amplitude (HB)':<35} {peak_amp_hb:>15.6f}")
    print(f"{'Resonance frequency (HB) [rad/s]':<35} {omega_res_hb:>15.6f}")
    print(f"{'HB continuation points':<35} {len(omega_hb):>15d}")
    print(f"{'Shooting continuation points':<35} {len(omega_sh):>15d}")
    print("-" * 52)
    print(f"{'Harmonic amplitudes at peak:':<35}")
    for h_idx, h_amp in enumerate(harmonic_amps):
        print(f"  H{h_idx + 1:<33} {h_amp:>15.6e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
