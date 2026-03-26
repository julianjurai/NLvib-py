"""
Example 02 — Two-DOF chain of oscillators with cubic spring nonlinearity.

System parameters (Krack & Gross 2019, §5):
    masses      = [1.0, 1.0]
    stiffnesses = [1.0, 0.5, 1.0]   (left boundary, inter-mass, right boundary)
    dampings    = [0.01, 0.01, 0.01]
    Nonlinearity: cubic spring k3=1.0 at DOF 0
    Excitation:   F=0.05 at DOF 0 (cosine, harmonic 1)
    Frequency range: omega in [0.3, 1.5] rad/s
    n_harmonics = 3

Outputs (saved to examples/02_two_dof_cubic/output/):
    frf_dof0.png        — FRF amplitude vs. omega for DOF 0
    frf_dof1.png        — FRF amplitude vs. omega for DOF 1
    harmonic_content.png — Harmonic content at the peak response point

Summary printed to stdout:
    Peak amplitudes (DOF 0, DOF 1), resonance frequencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output

# ---------------------------------------------------------------------------
# Make sure the installed package (or src layout) is importable.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.nonlinearities.elements import cubic_spring
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions
from nlvib.visualization.plots import plot_frf, plot_harmonic_content

# ---------------------------------------------------------------------------
# Constants — system parameters
# ---------------------------------------------------------------------------
MASSES: list[float] = [1.0, 1.0]
STIFFNESSES: list[float] = [1.0, 0.5, 1.0]
DAMPINGS: list[float] = [0.01, 0.01, 0.01]
CUBIC_K3: float = 1.0
CUBIC_DOF: int = 0
EXCITATION_AMPLITUDE: float = 0.05
EXCITATION_DOF: int = 0
N_HARMONICS: int = 3
OMEGA_START: float = 0.3
OMEGA_END: float = 1.5

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Helper: wrap ContinuationResult into the protocol expected by plot_frf
# ---------------------------------------------------------------------------

@dataclass
class FRFResult:
    """Adapter that satisfies the ContinuationResult protocol in plots.py.

    Attributes
    ----------
    omega:
        1-D array of angular frequencies (rad/s), shape (n_points,).
    amplitude:
        Array of amplitudes, shape (n_dof, n_points).
    stability:
        Boolean array, shape (n_points,).  True = potentially unstable.
    """

    omega: np.ndarray
    amplitude: np.ndarray
    stability: np.ndarray


def _extract_frf(
    solutions: np.ndarray,
    stability: np.ndarray,
    n_dof: int,
    n_harmonics: int,
) -> FRFResult:
    """Extract omega and peak-harmonic amplitudes from continuation solutions.

    The augmented state vector layout is [Q; omega] where Q has length
    n_dof * (2*H + 1).  The amplitude of DOF i at harmonic h is:

        A_i^h = sqrt(Q_{c,h,i}^2 + Q_{s,h,i}^2)

    For the FRF we report the fundamental (h=1) amplitude.

    Parameters
    ----------
    solutions:
        Shape (n_steps, n_dof*(2H+1) + 1).  Last column is omega.
    stability:
        Shape (n_steps,).
    n_dof, n_harmonics:
        System dimensions.

    Returns
    -------
    FRFResult
        omega (n_steps,), amplitude (n_dof, n_steps), stability (n_steps,).
    """
    n_steps = solutions.shape[0]
    omega_arr = solutions[:, -1].copy()  # last column is lambda = omega

    # Amplitude at each step: sqrt(Q_c1^2 + Q_s1^2) for each DOF
    # Q layout: [Q_0 (n_dof), Q_c1 (n_dof), Q_s1 (n_dof), Q_c2 (n_dof), Q_s2 (n_dof), ...]
    amplitude = np.zeros((n_dof, n_steps), dtype=np.float64)
    for i_step in range(n_steps):
        Q = solutions[i_step, :-1]  # shape (n_dof * (2H+1),)
        for i_dof in range(n_dof):
            # Cosine block for harmonic h: block index 2h-1, starting at (2h-1)*n_dof + i_dof
            # Sine block for harmonic h:   block index 2h,   starting at 2h*n_dof + i_dof
            amp_dof = 0.0
            for h in range(1, n_harmonics + 1):
                c_idx = (2 * h - 1) * n_dof + i_dof
                s_idx = 2 * h * n_dof + i_dof
                amp_h = float(np.sqrt(Q[c_idx] ** 2 + Q[s_idx] ** 2))
                if h == 1:  # report fundamental for FRF
                    amp_dof = amp_h
            amplitude[i_dof, i_step] = amp_dof

    return FRFResult(omega=omega_arr, amplitude=amplitude, stability=stability.copy())


def _extract_harmonics_at_peak(
    solutions: np.ndarray,
    n_dof: int,
    n_harmonics: int,
    dof: int = 0,
) -> tuple[np.ndarray, float]:
    """Extract per-harmonic amplitudes at the peak response step for a given DOF.

    Parameters
    ----------
    solutions:
        Shape (n_steps, n_dof*(2H+1) + 1).
    n_dof, n_harmonics:
        System dimensions.
    dof:
        DOF index at which to measure peak.

    Returns
    -------
    Q_harmonics : ndarray, shape (n_harmonics,)
        Amplitude of each harmonic (h=1..H) at the peak step.
    omega_peak : float
        Frequency at the peak step.
    """
    # Compute fundamental amplitude for the specified DOF at each step
    n_steps = solutions.shape[0]
    amp_fund = np.zeros(n_steps, dtype=np.float64)
    for i_step in range(n_steps):
        Q = solutions[i_step, :-1]
        c_idx = (2 * 1 - 1) * n_dof + dof
        s_idx = 2 * 1 * n_dof + dof
        amp_fund[i_step] = float(np.sqrt(Q[c_idx] ** 2 + Q[s_idx] ** 2))

    peak_idx = int(np.argmax(amp_fund))
    Q_peak = solutions[peak_idx, :-1]
    omega_peak = float(solutions[peak_idx, -1])

    Q_harmonics = np.zeros(n_harmonics, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        c_idx = (2 * h - 1) * n_dof + dof
        s_idx = 2 * h * n_dof + dof
        Q_harmonics[h - 1] = float(np.sqrt(Q_peak[c_idx] ** 2 + Q_peak[s_idx] ** 2))

    return Q_harmonics, omega_peak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Example 02: Two-DOF cubic spring HB continuation and plotting."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build the system
    # ------------------------------------------------------------------
    system = ChainOfOscillators(
        masses=MASSES,
        stiffnesses=STIFFNESSES,
        dampings=DAMPINGS,
    )
    system.add_nonlinear_element(cubic_spring(k3=CUBIC_K3, dof_index=CUBIC_DOF))

    n_dof = system.n_dof  # 2
    n_total = n_dof * (2 * N_HARMONICS + 1)

    print(f"System: {system}")
    print(f"  n_dof={n_dof}, n_harmonics={N_HARMONICS}, n_total_HB={n_total}")

    # ------------------------------------------------------------------
    # 2. Find initial solution at OMEGA_START using Newton iterations
    # ------------------------------------------------------------------
    excitation = {"dof": EXCITATION_DOF, "amplitude": EXCITATION_AMPLITUDE}

    # Linear initial guess: Q = 0 (good enough near low frequency)
    Q0 = np.zeros(n_total, dtype=np.float64)

    # Refine initial guess with a few Newton steps
    omega0 = OMEGA_START
    Q_curr = Q0.copy()
    for _ in range(50):
        R, J = hb_residual(Q_curr, omega0, system, N_HARMONICS, excitation)
        if np.linalg.norm(R) < 1e-10:
            break
        try:
            dQ = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dQ = np.linalg.lstsq(J, -R, rcond=None)[0]
        Q_curr = Q_curr + dQ

    print(f"  Initial residual norm at omega={omega0:.3f}: {np.linalg.norm(R):.3e}")

    # ------------------------------------------------------------------
    # 3. Run HB continuation from OMEGA_START to OMEGA_END
    # ------------------------------------------------------------------
    def residual_fn(x: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
        return hb_residual(x, lam, system, N_HARMONICS, excitation)

    opts = ContinuationOptions(
        ds_initial=0.02,
        ds_min=1e-5,
        ds_max=0.1,
        max_steps=800,
        max_newton_iter=25,
        newton_tol=1e-8,
        adapt_step=True,
        lambda_min=OMEGA_START - 0.05,
        lambda_max=OMEGA_END + 0.05,
    )

    print("  Running continuation...")
    solver = ContinuationSolver()
    result = solver.run(residual_fn, Q_curr, omega0, opts)

    print(f"  Continuation finished: {result.message}")
    print(f"  Accepted steps: {result.n_steps}")

    # ------------------------------------------------------------------
    # 4. Extract FRF data
    # ------------------------------------------------------------------
    frf = _extract_frf(result.solutions, result.stability, n_dof, N_HARMONICS)

    # Filter to the requested frequency range for plotting/summary
    mask = (frf.omega >= OMEGA_START) & (frf.omega <= OMEGA_END)
    frf_plot = FRFResult(
        omega=frf.omega[mask],
        amplitude=frf.amplitude[:, mask],
        stability=frf.stability[mask],
    )

    # ------------------------------------------------------------------
    # 5. Compute summary statistics
    # ------------------------------------------------------------------
    amp_dof0 = frf_plot.amplitude[0, :]
    amp_dof1 = frf_plot.amplitude[1, :]

    peak_amp_dof0 = float(np.max(amp_dof0))
    peak_amp_dof1 = float(np.max(amp_dof1))
    omega_peak_dof0 = float(frf_plot.omega[np.argmax(amp_dof0)])
    omega_peak_dof1 = float(frf_plot.omega[np.argmax(amp_dof1)])

    # ------------------------------------------------------------------
    # 6. Generate and save plots
    # ------------------------------------------------------------------

    # FRF at DOF 0
    fig0 = plot_frf(frf_plot, dof=0, harmonic=1)
    fig0.suptitle("Two-DOF Cubic Spring — FRF at DOF 0", fontsize=11)
    frf0_path = OUTPUT_DIR / "frf_dof0.png"
    fig0.savefig(frf0_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {frf0_path}")

    # FRF at DOF 1
    fig1 = plot_frf(frf_plot, dof=1, harmonic=1)
    fig1.suptitle("Two-DOF Cubic Spring — FRF at DOF 1", fontsize=11)
    frf1_path = OUTPUT_DIR / "frf_dof1.png"
    fig1.savefig(frf1_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {frf1_path}")

    # Harmonic content at peak (DOF 0)
    Q_harmonics, omega_at_peak = _extract_harmonics_at_peak(
        result.solutions, n_dof, N_HARMONICS, dof=0
    )
    fig_hc = plot_harmonic_content(Q_harmonics, omega_at_peak)
    fig_hc.suptitle("Harmonic Content at Peak (DOF 0)", fontsize=11)
    hc_path = OUTPUT_DIR / "harmonic_content.png"
    fig_hc.savefig(hc_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {hc_path}")

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("  SUMMARY — Example 02: Two-DOF Cubic Spring")
    print("=" * 55)
    print(f"  Peak amplitude DOF 0 : {peak_amp_dof0:.6f}  at omega = {omega_peak_dof0:.4f} rad/s")
    print(f"  Peak amplitude DOF 1 : {peak_amp_dof1:.6f}  at omega = {omega_peak_dof1:.4f} rad/s")
    print(f"  Resonance freq DOF 0 : {omega_peak_dof0:.4f} rad/s")
    print(f"  Resonance freq DOF 1 : {omega_peak_dof1:.4f} rad/s")
    print(f"  Harmonic content at peak (DOF 0):")
    for h, amp_h in enumerate(Q_harmonics, start=1):
        print(f"    H{h}: {amp_h:.6f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
