"""
Example 02 — Two-DOF chain of oscillators with cubic spring nonlinearity.

Matches MATLAB NLvib demo: twoDOFoscillator_cubicSpring.m

System parameters:
    masses      = [1.0, 0.05]
    stiffnesses = [1.0, 0.0453, 0.0]
    dampings    = [0.002, 0.013, 0.0]
    Nonlinearities: cubic spring k3=1.0 at DOF 0,
                    inter-DOF cubic spring k3=0.0042 (force_direction=[1;-1])
    Excitation:   F=0.11 at DOF 0 (cosine, harmonic 1)
    Frequency range: omega in [0.8, 1.4] rad/s
    n_harmonics = 7

Outputs (saved to examples/02_two_dof_cubic/output/):
    frequency_response.png — a_rms vs omega (matches MATLAB figure)

Summary printed to stdout:
    Peak amplitudes (DOF 0, DOF 1), resonance frequencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make sure the installed package (or src layout) is importable.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.nonlinearities.elements import cubic_spring, polynomial_stiffness
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Constants — system parameters
# ---------------------------------------------------------------------------
MASSES: list[float] = [1.0, 0.05]          # MATLAB: [1.0, 1.0]
STIFFNESSES: list[float] = [1.0, 0.0453, 0.0]  # MATLAB: [1.0, 0.5, 1.0]
DAMPINGS: list[float] = [0.002, 0.013, 0.0]    # MATLAB: [0.01, 0.01, 0.01]
CUBIC_K3: float = 1.0
CUBIC_DOF: int = 0
EXCITATION_AMPLITUDE: float = 0.11        # MATLAB: 0.05
EXCITATION_DOF: int = 0
N_HARMONICS: int = 7                       # MATLAB: 3
OMEGA_START: float = 0.8                   # MATLAB: 0.3
OMEGA_END: float = 1.4                     # MATLAB: 1.5

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_a_rms(solutions: np.ndarray, n_dof: int, n_harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute omega and a_rms matching MATLAB's ``a_rms_HB`` formula.

    MATLAB: ``a_rms_HB = sqrt(sum(Q_HB(1:2:end,:).^2))/sqrt(2)``
    which sums all harmonic coefficients of DOF 0 (every other row in the
    interleaved layout), giving RMS displacement of DOF 0.

    Returns
    -------
    omega : ndarray, shape (n_steps,)
    a_rms : ndarray, shape (n_steps,)
    """
    omega = solutions[:, -1]
    Q_all = solutions[:, :-1]  # (n_steps, n_dof*(2H+1))
    # Reshape to (n_steps, 2H+1, n_dof) then take DOF 0
    Q_dof0 = Q_all.reshape(Q_all.shape[0], 2 * n_harmonics + 1, n_dof)[:, :, 0]  # (n_steps, 2H+1)
    a_rms = np.sqrt(np.sum(Q_dof0 ** 2, axis=1)) / np.sqrt(2)
    return omega, a_rms


def _extract_h1_amplitude(solutions: np.ndarray, n_dof: int, n_harmonics: int) -> np.ndarray:
    """Return H1 amplitude for each DOF at each step, shape (n_dof, n_steps)."""
    n_steps = solutions.shape[0]
    amp = np.zeros((n_dof, n_steps))
    Q_all = solutions[:, :-1]
    for i_dof in range(n_dof):
        c_idx = (2 * 1 - 1) * n_dof + i_dof
        s_idx = 2 * 1 * n_dof + i_dof
        amp[i_dof, :] = np.sqrt(Q_all[:, c_idx] ** 2 + Q_all[:, s_idx] ** 2)
    return amp


def _extract_harmonics_at_peak(
    solutions: np.ndarray, n_dof: int, n_harmonics: int, dof: int = 0
) -> tuple[np.ndarray, float]:
    """Per-harmonic amplitudes at the peak H1 step for the given DOF."""
    amp_h1 = _extract_h1_amplitude(solutions, n_dof, n_harmonics)[dof]
    peak_idx = int(np.argmax(amp_h1))
    Q_peak = solutions[peak_idx, :-1]
    omega_peak = float(solutions[peak_idx, -1])
    Q_harmonics = np.zeros(n_harmonics)
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

    # MATLAB: inter-DOF cubic spring k3=0.0042 with force_direction=[1;-1]
    # Implemented via two polynomial_stiffness elements (relative displacement trick)
    k3_inter = 0.0042  # MATLAB: k3=0 (no inter-DOF cubic in original Python)
    _exp = np.array([[3, 0], [2, 1], [1, 2], [0, 3]], dtype=np.intp)
    _coeff = np.array([k3_inter, -3 * k3_inter, 3 * k3_inter, -k3_inter])
    # Element A: force on DOF 0 from (q0-q1)^3
    system.add_nonlinear_element(polynomial_stiffness(_exp, _coeff, np.array([0, 1], dtype=np.intp)))
    # Element B: force on DOF 1 from (q1-q0)^3
    system.add_nonlinear_element(polynomial_stiffness(_exp, _coeff, np.array([1, 0], dtype=np.intp)))

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
        verbose=True,
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
    omega_all, a_rms_all = _compute_a_rms(result.solutions, n_dof, N_HARMONICS)
    amp_h1 = _extract_h1_amplitude(result.solutions, n_dof, N_HARMONICS)

    # Filter to requested frequency range
    mask = (omega_all >= OMEGA_START) & (omega_all <= OMEGA_END)
    omega_plot = omega_all[mask]
    a_rms_plot = a_rms_all[mask]
    amp_h1_plot = amp_h1[:, mask]

    # ------------------------------------------------------------------
    # 5. Compute summary statistics
    # ------------------------------------------------------------------
    peak_amp_dof0 = float(np.max(amp_h1_plot[0]))
    peak_amp_dof1 = float(np.max(amp_h1_plot[1]))
    omega_peak_dof0 = float(omega_plot[np.argmax(amp_h1_plot[0])])
    omega_peak_dof1 = float(omega_plot[np.argmax(amp_h1_plot[1])])

    Q_harmonics, omega_at_peak = _extract_harmonics_at_peak(
        result.solutions, n_dof, N_HARMONICS, dof=0
    )

    # ------------------------------------------------------------------
    # 6. Generate and save single figure (matches MATLAB frequency_response.png)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(omega_plot, a_rms_plot, "g-", label="HB")
    ax.set_xlabel("excitation frequency")
    ax.set_ylabel("response amplitude")
    ax.set_xlim(OMEGA_START, OMEGA_END)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = OUTPUT_DIR / "frequency_response.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

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
