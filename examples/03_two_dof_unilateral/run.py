"""
Example 03 — Two-DOF chain with unilateral spring (impact nonlinearity).

System parameters
-----------------
- masses         : [1.0, 1.0]  kg
- stiffnesses    : [1.0, 0.5, 1.0]  N/m  (left-ground, inter-mass, right-ground)
- dampings       : [0.02, 0.02, 0.02]  N·s/m
- Nonlinearity   : unilateral spring k=5.0, gap=0.1 at DOF 1 (second mass)
- Excitation     : harmonic force F=0.1 at DOF 0 (first mass)
- Frequency range: omega in [0.5, 1.8]
- n_harmonics    : 7  (unilateral springs need more harmonics for accuracy)

Outputs (saved to examples/03_two_dof_unilateral/output/)
---------------------------------------------------------
- frf.png           — FRF amplitude at DOF 1 (where the unilateral spring is)
- phase_portrait.png — Phase portrait at peak frequency, reconstructed from HB coefficients

Reference: Krack & Gross (2019) Harmonic Balance for Nonlinear Vibration Problems.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no display needed

# ---------------------------------------------------------------------------
# Ensure project src is on sys.path when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nlvib.nonlinearities.elements import unilateral_spring
from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions
from nlvib.utils.transforms import freq_to_time
from nlvib.visualization.plots import plot_frf, plot_phase_portrait

# ---------------------------------------------------------------------------
# Named system constants
# ---------------------------------------------------------------------------
MASSES = [1.0, 1.0]
STIFFNESSES = [0.0, 1.0, 1.0]   # MATLAB: [1.0, 0.5, 1.0] (ki=[0,1,1])
DAMPINGS = [0.0, 0.03, 0.03]    # MATLAB: [0.02, 0.02, 0.02] (di=0.03*ki)
CONTACT_STIFFNESS = 100.0        # MATLAB: 5.0 (k=100)
CONTACT_GAP = 1.0                # MATLAB: 0.1 (gap=1)
CONTACT_DOF = 0                  # MATLAB: 1 (W=[1;0] → DOF 0)
EXCITATION_DOF = 1               # MATLAB: 0 (Fex1=[0;0.1] → DOF 1)
EXCITATION_AMPLITUDE = 0.1
N_HARMONICS = 21                 # MATLAB: 7 (H=21)
OMEGA_START = 0.5
OMEGA_END = 0.8                  # MATLAB: 1.8 (MATLAB sweeps 0.8→0.5; Python sweeps forward 0.5→0.8)

# Continuation options
DS_INITIAL = 0.02
DS_MIN = 1e-5
DS_MAX = 0.1
MAX_STEPS = 600
NEWTON_TOL = 1e-10

OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# 1. Build system
# ---------------------------------------------------------------------------


def build_system() -> ChainOfOscillators:
    """Construct the 2-DOF chain with an attached unilateral spring at DOF 1."""
    sys_ = ChainOfOscillators(
        masses=MASSES,
        stiffnesses=STIFFNESSES,
        dampings=DAMPINGS,
    )
    nl_element = unilateral_spring(k=CONTACT_STIFFNESS, gap=CONTACT_GAP, dof_index=CONTACT_DOF)
    sys_.add_nonlinear_element(nl_element)
    return sys_


# ---------------------------------------------------------------------------
# 2. Compute initial solution at OMEGA_START
# ---------------------------------------------------------------------------


def compute_initial_solution(
    system: ChainOfOscillators,
    omega0: float,
) -> np.ndarray:
    """Solve for the linear HB starting point at omega0.

    Returns the Q vector (n_dof * (2*H+1),) obtained by ignoring the
    nonlinear term as a first guess, then refining with Newton iterations.
    """
    n_dof = system.n_dof
    n_total = n_dof * (2 * N_HARMONICS + 1)
    excitation = {"dof": EXCITATION_DOF, "amplitude": EXCITATION_AMPLITUDE}

    # Zero initial guess
    Q = np.zeros(n_total, dtype=np.float64)

    # Newton iteration to converge at omega0
    for _ in range(50):
        R, J = hb_residual(Q, omega0, system, N_HARMONICS, excitation)
        if np.linalg.norm(R) < NEWTON_TOL:
            break
        try:
            dQ = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dQ = np.linalg.lstsq(J, -R, rcond=None)[0]
        Q = Q + dQ

    return Q


# ---------------------------------------------------------------------------
# 3. Run HB continuation
# ---------------------------------------------------------------------------


def run_continuation(
    system: ChainOfOscillators,
    Q0: np.ndarray,
) -> object:
    """Trace the FRF branch from OMEGA_START to OMEGA_END."""
    excitation = {"dof": EXCITATION_DOF, "amplitude": EXCITATION_AMPLITUDE}

    def residual_fn(Q: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
        return hb_residual(Q, omega, system, N_HARMONICS, excitation)

    opts = ContinuationOptions(
        verbose=True,
        ds_initial=DS_INITIAL,
        ds_min=DS_MIN,
        ds_max=DS_MAX,
        max_steps=MAX_STEPS,
        max_newton_iter=25,
        newton_tol=NEWTON_TOL,
        adapt_step=True,
        lambda_min=OMEGA_START,
        lambda_max=OMEGA_END,
    )

    solver = ContinuationSolver()
    result = solver.run(residual_fn, Q0, OMEGA_START, opts)
    return result


# ---------------------------------------------------------------------------
# 4. Extract amplitudes from continuation result
# ---------------------------------------------------------------------------


def extract_amplitudes(solutions: np.ndarray, n_dof: int) -> np.ndarray:
    """Extract fundamental harmonic amplitudes at each solution point.

    Parameters
    ----------
    solutions:
        Array of shape (n_steps, n_total + 1) from ContinuationResult.solutions.
        Last column is omega; first n_total columns are Fourier coefficients Q.
    n_dof:
        Number of degrees of freedom.

    Returns
    -------
    amplitudes:
        Array of shape (n_dof, n_steps) with the fundamental harmonic (h=1)
        amplitude sqrt(Q_c1^2 + Q_s1^2) for each DOF at each step.
    """
    n_steps = solutions.shape[0]
    n_total = n_dof * (2 * N_HARMONICS + 1)
    amplitudes = np.zeros((n_dof, n_steps), dtype=np.float64)

    for i in range(n_steps):
        Q = solutions[i, :n_total]
        for d in range(n_dof):
            # Cosine coeff for h=1 at DOF d: block (2*1 - 1), element d
            # Block index in flattened Q: block_start = (2*h - 1) * n_dof
            c_idx = (2 * 1 - 1) * n_dof + d
            s_idx = 2 * 1 * n_dof + d
            amplitudes[d, i] = np.sqrt(Q[c_idx] ** 2 + Q[s_idx] ** 2)

    return amplitudes


# ---------------------------------------------------------------------------
# 5. Reconstruct time series from HB coefficients at peak
# ---------------------------------------------------------------------------


def reconstruct_time_series(
    Q: np.ndarray,
    omega_peak: float,
    n_dof: int,
    n_periods: int = 2,
    n_points_per_period: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct q(t) and dq/dt(t) from Fourier coefficients Q.

    Parameters
    ----------
    Q:
        Fourier coefficient vector, shape (n_dof * (2*H+1),).
    omega_peak:
        Fundamental frequency at the peak point.
    n_dof:
        Number of DOFs.
    n_periods:
        Number of periods to reconstruct.
    n_points_per_period:
        Time-domain resolution per period.

    Returns
    -------
    t : (n_time,) time vector
    q : (n_dof, n_time) displacement
    dq : (n_dof, n_time) velocity (analytical derivative of Fourier series)
    """
    T = 2.0 * np.pi / omega_peak
    n_time = n_periods * n_points_per_period
    t = np.linspace(0.0, n_periods * T, n_time, endpoint=False)

    # Reshape Q into (n_dof, 2H+1) matrix layout
    Q_mat = Q.reshape(2 * N_HARMONICS + 1, n_dof).T  # (n_dof, 2H+1)

    # Build velocity coefficient matrix dQ by differentiating Fourier series
    dQ_mat = np.zeros_like(Q_mat)
    for h in range(1, N_HARMONICS + 1):
        h_omega = h * omega_peak
        c_idx = 2 * h - 1
        s_idx = 2 * h
        # d/dt [a_h cos(hωt) + b_h sin(hωt)] = hω·b_h cos(hωt) - hω·a_h sin(hωt)
        dQ_mat[:, c_idx] = h_omega * Q_mat[:, s_idx]
        dQ_mat[:, s_idx] = -h_omega * Q_mat[:, c_idx]

    # Evaluate time series by constructing the Fourier sum explicitly
    q_time = np.zeros((n_dof, n_time), dtype=np.float64)
    dq_time = np.zeros((n_dof, n_time), dtype=np.float64)

    # DC term
    q_time += Q_mat[:, 0:1]      # shape (n_dof, 1) broadcasts

    for h in range(1, N_HARMONICS + 1):
        h_omega = h * omega_peak
        cos_t = np.cos(h_omega * t)  # (n_time,)
        sin_t = np.sin(h_omega * t)
        c_idx = 2 * h - 1
        s_idx = 2 * h
        q_time += Q_mat[:, c_idx:c_idx+1] * cos_t + Q_mat[:, s_idx:s_idx+1] * sin_t
        dq_time += dQ_mat[:, c_idx:c_idx+1] * cos_t + dQ_mat[:, s_idx:s_idx+1] * sin_t

    return t, q_time, dq_time


# ---------------------------------------------------------------------------
# 6. FRF result wrapper (satisfies plots.ContinuationResult protocol)
# ---------------------------------------------------------------------------


class FRFResult:
    """Lightweight wrapper satisfying the ``ContinuationResult`` protocol
    expected by :func:`nlvib.visualization.plots.plot_frf`.

    Attributes
    ----------
    omega:
        1-D array of frequencies, shape (n_steps,).
    amplitude:
        Array of shape (n_dof, n_steps) for multi-DOF FRF.
    stability:
        Boolean array, shape (n_steps,).
        ``True`` = stable branch (note: plot_frf treats True=stable → solid line).
    """

    def __init__(
        self,
        omega: np.ndarray,
        amplitude: np.ndarray,
        stability: np.ndarray,
    ) -> None:
        self.omega = omega
        self.amplitude = amplitude
        self.stability = stability


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Example 03 — Two-DOF chain with unilateral spring")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build system
    print("\n[1/4] Building system...")
    system = build_system()
    print(f"      {system}")

    # 2. Compute initial solution
    print(f"\n[2/4] Computing initial solution at omega = {OMEGA_START}...")
    Q0 = compute_initial_solution(system, OMEGA_START)
    R0, _ = hb_residual(
        Q0, OMEGA_START, system, N_HARMONICS,
        {"dof": EXCITATION_DOF, "amplitude": EXCITATION_AMPLITUDE},
    )
    print(f"      Initial residual norm: {np.linalg.norm(R0):.3e}")

    # 3. Run continuation
    print(f"\n[3/4] Running HB continuation from omega={OMEGA_START} to {OMEGA_END}...")
    print(f"      n_harmonics={N_HARMONICS}, max_steps={MAX_STEPS}")
    raw_result = run_continuation(system, Q0)
    print(f"      Continuation: {raw_result.n_steps} steps accepted")  # type: ignore[union-attr]
    print(f"      Message: {raw_result.message}")  # type: ignore[union-attr]

    solutions = raw_result.solutions  # type: ignore[union-attr]
    stability_raw = raw_result.stability  # type: ignore[union-attr]  # True = unstable in solver

    # Extract omega and amplitudes
    omega_branch = solutions[:, -1]  # last column = lambda = omega
    n_dof = system.n_dof
    amplitudes = extract_amplitudes(solutions, n_dof)

    # Invert stability flag: solver marks True = unstable; plots module marks True = stable
    stability_for_plot = ~stability_raw

    # 4. Produce and save plots
    print("\n[4/4] Producing plots...")

    # --- FRF at DOF 1 (where the unilateral spring is) ---
    frf_result = FRFResult(
        omega=omega_branch,
        amplitude=amplitudes,   # shape (n_dof, n_steps)
        stability=stability_for_plot,
    )
    fig_frf = plot_frf(frf_result, dof=CONTACT_DOF, harmonic=1)
    fig_frf.suptitle(
        "Example 03 — Two-DOF unilateral spring\n"
        f"DOF {CONTACT_DOF} (contact side), H={N_HARMONICS}",
        fontsize=9,
    )
    frf_path = OUTPUT_DIR / "frf.png"
    fig_frf.savefig(frf_path, dpi=150, bbox_inches="tight")
    print(f"      Saved: {frf_path}")

    # --- Phase portrait at peak ---
    # Identify peak: largest amplitude at DOF 1
    amp_dof1 = amplitudes[CONTACT_DOF, :]
    peak_idx = int(np.argmax(amp_dof1))
    omega_peak = float(omega_branch[peak_idx])
    amp_peak = float(amp_dof1[peak_idx])

    Q_peak = solutions[peak_idx, :-1]  # Fourier coefficients at peak
    t, q_ts, dq_ts = reconstruct_time_series(Q_peak, omega_peak, n_dof)

    # Phase portrait uses last period to show steady-state orbit
    n_pts = t.size // 2  # second half (one full period)
    fig_phase = plot_phase_portrait(
        t[n_pts:],
        q_ts[:, n_pts:],
        dq_ts[:, n_pts:],
        dof=CONTACT_DOF,
    )
    fig_phase.suptitle(
        f"Example 03 — Phase portrait at peak\n"
        f"DOF {CONTACT_DOF},  omega={omega_peak:.4f} rad/s,  amp={amp_peak:.4f} m",
        fontsize=9,
    )
    phase_path = OUTPUT_DIR / "phase_portrait.png"
    fig_phase.savefig(phase_path, dpi=150, bbox_inches="tight")
    print(f"      Saved: {phase_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Peak amplitude (DOF {CONTACT_DOF}, H=1): {amp_peak:.6f} m")
    print(f"  Frequency at peak:                       {omega_peak:.6f} rad/s")
    print(f"  Total continuation steps:                {raw_result.n_steps}")  # type: ignore[union-attr]
    print(f"  Omega range covered:                     [{omega_branch.min():.4f}, {omega_branch.max():.4f}]")
    print(f"  Output directory:                        {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
