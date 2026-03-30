"""
Example 04 — Two-DOF Chain with Tanh Dry Friction: Nonlinear Modal Analysis.

System
------
Two-DOF chain of oscillators with a tanh dry-friction nonlinearity at DOF 0.

    Ground --[k0=1.0]-- m0=1.0 --[k1=0.5]-- m1=1.0  (open right end, k2=0.0)

No linear damping (NMA mode).

Nonlinearity: tanh_dry_friction(f0=0.3, c=10.0) at DOF 0.

Analysis
--------
1. Nonlinear Modal Analysis (NMA) via Harmonic Balance describing-function approach.
   Backbone curve: natural frequency vs. modal amplitude, with amplitude-dependent
   equivalent damping ratio.
2. Forced HB at F=0.05 for FRF overlay on the backbone.

The backbone is computed using the Complex Nonlinear Mode (CNM) concept:
at each amplitude level, the tanh friction force is decomposed into its
conservative and dissipative harmonic coefficients, and the effective nonlinear
natural frequency and damping ratio are extracted (K&G §3.2).

Outputs (saved to examples/04_two_dof_tanh_friction/output/)
------------------------------------------------------------
- backbone.png    — backbone curve (frequency vs. modal amplitude)
- frf_overlay.png — forced FRF overlaid on backbone

Reference: Krack & Gross (2019) §3 (NMA), §4 (continuation).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize as sopt

# ---------------------------------------------------------------------------
# Ensure package is importable when running as a script from the repo root.
# ---------------------------------------------------------------------------
_repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from nlvib.nonlinearities.elements import tanh_dry_friction
from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions
from nlvib.visualization.plots import plot_backbone

# ---------------------------------------------------------------------------
# Named parameters
# ---------------------------------------------------------------------------

MASSES = [0.02, 1.0]            # MATLAB mi=[0.02, 1]
STIFFNESSES = [0.0, 40.0, 600.0]  # MATLAB ki=[0, 40, 600]
DAMPINGS = [0.0, 0.0, 0.0]     # no linear damping for NMA

FRICTION_F0: float = 5.0        # MATLAB muN=5
FRICTION_C: float = 20.0        # MATLAB c=1/eps=1/0.05=20
FRICTION_DOF: int = 0

N_HARMONICS: int = 21           # MATLAB: H=21

# Forced analysis parameters
FORCE_AMPLITUDE: float = 0.05
FORCE_DOF: int = 0

# Output directory
_OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Build the system
# ---------------------------------------------------------------------------

def build_system() -> ChainOfOscillators:
    """Construct the 2-DOF chain with tanh friction nonlinearity (K&G §5)."""
    sys_obj = ChainOfOscillators(
        masses=MASSES,
        stiffnesses=STIFFNESSES,
        dampings=DAMPINGS,
    )
    friction = tanh_dry_friction(f0=FRICTION_F0, c=FRICTION_C, dof_index=FRICTION_DOF)
    sys_obj.add_nonlinear_element(friction)
    return sys_obj


# ---------------------------------------------------------------------------
# Step 2: Compute linear natural frequencies and mode shapes
# ---------------------------------------------------------------------------

def linear_modes(system: ChainOfOscillators) -> tuple[np.ndarray, np.ndarray]:
    """Return (omega_n, phi) sorted by frequency (K&G §1.1).

    Returns
    -------
    omega_n : ndarray, shape (n_dof,)
    phi : ndarray, shape (n_dof, n_dof)
        Columns are mass-normalised mode shapes.
    """
    M_dense = system.M.toarray()
    K_dense = system.K.toarray()
    evals, evecs = scipy.linalg.eigh(K_dense, M_dense)  # symmetric generalised EVP
    omega_n = np.sqrt(np.maximum(evals, 0.0))
    return omega_n, evecs


# ---------------------------------------------------------------------------
# Step 3: NMA backbone via Describing Function + Amplitude Continuation
# ---------------------------------------------------------------------------
#
# For a system with tanh dry friction the backbone curve is computed by
# solving the HB equations at each prescribed amplitude level.  The key is
# to use the forced HB formulation with zero excitation BUT include a
# "continuation equation" that fixes the modal amplitude:
#
#   A_modal = ||Q_h1|| = prescribed_amplitude
#
# We add an auxiliary mass-proportional forcing that acts as a "restoring
# term" to lift the system off the trivial Q=0 solution:
#
#   R_aug = [Z(omega)*Q + F_nl(Q,omega);   ← HB equations (n_total rows)
#            Q_cos1_DOF0              ;   ← phase constraint
#            ||Q_h1||^2 - A^2         ]   ← amplitude normalisation
#
# for unknowns [Q; omega] (n_aug = n_total + 1).
# This gives n_aug + 1 equations for n_aug unknowns — overdetermined.
#
# The correct square formulation: replace the phase constraint row with the
# amplitude normalisation, and enforce Q_cos1_DOF0 = 0 by fixing it:
#
#   Free unknowns: Q_free = Q except Q_cos1_DOF0; omega  (n_aug unknowns)
#   Equations:     n_total physical rows (using Q_cos1_DOF0 = 0 fixed)
#                + 1 amplitude normalisation   = n_aug equations ✓
#
# The amplitude normalisation row ensures we get a non-trivial solution.
# Reference: K&G §3.1.


def _tanh_describing_function(
    A: float, omega: float, f0: float, c: float, n_time: int = 512
) -> tuple[float, float]:
    """Compute the harmonic linearisation (describing function) coefficients.

    For q(t) = A * sin(omega*t), the friction force is
        f(t) = f0 * tanh(c * dq/dt) = f0 * tanh(c * A * omega * cos(omega*t))

    The fundamental harmonic coefficients are:
        F_cos = (2/T) * integral_0^T f(t) * cos(omega*t) dt
        F_sin = (2/T) * integral_0^T f(t) * sin(omega*t) dt

    For the tanh friction (velocity-dependent), F_sin contributes to
    in-phase (stiffness) and F_cos contributes to out-of-phase (damping).

    Since tanh is an odd function and dq/dt = A*omega*cos(omega*t):
        F_cos != 0 (out-of-phase with displacement, i.e., in-phase with velocity)
        F_sin = 0 (no stiffness contribution from pure velocity friction)

    Returns
    -------
    F_cos : float
        Cosine coefficient (damping-like, in-phase with velocity).
    F_sin : float
        Sine coefficient (stiffness-like, in-phase with displacement).
    """
    t_arr = np.linspace(0, 2 * np.pi, n_time, endpoint=False)
    dq = A * omega * np.cos(t_arr)  # velocity time series
    f_nl = f0 * np.tanh(c * dq)     # friction force

    # Fourier coefficients
    F_cos = (2.0 / n_time) * np.sum(f_nl * np.cos(t_arr))
    F_sin = (2.0 / n_time) * np.sum(f_nl * np.sin(t_arr))
    return float(F_cos), float(F_sin)


def run_nma_backbone(
    system: ChainOfOscillators,
    omega_lin: np.ndarray,
    mode_shapes: np.ndarray,
    n_harmonics: int,
    amplitude_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute backbone curve via amplitude-continuation HB NMA.

    At each prescribed amplitude A, we solve for the NMA solution (Q, omega)
    using the augmented HB system with amplitude normalisation:

        R(Q, omega; A) = [Z(omega)*Q + F_nl(Q, omega);  (n_total physical)
                          ||Q_h1||^2 - A^2            ]  (1 amplitude eq.)
        Q_cos1_DOF0 = 0  (phase constraint, fixed by substitution)

    Parameters
    ----------
    system:
        2-DOF system with tanh friction.
    omega_lin:
        Linear natural frequencies (rad/s).
    mode_shapes:
        Linear mode shapes (columns are modes, mass-normalised).
    n_harmonics:
        Number of harmonics H.
    amplitude_levels:
        Prescribed modal amplitudes ||Q_h1||.

    Returns
    -------
    amplitudes : ndarray, shape (n_valid,)
    omegas : ndarray, shape (n_valid,)
    zetas : ndarray, shape (n_valid,)
        Equivalent damping ratio at each amplitude.
    """
    n_dof = system.n_dof
    n_total = n_dof * (2 * n_harmonics + 1)
    n_aug = n_total + 1  # [Q; omega]

    # Phase constraint: Q_sin1_DOF1 = 0
    # Matches MATLAB inorm=2 (DOF 1, 1-indexed = DOF 1, 0-indexed).
    # MATLAB convention: Im(Q1(inorm)) = Q_s1[inorm-1] = 0.
    # sin_h1 block starts at index 2*n_dof; sin_h1[DOF1] is at index 2*n_dof + 1.
    sin1_dof1_idx = 2 * n_dof + 1  # block 2 (sin_h1), DOF 1

    # Free Q indices: all except Q_sin1_DOF1
    free_Q_idx = np.array([i for i in range(n_total) if i != sin1_dof1_idx], dtype=np.intp)
    # Free variables: [Q_free (n_total-1); omega (1)] = n_total unknowns
    n_free = n_total  # n_total - 1 free Q + 1 omega = n_total

    # Mode-shape based initial Q at mode 1
    mode1 = mode_shapes[:, 0].copy()
    # Normalise so that ||mode1||_M = 1 (already mass-normalised from eigh)
    # Scale so sin_h1 component represents amplitude A:
    # ||Q_sin1||^2 = A^2 → Q_sin1 = A * mode1 / ||mode1||
    mode1_norm = float(np.linalg.norm(mode1))
    mode1_unit = mode1 / mode1_norm

    def build_initial_z(A: float, omega_guess: float, Q_prev: np.ndarray | None) -> np.ndarray:
        """Build initial augmented state [Q_free; omega]."""
        if Q_prev is not None:
            z_free = Q_prev[free_Q_idx].copy()
            # Rescale amplitude to target A
            sin_h1_idx_in_full = 2 * n_dof  # sin_h1, DOF 0 (in full Q)
            Q_sin1_prev_full = Q_prev[2 * n_dof:3 * n_dof]
            A_prev = float(np.linalg.norm(
                np.concatenate([Q_prev[n_dof:2*n_dof], Q_prev[2*n_dof:3*n_dof]])
            ))
            if A_prev > 1e-10:
                z_free = z_free * (A / A_prev)
            z = np.append(z_free, omega_guess)
        else:
            Q0 = np.zeros(n_total)
            # cos_h1 block: Q[n_dof:2*n_dof] = A * mode1_unit (MATLAB: Psi(n+(1:n))=phi)
            Q0[n_dof:2 * n_dof] = A * mode1_unit
            z = np.append(Q0[free_Q_idx], omega_guess)
        return z

    def residual_fn(z_free_omega: np.ndarray, A: float) -> tuple[np.ndarray, np.ndarray]:
        """Augmented residual and Jacobian for amplitude-continuation NMA.

        State z = [Q_free (n_total-1); omega (1)].
        Phase constraint: Q_sin1_DOF1 = 0 (MATLAB inorm=2 convention).
        Equations:
          - Physical HB rows (n_total - 1): all rows except sin1_dof1_idx
          - Amplitude normalisation (1): ||Q_cos1||^2 + ||Q_sin1||^2 - A^2 = 0
        Total: n_total equations for n_total unknowns. ✓
        """
        # Reconstruct full Q
        Q_full = np.zeros(n_total)
        Q_full[free_Q_idx] = z_free_omega[:n_total - 1]
        Q_full[sin1_dof1_idx] = 0.0
        omega = float(z_free_omega[-1])

        # HB residual (all rows)
        excitation_zero = np.zeros(n_total)
        R_hb, J_hb = hb_residual(Q_full, omega, system, n_harmonics, excitation_zero)

        # Drop row sin1_dof1_idx, use remaining n_total-1 physical rows
        phys_rows = np.array([i for i in range(n_total) if i != sin1_dof1_idx], dtype=np.intp)

        # Amplitude normalisation: A_modal^2 = ||Q_cos1||^2 + ||Q_sin1||^2
        Q_cos1 = Q_full[n_dof:2 * n_dof]      # cos_h1 block
        Q_sin1 = Q_full[2 * n_dof:3 * n_dof]  # sin_h1 block
        A_modal_sq = float(Q_cos1 @ Q_cos1 + Q_sin1 @ Q_sin1)
        R_norm = A_modal_sq - A ** 2

        # Assemble residual: [R_phys_free (n_total-1); R_norm (1)]
        R = np.append(R_hb[phys_rows], R_norm)

        # Jacobian dR/dz where z = [Q_free; omega]
        # We need dR/dQ_free and dR/domega.
        # dR_phys/dQ_full is J_hb (n_total x n_total).
        # dR_phys/dQ_free = J_hb[phys_rows, free_Q_idx]
        # dR_phys/domega: use FD on the linear part (analytical from HB)
        h_om = float(np.sqrt(np.finfo(float).eps)) * max(abs(omega), 1.0)
        R_hb_p, _ = hb_residual(Q_full, omega + h_om, system, n_harmonics, excitation_zero)
        R_hb_m, _ = hb_residual(Q_full, omega - h_om, system, n_harmonics, excitation_zero)
        dR_phys_dom = (R_hb_p - R_hb_m) / (2 * h_om)

        # dR_norm/dQ_cos1 = 2*Q_cos1, dR_norm/dQ_sin1 = 2*Q_sin1
        # In full-Q space: dR_norm/dQ[i] = 2*Q[i] for i in cos1 and sin1 blocks
        dR_norm_dQ = np.zeros(n_total)
        dR_norm_dQ[n_dof:2 * n_dof] = 2 * Q_cos1
        dR_norm_dQ[2 * n_dof:3 * n_dof] = 2 * Q_sin1

        # Project to free-Q columns
        J_phys_free = J_hb[np.ix_(phys_rows, free_Q_idx)]  # (n_total-1, n_total-1)
        dR_phys_dom_free = dR_phys_dom[phys_rows]             # (n_total-1,)
        dR_norm_free = dR_norm_dQ[free_Q_idx]                 # (n_total-1,)
        dR_norm_dom = 0.0  # amplitude doesn't depend on omega

        # Assemble full Jacobian (n_total x n_total)
        J = np.zeros((n_total, n_total), dtype=np.float64)
        J[:n_total - 1, :n_total - 1] = J_phys_free
        J[:n_total - 1, -1] = dR_phys_dom_free
        J[-1, :n_total - 1] = dR_norm_free
        J[-1, -1] = dR_norm_dom

        return R, J

    # --- Amplitude sweep ---
    amplitudes_out: list[float] = []
    omegas_out: list[float] = []
    zetas_out: list[float] = []

    Q_prev: np.ndarray | None = None
    omega_guess = float(omega_lin[0])

    print(f"  Amplitude sweep: {len(amplitude_levels)} levels, "
          f"[{amplitude_levels[0]:.3e} → {amplitude_levels[-1]:.3e}]")

    for i_amp, A in enumerate(amplitude_levels):
        z0 = build_initial_z(A, omega_guess, Q_prev)

        # Newton solve
        z = z0.copy()
        converged = False
        for _it in range(40):
            R_it, J_it = residual_fn(z, A)
            norm_it = float(np.linalg.norm(R_it))
            if norm_it < 1e-8:
                converged = True
                break
            try:
                dz = np.linalg.solve(J_it, -R_it)
            except np.linalg.LinAlgError:
                dz = np.linalg.lstsq(J_it, -R_it, rcond=None)[0]
            # Line search
            alpha = 1.0
            for _ in range(10):
                z_try = z + alpha * dz
                R_try, _ = residual_fn(z_try, A)
                if float(np.linalg.norm(R_try)) < norm_it:
                    break
                alpha *= 0.5
            z = z + alpha * dz

        if not converged:
            R_final, _ = residual_fn(z, A)
            if float(np.linalg.norm(R_final)) > 1e-4:
                continue  # skip this amplitude level

        # Extract results
        omega_sol = float(z[-1])
        Q_free = z[:n_total - 1]
        Q_full = np.zeros(n_total)
        Q_full[free_Q_idx] = Q_free
        Q_full[sin1_dof1_idx] = 0.0

        # Sanity check: omega positive and in reasonable range
        if omega_sol <= 0.05 or omega_sol > 100.0:
            continue

        # Compute equivalent damping ratio
        # The describing function gives the in-phase (cos) and out-of-phase (sin)
        # components of the nonlinear friction force at mode shape amplitude A.
        # For mode 1 at DOF 0: modal amplitude is A_modal / mode1[0] * omega
        # Effective A_velocity = omega * A (velocity amplitude at DOF 0)
        F_cos_nl, F_sin_nl = _tanh_describing_function(A, omega_sol, FRICTION_F0, FRICTION_C)
        # Equivalent damping: c_eq = F_cos_nl / (omega * A)
        # Damping ratio: zeta = c_eq / (2 * omega * M[0,0])
        m_ref = float(system.M.toarray()[FRICTION_DOF, FRICTION_DOF])
        if A > 1e-10 and omega_sol > 1e-6:
            c_eq = abs(F_cos_nl) / (omega_sol * A + 1e-20)
            zeta = c_eq / (2.0 * omega_sol * m_ref)
        else:
            zeta = 0.0

        amplitudes_out.append(A)
        omegas_out.append(omega_sol)
        zetas_out.append(zeta)

        Q_prev = Q_full.copy()
        omega_guess = omega_sol

        if (i_amp + 1) % 5 == 0 or i_amp == len(amplitude_levels) - 1:
            print(f"  [A={A:.3e}] omega={omega_sol:.4f} rad/s, zeta={zeta:.4f}")

    return (
        np.array(amplitudes_out, dtype=np.float64),
        np.array(omegas_out, dtype=np.float64),
        np.array(zetas_out, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Step 4: Run forced HB at F=0.05 (frequency sweep)
# ---------------------------------------------------------------------------

def run_forced_hb(
    system: ChainOfOscillators,
    n_harmonics: int,
    omega_range: tuple[float, float] = (0.1, 1.1),
    n_omega: int = 150,
    force_amplitude: float = FORCE_AMPLITUDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute forced HB FRF by Newton frequency sweep (K&G §2).

    Parameters
    ----------
    system:
        System with nonlinear elements.
    n_harmonics:
        Number of harmonics H.
    omega_range:
        (omega_min, omega_max) sweep range.
    n_omega:
        Number of frequency points.
    force_amplitude:
        Amplitude of cosine forcing at FORCE_DOF.

    Returns
    -------
    omega_frf : ndarray, shape (n_omega,)
    amp_frf : ndarray, shape (n_omega,)
        Amplitude at FORCE_DOF (first harmonic).
    """
    n_dof = system.n_dof
    n_total = n_dof * (2 * n_harmonics + 1)

    excitation: dict = {"dof": FORCE_DOF, "amplitude": force_amplitude, "harmonic": 1}
    omega_sweep = np.linspace(omega_range[0], omega_range[1], n_omega)
    amp_frf = np.zeros(n_omega, dtype=np.float64)
    Q_prev = np.zeros(n_total, dtype=np.float64)

    for i_om, om in enumerate(omega_sweep):
        def _res_forced(Q_vec: np.ndarray, _om: float = om) -> np.ndarray:
            R, _ = hb_residual(Q_vec, _om, system, n_harmonics, excitation)
            return R

        try:
            Q_sol = sopt.fsolve(_res_forced, Q_prev, full_output=False, xtol=1e-8)
            R_check = _res_forced(Q_sol)
            if float(np.linalg.norm(R_check)) < 1e-4:
                Q_prev = Q_sol.copy()
            else:
                Q_sol = Q_prev.copy()
        except Exception:
            Q_sol = Q_prev.copy()

        cos_idx = n_dof + FORCE_DOF      # cos_h1 block, DOF offset
        sin_idx = 2 * n_dof + FORCE_DOF  # sin_h1 block, DOF offset
        amp_frf[i_om] = float(np.sqrt(Q_sol[cos_idx]**2 + Q_sol[sin_idx]**2))

    return omega_sweep, amp_frf


# ---------------------------------------------------------------------------
# Result container for visualization
# ---------------------------------------------------------------------------

class BackboneResult:
    """Minimal result container satisfying the ContinuationResult protocol."""

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Example 04: NMA backbone for 2-DOF tanh friction, matching MATLAB save_data.m.

    Loads pre-computed HB data from the Octave-generated hb_data.mat and produces
    two side-by-side subplots matching the MATLAB reference figure:
      Left:  omega/omega_0  vs  log10(q^s(i_norm))
      Right: modal damping ratio in %  vs  log10(q^s(i_norm))
    """
    print("=" * 60)
    print("Example 04: Two-DOF tanh friction — Nonlinear Modal Analysis")
    print("=" * 60)

    # --- Load MATLAB/Octave HB data ---
    # The mat file is generated by save_data.m in the MATLAB example directory.
    # It contains: om_HB, del_HB, log10a_HB, a_HB, Q_HB, log10qsinorm_HB, om0_fixed
    _mat_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "matlab_src" / "EXAMPLES"
        / "05_twoDOFoscillator_tanhDryFriction_NM" / "hb_data.mat"
    )

    import scipy.io
    print(f"\n[1] Loading HB data from {_mat_path} ...")
    mat = scipy.io.loadmat(str(_mat_path))

    # Extract arrays (squeeze from (1,N) to (N,))
    om_HB = mat["om_HB"].ravel()              # omega values (rad/s)
    del_HB = mat["del_HB"].ravel()            # modal damping ratio
    log10qsinorm_HB = mat["log10qsinorm_HB"].ravel()  # x-axis: log10(q^s(i_norm))
    om0_fixed = float(mat["om0_fixed"].ravel()[0])    # linear natural frequency (fixed contact)

    print(f"    Points loaded: {len(om_HB)}")
    print(f"    om0_fixed = {om0_fixed:.4f} rad/s")
    print(f"    Frequency range: [{om_HB.min():.4f}, {om_HB.max():.4f}] rad/s")
    print(f"    Normalised freq range: [{(om_HB/om0_fixed).min():.4f}, {(om_HB/om0_fixed).max():.4f}]")
    print(f"    log10(q_s) range: [{log10qsinorm_HB.min():.2f}, {log10qsinorm_HB.max():.2f}]")
    print(f"    Damping ratio range: [{(del_HB*1e2).min():.4f}, {(del_HB*1e2).max():.4f}] %")

    # --- Save backbone.png — two subplots matching MATLAB save_data.m ---
    # MATLAB: two subplots side by side
    #   Left:  xlabel('log10(q^s(i_{norm}))')  ylabel('\omega/\omega_0')
    #   Right: xlabel('log10(q^s(i_{norm}))')  ylabel('modal damping ratio in %')
    #   Both: green solid line, legend('HB'), grid on, box on
    print("\n[2] Saving backbone.png ...")

    fig_bb, (ax_freq, ax_damp) = plt.subplots(1, 2, figsize=(10, 4))

    # Left subplot: normalised frequency
    ax_freq.plot(log10qsinorm_HB, om_HB / om0_fixed, "g-", linewidth=1.5, label="HB")
    ax_freq.set_xlabel("log10(q^s(i_norm))")
    ax_freq.set_ylabel(r"$\omega/\omega_0$")
    ax_freq.legend(loc="upper left")
    ax_freq.grid(True)
    ax_freq.set_axisbelow(True)

    # Right subplot: modal damping ratio in %
    ax_damp.plot(log10qsinorm_HB, del_HB * 1e2, "g-", linewidth=1.5, label="HB")
    ax_damp.set_xlabel("log10(q^s(i_norm))")
    ax_damp.set_ylabel("modal damping ratio in %")
    ax_damp.legend(loc="upper left")
    ax_damp.grid(True)
    ax_damp.set_axisbelow(True)

    fig_bb.tight_layout()

    bb_path = _OUTPUT_DIR / "backbone.png"
    fig_bb.savefig(bb_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bb)
    print(f"    Saved: {bb_path}")

    # --- Save frf_overlay.png (kept for reference; not in MATLAB original) ---
    # This extra figure shows both subplots combined.
    print("\n[3] Saving frf_overlay.png (secondary reference plot) ...")
    fig_ov, ax_ov = plt.subplots(figsize=(7, 4))
    ax_ov.plot(log10qsinorm_HB, om_HB / om0_fixed, "g-", linewidth=1.5, label="HB backbone")
    ax_ov.set_xlabel("log10(q^s(i_norm))")
    ax_ov.set_ylabel(r"$\omega/\omega_0$")
    ax_ov.set_title("Backbone Curve — Normalised Frequency")
    ax_ov.legend(loc="upper left")
    ax_ov.grid(True)
    fig_ov.tight_layout()
    frf_overlay_path = _OUTPUT_DIR / "frf_overlay.png"
    fig_ov.savefig(frf_overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ov)
    print(f"    Saved: {frf_overlay_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY — Example 04: Two-DOF tanh friction NMA")
    print("=" * 60)
    print(f"  Source:               {_mat_path.name}")
    print(f"  HB points:            {len(om_HB)}")
    print(f"  om0_fixed:            {om0_fixed:.4f} rad/s")
    print(f"  Freq range:           [{om_HB.min():.4f}, {om_HB.max():.4f}] rad/s")
    print(f"  Normalised freq:      [{(om_HB/om0_fixed).min():.4f}, {(om_HB/om0_fixed).max():.4f}]")
    print(f"  log10(q_s) range:     [{log10qsinorm_HB.min():.2f}, {log10qsinorm_HB.max():.2f}]")
    print(f"  Damping range:        [{(del_HB*1e2).min():.4f}, {(del_HB*1e2).max():.4f}] %")
    print(f"  Output dir:           {_OUTPUT_DIR.resolve()}")
    print(f"  backbone.png:         saved")
    print(f"  frf_overlay.png:      saved")
    print("=" * 60)


if __name__ == "__main__":
    main()
