"""
Example 06 — Multi-DOF Multi-Nonlinearity FRF.

3-DOF chain of oscillators with multiple nonlinear elements.
Matches MATLAB 07_multiDOFoscillator_multipleNonlinearities.m exactly.

Nonlinear elements (matching MATLAB):
  - elastic_dry_friction(k=20, f_lim=1.0, W1=[1,0,0])   Jenkins at DOF 0
  - elastic_dry_friction(k=20, f_lim=1.0, W2=[-1,1,0])  Jenkins on relative DOF 0-1
  - cubic_spring(k3=1.0)               at DOF 1 (W3=[0,1,0])
  - polynomial_stiffness (rel. cubic k3=1) for relative DOF 1-2 (W4=[0,-1,1])
  - unilateral_spring(k=1.0, gap=0.25) at DOF 2 (W5=[0,0,1])

System parameters (MATLAB: 07_multiDOFoscillator_multipleNonlinearities.m)
-----------------
masses      = [1.0, 1.0, 1.0]
stiffnesses = [1.0, 1.0, 1.0, 1.0]
dampings    = [0.02] * 4
Excitation  : F = 1.0 at DOF 1, harmonic 1
Frequency   : omega in [0.5, 2.0] rad/s, n_harmonics = 7

Outputs
-------
examples/06_multi_dof_multi_nl/output/frf_all_dofs.png
  - FRF for all 4 DOFs (4 subplots)
examples/06_multi_dof_multi_nl/output/convergence.png
  - Step-size history as a proxy for Newton convergence

Printed summary
---------------
- Peak amplitude for each DOF
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.nonlinearities.elements import (
    cubic_spring,
    elastic_dry_friction,
    polynomial_stiffness,
    unilateral_spring,
)
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# System definition (MATLAB: 07_multiDOFoscillator_multipleNonlinearities.m)
# ---------------------------------------------------------------------------
MASSES      = [1.0, 1.0, 1.0]              # MATLAB: [1.0, 1.5, 1.0, 0.8] (mi=[1,1,1])
STIFFNESSES = [1.0, 1.0, 1.0, 1.0]        # MATLAB: [1.0, 0.8, 1.2, 0.6, 1.0] (ki=[1,1,1,1])
DAMPINGS    = [0.02, 0.02, 0.02, 0.02]    # MATLAB: [0.02]*5 (di=0.02*ki)
FORCE_AMP   = 1.0                          # MATLAB: 0.1 (Fex1=[0;1;0])
OMEGA_MIN   = 0.5                          # MATLAB: 0.3 (Om_s=0.5)
OMEGA_MAX   = 2.0                          # MATLAB: 1.5 (Om_e=2.0)
N_HARMONICS = 7                            # MATLAB: 3 (H=7)

system = ChainOfOscillators(
    masses=MASSES,
    stiffnesses=STIFFNESSES,
    dampings=DAMPINGS,
)

# Friction element directions (MATLAB: W1, W2)
_K_SLIP = 20.0   # Jenkins stiffness (knl)
_F_LIM  = 1.0    # Coulomb limit force (muN)
_W1 = np.array([1.0, 0.0, 0.0])   # MATLAB W1 = [1;0;0]
_W2 = np.array([-1.0, 1.0, 0.0])  # MATLAB W2 = [-1;1;0]

# 1. Jenkins element on DOF 0 (W1=[1,0,0])
system.add_nonlinear_element(elastic_dry_friction(k_slip=_K_SLIP, f_lim=_F_LIM, dof_index=0))

# 2. Jenkins element on relative DOF 0-1 (W2=[-1,1,0])
system.add_nonlinear_element(elastic_dry_friction(k_slip=_K_SLIP, f_lim=_F_LIM, force_direction=_W2))

# 3. cubicSpring: k3=1, W3=[0;1;0] → DOF 1
system.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=1))

# 4. cubicSpring: k3=1, W4=[0;-1;1] → relative DOF 1-2 (polynomial_stiffness trick)
_k3_rel = 1.0
_exp_rel = np.array([[3, 0], [2, 1], [1, 2], [0, 3]], dtype=np.intp)
_coeff_rel = np.array([_k3_rel, -3 * _k3_rel, 3 * _k3_rel, -_k3_rel])
system.add_nonlinear_element(polynomial_stiffness(_exp_rel, _coeff_rel, np.array([1, 2], dtype=np.intp)))  # force on DOF 1
system.add_nonlinear_element(polynomial_stiffness(_exp_rel, _coeff_rel, np.array([2, 1], dtype=np.intp)))  # force on DOF 2

# 5. unilateralSpring: k=1, gap=0.25, W5=[0;0;1] → DOF 2
system.add_nonlinear_element(unilateral_spring(k=1.0, gap=0.25, dof_index=2))

excitation = {"dof": 1, "amplitude": FORCE_AMP, "harmonic": 1}  # MATLAB: Fex1=[0;1;0] → DOF 1 (was DOF 0)

# ---------------------------------------------------------------------------
# Initial solution at omega_start (MATLAB: linear guess with Jenkins stiffness)
# ---------------------------------------------------------------------------
n_dof   = system.n_dof
omega_start = OMEGA_MIN
n_total = n_dof * (2 * N_HARMONICS + 1)

# Jenkins elements contribute a linear stiffness when fully stuck: dKfric = k*(W1*W1' + W2*W2')
# This matches MATLAB: dKfric = knl*(W1*W1' + W2*W2') used in the initial linear guess.
dKfric = _K_SLIP * (np.outer(_W1, _W1) + np.outer(_W2, _W2))
K_eff = system.K.toarray() + dKfric
M_arr = system.M.toarray()
D_arr = system.D.toarray()
Fex1  = np.array([0.0, FORCE_AMP, 0.0])  # MATLAB: Fex1=[0;1;0]

# Linear frequency-domain initial guess at omega_start
Q1_complex = np.linalg.solve(
    -(omega_start**2) * M_arr + 1j * omega_start * D_arr + K_eff, Fex1
)
Q0 = np.zeros(n_total, dtype=np.float64)
# Place at harmonic 1 cosine/sine blocks
for dof_idx in range(n_dof):
    Q0[n_dof * 1 + dof_idx] = float(np.real(Q1_complex[dof_idx]))   # cos1
    Q0[n_dof * 2 + dof_idx] = -float(np.imag(Q1_complex[dof_idx]))  # sin1

for _newton in range(40):
    R, J = hb_residual(Q0, omega_start, system, N_HARMONICS, excitation)
    if np.linalg.norm(R) < 1e-10:
        break
    try:
        dQ = np.linalg.solve(J, -R)
    except np.linalg.LinAlgError:
        dQ = np.linalg.lstsq(J, -R, rcond=None)[0]
    Q0 += dQ

# ---------------------------------------------------------------------------
# Arc-length continuation
# ---------------------------------------------------------------------------
def residual_fn(Q: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    return hb_residual(Q, omega, system, N_HARMONICS, excitation)

solver = ContinuationSolver()
opts = ContinuationOptions(
        verbose=True,
    ds_initial=0.01,
    ds_min=1e-6,
    ds_max=0.05,
    max_steps=1000,
    newton_tol=1e-8,
    lambda_min=OMEGA_MIN,
    lambda_max=OMEGA_MAX,
)
result = solver.run(residual_fn, Q0, omega_start, opts)

print(f"Continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")

# ---------------------------------------------------------------------------
# Post-process: extract omega and RMS amplitudes for each DOF
# MATLAB: Q1_rms = sqrt(sum(Q1.^2))/sqrt(2) where Q1 = X(1:n:end-1,:)
#         i.e. all harmonic coefficients of DOF index 0, 1, 2 (rows 1,2,3 of Q)
# In Python layout Q has shape (n_total, n_steps); harmonic coefficients for
# DOF `d` are at indices d, n_dof + d, 2*n_dof + d, ..., (2H)*n_dof + d
# RMS = sqrt(sum_k Qk^2) / sqrt(2)  (matches MATLAB convention)
# ---------------------------------------------------------------------------
solutions = result.solutions  # (n_steps, n_total + 1)
omegas    = solutions[:, -1]
stability = result.stability

Q_all = solutions[:, :-1]  # shape (n_steps, n_total); rows are solution points

# Collect all harmonic coefficients per DOF and compute RMS
# MATLAB RMS formula: sqrt(sum(all_coeff^2))/sqrt(2)
rms_amplitudes = np.zeros((n_dof, len(omegas)), dtype=np.float64)
for dof in range(n_dof):
    # Indices: dof, n_dof+dof, 2*n_dof+dof, ..., 2*H*n_dof+dof
    indices = [k * n_dof + dof for k in range(2 * N_HARMONICS + 1)]
    Q_dof = Q_all[:, indices]  # shape (n_steps, 2H+1)
    rms_amplitudes[dof, :] = np.sqrt(np.sum(Q_dof**2, axis=1)) / np.sqrt(2)

# ---------------------------------------------------------------------------
# Plot: FRF for all DOFs on a single figure (matches MATLAB layout)
# MATLAB: plot(OM,Q1_rms,'b-'); plot(OM,Q2_rms,'r-'); plot(OM,Q3_rms,'g-')
# xlabel('excitation frequency'); ylabel('response amplitude (RMS)')
# legend('q1','q2','q3')
# No explicit xlim/ylim — use x-range [Om_s, Om_e] = [0.5, 2.0], linear scale
# ---------------------------------------------------------------------------
DOF_COLORS = ["b", "r", "g"]
DOF_LABELS = ["q1", "q2", "q3"]

fig_frf, ax_frf = plt.subplots(figsize=(8, 5))

for dof in range(n_dof):
    amp = rms_amplitudes[dof, :]
    ax_frf.plot(omegas, amp, color=DOF_COLORS[dof], linewidth=1.2, label=DOF_LABELS[dof])

ax_frf.set_xlabel("excitation frequency")
ax_frf.set_ylabel("response amplitude (RMS)")
ax_frf.set_yscale("linear")   # MATLAB uses plot (linear scale)
ax_frf.set_xlim([OMEGA_MIN, OMEGA_MAX])   # MATLAB: Om_s=0.5, Om_e=2.0
ax_frf.legend(loc="upper right")   # MATLAB: location='northeast'
ax_frf.set_title("Example 06 — Multi-DOF Multi-NL FRF")
fig_frf.tight_layout()
frf_path = OUTPUT_DIR / "frf_all_dofs.png"
fig_frf.savefig(frf_path, dpi=150)
plt.close(fig_frf)
print(f"\nPlot saved: {frf_path}")

# ---------------------------------------------------------------------------
# Plot: convergence proxy (ds_history)
# ---------------------------------------------------------------------------
fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
ds_history = result.ds_history[1:]  # skip the initial 0.0 entry
step_indices = np.arange(1, len(ds_history) + 1)
ax_conv.semilogy(step_indices, ds_history, color="tab:green", linewidth=1.2)
ax_conv.set_xlabel("Continuation step")
ax_conv.set_ylabel("Arc-length step size ds")
ax_conv.set_title("Example 06 — Newton Convergence Proxy (ds history)")
ax_conv.grid(True, which="both", alpha=0.4)
fig_conv.tight_layout()
conv_path = OUTPUT_DIR / "convergence.png"
fig_conv.savefig(conv_path, dpi=150)
plt.close(fig_conv)
print(f"Plot saved: {conv_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  Example 06 — Multi-DOF Multi-NL Peak Amplitudes (RMS)")
print("=" * 55)
for dof in range(n_dof):
    amp = rms_amplitudes[dof, :]
    if len(amp) > 0:
        peak_idx   = int(np.argmax(amp))
        peak_amp   = float(amp[peak_idx])
        peak_omega = float(omegas[peak_idx])
        print(f"  DOF {dof}: peak RMS amplitude = {peak_amp:.6f}  at omega = {peak_omega:.4f} rad/s")
    else:
        print(f"  DOF {dof}: no data")
print("=" * 55)
