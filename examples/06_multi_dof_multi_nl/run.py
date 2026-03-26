"""
Example 06 — Multi-DOF Multi-Nonlinearity FRF.

3-DOF chain of oscillators with multiple nonlinear elements.
Matches MATLAB 07_multiDOFoscillator_multipleNonlinearities.m.

Nonlinear elements:
  - tanh_dry_friction(f0=1.0, c=20.0)  at DOF 0 (approx for elasticDryFriction W1)
  - tanh_dry_friction(f0=1.0, c=20.0)  at DOF 1 (approx for elasticDryFriction W2)
  - cubic_spring(k3=1.0)               at DOF 1 (cubicSpring W3)
  - polynomial_stiffness (rel. cubic k3=1) for relative DOF 1-2 (cubicSpring W4)
  - unilateral_spring(k=1.0, gap=0.25) at DOF 2 (unilateralSpring W5)

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
from nlvib.nonlinearities.elements import cubic_spring, tanh_dry_friction, polynomial_stiffness, unilateral_spring
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

# MATLAB nonlinear elements:
# 1. elasticDryFriction: knl=20, muN=1, W1=[1;0;0] → DOF 0 (approx with tanh)
system.add_nonlinear_element(tanh_dry_friction(f0=1.0, c=20.0, dof_index=0))  # MATLAB: f0=1.0, c=knl/muN=20

# 2. elasticDryFriction: knl=20, muN=1, W2=[-1;1;0] → relative DOF 0-1 (simplified: single-DOF approx at DOF 1)
system.add_nonlinear_element(tanh_dry_friction(f0=1.0, c=20.0, dof_index=1))  # MATLAB: relative DOF 0-1, simplified

# 3. cubicSpring: k3=1, W3=[0;1;0] → DOF 1
system.add_nonlinear_element(cubic_spring(k3=1.0, dof_index=1))  # MATLAB: k3=1 (was 0.5 at DOF 0)

# 4. cubicSpring: k3=1, W4=[0;-1;1] → relative DOF 1-2 (polynomial_stiffness trick)
_k3_rel = 1.0  # MATLAB: k3=1 for relative cubic DOF 1-2
_exp_rel = np.array([[3, 0], [2, 1], [1, 2], [0, 3]], dtype=np.intp)
_coeff_rel = np.array([_k3_rel, -3 * _k3_rel, 3 * _k3_rel, -_k3_rel])
system.add_nonlinear_element(polynomial_stiffness(_exp_rel, _coeff_rel, np.array([1, 2], dtype=np.intp)))  # force on DOF 1
system.add_nonlinear_element(polynomial_stiffness(_exp_rel, _coeff_rel, np.array([2, 1], dtype=np.intp)))  # force on DOF 2

# 5. unilateralSpring: k=1, gap=0.25, W5=[0;0;1] → DOF 2
system.add_nonlinear_element(unilateral_spring(k=1.0, gap=0.25, dof_index=2))  # MATLAB: k=1, gap=0.25

excitation = {"dof": 1, "amplitude": FORCE_AMP, "harmonic": 1}  # MATLAB: Fex1=[0;1;0] → DOF 1 (was DOF 0)

# ---------------------------------------------------------------------------
# Initial solution at omega_start
# ---------------------------------------------------------------------------
n_dof   = system.n_dof
omega_start = OMEGA_MIN
n_total = n_dof * (2 * N_HARMONICS + 1)
Q0      = np.zeros(n_total, dtype=np.float64)

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
# Post-process: extract omega and amplitudes for each DOF (harmonic 1)
# ---------------------------------------------------------------------------
solutions = result.solutions  # (n_steps, n_total + 1)
omegas    = solutions[:, -1]
stability = result.stability

# Harmonic 1 amplitude for each DOF:
# cos1 block starts at n_dof * 1, sin1 block at n_dof * 2
amplitudes = np.zeros((n_dof, len(omegas)), dtype=np.float64)
for dof in range(n_dof):
    cos1 = solutions[:, n_dof * 1 + dof]
    sin1 = solutions[:, n_dof * 2 + dof]
    amplitudes[dof, :] = np.sqrt(cos1**2 + sin1**2)

# ---------------------------------------------------------------------------
# Plot: FRF for all DOFs
# ---------------------------------------------------------------------------
fig_frf, axes_frf = plt.subplots(n_dof, 1, figsize=(8, 10), sharex=True)

for dof in range(n_dof):
    ax = axes_frf[dof]
    amp = amplitudes[dof, :]
    for i in range(len(omegas) - 1):
        is_stable = not bool(stability[i])
        color = "tab:blue" if is_stable else "tab:red"
        ls    = "-" if is_stable else "--"
        ax.plot(omegas[i:i+2], amp[i:i+2], color=color, linestyle=ls, linewidth=1.2)
    ax.set_ylabel(f"$|q_{dof}|$")
    ax.set_title(f"DOF {dof}")

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="tab:blue", linestyle="-",  label="stable"),
    Line2D([0], [0], color="tab:red",  linestyle="--", label="unstable"),
]
axes_frf[0].legend(handles=handles, loc="upper left")
axes_frf[-1].set_xlabel(r"Excitation frequency $\Omega$ (rad/s)")
fig_frf.suptitle("Example 06 — Multi-DOF Multi-NL FRF (all DOFs)")
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
print("  Example 06 — Multi-DOF Multi-NL Peak Amplitudes")
print("=" * 55)
for dof in range(n_dof):
    amp = amplitudes[dof, :]
    if len(amp) > 0:
        peak_idx   = int(np.argmax(amp))
        peak_amp   = float(amp[peak_idx])
        peak_omega = float(omegas[peak_idx])
        print(f"  DOF {dof}: peak amplitude = {peak_amp:.6f}  at omega = {peak_omega:.4f} rad/s")
    else:
        print(f"  DOF {dof}: no data")
print("=" * 55)
