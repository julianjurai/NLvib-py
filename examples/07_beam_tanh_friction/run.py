"""
Example 07 — FE Euler-Bernoulli Beam with Tanh Dry Friction.

Cantilever beam (clamped-free) with tanh dry friction at midpoint node (node 5).
Demonstrates FRF near first bending resonance and the spatial mode shape.

System parameters
-----------------
n_elements = 10
L = 1.0 m
E = 2.1e11 Pa  (steel Young's modulus)
I_area = 1e-8 m^4
rho = 7800 kg/m^3
A = 1e-4 m^2
bc = "clamped-free"

Nonlinearity: tanh_dry_friction(f0=5.0, c=100.0) at node 5, dof_type="w"
Excitation  : F = 10.0 at tip (node 10), dof_type="w", harmonic 1
Frequency   : omega in [150, 250] rad/s, n_harmonics = 3

Outputs
-------
examples/07_beam_tanh_friction/output/frf.png
  - FRF at tip node (node 10)
examples/07_beam_tanh_friction/output/mode_shape.png
  - Mode shape (transverse displacement DOFs) at resonance peak

Printed summary
---------------
- Resonance frequency
- Tip amplitude at resonance
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nlvib.systems.fe_beam import FE_EulerBernoulliBeam
from nlvib.nonlinearities.elements import tanh_dry_friction
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------------
N_ELEMENTS = 19                  # MATLAB: 10 (n_nodes=20 → n_elements=19)
L_BEAM     = 0.7                 # MATLAB: 1.0 (len=0.7)
E_MOD      = 2.05e11             # MATLAB: 2.1e11 (E=2.05e11)
I_AREA     = 3.201e-9            # MATLAB: 1e-8 (I=0.014*0.014^3/12≈3.201e-9)
RHO        = 7800.0
A_SECT     = 1.96e-4             # MATLAB: 1e-4 (A=0.014*0.014=1.96e-4)
BC         = "clamped-free"

# Nonlinearity at midpoint node 10 (midpoint of 19 elements)
FRICTION_F0 = 1.5                # MATLAB: 5.0 (muN=1.5)
FRICTION_C  = 1666667.0          # MATLAB: 100.0 (c=1/eps=1/6e-7≈1666667)
FRICTION_NODE = 10               # MATLAB: 5 (midpoint of 19 elements)

# Excitation at tip (node 19)
FORCE_AMP   = 10.0
FORCE_NODE  = N_ELEMENTS         # = 19 (MATLAB: 10)

N_HARMONICS = 7                  # MATLAB: 3 (H=7)
OMEGA_MIN   = 150.0
OMEGA_MAX   = 250.0

beam = FE_EulerBernoulliBeam(
    n_elements=N_ELEMENTS,
    L=L_BEAM,
    E=E_MOD,
    I_area=I_AREA,
    rho=RHO,
    A=A_SECT,
    bc=BC,
)

# Attach tanh friction at midpoint
midpoint_dof = beam.find_dof(FRICTION_NODE, "w")
nl_element   = tanh_dry_friction(f0=FRICTION_F0, c=FRICTION_C, dof_index=midpoint_dof)
beam.add_nonlinear_attachment(FRICTION_NODE, "w", nl_element)

# Add tip forcing
beam.add_forcing(FORCE_NODE, "w", FORCE_AMP)

# Build excitation dict for hb_residual
tip_dof   = beam.find_dof(FORCE_NODE, "w")
excitation = {"dof": tip_dof, "amplitude": FORCE_AMP, "harmonic": 1}

# ---------------------------------------------------------------------------
# Linear eigenfrequency (first mode) for reference
# ---------------------------------------------------------------------------
K_dense = beam.K.toarray()
M_dense = beam.M.toarray()
eigvals, eigvecs = eigh(K_dense, M_dense)
omega1_linear = float(np.sqrt(np.abs(eigvals[0])))
print(f"Linear first natural frequency: {omega1_linear:.2f} rad/s")

# Adjust omega range if needed to bracket the resonance
if not (OMEGA_MIN <= omega1_linear <= OMEGA_MAX):
    # Widen the search window around the resonance
    sweep_center = omega1_linear
    half_width   = max(50.0, 0.3 * sweep_center)
    omega_start  = max(1.0, sweep_center - half_width)
    omega_end    = sweep_center + half_width
    print(f"Adjusting frequency range to [{omega_start:.1f}, {omega_end:.1f}] rad/s")
else:
    omega_start = OMEGA_MIN
    omega_end   = OMEGA_MAX

# ---------------------------------------------------------------------------
# Initial solution
# ---------------------------------------------------------------------------
n_dof   = beam.n_dof
n_total = n_dof * (2 * N_HARMONICS + 1)
Q0      = np.zeros(n_total, dtype=np.float64)

for _newton in range(40):
    R, J = hb_residual(Q0, omega_start, beam, N_HARMONICS, excitation)
    if np.linalg.norm(R) < 1e-8:
        break
    try:
        dQ = np.linalg.solve(J, -R)
    except np.linalg.LinAlgError:
        dQ = np.linalg.lstsq(J, -R, rcond=None)[0]
    # Guard against divergence
    step_norm = np.linalg.norm(dQ)
    if step_norm > 10.0:
        dQ *= 10.0 / step_norm
    Q0 += dQ

print(f"Initial residual at omega={omega_start:.1f}: {np.linalg.norm(R):.3e}")

# ---------------------------------------------------------------------------
# Arc-length continuation
# ---------------------------------------------------------------------------
def residual_fn(Q: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    return hb_residual(Q, omega, beam, N_HARMONICS, excitation)

solver = ContinuationSolver()
opts = ContinuationOptions(
        verbose=True,
    ds_initial=0.5,
    ds_min=1e-4,
    ds_max=5.0,
    max_steps=800,
    newton_tol=1e-6,
    lambda_min=omega_start,
    lambda_max=omega_end,
)
result = solver.run(residual_fn, Q0, omega_start, opts)

print(f"Continuation: {result.n_steps} steps, converged={result.converged}")
print(f"  Termination: {result.message}")

# ---------------------------------------------------------------------------
# Post-process: extract FRF at tip DOF
# ---------------------------------------------------------------------------
solutions = result.solutions  # (n_steps, n_total + 1)
omegas    = solutions[:, -1]
stability = result.stability

cos1_tip = solutions[:, n_dof * 1 + tip_dof]
sin1_tip = solutions[:, n_dof * 2 + tip_dof]
amp_tip  = np.sqrt(cos1_tip**2 + sin1_tip**2)

if len(amp_tip) > 0:
    peak_idx       = int(np.argmax(amp_tip))
    peak_amp_tip   = float(amp_tip[peak_idx])
    resonance_freq = float(omegas[peak_idx])
else:
    peak_idx       = 0
    peak_amp_tip   = float("nan")
    resonance_freq = float("nan")

# ---------------------------------------------------------------------------
# Plot: FRF at tip
# ---------------------------------------------------------------------------
fig_frf, ax_frf = plt.subplots(figsize=(8, 5))
for i in range(len(omegas) - 1):
    is_stable = not bool(stability[i])
    color = "tab:blue" if is_stable else "tab:red"
    ls    = "-" if is_stable else "--"
    ax_frf.plot(omegas[i:i+2], amp_tip[i:i+2], color=color, linestyle=ls, linewidth=1.5)

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="tab:blue", linestyle="-",  label="stable"),
    Line2D([0], [0], color="tab:red",  linestyle="--", label="unstable"),
]
ax_frf.legend(handles=handles)
ax_frf.set_xlabel(r"Excitation frequency $\Omega$ (rad/s)")
ax_frf.set_ylabel(r"Tip amplitude $|w_{\rm tip}|$ (harmonic 1)")
ax_frf.set_title("Example 07 — Beam Tanh Friction FRF (tip)")
ax_frf.axvline(resonance_freq, color="gray", linestyle=":", linewidth=0.8)

frf_path = OUTPUT_DIR / "frf.png"
fig_frf.tight_layout()
fig_frf.savefig(frf_path, dpi=150)
plt.close(fig_frf)
print(f"\nPlot saved: {frf_path}")

# ---------------------------------------------------------------------------
# Mode shape at resonance peak
# ---------------------------------------------------------------------------
# Extract displacement Fourier coefficients (cosine h=1 block) at peak step
Q_peak   = solutions[peak_idx, :n_total]  # (n_total,)

# Transverse displacement DOFs (dof_type="w") are at even indices in free_dofs
# We extract the amplitude (sqrt(cos^2 + sin^2)) for each "w" DOF
free_dofs = beam.free_dofs  # global unreduced DOF indices

# Collect (node_position, amplitude) pairs for "w" DOFs
node_positions = []
mode_shape_amp = []

n_nodes = N_ELEMENTS + 1
# Include clamped node 0 at position 0 with displacement = 0
node_positions.append(0.0)
mode_shape_amp.append(0.0)

for node_i in range(1, n_nodes):
    try:
        reduced_dof = beam.find_dof(node_i, "w")
        cos1_val    = Q_peak[n_dof * 1 + reduced_dof]
        sin1_val    = Q_peak[n_dof * 2 + reduced_dof]
        amp_val     = float(np.sqrt(cos1_val**2 + sin1_val**2))
        node_pos    = node_i * L_BEAM / N_ELEMENTS
        node_positions.append(node_pos)
        mode_shape_amp.append(amp_val)
    except ValueError:
        # DOF is constrained — skip
        pass

node_positions_arr = np.array(node_positions)
mode_shape_arr     = np.array(mode_shape_amp)

fig_mode, ax_mode = plt.subplots(figsize=(8, 4))
ax_mode.plot(node_positions_arr, np.zeros_like(node_positions_arr), "k--",
             linewidth=0.6, label="undeformed")
ax_mode.plot(node_positions_arr, mode_shape_arr, "o-", color="tab:green",
             linewidth=1.5, label=f"mode shape at resonance ({resonance_freq:.1f} rad/s)")
ax_mode.fill_between(node_positions_arr, mode_shape_arr, alpha=0.2, color="tab:green")
ax_mode.set_xlabel("Position along beam (m)")
ax_mode.set_ylabel(r"Transverse amplitude (m)")
ax_mode.set_title("Example 07 — Mode Shape at Resonance Peak")
ax_mode.legend()
ax_mode.grid(True, alpha=0.3)

mode_path = OUTPUT_DIR / "mode_shape.png"
fig_mode.tight_layout()
fig_mode.savefig(mode_path, dpi=150)
plt.close(fig_mode)
print(f"Plot saved: {mode_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  Example 07 — Beam Tanh Friction Summary")
print("=" * 55)
print(f"  Linear 1st natural frequency : {omega1_linear:.2f} rad/s")
print(f"  Resonance frequency (HB)     : {resonance_freq:.2f} rad/s")
print(f"  Tip amplitude at resonance   : {peak_amp_tip:.6e} m")
print("=" * 55)
