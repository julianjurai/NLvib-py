# Comparison Notebooks — Agent Context

## Goal
Each notebook in `examples/comparison/` runs the **Python HB continuation** and the
**MATLAB/Octave reference** for the same example, then overlays both curves on one
figure so numerical agreement (or discrepancy) is immediately visible.

---

## MATLAB ↔ Python Example Mapping

| Notebook to create | MATLAB script | Python run.py |
|--------------------|---------------|---------------|
| `01_duffing.ipynb` | `matlab_src/EXAMPLES/01_Duffing/Duffing.m` | `examples/01_duffing/run.py` |
| `02_two_dof_cubic.ipynb` | `matlab_src/EXAMPLES/02_twoDOFoscillator_cubicSpring/twoDOFoscillator_cubicSpring.m` | `examples/02_two_dof_cubic/run.py` |
| `03_two_dof_unilateral.ipynb` | `matlab_src/EXAMPLES/03_twoDOFoscillator_unilateralSpring/twoDOFoscillator_unilateralSpring.m` | `examples/03_two_dof_unilateral/run.py` |
| `04_two_dof_tanh_friction.ipynb` | `matlab_src/EXAMPLES/05_twoDOFoscillator_tanhDryFriction_NM/twoDOFoscillator_tanhDryFriction_NM.m` | `examples/04_two_dof_tanh_friction/run.py` |
| `05_geometric_nonlinearity.ipynb` | `matlab_src/EXAMPLES/06_twoSprings_geometricNonlinearity/twoSprings_geometricNonlinearity.m` | `examples/05_geometric_nonlinearity/run.py` |
| `06_multi_dof_multi_nl.ipynb` | `matlab_src/EXAMPLES/07_multiDOFoscillator_multipleNonlinearities/multiDOFoscillator_multipleNonlinearities.m` | `examples/06_multi_dof_multi_nl/run.py` |
| `07_beam_tanh_friction.ipynb` | `matlab_src/EXAMPLES/08_beam_tanhDryFriction/beam_tanhDryFriction_simple.m` | `examples/07_beam_tanh_friction/run.py` |
| `08_beam_cubic_spring_nma.ipynb` | `matlab_src/EXAMPLES/09_beam_cubicSpring_NM/beam_cubicSpring_NM1.m` | `examples/08_beam_cubic_spring_nma/run.py` |

---

## Approach: How to Get MATLAB Data into Python

### Step 1 — Create a data-saving wrapper script
For each example, create a thin Octave wrapper next to the MATLAB script that:
1. `addpath`s the NLvib source (`matlab/NLvib/SRC/`)
2. Runs the original example script
3. Saves the key variables to a `.mat` file

Example wrapper (`matlab_src/EXAMPLES/02_twoDOFoscillator_cubicSpring/save_data.m`):
```matlab
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'SRC')));
run('twoDOFoscillator_cubicSpring.m');
save('hb_data.mat', 'Om_HB', 'Q_HB', 'a_rms_HB');
```

### Step 2 — Run with Octave from Python (subprocess)
```python
import subprocess, shutil
script_dir = repo_root / 'matlab_src/EXAMPLES/02_twoDOFoscillator_cubicSpring'
subprocess.run(
    ['octave', '--no-gui', '--path', str(script_dir), 'save_data.m'],
    cwd=str(script_dir), check=True, capture_output=True
)
```
Octave binary is at `/usr/local/bin/octave` (version 11.1.0 installed).

### Step 3 — Load .mat in Python
```python
import scipy.io
data = scipy.io.loadmat(script_dir / 'hb_data.mat')
Om_HB_matlab = data['Om_HB'].ravel()
a_rms_HB_matlab = data['a_rms_HB'].ravel()
```

### Step 4 — Run Python continuation inline
Use the same system setup from the corresponding `run.py` (copy its parameters
directly — do NOT import run.py, reproduce the setup inline in the notebook).

### Step 5 — Overlay both curves
```python
fig, ax = plt.subplots()
ax.plot(Om_HB_matlab, a_rms_HB_matlab, 'g-', label='MATLAB/Octave HB')
ax.plot(omega_py, a_rms_py,            'b--', label='Python HB')
ax.set_xlabel('excitation frequency'); ax.set_ylabel('response amplitude')
ax.legend(); ax.set_title('Example XX — HB Comparison')
```

---

## MATLAB a_rms Formula (critical — match exactly)

MATLAB computes:
```matlab
a_rms_HB = sqrt(sum(Q_HB(1:2:end,:).^2)) / sqrt(2);
```
For `n_dof=2`, `1:2:end` picks **DOF 0 coefficients** (DC + all harmonics).

Python equivalent:
```python
# Q_all shape: (n_steps, n_dof*(2H+1))
# Reshape to (n_steps, 2H+1, n_dof), take DOF 0
Q_dof0 = Q_all.reshape(n_steps, 2*H+1, n_dof)[:, :, 0]
a_rms = np.sqrt(np.sum(Q_dof0**2, axis=1)) / np.sqrt(2)
```

For `n_dof=1` (Duffing), ALL Q entries are DOF 0, so:
```python
a_rms = np.sqrt(np.sum(Q_all**2, axis=1)) / np.sqrt(2)
```

---

## Nonlinear Element: elastic_dry_friction (Jenkins/Masing)

The `elastic_dry_friction` factory creates a hysteretic friction element — a spring of
stiffness *k_slip* in series with a Coulomb slider of slip-force limit *f_lim*. This is the
Jenkins (1962) / Masing (1926) model, distinct from `tanh_dry_friction` (which is
velocity-based and smooth).

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `k_slip` | Elastic stiffness of the stuck spring [N/m] |
| `f_lim` | Coulomb slip-force limit [N] |
| `dof_index` | For axis-aligned single-DOF elements |
| `force_direction` | Direction vector for multi-DOF relative-slip elements (e.g. `[-1, 1, 0]`) |

**Force law (Masing's rule)**

- *Stuck* regime: `|k_slip * (q - z)| < f_lim` → force = `k_slip * (q - z)`, slider `z`
  unchanged.
- *Sliding* regime: `|k_slip * (q - z)| >= f_lim` → force = `±f_lim`, slider `z` updates to
  `q ∓ f_lim / k_slip`.

**Hysteretic state in AFT**

The Jenkins element is path-dependent: `eval_batch` integrates **two periods** from zero
initial conditions and uses only the second (settled) period for the Fourier transform.
This matches the MATLAB `elasticDryFriction` function inside `HB_residual.m`. The scalar
`eval` method provides an instantaneous approximation (for Jacobian assembly) and should
not be used stand-alone for AFT.

**Reference**: Jenkins (1962); Masing (1926); Krack & Gross (2019) §C.2.

---

## Key Technical Learnings From Implementation

### 1. polynomial_stiffness target_dof bug (CRITICAL)
**Bug**: When `polynomial_stiffness` elements are used to implement an inter-DOF
spring with `dof_indices=[1,0]`, the old code placed the force at `min(dof_indices)=0`
(via `np.flatnonzero` of the gradient — always returns sorted indices). Both elements
of the pair targeted DOF 0, their forces cancelled (+f and −f), and DOF 1 received
**nothing**. This completely eliminated the fold structure from the FRF.

**Fix** (committed `0f3a25f`): `NonlinearElement` now has a `target_dof: int | None`
field. `polynomial_stiffness` sets `target_dof = dof_indices[0]`. `eval_nonlinear_forces`
uses `element.target_dof` when present, bypassing the gradient-based inference.

**Symptom**: Python FRF curve monotonically increasing vs MATLAB S-curve with fold.

### 2. AFT vectorization (220× speedup)
**Before**: `harmonic_balance.py` looped over `n_time=128` time steps calling
`eval_nonlinear_forces` once per step → 7,808 Python function calls per `hb_residual`.

**After**: Each `NonlinearElement` has an `eval_batch(q_time, dq_time) -> f_time`
method. `MechanicalSystem.eval_nonlinear_forces_batch()` calls batch methods directly
(numpy vectorized), falling back to scalar loop only for elements without `eval_batch`.

**Result**: 0.033 ms vs 7.2 ms per AFT call. Example 02 runs in ~4.5s (was ~30s+).

### 3. Continuation step direction / fold tracking
The arc-length continuation correctly tracks folds (turning points) via sign change of
`t_lam`. When the step size `ds_max` was too large relative to the amplitude changes near
the fold, Newton's corrector jumped to the wrong branch. This was resolved by the fix
above (once the correct physics were restored, the fold was sharp enough for ds_max=0.1
to track correctly).

### 4. Single-graph output convention
MATLAB example 02 produces ONE figure with `a_rms` (DOF 0, all harmonics) vs omega.
Python originally produced 3 separate PNGs. Fixed: Python now outputs `frequency_response.png`
with the same a_rms metric.

### 5. _FD_STEP patching for small-amplitude beam problems

The arc-length continuation uses a finite-difference step size `_FD_STEP` (default ~1e-7)
to estimate Jacobians numerically. For beam problems (Example 07, 08) where the harmonic
amplitudes are in the **Q ~ 1e-8 regime**, the default step is far too large relative to
the signal and produces severely degraded Jacobians.

**Fix**: Monkey-patch `_FD_STEP` to `1.5e-15` before running the beam continuation:

```python
import nlvib.continuation.predictor_corrector as _pc
_pc._FD_STEP = 1.5e-15
```

This must be applied before calling `continuation(...)`. Without it, beam FRF curves
may be distorted or fail to converge.

### 6. ChainOfOscillators stiffness convention
`stiffnesses=[k_left_ground, k_inter, k_right_ground]` — length = n_dof + 1.
In MATLAB example 02: `ki=[1, 0.0453, 0]` → Python `STIFFNESSES=[1.0, 0.0453, 0.0]`.

---

## Notebook Structure Template

Each `examples/comparison/XX_<name>.ipynb` should have these cells:

1. **Markdown header**: example name, what's being compared, reference
2. **Imports + path setup**
3. **Run Octave → save .mat** (subprocess call)
4. **Load MATLAB data** (scipy.io.loadmat)
5. **Python system setup + HB continuation** (inline, from run.py parameters)
6. **Compute a_rms** for both (using the formula above)
7. **Overlay plot** on one figure
8. **Max amplitude / peak frequency table** comparing both
9. **Pass/fail assertion**: `assert abs(peak_py - peak_matlab) / peak_matlab < 0.05`

### ## MATLAB vs Python section (6-cell standard block)

The final section of every notebook must follow this 6-cell layout:

| Cell # | Type | Content |
|--------|------|---------|
| 1 | Markdown | `## MATLAB vs Python` header — states example name, MATLAB source file, Python run.py |
| 2 | Code | Side-by-side or overlaid figure: MATLAB curve (green solid) + Python curve (blue dashed) with legend, x-label `ω [rad/s]`, y-label `a_rms [m]` |
| 3 | Code | Metrics table — peak amplitude (MATLAB, Python, abs diff, % relative error) and peak frequency (MATLAB, Python, abs diff, % relative error) |
| 4 | Code | Runtime cell — `%%time` magic or `time.perf_counter()` block recording wall time for Python continuation |
| 5 | Code | Harmonic content cell — bar chart or table of harmonic amplitudes (H1, H3, H5, …) at the peak frequency point for both MATLAB and Python |
| 6 | Code | MOE summary — `print(f"Peak amplitude error: ±{err:.2f}% (95% CI)")` for each metric; `assert` all errors below tolerance |

---

## Running the Notebooks

```bash
# From repo root
jupyter notebook examples/comparison/02_two_dof_cubic.ipynb
# or non-interactively:
jupyter nbconvert --to notebook --execute examples/comparison/02_two_dof_cubic.ipynb
```

## Prerequisite: MATLAB .mat save scripts

Each agent must create `matlab_src/EXAMPLES/<XX>/save_data.m` before running.
The SRC path to add is always: `../../SRC` relative to the EXAMPLES subfolder.
Check which variables the .m script outputs by reading it first.

---

## Example 02 is the Reference Implementation
Task `02_two_dof_cubic` was completed manually and validated. Use it as the template
for all other comparison notebooks. The Python curve now matches MATLAB's S-shaped
FRF with two peaks/folds. See `examples/comparison/02_two_dof_cubic.ipynb` once
that task is done.

---

## Notebook Status — All 8 Passing

All comparison notebooks pass `nbconvert --execute` with the following final peak-amplitude
errors (as of T-29–T-43):

| Notebook | Example | MATLAB Source | Peak Error | Notes |
|----------|---------|---------------|------------|-------|
| `02_two_dof_cubic.ipynb` | 02 Two-DOF cubic spring | `02_twoDOFoscillator_cubicSpring` | 0.01% | Reference template |
| `01_duffing.ipynb` | 01 Duffing oscillator | `01_Duffing` | 0.0007% | |
| `03_two_dof_unilateral.ipynb` | 03 Two-DOF unilateral spring | `03_twoDOFoscillator_unilateralSpring` | 0.08% | |
| `04_two_dof_tanh_friction.ipynb` | 04 Two-DOF tanh friction NMA | `05_twoDOFoscillator_tanhDryFriction_NM` | 0.09% | Phase constraint index fix |
| `05_geometric_nonlinearity.ipynb` | 05 Geometric nonlinearity | `06_twoSprings_geometricNonlinearity` | <1% | T-41: reduced ds_max near peak |
| `06_multi_dof_multi_nl.ipynb` | 06 Multi-DOF multi-NL | `07_multiDOFoscillator_multipleNonlinearities` | <5% | T-37/T-38: Jenkins element required |
| `07_beam_tanh_friction.ipynb` | 07 Beam tanh friction | `08_beam_tanhDryFriction` | 0.29% | _FD_STEP patched to 1.5e-15 |
| `08_beam_cubic_spring_nma.ipynb` | 08 Beam cubic spring NMA | `09_beam_cubicSpring_NM` | <5% | Galerkin reduction at high amplitude |

**Assertion tolerances**: examples 05, 06, 08 use `< 5%`; all others use `< 1%` (or tighter).

**Jenkins element caveat (example 06)**: MATLAB uses `elasticDryFriction` (hysteretic,
k_slip + f_lim). Python originally used `tanh_dry_friction` (smooth, velocity-based) which
produced 60.7% error. T-37 added `elastic_dry_friction` and T-38 updated the notebook —
error is now < 5%.
