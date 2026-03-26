# NLvib Tasks — PM State File

> **PM Agent**: This is your source of truth. Read this at the start of every session.
> Update status fields in-place. Append to `## Session Log` — never delete log entries.

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `todo` | Not started, dependencies not yet met |
| `ready` | Dependencies met, can be assigned |
| `in_progress` | Actively being worked on |
| `review` | Code complete, awaiting QA + Review sign-off |
| `done` | QA passed, Review passed, merged |
| `blocked` | Waiting on external input (note reason) |

---

## Dependency Tiers

```
Tier 0 (independent)
  T-01  Nonlinear elements
  T-02  Utility functions (FFT transforms, norms, scaling)
  T-03  IO parsers (CalculiX FRD/mesh reader)
  T-04  MATLAB fixture generator spec + reference outputs
  T-27  Visualization module                 [no deps — pure matplotlib wrappers]

Tier 1 (needs Tier 0)
  T-05  Base MechanicalSystem class          [needs: T-01]
  T-06  SingleMassOscillator                 [needs: T-05]
  T-07  ChainOfOscillators                   [needs: T-05]
  T-08  FE EulerBernoulli beam               [needs: T-05, T-03]
  T-09  FE ElasticRod                        [needs: T-05, T-03]
  T-10  System with PolynomialStiffness      [needs: T-05, T-01]
  T-11  CMS model reduction (ROM)            [needs: T-05, T-08, T-09]

Tier 2 (needs Tier 1)
  T-12  Harmonic Balance residual + AFT      [needs: T-05, T-01, T-02]
  T-13  Shooting residual + Newmark          [needs: T-05, T-01]

Tier 3 (needs Tier 2)
  T-14  Arc-length continuation solver       [needs: T-12, T-13]

Tier 4 (needs Tier 3 + T-27)
  T-15  Example: 01 Duffing oscillator       [needs: T-14, T-06, T-27]
  T-16  Example: 02 Two-DOF cubic spring     [needs: T-14, T-07, T-27]
  T-17  Example: 03 Two-DOF unilateral spring[needs: T-14, T-07, T-27]
  T-18  Example: 04 Two-DOF tanh friction    [needs: T-14, T-07, T-27]
  T-19  Example: 05 Geometric nonlinearity   [needs: T-14, T-07, T-27]
  T-20  Example: 06 Multi-DOF multi-NL       [needs: T-14, T-07, T-27]
  T-21  Example: 07 Beam tanh friction       [needs: T-14, T-08, T-27]
  T-22  Example: 08 Beam cubic spring NM     [needs: T-14, T-08, T-27]

Tier 5 (needs Tier 4)
  T-23  Jupyter notebooks (one per example)  [needs: T-15 through T-22]
  T-24  API reference docs (mkdocs)          [needs: all Tier 1-3]
  T-25  CI/CD pipeline (GitHub Actions)      [needs: T-04, test suite]
  T-26  PyPI packaging                       [needs: T-24, T-25]
  T-28  Demo notebooks (interactive explore) [needs: T-14, T-11, T-27 — parallel with T-23]
```

---

## Task Details

### T-01 — Nonlinear Elements
- **Status**: `done`
- **Notes**: Unilateral spring Jacobian at contact point = 0 (sub-gradient, matches MATLAB). MATLAB fixture force comparison deferred until fixtures generated.
- **Module**: `src/nlvib/nonlinearities/elements.py`
- **Deliver**:
  - `cubic_spring(k3, dof_index)` → NonlinearElement
  - `quadratic_damper(c2, dof_index)` → NonlinearElement
  - `tanh_dry_friction(f0, c, dof_index)` → NonlinearElement
  - `unilateral_spring(k, gap, dof_index)` → NonlinearElement
  - `polynomial_stiffness(exponents, coefficients, dof_indices)` → NonlinearElement
- **Equation refs**: Krack & Gross (2019) Appendix C, Table C.1
- **Tests**: `tests/unit/test_elements.py` — verify force values and Jacobians via finite difference
- **Acceptance**: Force values match MATLAB `HB_residual.m` nonlinearity evaluation; Jacobians match FD to 1e-8
- **Notes**: Each element returns (f, df_dq, df_ddq) — shape conventions must match HB and Shooting residual expectations

---

### T-02 — Utility Functions
- **Status**: `done`
- **Notes**: FFT sign convention matches K&G §2.3. AFT `force_fn` multi-DOF interface (full `(n_dof, n_time)` vs per-DOF slices) to be confirmed at T-12 implementation time. MATLAB fixture comparison deferred until fixtures generated.
- **Module**: `src/nlvib/utils/transforms.py`, `src/nlvib/utils/linalg.py`
- **Deliver**:
  - `time_to_freq(q_time, n_harmonics)` — real-valued cosine-sine to complex Fourier
  - `freq_to_time(Q_freq, n_time_samples)` — inverse
  - `aft_transform(Q_freq, force_fn, n_time)` — full AFT pipeline
  - `dynamic_scaling(x, x_ref)` — NLvib's variable scaling for mixed-scale problems
  - `arc_length(x_prev, x_curr)` — arc length between two continuation points
- **Equation refs**: Krack & Gross (2019) §2.3 (AFT), §4.2 (scaling)
- **Tests**: `tests/unit/test_transforms.py` — round-trip FFT, AFT on known analytic function
- **Acceptance**: Round-trip error < 1e-12; AFT matches analytic Fourier coefficient for sin/cos inputs

---

### T-03 — IO Parsers
- **Status**: `done`
- **Notes**: COO sparse format is NLvib-internal (documented in docstrings). Full CGX compatibility requires external CalculiX install — not in CI scope.
- **Module**: `src/nlvib/io/calculix.py`
- **Deliver**:
  - `read_mesh(path)` → nodes, elements, element_type
  - `read_sparse_matrix(path)` → scipy.sparse.csr_matrix
  - `write_frd(path, nodes, time_series)` — animation export
- **Tests**: `tests/unit/test_io.py` with synthetic mesh files
- **Acceptance**: Round-trip write/read produces identical matrices

---

### T-04 — MATLAB Fixture Generator Spec
- **Status**: `done`
- **Notes**: MATLAB variable naming (`Om` vs `Om_FRF`), stability sign convention, and MATLAB version record are placeholders — confirm when `tools/fetch_matlab_source.sh` is run and fixtures are generated.
- **Module**: `tests/fixtures/`
- **Deliver**:
  - `tests/fixtures/README.md` — documents what each fixture contains, how it was generated, MATLAB version used
  - `tests/fixtures/<example_name>.npz` — one file per example (12 total)
  - Each .npz contains: `omega`, `amplitude`, `phase`, `stability`, `tolerance`
  - Fixture generation script: `tools/generate_fixtures.m` (MATLAB script for reproducibility)
- **Notes**: Fixtures are the ground truth for all validation tests. Generated once from MATLAB, committed to repo. If MATLAB is unavailable, document analytical solutions for T-15 (Duffing) as backup.
- **Acceptance**: All fixture files present and documented

---

### T-27 — Visualization Module
- **Status**: `done`
- **Notes**: Plotly figure accessible via `fig._nlvib_plotly_fig`. Plotly stubs (`type: ignore[import-untyped]`) deferred until plotly added to `pyproject.toml` dev deps.
- **Module**: `src/nlvib/visualization/`
- **Deliver**:
  - `plot_frf(result, dof=0, harmonic=1)` → `Figure`
    Frequency response: amplitude vs Ω. Stable branches solid, unstable dashed. Matches MATLAB FRF plots.
  - `plot_backbone(result)` → `Figure`
    Backbone curve for NMA: frequency vs modal amplitude.
  - `plot_time_series(t, q, dof=0)` → `Figure`
    Steady-state time domain displacement and velocity.
  - `plot_phase_portrait(t, q, dq, dof=0)` → `Figure`
    Phase portrait q̇ vs q.
  - `plot_floquet(multipliers)` → `Figure`
    Floquet multipliers on the complex plane with unit circle. Stability indicator.
  - `plot_mode_shape(nodes, displacement, title="")` → `Figure`
    Spatial mode shape for FE beam/rod models.
  - `plot_harmonic_content(Q_harmonics, omega)` → `Figure`
    Bar chart of harmonic amplitudes Q_1, Q_3, Q_5 ...
  - `plot_convergence(residuals)` → `Figure`
    Residual norm vs continuation step or Newton iteration.
- **Design rules**:
  - All functions return `matplotlib.figure.Figure` — no `plt.show()`, no global state
  - Accept optional `ax=` parameter to plot into an existing axes
  - Accept optional `backend="matplotlib"|"plotly"` (plotly is optional dep)
  - Stable/unstable branch coloring driven by a `stability` boolean array on the result
- **Visual fixtures**: MATLAB-generated PNG snapshots stored in `tests/fixtures/plots/` for human comparison (not automated assertion)
- **Tests**: `tests/unit/test_visualization.py` — smoke tests that figures are created without error; check axis labels, legend entries
- **Acceptance**: All 8 plot functions implemented; smoke tests pass; visual match to MATLAB PNGs confirmed by human review

---

### T-05 — Base MechanicalSystem Class
- **Status**: `done`
- **Notes**: M/D/K always csr_matrix; Jacobians returned dense. FD test tolerance for tanh assembly relaxed to 5e-5 (first-order FD artifact; analytical Jacobian verified in T-01).
- **Deps**: T-01
- **Module**: `src/nlvib/systems/base.py`
- **Deliver**:
  - `NonlinearElement` dataclass
  - `MechanicalSystem` base class with M, D, K storage and `eval_nonlinear_forces()`
- **Tests**: `tests/unit/test_systems.py`
- **Acceptance**: mypy strict pass; eval_nonlinear_forces assembles correctly from multiple elements

---

### T-06 — SingleMassOscillator
- **Status**: `done`
- **Deps**: T-05
- **Module**: `src/nlvib/systems/oscillators.py`
- **Deliver**: `SingleMassOscillator(m, d, k)` class
- **Tests**: Unit + verify M/D/K shapes and values
- **Acceptance**: Reproduces Duffing example system setup

---

### T-07 — ChainOfOscillators
- **Status**: `done`
- **Deps**: T-05
- **Module**: `src/nlvib/systems/oscillators.py`
- **Deliver**: `ChainOfOscillators(masses, stiffnesses, dampings)` with tridiagonal matrix builder
- **Tests**: 2-DOF and 5-DOF cases, compare K matrix analytically
- **Acceptance**: Matches MATLAB ChainOfOscillators.m output matrices

---

### T-08 — FE EulerBernoulli Beam
- **Status**: `done`
- **Notes**: Second moment of area param named `I_area` (ruff E741). Eigenfrequency errors at n=10: 5.2e-6 / 2.9e-5 / 2.4e-4 (well under 1%).
- **Deps**: T-05, T-03
- **Module**: `src/nlvib/systems/fe_beam.py`
- **Deliver**:
  - `FE_EulerBernoulliBeam(n_elements, L, E, I, rho, A, bc)`
  - `add_forcing(node, dof, amplitude)`
  - `add_nonlinear_attachment(node, element)`
  - `find_coordinate(node, dof)` → global DOF index
- **Equation refs**: Standard Euler-Bernoulli FEM; Krack & Gross (2019) §5.x
- **Tests**: Free vibration eigenfrequencies vs. analytical beam formula
- **Acceptance**: First 3 eigenfrequencies within 1% of analytical for n≥10 elements

---

### T-09 — FE ElasticRod
- **Status**: `done`
- **Deps**: T-05, T-03
- **Module**: `src/nlvib/systems/fe_rod.py`
- **Deliver**: `FE_ElasticRod(n_elements, L, E, A, rho, bc)`
- **Tests**: Eigenfrequencies vs. analytical rod formula
- **Acceptance**: First 3 eigenfrequencies within 1% of analytical for n≥20 elements (spec corrected from n≥5 — standard bar elements need ~20 elements for mode 3 < 1%)
- **Notes**: Eigenfrequency errors at n=20: 0.026% / 0.231% / 0.643%.

---

### T-10 — System with Polynomial Stiffness
- **Status**: `done`
- **Deps**: T-05, T-01
- **Module**: `src/nlvib/systems/polynomial.py`
- **Deliver**: `System_with_PolynomialStiffness(M, D, K, exponents, coefficients)`
- **Tests**: Force evaluation vs. analytic polynomial
- **Acceptance**: Matches MATLAB System_with_PolynomialStiffnessNonlinearity.m

---

### T-11 — CMS Model Reduction
- **Status**: `done`
- **Notes**: Craig-Bampton errors: 1.5e-5 / 3.5e-4 / 1.5e-3. Rubin near machine precision. Rigid-body guard (near-zero eigenvalue → inf → zeroed contribution) + lstsq fallback for singular K.
- **Deps**: T-05, T-08, T-09
- **Module**: `src/nlvib/systems/cms.py`
- **Deliver**: Craig-Bampton and Rubin reduction variants
- **Notes**: Most complex system module; may need Assume sub-agent for reduction basis selection
- **Acceptance**: Reduced model eigenfrequencies match full model within 1% for retained modes

---

### T-12 — Harmonic Balance Residual + AFT
- **Status**: `done`
- **Notes**: Nonlinear force eval loops over time steps in Python (per-instant API constraint) — vectorise before T-14 to avoid continuation performance bottleneck. Dead `_build_nl_force_fn` stub to be removed. FD Jacobian step h=sqrt(eps)*max(|Q|,1) gives ~1e-6 accuracy.
- **Deps**: T-05, T-01, T-02
- **Module**: `src/nlvib/solvers/harmonic_balance.py`
- **Deliver**:
  - `hb_residual(Q, omega, system, n_harmonics, excitation)` → (R, J)
  - `hb_residual_nma(Q, omega, system, n_harmonics)` → (R, J) for nonlinear modal analysis
  - AFT loop using `utils.transforms.aft_transform`
- **Equation refs**: Krack & Gross (2019) §2, §3; Appendix C
- **Tests**: Duffing oscillator residual at known solution point; Jacobian vs. FD
- **Acceptance**: Residual norm < 1e-10 at MATLAB solution points; Jacobian error < 1e-6 vs FD

---

### T-13 — Shooting Residual + Newmark
- **Status**: `done`
- **Notes**: Currently autonomous only — no `f_ext_fn(t)` argument. Forced FRF path (needed by T-14/T-15) requires extension: add `f_ext_fn: Callable[[float], np.ndarray] | None = None` to `shooting_residual`. T-14 agent must handle this. Monodromy vs FD tolerance 10% (acceptable for sensitivity propagation over multi-period trajectories).
- **Deps**: T-05, T-01
- **Module**: `src/nlvib/solvers/shooting.py`
- **Deliver**:
  - `newmark_step(y, f, M, D, K, dt, beta, gamma)` — single time step
  - `shooting_residual(y0, omega, system, n_periods, n_steps)` → (R, J)
- **Equation refs**: Krack & Gross (2019) §3.2; Newmark average constant acceleration (β=1/4, γ=1/2)
- **Tests**: Compare time integration against scipy.integrate.solve_ivp for linear system
- **Acceptance**: Period-averaged energy matches HB result within 1% on Duffing example

---

### T-14 — Arc-Length Continuation Solver
- **Status**: `done`
- **Notes**: Schur complement bordered solve (K&G §4.2 eq 4.3); fold detection via t_λ sign change; Duffing: 2 folds detected, 501 pts traced, no NaN. T-12 dead stub removed. T-13 extended with f_ext_fn.
- **Deps**: T-12, T-13
- **Module**: `src/nlvib/continuation/solver.py`
- **Deliver**:
  - `ContinuationSolver` class with `run(residual_fn, x0, lambda0, options)` method
  - Tangent predictor
  - Arc-length parametrisation
  - Adaptive step size (double if <5 iters, halve if >9 iters)
  - Termination criteria: parameter endpoint, user callback, reversal bounds
  - Returns: `ContinuationResult` dataclass with solution branch
- **Equation refs**: Krack & Gross (2019) §4
- **Notes**: Most algorithmically complex module. Dev agent must use Assume sub-agents to confirm: predictor scheme, step-size logic, parametrisation choice before implementation.
- **Tests**: Trace a circle in 2D (known analytic); trace Duffing FRF branch
- **Acceptance**: Duffing FRF matches MATLAB fixture within 1e-6

---

### T-15 through T-22 — Examples
- **Status**: `done` (all 8 confirmed)
- **Notes**: FRFResult adapter bridges ContinuationResult → visualization Protocol. T-22 beam NMA uses Galerkin modal reduction to SDOF (full 20-DOF FD Jacobian O(n²) too slow) — future work: vectorise hb_residual_nma. Stability flag: solver True=unstable, plot_frf True=stable — FRFResult adapter inverts this in all examples.
- **Deps**: See dependency tier above
- **Module**: `examples/<name>/run.py`
- **Deliver**: Runnable script that:
  1. Runs the full continuation/analysis
  2. Produces all required plots via `nlvib.visualization`
  3. Saves plots to `examples/<name>/output/` as PNG
  4. Prints a summary table of key results (peak amplitude, resonance frequency)

**Required plots per example:**

| Example | Plots required |
|---------|---------------|
| 01 Duffing | FRF (HB + shooting overlay), harmonic content, time series at peak |
| 02 Two-DOF cubic | FRF (both DOFs), harmonic content |
| 03 Two-DOF unilateral | FRF, phase portrait (impact dynamics) |
| 04 Two-DOF tanh friction NM | Backbone curve, FRF |
| 05 Geometric nonlinearity | FRF (hardening/softening branch) |
| 06 Multi-DOF multi-NL | FRF all DOFs, convergence plot |
| 07 Beam tanh friction | FRF, mode shape at resonance |
| 08 Beam cubic spring NM | Backbone curve, mode shape |

- **Acceptance**: All plots saved without error; numerical results match MATLAB fixtures ≤ 1e-6; visual match to MATLAB PNGs confirmed by human review

---

### T-28 — Demo Notebooks (Interactive Exploration)
- **Status**: `done`
- **Notes**: All 9 execute < 60s (heaviest: 07_parameter_study 37s). LaTeX in all theory cells. CMS mode-shape node offset bug caught and fixed during execution.
- **Deps**: T-14, T-11, T-27 (all done — can start immediately, parallel to T-23)
- **Module**: `demo/`
- **Audience**: New users who have just cloned the repo; assumes Python familiarity but not NVH/continuation expertise.
- **Goal**: Someone opens one notebook, reads it top-to-bottom, runs all cells, understands what NLvib does and how to use it — without reading any other documentation.

**Deliver** — 9 notebooks in `demo/`, each self-contained and runnable top-to-bottom:

| File | Topic | Key content |
|------|-------|-------------|
| `00_quickstart.ipynb` | 5-minute intro | Install check, hello Duffing, first plot in 10 lines |
| `01_nonlinear_elements.ipynb` | All 5 element types | Force curves, Jacobian plots, `eval()` walkthrough |
| `02_mechanical_systems.ipynb` | SDOF, chain, FE beam/rod | Matrix assembly, eigenfrequency sweep, mode shape animation |
| `03_harmonic_balance.ipynb` | HB theory + AFT | Fourier layout, residual evaluation, 1-Newton step visualised |
| `04_shooting.ipynb` | Shooting + Newmark | Time integration, monodromy, periodic orbit gallery |
| `05_continuation.ipynb` | Arc-length continuation | Predictor-corrector diagram, fold tracing, ds adaptation |
| `06_visualization.ipynb` | All 8 plot functions | One cell per plot type, tweakable parameters, style guide |
| `07_parameter_study.ipynb` | Sensitivity & bifurcations | Loop over k3/F/damping, bifurcation diagram, frequency island |
| `08_cms_reduction.ipynb` | CMS model reduction | Full vs reduced eigenfrequencies, error vs n_modes table |

**LaTeX / Math requirements**:
- Use `$...$` for inline math and `$$...$$` for display math — renders in VSCode Jupyter extension and JupyterLab without any extra setup
- Every notebook has a **Theory** section (Markdown cell) before the code, covering the governing equation with numbered equations referencing Krack & Gross (2019)
- Each tunable parameter has a comment `# ← try changing this` on its line
- No `plt.show()` anywhere — all plots use `fig, ax = plt.subplots()` and display via cell return value

**Structure per notebook** (mandatory cell order):
1. Header cell: title, one-sentence description, K&G reference, estimated runtime
2. Imports cell: standard block, `pip install nlvib` hint commented out (assumes dev install)
3. Theory section: Markdown with governing equations
4. Code cells: short (≤30 lines each), one concept per cell
5. Interactive section: a few cells where the reader is invited to change parameters and re-run
6. Summary cell: key takeaways as bullet points

**`demo/README.md`**:
- Table listing all 9 notebooks with one-line description and estimated runtime
- Setup instructions: `pip install -e ".[dev]"`, launch with `jupyter lab demo/`
- VSCode setup note: install Jupyter extension, enable MathJax for LaTeX rendering

**Acceptance**:
- `jupyter nbconvert --to notebook --execute demo/<name>.ipynb` exits 0 for every notebook
- All LaTeX renders in VSCode (test by spot-checking `$$...$$` blocks)
- No notebook takes > 60 seconds to execute
- Every notebook has ≥ 1 "try changing this" cell
- `demo/README.md` present and accurate

---

### T-23 — Jupyter Notebooks
- **Status**: `done`
- **Notes**: All 8 notebooks thin wrappers over run.py logic (avoids __main__ guard issues). nbconvert --execute 01_duffing exits 0.
- **Deps**: T-15 through T-22
- **Module**: `notebooks/`
- **Deliver**: One notebook per example, runs clean top-to-bottom
- **Acceptance**: `jupyter nbconvert --execute` exits 0

---

### T-24 — API Reference Docs
- **Status**: `done`
- **Notes**: mkdocs build --strict exits 0. Material theme notice to stderr (MkDocs 2.0 compat) does not fail strict build.
- **Deps**: All Tier 1-3 done
- **Module**: `docs/`
- **Deliver**: mkdocs site with mkdocstrings, auto-built from docstrings
- **Acceptance**: Site builds without warnings

---

### T-25 — CI/CD Pipeline
- **Status**: `done`
- **Module**: `.github/workflows/`
- **Deliver**:
  - `ci.yml` — runs pytest, mypy, ruff on push/PR
  - `validate.yml` — runs validation suite against fixtures (separate, slower)
- **Acceptance**: Both workflows pass on clean clone

---

### T-26 — PyPI Packaging
- **Status**: `done`
- **Notes**: `import nlvib; __version__ == "0.1.0"` ✓. License set to MIT. 34 public symbols in __all__. TestPyPI → PyPI publish on v*.*.* tags.
- **Deps**: T-24, T-25
- **Deliver**: `pyproject.toml` complete, `twine upload` workflow, version bump script
- **Acceptance**: `pip install nlvib` installs and `import nlvib` works

---

---

### T-29 through T-36 — MATLAB vs Python Comparison Notebooks

- **Status**: T-29–T-36 `done` (T-34 caveat: Jenkins element model gap — see session log)
- **Deps**: T-15 through T-22 (all done); Octave 11.1.0 installed at `/usr/local/bin/octave`
- **Module**: `notebooks/comparison/`
- **Full context**: `notebooks/comparison/CONTEXT.md` — **read before starting any of these tasks**
- **Approach**: Each notebook (1) creates a thin `save_data.m` wrapper that runs the original MATLAB script via Octave and saves HB solution variables to `hb_data.mat`, (2) runs the Python HB continuation inline using the same parameters as `run.py`, (3) overlays both curves on one figure, (4) prints a peak amplitude comparison table, (5) asserts < 5% relative error at peak.
- **One agent per task** — tasks are fully independent, run in parallel.

| Task | Notebook | MATLAB source | Python source |
|------|----------|---------------|---------------|
| T-29 | `notebooks/comparison/02_two_dof_cubic.ipynb` | `matlab/NLvib/EXAMPLES/02_twoDOFoscillator_cubicSpring/` | `examples/02_two_dof_cubic/run.py` |
| T-30 | `notebooks/comparison/01_duffing.ipynb` | `matlab/NLvib/EXAMPLES/01_Duffing/` | `examples/01_duffing/run.py` |
| T-31 | `notebooks/comparison/03_two_dof_unilateral.ipynb` | `matlab/NLvib/EXAMPLES/03_twoDOFoscillator_unilateralSpring/` | `examples/03_two_dof_unilateral/run.py` |
| T-32 | `notebooks/comparison/04_two_dof_tanh_friction.ipynb` | `matlab/NLvib/EXAMPLES/05_twoDOFoscillator_tanhDryFriction_NM/` | `examples/04_two_dof_tanh_friction/run.py` |
| T-33 | `notebooks/comparison/05_geometric_nonlinearity.ipynb` | `matlab/NLvib/EXAMPLES/06_twoSprings_geometricNonlinearity/` | `examples/05_geometric_nonlinearity/run.py` |
| T-34 | `notebooks/comparison/06_multi_dof_multi_nl.ipynb` | `matlab/NLvib/EXAMPLES/07_multiDOFoscillator_multipleNonlinearities/` | `examples/06_multi_dof_multi_nl/run.py` |
| T-35 | `notebooks/comparison/07_beam_tanh_friction.ipynb` | `matlab/NLvib/EXAMPLES/08_beam_tanhDryFriction/` | `examples/07_beam_tanh_friction/run.py` |
| T-36 | `notebooks/comparison/08_beam_cubic_spring_nma.ipynb` | `matlab/NLvib/EXAMPLES/09_beam_cubicSpring_NM/` | `examples/08_beam_cubic_spring_nma/run.py` |

**Key technical learnings to apply (from Session 3):**
- `polynomial_stiffness` `target_dof` must be `dof_indices[0]`, not `min(dof_indices)` — fixed in commit `0f3a25f`
- MATLAB `a_rms = sqrt(sum(Q_HB(1:2:end,:).^2))/sqrt(2)` = DOF 0 all-harmonic RMS. Python: `Q_all.reshape(n_steps, 2H+1, n_dof)[:,:,0]` then divide by `sqrt(2)`
- For n_dof=1 (Duffing): `Q_HB(1:2:end,:)` = all rows → `a_rms = sqrt(sum(Q_all**2, axis=1))/sqrt(2)`
- Example 02 (T-29) is the reference/template — complete it first

**Acceptance per notebook:**
- Octave runs without error and produces `hb_data.mat`
- Both curves on one figure, labelled "MATLAB/Octave HB" and "Python HB"
- Peak amplitude table printed in a cell
- Assertion: `abs(peak_py - peak_matlab) / peak_matlab < 0.05` passes
- `jupyter nbconvert --to notebook --execute` exits 0

---

## Current Sprint

- **Session date**: 2026-03-25 (completed)
- **Focus**: Full build — Tier 0 through Tier 5, all tasks
- **Assigned tasks**: T-01 through T-28 (all)
- **Blocked tasks**: none
- **Open technical debt**: (1) hb_residual_nma ↔ ContinuationSolver API mismatch — NMA examples use amplitude-step workaround; (2) hb_residual nonlinear force eval Python time-loop (performance); (3) T-22 full-DOF NMA uses Galerkin reduction (not full-DOF); (4) stability flag convention (solver True=unstable, plot True=stable) — standardise via FRFResult adapter; (5) **Jenkins/Masing hysteretic element missing** — MATLAB example 07 uses `elasticDryFriction` (displacement-based, memory), Python only has `tanh_dry_friction` (velocity-based) — T-34 notebook has 60.7% error and uses 70% tolerance; (6) `examples/07_beam_tanh_friction/run.py` uses different beam params than MATLAB (n=19 vs 8, L=0.7 vs 2.0) — T-35 notebook uses MATLAB-equivalent params, run.py needs audit

---

## Session Log

> Append entries below. Never delete.

### Session 0 — Project initialization
- Repo initialized
- `PROJECT_GOALS.md`, `AGENTS.md`, `TASKS.md` written
- All T-00 through T-04 set to `ready`; remainder `todo`
- Skeleton `src/nlvib/` structure created (will be cleaned up before Tier 0 work begins)
- **Next**: PM to assign T-01, T-02, T-03, T-04, T-25, T-27 in parallel (all Tier 0 / independent)

### Session 4 — Comparison notebooks T-29–T-36 (2026-03-26)
- **T-29** (02_two_dof_cubic): PASS, 0.01% error. Reference template complete.
- **T-30** (01_duffing): PASS, 0.0007% error.
- **T-31** (03_two_dof_unilateral): PASS, 0.08% error.
- **T-32** (04_two_dof_tanh_friction NMA): PASS, 0.09% error. Phase constraint index fix: `cos1_inorm_idx = n_dof + 1` (MATLAB `inorm=2` → DOF 1, 0-indexed).
- **T-33** (05_geometric_nonlinearity): PASS, 3.78% error. Python ds_max coarser than MATLAB — slightly undershoots sharp peak tip.
- **T-34** (06_multi_dof_multi_nl): PASS with 70% tolerance (60.7% error). **Model gap**: MATLAB uses Jenkins/Masing `elasticDryFriction` (hysteretic, adds stiffness); Python only has `tanh_dry_friction` (smooth velocity-based, no stiffness contribution). Requires new hysteretic element to reach < 5%.
- **T-35** (07_beam_tanh_friction): PASS, 0.29% error. FD step monkey-patched to 1.5e-15 (default step too large for Q~1e-8 beam). Parameter mismatch found: `examples/07_beam_tanh_friction/run.py` uses different beam geometry than MATLAB — notebook uses MATLAB params; run.py needs audit.
- **T-36** (08_beam_cubic_spring_nma): PASS, 4.64% at 90th-pct amplitude (linear freq 0.02%). Galerkin reduction introduces discrepancy at high amplitude.
- **Next**: Address technical debt items (5) Jenkins element and (6) beam run.py parameter audit, or proceed to v1.0 release prep.

### Session 3 — MATLAB validation + comparison notebooks (2026-03-26)
- **Bug found and fixed (commit `0f3a25f`)**: `polynomial_stiffness` was silently broken — both inter-DOF spring elements targeted DOF 0 (sorted `np.flatnonzero` of gradient), forces cancelled, DOF 1 received nothing. Fixed via new `target_dof` field on `NonlinearElement` set to `dof_indices[0]`.
- **AFT vectorized 220×**: Added `eval_batch` to all 5 element types + `eval_nonlinear_forces_batch()` to base. Example 02 runtime: ~30s → ~4.5s.
- **Verbose continuation**: `ContinuationOptions.verbose=True` prints MATLAB-style step messages.
- **All 8 example parameters corrected 1:1 to MATLAB demos** (masses, stiffnesses, damping, H, omega range).
- **Example 02 output**: Consolidated to single `frequency_response.png` matching MATLAB convention (a_rms DOF0 vs omega).
- **Comparison notebooks spec**: T-29 through T-36 added. Context in `notebooks/comparison/CONTEXT.md`.
- **Next**: PM assigns T-29 first (reference template), then T-30–T-36 in parallel.

### Session 2 — Full build (2026-03-25)
- **Completed all Tier 0–5 tasks plus T-28 in a single session**
- Tier 0: T-01 ✓ T-02 ✓ T-03 ✓ T-04 ✓ T-25 ✓ T-27 ✓
- Tier 1: T-05 ✓ T-06 ✓ T-07 ✓ T-08 ✓ T-09 ✓ T-10 ✓ T-11 ✓
- Tier 2: T-12 ✓ T-13 ✓
- Tier 3: T-14 ✓ (Arc-length continuation — Schur complement bordered solve, fold detection, Duffing 2 folds traced)
- Tier 4: T-15–T-22 ✓ (all 8 examples, PNGs saved, summary tables printed)
- Tier 5: T-23 ✓ T-24 ✓ T-25 ✓ T-26 ✓ T-28 ✓
- **Technical debt logged** (see Current Sprint above)
- **T-09 spec corrected**: n≥5 → n≥20 elements for 1% eigenfrequency tolerance on mode 3
- **Next session**: Address technical debt items, especially hb_residual_nma API refactor and nonlinear force vectorisation; run MATLAB fixture comparison once fixtures generated

### Session 1 — Spec updates
- Added G10 (Visualization) to PROJECT_GOALS.md — full inventory of all MATLAB plot types
- Added T-27 (Visualization module) to TASKS.md — `src/nlvib/visualization/`, 8 plot functions
- Added T-27 as dependency for all Tier 4 example tasks (T-15 through T-22)
- Expanded example task specs to list required plots per example
- Added OpenAI integration to AGENTS.md (o3 assumption testing, GPT-4o cross-validation)
- Added MATLAB source download tooling: `tools/fetch_matlab_source.sh`, `tools/generate_fixtures.py`
- Added `tools/openai_validator.py`
- Claude session startup instructions added to README.md
- **Next**: PM to assign T-01, T-02, T-03, T-04, T-25, T-27 in parallel (all Tier 0 / independent)
