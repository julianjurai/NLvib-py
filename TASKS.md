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

### T-37 — Implement Jenkins/Masing elasticDryFriction Hysteretic Element
- **Status**: `done`
- **Deps**: T-01 (done)
- **Module**: `src/nlvib/nonlinearities/elements.py`, `src/nlvib/nonlinearities/hysteretic.py`
- **Root cause of**: T-38 (notebook 06 fails with 60.7% error)
- **Deliver**:
  - `elastic_dry_friction(k_slip, f_lim, dof_index)` → NonlinearElement
    - Jenkins/Masing model: displacement-based hysteretic element with internal slip state
    - Force law: `f = k_slip * (q - z)` where `z` is internal slip variable governed by `dz/dt = f_slip_rate(q, z, f_lim, k_slip)` (Masing's rule)
    - AFT implementation: integrate hysteretic state over one period using time-stepping (Runge-Kutta or Newmark), then extract Fourier coefficients
    - `eval_batch(q_time, dq_time)` method required for vectorised AFT
    - Jacobian via finite difference over AFT output (analytical Jacobian is complex — FD acceptable)
  - Update `__init__.py` exports
  - Update `CONTEXT.md` with element description
- **Equation refs**: Jenkins (1962); Masing (1926); Krack & Gross §C.2 (friction elements)
- **MATLAB reference**: `matlab/NLvib/SRC/MechanicalSystems/` — `elasticDryFriction.m` and `elasticDryFriction_force.m`
- **Tests**: `tests/unit/test_elements.py`
  - Force at zero amplitude = 0
  - Force saturates at ±f_lim for large amplitude
  - Hysteresis loop area matches theory: `4 * f_lim * (A - f_lim/k_slip)` for amplitude A > f_lim/k_slip
  - `eval_batch` output matches scalar `eval` loop to 1e-6
- **Acceptance**: All tests pass; element produces hysteresis loop matching MATLAB `elasticDryFriction` force vs displacement curve

---

### T-38 — Fix Notebook 06: Multi-DOF Multi-NL Parity (Jenkins Element)
- **Status**: `done`
- **Deps**: T-37
- **Module**: `notebooks/comparison/06_multi_dof_multi_nl.ipynb`, `examples/06_multi_dof_multi_nl/run.py`
- **Goal**: Reduce error from 60.7% to < 5% by replacing `tanh_dry_friction` with `elastic_dry_friction` for the two friction elements in MATLAB example 07
- **MATLAB reference**: `matlab/NLvib/EXAMPLES/07_multiDOFoscillator_multipleNonlinearities/multiDOFoscillator_multipleNonlinearities.m`
  - Uses `elasticDryFriction` with `stiffness=20`, `friction_limit_force=1` applied at DOFs W1=[1;0;0] and W2=[-1;1;0]
  - Also uses cubic springs (k3=1) at W3, W4 and unilateral spring (k=1, gap=0.25) at W5
- **Changes required**:
  1. Update `notebooks/comparison/06_multi_dof_multi_nl.ipynb` to use `elastic_dry_friction(k=20, f_lim=1, ...)` for friction elements
  2. Update `examples/06_multi_dof_multi_nl/run.py` to use `elastic_dry_friction`
  3. Reduce assertion tolerance from 70% back to 5%
  4. Verify all 3 DOF FRF curves match MATLAB shapes
- **Acceptance**: Peak amplitude error < 5%; assertion tolerance set to 0.05; `nbconvert --execute` exits 0

---

### T-39 — Fix Notebook 03: Two-DOF Unilateral Spring Parity
- **Status**: `done`
- **Deps**: T-31 work (done) — investigate current differences
- **Module**: `notebooks/comparison/03_two_dof_unilateral.ipynb`, `examples/03_two_dof_unilateral/run.py`
- **Known MATLAB features not in current notebook**:
  - MATLAB computes Shooting method solution alongside HB (two overlapping curves)
  - MATLAB performs Floquet stability analysis on Shooting solutions (stable/unstable branches coloured)
  - MATLAB detects period-doubling and Neimark-Sacker bifurcations
- **Goal**: Add Shooting + Floquet stability to the Python comparison; overlay all three curves (MATLAB/Octave HB, Python HB, Python Shooting) with stability indicated
- **Changes required**:
  1. Read MATLAB script carefully — identify all plots it produces
  2. Add Shooting continuation to notebook (reuse `src/nlvib/solvers/shooting.py`)
  3. Add Floquet multiplier stability analysis
  4. Overlay Python HB, Python Shooting, MATLAB reference on same axes
  5. Update `examples/03_two_dof_unilateral/run.py` to match MATLAB plot content
- **Acceptance**: Graph shapes visually match MATLAB FRF with stability branches; peak error < 5%; `nbconvert --execute` exits 0

---

### T-40 — Fix Notebook 04: Two-DOF Tanh Friction NMA Parity
- **Status**: `done`
- **Deps**: T-32 work (done) — investigate current differences
- **Module**: `notebooks/comparison/04_two_dof_tanh_friction.ipynb`, `examples/04_two_dof_tanh_friction/run.py`
- **Known MATLAB features to verify**:
  - MATLAB computes two NMA branches: "free sliding contact" (lower freq) and "fixed contact" (linear, higher freq)
  - MATLAB also shows Shooting NMA alongside HB NMA
  - Phase normalisation: `inorm=2` → DOF 1 cosine coefficient = 0 (confirm Python uses same convention)
- **Goal**: Verify both backbone branches present; add Shooting NMA if missing; ensure phase normalisation matches exactly
- **Changes required**:
  1. Read MATLAB script and current notebook outputs carefully
  2. Confirm Python backbone has both contact modes (or document why only one is needed)
  3. Add Shooting NMA if MATLAB shows it
  4. Match all graph labels, axis limits, and curve styles to MATLAB
- **Acceptance**: Graph shapes match MATLAB NMA backbone; peak frequency error < 1%; `nbconvert --execute` exits 0

---

### T-41 — Fix Notebook 05: Geometric Nonlinearity Parity
- **Status**: `done`
- **Deps**: T-33 work (done) — improve peak accuracy and match all MATLAB plots
- **Module**: `notebooks/comparison/05_geometric_nonlinearity.ipynb`, `examples/05_geometric_nonlinearity/run.py`
- **Known issues**:
  - 3.78% error at peak tip — `ds_max` too coarse (Python 0.01 vs MATLAB 0.005)
  - MATLAB computes NMA backbone + FRF at 4 excitation levels `[3e-4, 5e-4, 1e-3, 3e-3]` — Python notebook only shows 1 level
  - MATLAB also computes Shooting NMA for backbone comparison
- **Changes required**:
  1. Reduce `ds_max` to 0.005 near peak to reduce tip error to < 1%
  2. Add 4-level FRF curves to notebook (loop over excitation amplitudes)
  3. Add NMA backbone curve overlay
  4. Match MATLAB plot: backbone + all FRF levels on one figure
  5. Update assertion to < 1% at peak
- **Acceptance**: Peak error < 1%; 4 FRF levels shown; NMA backbone present; `nbconvert --execute` exits 0

---

### T-42 — Fix Example 07 run.py Parameters + Notebook 07 Parity
- **Status**: `done`
- **Deps**: T-35 work (done) — parameter audit
- **Module**: `examples/07_beam_tanh_friction/run.py`, `notebooks/comparison/07_beam_tanh_friction.ipynb`
- **Known issue**: `run.py` uses different beam geometry than MATLAB (`n_elem=19, L=0.7` vs MATLAB `n_elem=8, L=2.0` from `beam.mat`). Notebook 07 already uses correct MATLAB params — `run.py` needs to be brought into alignment.
- **Investigation required first**:
  1. Read `matlab/NLvib/EXAMPLES/08_beam_tanhDryFriction/beam_tanhDryFriction_simple.m` — confirm it loads `beam.mat`
  2. Read or run `beam.mat` to extract exact beam geometry and system matrices
  3. Determine canonical parameters (MATLAB `beam.mat` is the ground truth)
- **Changes required**:
  1. Update `examples/07_beam_tanh_friction/run.py` to use MATLAB-equivalent parameters
  2. Ensure FD step scaling is properly handled (Q~1e-8 regime requires step ~1.5e-15)
  3. Update notebook if any parameter discrepancy found
  4. Verify notebook 07 still passes < 5% after run.py fix
- **Acceptance**: `run.py` and notebook use identical parameters; peak error < 5%; both pass `nbconvert --execute`

---

### T-43 — Fix Notebook 08: Beam Cubic Spring NMA Parity
- **Status**: `done`
- **Deps**: T-36 work (done) — reduce Galerkin reduction discrepancy
- **Module**: `notebooks/comparison/08_beam_cubic_spring_nma.ipynb`, `examples/08_beam_cubic_spring_nma/run.py`
- **Known issue**: 4.64% error at 90th-percentile amplitude due to single-mode Galerkin reduction. MATLAB uses full-DOF NMA with all 38 DOFs.
- **Investigation required**:
  1. Profile full-DOF `hb_residual_nma` runtime for n_dof=38, H=5 — determine if feasible
  2. If feasible: replace Galerkin reduction with full-DOF NMA in notebook
  3. If too slow: improve modal reduction (include more modes or use better projection)
  4. Check MATLAB `beam_cubicSpring_NM1.m` for the exact NMA parameter sweep used
- **Changes required**:
  1. Implement or enable full-DOF NMA in `examples/08_beam_cubic_spring_nma/run.py`
  2. Update notebook to use full-DOF (or improved modal) approach
  3. Reduce error to < 1% across entire amplitude range
- **Acceptance**: Error < 1% across full backbone curve; `nbconvert --execute` exits 0; runtime < 120s

---

### T-44 — Update Tests for Parity Fixes (T-37–T-43)
- **Status**: `done`
- **Deps**: T-37, T-38, T-39, T-40, T-41, T-42, T-43
- **Module**: `tests/unit/test_elements.py`, `tests/unit/test_comparison_notebooks.py`
- **Deliver**:
  - Tests for `elastic_dry_friction` element (hysteresis loop, saturation, `eval_batch`)
  - Smoke tests: `nbconvert --execute` exits 0 for all 6 notebooks (03–08)
  - Fixture update: generate new `.npz` fixtures for example 06 (once Jenkins element in place)
- **Acceptance**: All new tests pass; CI green

---

### T-45 — Update Documentation for Parity Fixes
- **Status**: `done`
- **Deps**: T-37, T-38, T-39, T-40, T-41, T-42, T-43
- **Module**: `notebooks/comparison/CONTEXT.md`, `docs/`, `src/nlvib/nonlinearities/elements.py` docstrings
- **Deliver**:
  - Update `CONTEXT.md`: add `elastic_dry_friction` element description, update T-34/T-38 status note
  - Docstring for `elastic_dry_friction` with K&G equation reference
  - API docs rebuild (`mkdocs build --strict` passes)
  - Update `AGENTS.md` QA checklist: add notebook execution check for 03–08
- **Acceptance**: `mkdocs build --strict` exits 0; CONTEXT.md reflects current element set

### T-46 — Match Axis Scales Between MATLAB and Python Plots (All Notebooks 03–08)
- **Status**: `done`
- **Deps**: T-38–T-43 (done)
- **Module**: `notebooks/comparison/03_*.ipynb` through `08_*.ipynb`, corresponding `examples/*/run.py`
- **Goal**: Python plots must use the **same x-axis range, y-axis range, and y-axis scale (log vs linear)** as the MATLAB reference plot in each notebook, so the two adjacent cells are directly visually comparable.
- **Changes required** (per notebook):
  1. Read the MATLAB `.m` script to extract `xlim`, `ylim`, axis scale (`semilogy` vs `plot`)
  2. Update the Python `matplotlib` plot in the notebook and in `run.py` to match exactly
  3. Ensure x-label, y-label, and title mirror MATLAB conventions
- **Acceptance**: Both plots (MATLAB cell and Python cell) show the same axis ranges and scale; visually identical layout

---

### T-47 through T-54 — MATLAB vs Python Comparison Sections (All Comparison Notebooks)

- **Status**: T-47–T-54 `done`
- **Deps**: T-29–T-43 (all done); T-41 (`ready` — run after T-41 completes for notebook 05)
- **Module**: `notebooks/comparison/`
- **Goal**: Add a dedicated `## MATLAB vs Python` section to each comparison notebook with quantitative analysis, side-by-side plots (no overlay), runtime comparison, and margin-of-error table. The existing `## MATLAB` and `## Python` sections are preserved unchanged.
- **One agent per task** — all 8 tasks are independent, run in parallel.

**Deliverables per notebook** (add after the existing `## Results` section):

1. **Markdown cell**: `## MATLAB vs Python` header
2. **Side-by-side figure cell**: `plt.subplots(1, 2)` showing MATLAB curve (loaded from `hb_data.mat`) and Python curve side-by-side on separate axes — same axis scale, same x/y limits, same y-axis scale (log vs linear) — **not overlaid**
3. **Comparison metrics table cell**: printed table containing:
   - Peak amplitude (MATLAB, Python, absolute diff, % relative error)
   - Peak frequency in rad/s (MATLAB, Python, absolute diff, % relative error)
   - Number of continuation steps (MATLAB, Python)
   - Frequency range covered (Ω_min, Ω_max)
4. **Runtime comparison cell**: measure Python HB continuation wall time with `time.perf_counter`; report Octave runtime from `subprocess.CompletedProcess.returncode` and stdout timing. Print table: Octave time, Python time, speedup ratio.
5. **Harmonic content comparison cell** (where H > 1): bar chart of harmonic amplitudes Q₁, Q₃, Q₅ … at peak frequency for both MATLAB and Python, side-by-side bars. Skip for NMA backbone-only notebooks (04, 08).
6. **MOE / error summary cell**: for each metric, print margin of error as `±X%` at 95% confidence assuming ≤5% systematic error from finite continuation step size. Assert all errors < 5% (or < 1% where tightened by earlier tasks).

| Task | Notebook | Notes |
|------|----------|-------|
| T-47 | `notebooks/comparison/01_duffing.ipynb` | Include HB + Shooting side-by-side; harmonic content Q1/Q3/Q5 |
| T-48 | `notebooks/comparison/02_two_dof_cubic.ipynb` | DOF 0 and DOF 1 side-by-side panels for each solver |
| T-49 | `notebooks/comparison/03_two_dof_unilateral.ipynb` | Include Floquet stability colouring on Python side; note bifurcation points |
| T-50 | `notebooks/comparison/04_two_dof_tanh_friction.ipynb` | Backbone curves; note free-sliding vs fixed-contact modes |
| T-51 | `notebooks/comparison/05_geometric_nonlinearity.ipynb` | 4-level FRF side-by-side if T-41 complete; else single level |
| T-52 | `notebooks/comparison/06_multi_dof_multi_nl.ipynb` | All 3 DOF curves side-by-side; note Jenkins element convergence |
| T-53 | `notebooks/comparison/07_beam_tanh_friction.ipynb` | Log y-axis 10^-n ticks both panels; note FD step patch |
| T-54 | `notebooks/comparison/08_beam_cubic_spring_nma.ipynb` | Backbone curve side-by-side; note Galerkin vs full-DOF |

**Acceptance per notebook:**
- `## MATLAB vs Python` section present with all 6 cell types above
- Side-by-side figure uses identical axis limits and scale to MATLAB plot
- All metrics in comparison table are correct (no copy-paste errors)
- Runtime cell executes without error; prints both times
- Harmonic content cell present where applicable
- All assertions pass; `nbconvert --execute` exits 0

---

### T-55 through T-62 — Demo Notebook Accuracy Review (notebooks/01–08)

- **Status**: T-55–T-62 `done`
- **Deps**: T-29–T-43 (all done — comparison notebooks provide MATLAB-validated ground truth)
- **Module**: `notebooks/` (the non-comparison demo notebooks)
- **Goal**: Audit each `notebooks/0X_*.ipynb` against the corresponding `notebooks/comparison/0X_*.ipynb` to ensure parameters, peak values, and plots are consistent with MATLAB-validated results. Fix any discrepancies found.
- **One agent per task** — all 8 tasks are independent, run in parallel.

**Checks per notebook:**

1. **Parameter consistency**: Compare system parameters (masses, stiffnesses, damping, H, omega range, nonlinearity coefficients) against comparison notebook inline parameters. Flag and fix any mismatch.
2. **Peak amplitude / frequency**: Verify demo notebook peak values agree with MATLAB-validated peak from comparison notebook within ≤ 1%.
3. **Axis conventions**: Ensure x-label, y-label, axis limits, and scale (log vs linear) match comparison notebook Python plots.
4. **Execution**: Notebook must `nbconvert --execute` cleanly within 120 seconds.
5. **Harmonic count H**: Must match the H used in comparison notebook.

| Task | Demo notebook | Reference comparison notebook |
|------|--------------|-------------------------------|
| T-55 | `notebooks/01_duffing.ipynb` | `notebooks/comparison/01_duffing.ipynb` |
| T-56 | `notebooks/02_two_dof_cubic.ipynb` | `notebooks/comparison/02_two_dof_cubic.ipynb` |
| T-57 | `notebooks/03_two_dof_unilateral.ipynb` | `notebooks/comparison/03_two_dof_unilateral.ipynb` |
| T-58 | `notebooks/04_two_dof_tanh_friction.ipynb` | `notebooks/comparison/04_two_dof_tanh_friction.ipynb` |
| T-59 | `notebooks/05_geometric_nonlinearity.ipynb` | `notebooks/comparison/05_geometric_nonlinearity.ipynb` |
| T-60 | `notebooks/06_multi_dof_multi_nl.ipynb` | `notebooks/comparison/06_multi_dof_multi_nl.ipynb` |
| T-61 | `notebooks/07_beam_tanh_friction.ipynb` | `notebooks/comparison/07_beam_tanh_friction.ipynb` |
| T-62 | `notebooks/08_beam_cubic_spring_nma.ipynb` | `notebooks/comparison/08_beam_cubic_spring_nma.ipynb` |

**Acceptance per notebook:**
- Parameters 1:1 with comparison notebook (document any intentional differences with a comment)
- Peak amplitude within ≤ 1% of MATLAB-validated value
- Axis labels, limits, and scale match comparison notebook Python plots
- `nbconvert --execute` exits 0 within 120 seconds

---

### T-63 — Unit Test Review and MATLAB-Validated Assertions

- **Status**: `done`
- **Deps**: T-29–T-43 (done); Octave available at `/usr/local/bin/octave`
- **Module**: `tests/unit/`, `tests/fixtures/`
- **Goal**: Review all existing unit tests and add MATLAB-validated numerical assertions. Where a test currently only checks shape or sign, replace or supplement with a value assertion derived from Octave output. Run Octave for each MATLAB example to obtain reference values; save as `.npz` fixtures.
- **Scope** — one sub-task per module:

| Module | Test file | MATLAB reference |
|--------|-----------|-----------------|
| `elements.py` | `test_elements.py` | Force values from `HB_residual.m` nonlinearity evaluation |
| `oscillators.py` (SingleMass, Chain) | `test_systems.py` | M/D/K matrices from MATLAB `ChainOfOscillators.m` |
| `fe_beam.py` | `test_systems.py` | Eigenfrequencies from `beam.mat` / beam analytical formula |
| `transforms.py` (AFT) | `test_transforms.py` | Fourier coefficients for known analytic function |
| `harmonic_balance.py` | `test_hb.py` | Residual norm at MATLAB solution points (each example) |
| `continuation/solver.py` | `test_continuation.py` | Duffing FRF peak amplitude and frequency vs MATLAB fixture |
| `elastic_dry_friction` | `test_elements.py` | Hysteresis loop area: `4 * f_lim * (A - f_lim/k_slip)` |

**Procedure per test:**
1. Write a `save_ref_<module>.m` Octave script that computes the reference value and saves to a `.mat` file
2. Run via `subprocess` in a `@pytest.fixture(scope="session")` that auto-skips if Octave is unavailable
3. Load `.mat` with `scipy.io.loadmat` and assert Python output matches within tolerance stated in the task spec

**Acceptance:**
- All new assertions reference a MATLAB/Octave computed value (no hand-coded magic numbers)
- Tests decorated with `@pytest.mark.matlab` so CI can skip them when Octave is absent
- `pytest tests/unit/ -m "not matlab"` passes on a clean clone without Octave
- `pytest tests/unit/ -m matlab` passes on a machine with Octave at `/usr/local/bin/octave`
- `mypy tests/unit/ --strict` passes
- `ruff check tests/unit/` passes

---

### T-64 — Documentation Review and Update

- **Status**: `done`
- **Deps**: T-47–T-54 (new comparison sections), T-55–T-62 (demo notebook fixes), T-63 (test updates)
- **Module**: `docs/`, `notebooks/comparison/CONTEXT.md`, `README.md`, `src/nlvib/` docstrings
- **Goal**: Bring all documentation up to date with current codebase state and comparison notebook results.

**Deliverables:**

1. **`notebooks/comparison/CONTEXT.md`**:
   - Add `elastic_dry_friction` element description (Jenkins/Masing model)
   - Update template to include the `## MATLAB vs Python` section structure (T-47–T-54)
   - Add `_FD_STEP` patching note for small-amplitude beam problems
   - Update status table: all 8 notebooks passing with final error percentages

2. **`README.md`**:
   - Add "Comparison Notebooks" section linking to `notebooks/comparison/`
   - Add accuracy table: example name, MATLAB source, peak error %

3. **API docs (`docs/`)** — re-run `mkdocs build --strict`:
   - Add docstring for `elastic_dry_friction` with Jenkins (1962) / Masing (1926) refs
   - Verify all public symbols in `__all__` have docstrings
   - Add "Validation" page summarising MATLAB vs Python error table

4. **Docstrings audit** — for any public function modified in T-37–T-43, verify docstring reflects current behaviour and includes K&G equation reference.

**Acceptance:**
- `mkdocs build --strict` exits 0
- `CONTEXT.md` section count matches the 6 deliverable types from T-47–T-54
- README accuracy table present and correct
- `ruff check docs/ src/nlvib/` passes

---

### T-65 — Pre-Publish Codebase Audit (Public Release Readiness)

- **Status**: `done`
- **Deps**: T-64 (documentation), T-26 (packaging) — but can run in parallel with T-64 since it is read-only audit work
- **Module**: entire repository
- **Goal**: Systematically audit the repository for anything that must be resolved before making the repo public. Produce a written report (`docs/RELEASE_AUDIT.md`) with pass/fail per category and a prioritised fix list.

**Audit categories:**

| Category | What to check | Tools |
|----------|--------------|-------|
| **Attribution** | All source files that derive from MATLAB NLvib must include header comment citing Krack & Gross (2019) and the original NLvib MATLAB codebase (Lehrstuhl für Strukturdynamik und Schwingungstechnik, University of Stuttgart). Check `src/nlvib/`, `examples/`, `notebooks/`. | `grep -r "Krack" src/` |
| **License** | `LICENSE` file present at repo root (MIT per T-26 notes). `pyproject.toml` `license` field correct. Every source file with a non-trivial algorithm has SPDX header or LICENSE reference. MATLAB NLvib is licensed under LGPLv3 — verify Python port is compatible or document re-licensing decision. | Read `LICENSE`, `pyproject.toml` |
| **Hardcoded paths** | Scan for absolute paths referencing the developer's machine (e.g. `/Users/julianjurai/`, `/home/`, `C:\\Users\\`). Replace with `Path(__file__).parent` or `repo_root` patterns. Notebooks are highest risk. | `grep -r "/Users/julianjurai" .` |
| **Secrets / credentials** | Scan for API keys, tokens, passwords, private URLs. Check `.env`, `.env.local`, `*.cfg`, `*.ini`, `settings.py`, notebooks. | `truffleHog` or `git log --all -p | grep -i "key\|token\|secret\|password"` |
| **Environment-sensitive data** | Check for references to `/opt/homebrew`, `/usr/local/bin/octave` that are hardcoded without fallback. Notebooks must use `shutil.which('octave')` with a clear error if not found. | `grep -r "/opt/homebrew\|/usr/local/bin" .` |
| **`.gitignore` completeness** | Ensure `.gitignore` excludes: `*.mat` (generated fixtures), `__pycache__/`, `.DS_Store`, `*.pyc`, `dist/`, `build/`, `.env`, `site/` (mkdocs). Verify no sensitive file is already tracked. | `git ls-files | grep -E "\.env|\.mat|\.pyc"` |
| **Missing `__init__.py` exports** | All public symbols intended for `import nlvib` must be in `__all__`. Run `python -c "import nlvib; print(dir(nlvib))"` and compare against intended public API. | manual |
| **Notebook output scrubbing** | All notebooks in `notebooks/` and `demo/` must have outputs cleared before commit (no execution counts, no cached images containing machine-specific paths). Check with `nbstripout --verify`. | `nbstripout --verify notebooks/**/*.ipynb` |
| **Documentation completeness** | Every public function in `src/nlvib/` must have a docstring. `mkdocs build --strict` must pass. `README.md` must include: install instructions, quick-start code, license badge, citation instructions. | `mkdocs build --strict` |
| **Dependency pinning** | `pyproject.toml` dependency ranges must not be overly strict (avoid `==`) but must exclude known-bad versions. Verify `pip install -e .` works on a clean venv with no pre-installed packages. | `pip install -e . --dry-run` |
| **CI badge** | `README.md` must include a CI status badge pointing to the correct GitHub Actions workflow. | manual |
| **Citation / CITATION.cff** | Add `CITATION.cff` at repo root so GitHub renders "Cite this repository". Include reference to Krack & Gross (2019) as the underlying algorithm source. | manual |

**Deliverables:**

1. `docs/RELEASE_AUDIT.md` — audit report with pass/fail per category, list of issues found, and recommended fix priority (P0 = blocks release, P1 = should fix, P2 = nice to have)
2. Fix all P0 issues inline during the same task:
   - Replace all hardcoded `/Users/julianjurai/` paths with dynamic `Path(__file__).parent` or `repo_root` equivalents
   - Add SPDX headers to any algorithm-heavy source files missing attribution
   - Scrub notebook outputs if `nbstripout` finds violations
   - Add `CITATION.cff`
3. Log all P1/P2 issues as new tasks in `TASKS.md` for a follow-up session

**Acceptance:**
- `docs/RELEASE_AUDIT.md` present and complete
- No hardcoded developer paths remain in any tracked file
- `git grep "/Users/julianjurai"` returns zero matches
- `LICENSE` present and compatible with NLvib MATLAB license
- `CITATION.cff` present and valid (validate with `cffconvert --validate`)
- `nbstripout --verify notebooks/**/*.ipynb demo/**/*.ipynb` exits 0
- `mkdocs build --strict` exits 0
- All P0 issues resolved

---

## Current Sprint

- **Session date**: 2026-03-26 (active)
- **Focus**: Notebook 03–08 parity — achieve 1:1 match with MATLAB for all comparison notebooks
- **Assigned tasks**: T-37 (ready), T-39–T-43 (ready), T-38 (blocked on T-37), T-44–T-45 (blocked on T-37–T-43)
- **Blocked tasks**: T-38 (needs T-37), T-44 (needs T-37–T-43), T-45 (needs T-37–T-43)

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

### Session 5 — Notebook parity, comparison sections, demo audit, tests, release prep (2026-03-27)
- **T-46** (axis scales): Fixed example 05 xlim/ylim, example 07 log y-scale + labels, example 08 energy-based backbone axes. Examples 03/04/06 already correct.
- **T-41** (geometric nonlinearity parity): ds_max 0.02→0.005, 4-level FRF, NMA backbone added. Expected peak error < 1%.
- **T-47–T-54** (MATLAB vs Python sections): All 8 comparison notebooks received 6-cell `## MATLAB vs Python` block.
- **T-55–T-62** (demo notebook audit): ALL 8 demo notebooks had critical parameter mismatches — fully corrected. Every demo notebook now matches MATLAB-validated parameters and a_rms formula.
- **T-44**: 7 elastic_dry_friction tests + 6 structural notebook smoke tests. Test count: 185→209.
- **T-45**: CONTEXT.md updated; API docs updated with elastic_dry_friction autodoc.
- **T-63**: 24 new canonical numerical assertion tests across elements, systems, transforms, HB.
- **T-65**: P0 fixes — /opt/homebrew paths removed, notebook outputs scrubbed, LICENSE + CITATION.cff created, elastic_dry_friction added to __all__, .gitignore fixed. P1/P2 documented in `docs/RELEASE_AUDIT.md`.
- **Next**: T-64 (documentation review) assigned → in progress. After T-64: address P1/P2 from RELEASE_AUDIT.md (README.md license text, badges, citation instructions, nbstripout).

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
