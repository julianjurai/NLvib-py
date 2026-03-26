# NLvib Demo Notebooks

Interactive Jupyter notebooks for new users exploring the NLvib Python toolbox.
Each notebook is self-contained and runs top-to-bottom without errors.

---

## Setup

```bash
# 1. Clone the repo and install in development mode
pip install -e ".[dev]"

# 2. Launch JupyterLab from the repo root
jupyter lab demo/

# 3. Or launch classic Jupyter Notebook
jupyter notebook demo/
```

> **Note**: All notebooks use `sys.path.insert(0, '../src')` so they work from a dev install
> without needing `pip install nlvib`. If you have already installed the package, remove that line.

---

## Notebooks

| File | Topic | Description | Est. Runtime |
|------|-------|-------------|-------------|
| `00_quickstart.ipynb` | 5-minute intro | Install check, build Duffing oscillator, 10 Newton steps, first FRF plot | < 10 s |
| `01_nonlinear_elements.ipynb` | Nonlinear elements | All 5 element types: force curves, Jacobian verification, `eval()` walkthrough | < 5 s |
| `02_mechanical_systems.ipynb` | Mechanical systems | SDOF, chain-of-oscillators, FE beam assembly, eigenfrequency convergence | < 15 s |
| `03_harmonic_balance.ipynb` | Harmonic Balance | Fourier ansatz, AFT pipeline, Newton convergence, effect of n_harmonics | < 10 s |
| `04_shooting.ipynb` | Shooting + Newmark | Newmark vs RK45 comparison, period-1 orbit, phase portrait | < 20 s |
| `05_continuation.ipynb` | Arc-length continuation | Full Duffing FRF, predictor-corrector steps, adaptive step size history | < 30 s |
| `06_visualization.ipynb` | Visualization | One cell per plot function (8 total), synthetic data, plotly fallback | < 10 s |
| `07_parameter_study.ipynb` | Parameter study | k3 and damping sweep, bifurcation diagram, resonance frequency vs k3 | < 45 s |
| `08_cms_reduction.ipynb` | CMS reduction | Craig-Bampton, error vs n_modes table, mode shape comparison | < 20 s |

---

## Recommended Reading Order

If you are new to NLvib and nonlinear vibrations:

1. **00_quickstart** — get something running in 5 minutes
2. **01_nonlinear_elements** — understand the building blocks
3. **02_mechanical_systems** — see how systems are assembled
4. **03_harmonic_balance** — the core solver algorithm
5. **05_continuation** — how to trace full solution branches
6. **06_visualization** — all the plotting tools
7. **04_shooting** — alternative time-domain approach
8. **07_parameter_study** — sensitivity analysis
9. **08_cms_reduction** — model reduction for large systems

---

## VSCode Setup

1. Install the **Jupyter** extension: `ms-toolsai.jupyter`
2. Install the **Python** extension: `ms-python.python`
3. Select the correct Python interpreter (the one with nlvib installed).
4. LaTeX rendering: VSCode's Jupyter extension renders `$...$` and `$$...$$` math via MathJax automatically — no extra configuration needed.
5. Open any `.ipynb` file and click **Run All** to execute.

---

## Dependencies

All notebooks require:

```
numpy
scipy
matplotlib
```

The plotly backend in `06_visualization.ipynb` requires:

```bash
pip install plotly
```

If plotly is not installed the notebook falls back to matplotlib gracefully.

---

## Running All Notebooks (CI)

```bash
for nb in demo/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" \
        --output /tmp/test_nb.ipynb \
        --ExecutePreprocessor.timeout=120 \
        && echo "PASS: $nb" || echo "FAIL: $nb"
done
```
