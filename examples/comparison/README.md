# MATLAB vs Python Comparison Notebooks

This directory contains Jupyter notebooks that validate the Python implementation against the original MATLAB/Octave reference code from the NLvib toolbox.

## Purpose

Each notebook:
1. Runs the **MATLAB/Octave reference** implementation via subprocess
2. Runs the **Python implementation** with identical parameters
3. Overlays both frequency response curves on a single plot
4. Computes numerical accuracy metrics (peak amplitude error, peak frequency error)
5. Asserts that the Python port matches MATLAB within specified tolerances

This provides automated validation that the Python port produces numerically correct results.

## Prerequisites

### 1. Install Octave

The comparison notebooks require **GNU Octave** to run the MATLAB reference code. Octave is a free, open-source alternative to MATLAB that runs the original `.m` scripts.

#### macOS
```bash
brew install octave
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install octave
```

#### Windows
Download the installer from [octave.org/download](https://octave.org/download)

#### Verify Installation
```bash
octave --version
```

The notebooks expect `octave` to be available on your PATH. If installed correctly, running `which octave` (macOS/Linux) or `where octave` (Windows) should show the octave binary location.

### 2. Install Python Dependencies

The comparison notebooks require the NLvib Python package and Jupyter:

```bash
# From repository root
pip install -e ".[dev]"
```

This installs:
- `nlvib` Python package (editable mode)
- `numpy`, `scipy`, `matplotlib`
- `jupyter` notebook environment

## Running the Notebooks

### Interactive Mode

```bash
# From repository root
jupyter notebook examples/comparison/
```

Then open any notebook (e.g., `01_duffing.ipynb`) and run all cells.

### Non-Interactive (Command Line)

To run a notebook from the command line and save outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/comparison/01_duffing.ipynb
```

To run all comparison notebooks:

```bash
for nb in examples/comparison/*.ipynb; do
    echo "Running $nb..."
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## Notebook Structure

Each comparison notebook follows this standard structure:

1. **Setup** — Import libraries, locate repository root
2. **Run MATLAB** — Execute Octave subprocess to generate `.mat` file with reference data
3. **Load MATLAB Data** — Read `.mat` file using `scipy.io.loadmat`
4. **Run Python** — Execute Python harmonic balance continuation with same parameters
5. **Compute Metrics** — Calculate RMS amplitude for both MATLAB and Python results
6. **Comparison Plot** — Overlay both curves (MATLAB: green solid, Python: blue dashed)
7. **Accuracy Table** — Peak amplitude/frequency comparison with % error
8. **Assertion** — Verify Python matches MATLAB within tolerance (typically <1% or <5%)

## Expected Validation Results

All 8 comparison notebooks pass with the following peak amplitude errors:

| Notebook | Example | Peak Error | Tolerance | Notes |
|----------|---------|------------|-----------|-------|
| `01_duffing.ipynb` | Duffing oscillator | 0.0007% | <1% | |
| `02_two_dof_cubic.ipynb` | Two-DOF cubic spring | 0.01% | <1% | Reference template |
| `03_two_dof_unilateral.ipynb` | Two-DOF unilateral spring | 0.08% | <1% | |
| `04_two_dof_tanh_friction.ipynb` | Two-DOF tanh friction NMA | 0.09% | <1% | |
| `05_geometric_nonlinearity.ipynb` | Geometric nonlinearity | <1% | <5% | |
| `06_multi_dof_multi_nl.ipynb` | Multi-DOF multi-NL | <5% | <5% | Jenkins element |
| `07_beam_tanh_friction.ipynb` | Beam tanh friction | 0.29% | <1% | |
| `08_beam_cubic_spring_nma.ipynb` | Beam cubic spring NMA | <5% | <5% | Galerkin reduction |

## Troubleshooting

### Octave Not Found

If you see:
```
RuntimeError: Octave not found on PATH. Install Octave and ensure it is on your PATH.
```

**Solution**: Install Octave (see Prerequisites above) and verify it's on your PATH:
```bash
which octave  # macOS/Linux
where octave  # Windows
```

If installed but not on PATH, add it to your shell configuration (`.bashrc`, `.zshrc`, etc.).

### MATLAB Source Files Not Found

If you see:
```
FileNotFoundError: .../matlab_src/EXAMPLES/01_Duffing/...
```

**Solution**: Ensure you're running the notebook from the repository root or that the dynamic path detection is working. The notebooks use:
```python
repo_root = Path(os.getcwd()).parent.parent
```

If running from a different location, you may need to adjust this.

### JSON Parse Errors

If Jupyter cannot open a notebook:
```
Unable to open '01_duffing.ipynb' JSON at position...
```

**Solution**: The notebook file may be corrupted. Restore from git:
```bash
git checkout examples/comparison/01_duffing.ipynb
```

### Import Errors

If you see:
```
ModuleNotFoundError: No module named 'nlvib'
```

**Solution**: Install the package in editable mode:
```bash
pip install -e .
```

Or ensure the notebook's `sys.path` modification is working correctly.

## What Gets Generated

When you run a comparison notebook, it creates:
- `matlab_src/EXAMPLES/<example_name>/hb_data.mat` — MATLAB reference data
- `matlab_src/EXAMPLES/<example_name>/save_data.m` — Octave wrapper script (auto-generated)
- Updated notebook outputs with plots and metrics

The `.mat` files are gitignored and regenerated each time you run the notebook.

## Reference

For implementation details and technical context, see:
- `CONTEXT.md` — Developer documentation for comparison notebooks
- `../../docs/validation.md` — Validation methodology and accuracy details

## Related Documentation

- **Python examples**: `../01_Duffing/`, `../02_two_dof_cubic/`, etc. — Standalone Python examples
- **MATLAB source**: `../../matlab_src/EXAMPLES/` — Original MATLAB reference code
- **Validation tests**: `../../tests/validation/` — Automated fixture-based validation tests
