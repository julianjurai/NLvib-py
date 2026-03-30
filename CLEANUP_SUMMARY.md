# Examples Directory Cleanup Summary

Date: 2026-03-30

## Changes Made

### ✅ Removed MATLAB Duplicates

Deleted directories that duplicated MATLAB source from `matlab_src/EXAMPLES/`:

**Removed (12 directories):**
1. `examples/02_twoDOFoscillator_cubicSpring/` → Use `matlab_src/EXAMPLES/02_twoDOFoscillator_cubicSpring/`
2. `examples/03_twoDOFoscillator_unilateralSpring/` → Use `matlab_src/EXAMPLES/03_twoDOFoscillator_unilateralSpring/`
3. `examples/04_beam_friction/`
4. `examples/04_twoDOFoscillator_cubicSpring_NM/` → Use `matlab_src/EXAMPLES/04_twoDOFoscillator_cubicSpring_NM/`
5. `examples/05_twoDOFoscillator_tanhDryFriction_NM/` → Use `matlab_src/EXAMPLES/05_twoDOFoscillator_tanhDryFriction_NM/`
6. `examples/06_twoSprings_geometricNonlinearity/` → Use `matlab_src/EXAMPLES/06_twoSprings_geometricNonlinearity/`
7. `examples/07_multiDOFoscillator_multipleNonlinearities/` → Use `matlab_src/EXAMPLES/07_multiDOFoscillator_multipleNonlinearities/`
8. `examples/08_beam_tanhDryFriction/` → Use `matlab_src/EXAMPLES/08_beam_tanhDryFriction/`
9. `examples/09_beam_cubicSpring_NM/` → Use `matlab_src/EXAMPLES/09_beam_cubicSpring_NM/`
10. `examples/10_RubBeR/` → Use `matlab_src/EXAMPLES/10_RubBeR/`
11. `examples/11_Timoshenko/` → Use `matlab_src/EXAMPLES/11_Timoshenko/`
12. `examples/12_condensationToLocalNonlinearity/` → Use `matlab_src/EXAMPLES/12_condensationToLocalNonlinearity/`

**Rationale:** These directories contained only MATLAB `.m` files that duplicated content already in `matlab_src/EXAMPLES/`. All MATLAB source should be referenced from the single source of truth in `matlab_src/`.

### ✅ Removed MATLAB Files from Python Examples

Deleted `.m` files from Python example directories that had both Python and MATLAB code:

**Cleaned directories:**
- `examples/01_Duffing/` - Removed `Duffing.m`, `HB_residual_Duffing.m`
- Other Python examples verified clean

**Kept:** Python scripts (`run.py`) and output directories

### ✅ Removed Redundant Notebooks Directory

Deleted `examples/notebooks/` containing 8 notebooks:
- `01_duffing.ipynb`
- `02_two_dof_cubic.ipynb`
- `03_two_dof_unilateral.ipynb`
- `04_two_dof_tanh_friction.ipynb`
- `05_geometric_nonlinearity.ipynb`
- `06_multi_dof_multi_nl.ipynb`
- `07_beam_tanh_friction.ipynb`
- `08_beam_cubic_spring_nma.ipynb`

**Rationale:** Content redundant with:
- `examples/demo/` - Tutorial notebooks covering concepts
- `examples/comparison/` - MATLAB vs Python validation notebooks
- Python example scripts (`examples/*/run.py`) - Runnable Python code

### ✅ Moved MATLAB Documentation

Moved `examples/EXAMPLES_overview.pdf` → `matlab_src/EXAMPLES/EXAMPLES_overview.pdf`

**Rationale:** This is MATLAB example documentation, belongs with MATLAB source.

## Final Structure

```
examples/
├── demo/                           # Tutorial notebooks (9 notebooks)
│   ├── 00_quickstart.ipynb
│   ├── 01_nonlinear_elements.ipynb
│   ├── 02_mechanical_systems.ipynb
│   ├── 03_harmonic_balance.ipynb
│   ├── 04_shooting.ipynb
│   ├── 05_continuation.ipynb
│   ├── 06_visualization.ipynb
│   ├── 07_parameter_study.ipynb
│   ├── 08_cms_reduction.ipynb
│   └── README.md
│
├── comparison/                     # MATLAB vs Python validation (8 notebooks)
│   ├── 01_duffing.ipynb
│   ├── 02_two_dof_cubic.ipynb
│   ├── 03_two_dof_unilateral.ipynb
│   ├── 04_two_dof_tanh_friction.ipynb
│   ├── 05_geometric_nonlinearity.ipynb
│   ├── 06_multi_dof_multi_nl.ipynb
│   ├── 07_beam_tanh_friction.ipynb
│   ├── 08_beam_cubic_spring_nma.ipynb
│   └── CONTEXT.md
│
└── 01_Duffing/, 02_two_dof_cubic/, ...  # Python example scripts (8 examples)
    ├── run.py                      # Runnable Python script
    └── output/                     # Generated outputs
```

## Benefits

1. **Single Source of Truth**: All MATLAB code in `matlab_src/EXAMPLES/`
2. **No Duplication**: MATLAB files not scattered across multiple locations
3. **Clear Organization**:
   - `demo/` = Tutorials
   - `comparison/` = MATLAB validation
   - `XX_example/` = Python runnable scripts
4. **Reduced Confusion**: No redundant notebooks or duplicate directories
5. **Easier Maintenance**: Update MATLAB examples in one place only

## Documentation Updates

Updated references in:
- ✅ `README.md` - Removed `notebooks/` from structure
- ✅ `docs/getting-started.md` - Updated example running instructions
- ✅ `docs/examples/index.md` - Removed `notebooks/` references
- ✅ `.gitignore` - Removed `!examples/notebooks/*.ipynb` pattern

## MATLAB Source Reference

All MATLAB examples are now in:
```
matlab_src/EXAMPLES/
├── 01_Duffing/
├── 02_twoDOFoscillator_cubicSpring/
├── 03_twoDOFoscillator_unilateralSpring/
├── 04_twoDOFoscillator_cubicSpring_NM/
├── 05_twoDOFoscillator_tanhDryFriction_NM/
├── 06_twoSprings_geometricNonlinearity/
├── 07_multiDOFoscillator_multipleNonlinearities/
├── 08_beam_tanhDryFriction/
├── 09_beam_cubicSpring_NM/
├── 10_RubBeR/
├── 11_Timoshenko/
├── 12_condensationToLocalNonlinearity/
└── EXAMPLES_overview.pdf
```

Comparison notebooks reference these via: `matlab_src/EXAMPLES/XX_example/`

---

**Status**: ✅ Complete - examples/ cleaned and simplified
