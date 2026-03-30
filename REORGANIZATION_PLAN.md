# Repository Reorganization Plan

## Overview
Transform this repository from a MATLAB-to-Python port structure into a proper Python library with well-organized MATLAB source reference and examples.

---

## Current Structure

```
.
├── AGENTS.md                   # Agent framework docs
├── PM.md                       # PM agent guide
├── TASKS.md                    # Task tracking
├── demo/                       # Python demo notebooks (12 .ipynb files)
├── DOC/                        # MATLAB manual (NLvibManual.pdf)
├── examples/                   # Python examples (numbered 01-12)
├── matlab/
│   ├── NLvib/                  # Original MATLAB repo
│   │   ├── DOC/
│   │   ├── EXAMPLES/
│   │   ├── SRC/
│   │   ├── README.md
│   │   └── LICENSE
│   ├── demo/
│   └── runtime/
├── notebooks/
│   ├── comparison/             # MATLAB vs Python comparison notebooks
│   └── (8 individual .ipynb files)
├── src/
│   ├── nlvib/                  # Python package
│   ├── MechanicalSystems/      # MATLAB code
│   ├── *.m files               # MATLAB source files
│   └── nlvib.egg-info/
├── tests/                      # Python tests
└── tools/                      # Build tools + openai_validator.py
```

---

## Target Structure

```
.
├── agents/                     # Agent framework files
│   ├── AGENTS.md
│   ├── PM.md
│   ├── TASKS.md
│   └── tools/
│       └── openai_validator.py
├── matlab_src/                 # Original MATLAB source (reference only)
│   ├── DOC/
│   │   └── NLvibManual.pdf
│   ├── EXAMPLES/               # MATLAB examples (01_Duffing, etc.)
│   ├── SRC/                    # MATLAB source code
│   │   ├── MechanicalSystems/
│   │   └── (all .m files)
│   ├── README.md               # Original MATLAB README
│   └── LICENSE                 # Original MATLAB LICENSE
├── src/
│   └── nlvib/                  # Python package (clean, no MATLAB)
│       ├── nonlinearities/
│       ├── systems/
│       ├── solvers/
│       ├── continuation/
│       ├── io/
│       └── utils/
├── examples/                   # All Python examples and notebooks
│   ├── demo/                   # Demo notebooks (00-08)
│   ├── comparison/             # MATLAB vs Python comparison notebooks
│   ├── 01_duffing/
│   ├── 02_two_dof_cubic/
│   ├── ... (all numbered examples)
│   └── notebooks/              # Individual example notebooks (01-08)
├── tests/                      # Python tests (unchanged)
├── tools/                      # Build tools (excluding openai_validator.py)
│   ├── bump_version.py
│   ├── fetch_matlab_source.sh
│   ├── generate_fixtures.m
│   ├── generate_fixtures.py
│   └── reference_scripts/
├── docs/                       # Documentation (reorganized)
│   ├── index.md                # Main landing page
│   ├── getting-started.md      # Installation & quickstart
│   ├── user-guide/             # User guides
│   │   ├── nonlinear-elements.md
│   │   ├── mechanical-systems.md
│   │   ├── harmonic-balance.md
│   │   ├── shooting-method.md
│   │   ├── continuation.md
│   │   └── visualization.md
│   ├── examples/               # Example documentation
│   │   ├── index.md            # Examples overview
│   │   ├── duffing.md
│   │   └── ... (one per example)
│   ├── api/                    # API reference (auto-generated)
│   │   ├── nonlinearities.md
│   │   ├── systems.md
│   │   ├── solvers.md
│   │   ├── continuation.md
│   │   ├── cms.md
│   │   ├── visualization.md
│   │   ├── io.md
│   │   └── utils.md
│   ├── validation.md           # MATLAB comparison & validation
│   ├── differences-from-matlab.md  # API differences from original
│   └── contributing.md         # Development guide
├── README.md
├── LICENSE
└── ...
```

---

## Migration Steps

### Phase 1: Agent Files
1. Create `agents/` directory
2. Move `AGENTS.md` → `agents/AGENTS.md`
3. Move `PM.md` → `agents/PM.md`
4. Move `TASKS.md` → `agents/TASKS.md`
5. Create `agents/tools/` directory
6. Move `tools/openai_validator.py` → `agents/tools/openai_validator.py`

### Phase 2: MATLAB Source Consolidation
7. Create `matlab_src/` directory structure
8. Move `matlab/NLvib/DOC/` → `matlab_src/DOC/`
9. Move `matlab/NLvib/EXAMPLES/` → `matlab_src/EXAMPLES/`
10. Move `matlab/NLvib/SRC/` → `matlab_src/SRC/`
11. Move root `DOC/NLvibManual.pdf` → `matlab_src/DOC/` (merge with existing)
12. Move `src/MechanicalSystems/` → `matlab_src/SRC/MechanicalSystems/`
13. Move all `.m` files from `src/` → `matlab_src/SRC/`
14. Copy `matlab/NLvib/README.md` → `matlab_src/README.md`
15. Copy `matlab/NLvib/LICENSE` → `matlab_src/LICENSE`

### Phase 3: Python Examples Reorganization
16. Create `examples/demo/` directory
17. Move `demo/*.ipynb` → `examples/demo/` (12 notebooks)
18. Create `examples/comparison/` directory
19. Move `notebooks/comparison/*` → `examples/comparison/` (8 notebooks + CONTEXT.md)
20. Create `examples/notebooks/` directory
21. Move individual `notebooks/*.ipynb` → `examples/notebooks/` (8 notebooks)
22. Keep existing `examples/01_Duffing/`, `examples/02_two_dof_cubic/`, etc. in place

### Phase 4: Path Updates
23. Update all comparison notebooks:
    - Replace `matlab/NLvib/EXAMPLES/` → `matlab_src/EXAMPLES/`
    - Update relative paths to account for new location
24. Update agent files:
    - Update any references to `TASKS.md`, `AGENTS.md`, `PM.md` in tools
    - Update `tools/openai_validator.py` imports if needed

### Phase 5: Cleanup
25. Remove empty `matlab/` directory (after verifying runtime isn't needed)
26. Remove empty `demo/` directory
27. Remove empty `DOC/` directory
28. Remove empty `notebooks/` directory
29. Clean up `src/nlvib.egg-info/` if needed

### Phase 6: Documentation Reorganization
30. Create `docs/getting-started.md` (extract from index.md)
31. Create `docs/user-guide/` directory
32. Move/create user guide pages:
    - Extract from existing docs or create new guides
    - Organize by workflow rather than API structure
33. Create `docs/examples/` directory
34. Create `docs/examples/index.md` (overview of all examples)
35. Create individual example docs if needed
36. Create `docs/differences-from-matlab.md`
37. Create `docs/contributing.md` (development guide)
38. Update `mkdocs.yml` to reflect new structure

### Phase 7: General Documentation Updates
39. Update `README.md`:
    - Update project structure section
    - Update example paths
    - Update reference to MATLAB source location
40. Update `PROJECT_GOALS.md` if it references old paths
41. Update any paths in `mkdocs.yml`
42. Update `.gitignore` if needed for new structure

---

## Critical Path Updates

### Comparison Notebooks Path Changes

All comparison notebooks currently reference:
```python
script_dir = repo_root / 'matlab/NLvib/EXAMPLES/01_Duffing'
```

Must become:
```python
script_dir = repo_root / 'matlab_src/EXAMPLES/01_Duffing'
```

Files to update:
- `examples/comparison/01_duffing.ipynb`
- `examples/comparison/02_two_dof_cubic.ipynb`
- `examples/comparison/03_two_dof_unilateral.ipynb`
- `examples/comparison/04_two_dof_tanh_friction.ipynb`
- `examples/comparison/05_geometric_nonlinearity.ipynb`
- `examples/comparison/06_multi_dof_multi_nl.ipynb`
- `examples/comparison/07_beam_tanh_friction.ipynb`
- `examples/comparison/08_beam_cubic_spring_nma.ipynb`

### Agent Tools Path Changes

Update references in:
- `agents/tools/openai_validator.py`
- Any scripts that reference MATLAB examples for fixture generation

---

## Verification Checklist

After reorganization, verify:
- [ ] All comparison notebooks run successfully with new paths
- [ ] All Python examples still run (`examples/0X_*/run.py`)
- [ ] All tests pass (`pytest`)
- [ ] Agent tools work with new paths
- [ ] Documentation builds successfully (`mkdocs build`)
- [ ] No broken symlinks or empty directories
- [ ] Git status shows expected changes
- [ ] README accurately reflects new structure

---

## Risks & Mitigations

**Risk**: Breaking comparison notebooks
- **Mitigation**: Systematic path update with verification after each notebook

**Risk**: Breaking agent tooling
- **Mitigation**: Update and test `openai_validator.py` after move

**Risk**: Losing MATLAB runtime if needed
- **Mitigation**: Check if `matlab/runtime/` is used before deletion

**Risk**: Git history confusion
- **Mitigation**: Use `git mv` instead of manual moves where possible

---

## Questions for Review

1. Should `matlab/runtime/` be kept, moved to `matlab_src/runtime/`, or deleted?
   - **Recommendation**: Check if it's needed for running MATLAB code, otherwise delete

2. Should `examples/notebooks/` be a separate folder or merged with `examples/demo/`?
   - **Current plan**: Keep separate as they serve different purposes
   - `examples/demo/` = Tutorial notebooks
   - `examples/notebooks/` = Example-specific notebooks
   - `examples/comparison/` = MATLAB vs Python comparison

3. Should we update import paths in any Python code that references the old structure?
   - **Need to check**: Any hardcoded paths in Python source code

4. Are there any CI/CD configurations that reference the current paths?
   - **Need to check**: `.github/workflows/` for path dependencies

5. Should agent files be in `agents/` at root or in a hidden `.agents/` directory?
   - **Current plan**: `agents/` visible at root for transparency

6. Should docs reorganization be more granular (e.g., tutorials/ vs user-guide/)?
   - **Current plan**: `user-guide/` for conceptual docs, `examples/` for hands-on

7. Should `TASKS.md` stay accessible at root for quick reference?
   - **Current plan**: Move to `agents/TASKS.md` for organization

---

## Rollback Plan

If reorganization fails:
1. All changes will be in a single commit or branch
2. Can revert with `git reset --hard HEAD~1` or `git checkout main`
3. No files will be deleted permanently before verification
