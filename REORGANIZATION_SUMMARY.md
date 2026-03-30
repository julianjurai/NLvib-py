# Repository Reorganization Summary

## Goal
Transform the repository into a clean Python library with:
- Organized MATLAB source reference (`matlab_src/`)
- Clean Python package (`src/nlvib/`)
- Well-structured examples (`examples/`)
- Separate agent framework (`agents/`)
- Better documentation structure (`docs/`)

## Key Changes

### 1. Agent Framework → `agents/`
Move all agent-related files out of root:
- `AGENTS.md`, `PM.md`, `TASKS.md` → `agents/`
- `tools/openai_validator.py` → `agents/tools/`

### 2. MATLAB Source → `matlab_src/`
Consolidate all MATLAB code in one place:
- `matlab/NLvib/DOC/`, `DOC/` → `matlab_src/DOC/`
- `matlab/NLvib/EXAMPLES/` → `matlab_src/EXAMPLES/`
- `matlab/NLvib/SRC/`, `src/*.m`, `src/MechanicalSystems/` → `matlab_src/SRC/`
- Include original README and LICENSE

### 3. Python Examples → `examples/`
Centralize all examples and notebooks:
- `demo/` → `examples/demo/` (tutorial notebooks)
- `notebooks/comparison/` → `examples/comparison/` (MATLAB vs Python)
- `notebooks/*.ipynb` → `examples/notebooks/` (individual examples)
- Keep existing `examples/01_Duffing/`, etc. in place

### 4. Clean Python Source → `src/`
Keep only Python package:
- `src/nlvib/` stays (clean Python code)
- Remove all `.m` files and MATLAB directories

### 5. Documentation → `docs/`
Reorganize for better navigation:
```
docs/
├── index.md                    # Landing page
├── getting-started.md          # Installation & quickstart
├── user-guide/                 # Conceptual guides
│   ├── nonlinear-elements.md
│   ├── harmonic-balance.md
│   └── ... (workflow-oriented)
├── examples/                   # Example docs
│   └── index.md
├── api/                        # API reference (existing)
├── validation.md               # MATLAB comparison
├── differences-from-matlab.md  # API differences
└── contributing.md             # Development guide
```

## Path Updates Required

### Critical: Comparison Notebooks
All 8 comparison notebooks need path updates:
```python
# Current
script_dir = repo_root / 'matlab/NLvib/EXAMPLES/01_Duffing'

# New
script_dir = repo_root / 'matlab_src/EXAMPLES/01_Duffing'
```

### Critical: Repository URL
Fix incorrect repo path in comparison notebooks:
```python
# Current (WRONG)
/Users/julianjurai/Desktop/CustomApps/nonlinear_vibration_analysis_toolbox

# Correct
https://github.com/julianjurai/NLvib-py
```

Files to update:
- All 8 `notebooks/comparison/*.ipynb`
- Verify `CITATION.cff` and `README.md` have correct URL

## Benefits

1. **Clearer structure**: Python library vs MATLAB reference vs agent framework
2. **Better discoverability**: All examples in one place
3. **Cleaner Python package**: No MATLAB files in `src/`
4. **Professional appearance**: Standard Python project layout
5. **Better documentation**: User-guide focused, not just API reference

## Risks

1. Breaking comparison notebooks → Mitigated by systematic path updates
2. Breaking agent tools → Mitigated by testing openai_validator.py after move
3. Breaking documentation build → Mitigated by updating mkdocs.yml
4. Git history → Mitigated by using `git mv` where possible

## Open Questions

1. Keep or delete `matlab/runtime/`? (Check if needed for running MATLAB code)
2. Should agent files be in `agents/` or `.agents/` (hidden)?
3. Should `TASKS.md` stay at root for quick access?
4. Need to check `.github/workflows/` for CI path dependencies
5. Need to check Python source for any hardcoded paths

## Next Steps

1. ✅ Review plan (current step)
2. Get approval to proceed
3. Execute in phases with verification at each step
4. Test critical paths (comparison notebooks, examples)
5. Update documentation
6. Verify git status and commit

## Estimated Impact

- Files moved: ~100+
- Files modified: ~20-30 (path updates)
- New directories: 5-6
- Removed directories: 4-5 (empty after moves)
