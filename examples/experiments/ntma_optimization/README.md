# NTMA Grid Search Optimization

Comprehensive grid search optimization for Nonlinear Tuned Mass Absorber (NTMA) parameters using parallel harmonic balance continuation.

## Overview

This project performs a systematic optimization of a two-degree-of-freedom NTMA system to minimize peak vibration amplitude. The optimization explores 3,000 parameter combinations using parallel processing and harmonic balance methods with arc-length continuation.

### System Description

- **Primary mass** (m1): 1.0 kg
- **Secondary mass** (m2): 0.05 kg (5% of primary)
- **Excitation**: Harmonic force at DOF 0, amplitude 0.11 N
- **Frequency range**: 0.8 to 1.4 rad/s
- **Nonlinearity**: Cubic stiffness between masses

### Optimization Parameters

The grid search varies three key parameters:

| Parameter | Description | Range | Step | Values |
|-----------|-------------|-------|------|--------|
| **k2** | Linear stiffness between masses | 0 to 1 | 0.5 | 3 |
| **d2** | Damping between masses | 0.001 to 0.05 | 0.001 | 50 |
| **α1** | Cubic stiffness coefficient | 0.001 to 0.01 | 0.0005 | 20 |

**Total combinations**: 3 × 50 × 20 = **3,000**

## Results

### Optimum Configuration

| Configuration | k2 | d2 | α1 | Peak Amplitude | Reduction |
|---------------|-----|-----|-----|----------------|-----------|
| **DEFAULT** | 0.0 | 0.013 | 0.0042 | 1.195124 | 0% (baseline) |
| **OPTIMUM** | 0.0 | 0.001 | 0.010 | 0.663075 | **44.5%** |

###Key Findings

1. **No linear stiffness needed**: Optimal k2 = 0
   - Vibration suppression achieved purely through nonlinear cubic stiffness
   - Linear stiffness between masses does not improve performance

2. **Minimal damping optimal**: d2 = 0.001
   - Lowest damping in search range minimizes energy dissipation
   - Higher damping reduces effectiveness of nonlinear energy transfer

3. **Optimum at boundary**: α1 = 0.010 at upper search limit
   - Suggests slightly higher values might improve performance further
   - Trade-off: larger α1 (e.g., 4.801) gives 62.8% reduction but causes MATLAB/Octave numerical issues

4. **Practical design**: Current optimum works reliably in all tools
   - Python (nlvib-py): Full continuation with 150+ points
   - MATLAB/Octave: Stable continuation (tested compatible)
   - Previous higher α1 values caused premature termination in MATLAB (only 3 points)

### Performance

- **Method**: Parallel grid search with 8 CPU cores
- **Duration**: 14.1 minutes
- **Success rate**: 100% (3,000/3,000 successful continuations)
- **Average rate**: 0.4 evaluations/second
- **Speedup**: ~8x faster than sequential

## Files

### Main Files

- **`ntma_grid_search.ipynb`** - Comprehensive results notebook with visualizations
- **`grid_search_parallel.py`** - Parallel grid search implementation
- **`grid_search_parallel.log`** - Execution log with progress tracking

### Data Files

- **`grid_search_final_cache.pkl`** (9.5 MB) - Complete results for all 3,000 configurations
  - Contains frequency response data for every parameter combination
  - Includes success/failure status for each evaluation
  - Used for post-processing and analysis

- **`grid_search_final_optimum.pkl`** (778 B) - Best configuration with full frequency response
  - Optimum parameters and performance metrics
  - Complete omega and amplitude arrays for plotting

### Visualization Outputs

- **`default_frequency_response.png`** - Baseline DEFAULT configuration
- **`optimization_progression.png`** - Evolution from DEFAULT to OPTIMUM
- **`matlab_comparison.png`** - Side-by-side DEFAULT vs OPTIMUM comparison

## Usage

### Running the Optimization

```bash
# Execute parallel grid search (requires 8+ CPU cores for full speedup)
python grid_search_parallel.py

# Results saved to:
#   - grid_search_final_cache.pkl
#   - grid_search_final_optimum.pkl
#   - grid_search_parallel.log
```

### Viewing Results

```bash
# Open the Jupyter notebook to see all visualizations
jupyter notebook ntma_grid_search.ipynb

# Or convert to HTML for static viewing
jupyter nbconvert --to html ntma_grid_search.ipynb
```

### Loading Results in Python

```python
import pickle

# Load optimum configuration
with open('grid_search_final_optimum.pkl', 'rb') as f:
    optimum = pickle.load(f)

print(f"Optimum peak amplitude: {optimum['peak']:.6f}")
print(f"Reduction from DEFAULT: {optimum['reduction']:.1f}%")

# Load complete cache
with open('grid_search_final_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

print(f"Total evaluations: {len(cache['results'])}")
print(f"DEFAULT baseline: {cache['default_peak']:.6f}")
```

## Methodology

### Harmonic Balance Continuation

Each parameter combination is evaluated using:

1. **Initial Newton solve** at ω = 0.8 rad/s
   - Converges initial guess to periodic solution
   - Uses H = 7 harmonics for accuracy

2. **Arc-length continuation** from ω = 0.8 to 1.4 rad/s
   - Adaptive step size control (ds: 1e-4 to 0.2)
   - Newton tolerance: 1e-6
   - Maximum 300 steps per continuation

3. **Peak extraction** from frequency response
   - Find maximum RMS amplitude in frequency range
   - Filter to region of interest (0.8 ≤ ω ≤ 1.4)

### Parallel Processing

- **Workers**: 8 parallel processes (configurable)
- **Batch size**: 50 evaluations per checkpoint
- **Caching**: Automatic resume from interruptions
- **Progress tracking**: Real-time updates with ETA

## Comparison: Small vs Large Cubic Stiffness

| Search Range | Optimum α1 | Peak | Reduction | MATLAB Status |
|--------------|------------|------|-----------|---------------|
| 0.001-5.0 (Previous) | 4.801 | 0.445 | 62.8% | Fails (3 points) |
| 0.001-0.01 (Current) | 0.010 | 0.663 | 44.5% | Works reliably |

**Trade-off**: The current optimum sacrifices 18.3% additional reduction for numerical reliability across all platforms.

## Technical Details

### System Equations

The NTMA system is modeled as a two-DOF oscillator with cubic nonlinearity:

```
M·ẍ + C·ẋ + K·x + f_nl(x) = F·cos(ωt)
```

Where:
- M = diag([1.0, 0.05])
- C = [[c1+c2, -c2], [-c2, c2]]
- K = [[k1+k2, -k2], [-k2, k2]]
- f_nl = α1·(x1-x2)³ (cubic spring between masses)

### Harmonic Balance Formulation

Solution assumed as truncated Fourier series:

```
x(t) = a0 + Σ[ak·cos(kωt) + bk·sin(kωt)]  for k=1 to H
```

With H=7 harmonics, this yields (2·H+1)·n_dof = 30 unknowns per configuration.

## Dependencies

- **nlvib-py**: Nonlinear vibration analysis library
  - Harmonic balance solver
  - Arc-length continuation
  - Polynomial nonlinearity elements

- **Python 3.8+**
- **NumPy**: Array operations
- **Matplotlib**: Visualization
- **Multiprocessing**: Parallel execution
- **Pickle**: Data serialization

## Future Work

1. **Extended α1 range**: Test 0.01 < α1 < 0.05 to find true optimum
2. **Multi-objective optimization**: Consider bandwidth and robustness
3. **Experimental validation**: Compare with physical NTMA prototype
4. **Gradient-based refinement**: Local optimization around grid optimum
5. **Uncertainty quantification**: Sensitivity to parameter variations

## References

- **Grid search implementation**: `grid_search_parallel.py`
- **Execution log**: `grid_search_parallel.log`
- **Results notebook**: `ntma_grid_search.ipynb`

## Citation

This optimization uses the **nlvib-py** library for nonlinear vibration analysis:

```
@software{nlvib_py,
  title = {nlvib-py: Nonlinear Vibration Analysis in Python},
  author = {nlvib-py contributors},
  year = {2024},
  url = {https://github.com/nlvib/nlvib-py}
}
```

## License

Part of the nlvib-py examples collection.

---

**Last updated**: 2026-04-15
**Optimization completed**: 2026-04-14 (14.1 minutes)
**Success rate**: 100% (3,000/3,000 evaluations)
