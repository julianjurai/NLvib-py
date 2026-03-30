# Differences from MATLAB

This document outlines the intentional API differences between NLvib-py and the original MATLAB toolbox.

## Philosophy

NLvib-py is **not** a literal translation of the MATLAB code. Instead, it:
- Uses Pythonic idioms and conventions
- Provides a modern, type-annotated API
- Leverages Python's ecosystem (NumPy, SciPy, matplotlib)
- Maintains numerical equivalence while improving usability

## Key Differences

### 1. Object-Oriented Design

**MATLAB**: Procedural functions operating on structs
```matlab
system.M = [1 0; 0 1];
system.K = [1 -0.5; -0.5 1];
```

**Python**: Classes with methods
```python
system = ChainOfOscillators(
    n_dof=2,
    mass=[1.0, 1.0],
    stiffness=[1.0, 1.0]
)
```

### 2. Return Values

**MATLAB**: Multiple return values as separate variables
```matlab
[Q, R, Om_HB] = solve_and_continue(Q, R, Om, ds, Np, ...);
```

**Python**: Dataclasses for structured results
```python
result = arc_length_continuation(
    system=system,
    omega_start=omega0,
    ...
)
# Access via: result.displacements, result.omega, etc.
```

### 3. Indexing

**MATLAB**: 1-based indexing
```matlab
Q(1:n)  % First n elements
```

**Python**: 0-based indexing
```python
Q[0:n]  # First n elements
```

### 4. Array Layout

**MATLAB**: Column vectors by default
```matlab
x = [1; 2; 3]  % Column vector
```

**Python**: 1D arrays, explicit reshaping when needed
```python
x = np.array([1, 2, 3])  # 1D array
x_col = x.reshape(-1, 1)  # Explicit column vector
```

### 5. Function Naming

| MATLAB | Python | Notes |
|--------|--------|-------|
| `solve_and_continue` | `arc_length_continuation` | More descriptive |
| `HB_residual` | `harmonic_balance_residual` | Full words preferred |
| `shooting_residual` | `shooting_residual` | Same |
| `FE_EulerBernoulliBeam` | `EulerBernoulliBeam` | No FE_ prefix |

### 6. Nonlinearity Definition

**MATLAB**: Inline anonymous functions
```matlab
fnl = @(t, x, xdot) k3 * x.^3;
```

**Python**: Explicit nonlinearity classes
```python
nl = CubicSpring(dof=0, stiffness=k3)
system.add_nonlinearity(nl)
```

### 7. Solver Options

**MATLAB**: Options struct
```matlab
options.Np = 1024;
options.tol = 1e-6;
options.max_iter = 20;
```

**Python**: Keyword arguments with defaults
```python
harmonic_balance(
    system=system,
    n_points=1024,
    tol=1e-6,
    max_iter=20
)
```

### 8. Sparse Matrices

**MATLAB**: Automatic sparse detection
```matlab
K = sparse(i, j, v, n, n);
```

**Python**: Explicit scipy.sparse usage
```python
from scipy.sparse import csr_matrix
K = csr_matrix((v, (i, j)), shape=(n, n))
```

### 9. FFT Conventions

Both use the same FFT conventions, but:
- **MATLAB**: `fft`, `ifft` built-in
- **Python**: `numpy.fft.fft`, `numpy.fft.ifft`

FFT scaling and Hermitian symmetry enforcement are handled identically.

### 10. Visualization

**MATLAB**: Direct plotting with `plot`, `figure`
```matlab
figure;
plot(Om_HB, a_rms, 'b-');
```

**Python**: matplotlib with object-oriented API
```python
fig, ax = plt.subplots()
ax.plot(omega, amplitude, 'b-')
ax.set_xlabel('Frequency')
```

## Numerical Equivalence

Despite these API differences, **numerical results are equivalent**:
- Same harmonic balance residual formulation
- Same shooting method integration (Newmark-β)
- Same arc-length continuation algorithm
- Same Floquet stability analysis

See [Validation](validation.md) for quantitative comparison against MATLAB outputs.

## Migration Guide

For users familiar with the MATLAB toolbox:

1. **Replace procedural calls with OOP**
   ```python
   system = ChainOfOscillators(...)
   system.add_nonlinearity(...)
   ```

2. **Use keyword arguments instead of positional**
   ```python
   harmonic_balance(system=system, omega=1.0, ...)
   ```

3. **Access results via attributes**
   ```python
   result.omega, result.amplitude
   ```

4. **Use 0-based indexing**
   ```python
   dof_indices = [0, 1]  # First two DOFs
   ```

## Original MATLAB Source

The original MATLAB source is preserved in `matlab_src/` for reference:
- `matlab_src/SRC/` - Original MATLAB source code
- `matlab_src/EXAMPLES/` - Original MATLAB examples
- `matlab_src/DOC/` - Original MATLAB manual (NLvibManual.pdf)
