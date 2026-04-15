#!/usr/bin/env python3
"""
PARALLEL Grid Search Optimization for NTMA Parameters
Uses multiprocessing to evaluate multiple configurations simultaneously

Updated parameter ranges (focusing on small cubic stiffness):
- k2: 0 to 1 in steps of 0.5 → [0, 0.5, 1.0]
- d2: 0.001 to 0.05 in steps of 0.001 → [0.001, 0.002, ..., 0.05]
- alpha1: 0.001 to 0.01 in steps of 0.0005 → [0.001, 0.0015, ..., 0.01]

Total: 3 × 50 × 20 = 3,000 combinations
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Add repo to path
repo_root = Path.cwd().parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from nlvib.systems.oscillators import ChainOfOscillators
from nlvib.nonlinearities.elements import polynomial_stiffness
from nlvib.solvers.harmonic_balance import hb_residual
from nlvib.continuation.solver import ContinuationSolver, ContinuationOptions

# Fixed parameters
MASSES = [1.0, 0.05]
F_AMP = 0.11
H = 7
OMEGA_START = 0.8
OMEGA_END = 1.4

# DEFAULT configuration (baseline)
DEFAULT_K2 = 0.0
DEFAULT_D2 = 0.013
DEFAULT_ALPHA1 = 0.0042

# FINAL grid search ranges (focusing on small alpha1)
K2_VALUES = np.arange(0, 1.0 + 0.5, 0.5)  # [0, 0.5, 1.0]
D2_VALUES = np.arange(0.001, 0.05 + 0.001, 0.001)  # [0.001, 0.002, ..., 0.05]
ALPHA1_VALUES = np.arange(0.001, 0.01 + 0.0005, 0.0005)  # [0.001, 0.0015, ..., 0.01]

# Parallel processing settings
NUM_WORKERS = min(8, cpu_count() - 2)  # Leave 2 cores for system
BATCH_SIZE = 50  # Save checkpoint after this many completed tasks

print("="*80)
print("PARALLEL GRID SEARCH OPTIMIZATION: NTMA Parameters")
print("="*80)
print(f"CPU cores available: {cpu_count()}")
print(f"Workers: {NUM_WORKERS}")
print(f"k2 values: {len(K2_VALUES)} → {K2_VALUES.tolist()}")
print(f"d2 values: {len(D2_VALUES)} → [{D2_VALUES[0]:.4f}, {D2_VALUES[1]:.4f}, ..., {D2_VALUES[-1]:.4f}]")
print(f"alpha1 values: {len(ALPHA1_VALUES)} → [{ALPHA1_VALUES[0]:.4f}, {ALPHA1_VALUES[1]:.4f}, ..., {ALPHA1_VALUES[-1]:.4f}]")
print(f"\nTotal combinations: {len(K2_VALUES)} × {len(D2_VALUES)} × {len(ALPHA1_VALUES)} = {len(K2_VALUES) * len(D2_VALUES) * len(ALPHA1_VALUES):,}")
print(f"DEFAULT baseline: k2={DEFAULT_K2}, d2={DEFAULT_D2:.4f}, α1={DEFAULT_ALPHA1:.4f}")
print(f"Expected speedup: ~{NUM_WORKERS}x faster")
print("="*80)

# Cache file
CACHE_FILE = Path('grid_search_final_cache.pkl')

def load_cache():
    """Load existing cache or create new one"""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        print(f"\n✓ Loaded cache: {len(cache['results'])} evaluations already completed")
        return cache
    else:
        return {
            'results': {},
            'completed': 0,
            'start_time': time.time(),
            'default_peak': None
        }

def save_cache(cache):
    """Save cache to disk"""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def evaluate_parameters(params):
    """
    Evaluate peak amplitude for given parameters using arc-length continuation.
    This function must be picklable for multiprocessing.

    Args:
        params: tuple of (k2, d2, alpha1)

    Returns:
        tuple: (params, result_dict)
    """
    k2, d2, alpha1 = params

    try:
        stiffnesses = [1.0, k2, 0.0]
        dampings = [0.002, d2, 0.0]

        system = ChainOfOscillators(masses=MASSES, stiffnesses=stiffnesses, dampings=dampings)

        # Cubic stiffness between masses (alpha1)
        _exp = np.array([[3,0],[2,1],[1,2],[0,3]], dtype=np.intp)
        _coeff = np.array([alpha1, -3*alpha1, 3*alpha1, -alpha1])
        system.add_nonlinear_element(polynomial_stiffness(_exp, _coeff, np.array([0,1], dtype=np.intp)))
        system.add_nonlinear_element(polynomial_stiffness(_exp, _coeff, np.array([1,0], dtype=np.intp)))

        n_dof = system.n_dof
        n_total = n_dof * (2*H + 1)
        excitation = {'dof': 0, 'amplitude': F_AMP}

        # Initial Newton solve
        Q = np.zeros(n_total)
        for _ in range(50):
            R, J = hb_residual(Q, OMEGA_START, system, H, excitation)
            if np.linalg.norm(R) < 1e-10:
                break
            Q += np.linalg.solve(J, -R)

        # Arc-length continuation (fast settings for grid search)
        opts = ContinuationOptions(
            verbose=False,
            ds_initial=0.05,
            ds_min=1e-4,
            ds_max=0.2,
            max_steps=300,
            max_newton_iter=15,
            newton_tol=1e-6,
            adapt_step=True,
            lambda_min=OMEGA_START-0.05,
            lambda_max=OMEGA_END+0.05
        )

        result = ContinuationSolver().run(
            lambda x, lam: hb_residual(x, lam, system, H, excitation),
            Q, OMEGA_START, opts
        )

        solutions = result.solutions
        omega = solutions[:, -1]
        Q_all = solutions[:, :-1]
        Q_dof0 = Q_all.reshape(Q_all.shape[0], 2*H+1, n_dof)[:, :, 0]
        a_rms = np.sqrt(np.sum(Q_dof0**2, axis=1)) / np.sqrt(2)

        # Find peak in frequency range
        mask = (omega >= OMEGA_START) & (omega <= OMEGA_END)
        if mask.sum() > 0:
            peak = a_rms[mask].max()
        else:
            peak = a_rms.max()

        return (params, {'peak': peak, 'omega': omega, 'a_rms': a_rms, 'success': True})

    except Exception as e:
        # Continuation failed - return high penalty
        return (params, {'peak': 1e6, 'omega': np.array([]), 'a_rms': np.array([]), 'success': False, 'error': str(e)})


def main():
    # Load cache
    cache = load_cache()

    # Evaluate DEFAULT first if not already done
    default_key = (DEFAULT_K2, DEFAULT_D2, DEFAULT_ALPHA1)
    if default_key not in cache['results']:
        print("\n=== Evaluating DEFAULT Configuration ===")
        _, result = evaluate_parameters(default_key)
        cache['results'][default_key] = result
        if result['success']:
            cache['default_peak'] = result['peak']
            print(f"DEFAULT: Peak={result['peak']:.6f}")
        save_cache(cache)

    default_peak = cache.get('default_peak', None)

    # Generate all parameter combinations to evaluate
    all_params = []
    for k2 in K2_VALUES:
        for d2 in D2_VALUES:
            for alpha1 in ALPHA1_VALUES:
                key = (float(k2), float(d2), float(alpha1))
                if key not in cache['results']:
                    all_params.append(key)

    total_to_evaluate = len(all_params)
    total_combinations = len(K2_VALUES) * len(D2_VALUES) * len(ALPHA1_VALUES)
    already_done = len(cache['results'])

    print(f"\n=== Starting Parallel Grid Search ===")
    print(f"Already completed: {already_done}/{total_combinations}")
    print(f"To evaluate: {total_to_evaluate}")
    print(f"Workers: {NUM_WORKERS}")
    print("="*80 + "\n")

    if total_to_evaluate == 0:
        print("✓ All evaluations already complete!")
        return cache

    start_time = time.time()
    completed = 0
    best_peak = float('inf')
    best_params = None

    # Find current best from cache
    for key, result in cache['results'].items():
        if result['success'] and result['peak'] < best_peak:
            best_peak = result['peak']
            best_params = key

    # Process in parallel batches
    with Pool(processes=NUM_WORKERS) as pool:
        for batch_start in range(0, total_to_evaluate, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_to_evaluate)
            batch_params = all_params[batch_start:batch_end]

            batch_results = pool.map(evaluate_parameters, batch_params)

            # Update cache with batch results
            for params, result in batch_results:
                cache['results'][params] = result
                completed += 1

                # Track best
                if result['success'] and result['peak'] < best_peak:
                    best_peak = result['peak']
                    best_params = params
                    reduction = 100 * (1 - best_peak / default_peak) if default_peak else 0
                    print(f"\n★ NEW BEST: k2={params[0]:.2f}, d2={params[1]:.4f}, α1={params[2]:.4f} → Peak={best_peak:.6f} ({reduction:.1f}% reduction)")

            # Progress update
            elapsed = time.time() - start_time
            total_done = already_done + completed
            progress_pct = total_done / total_combinations * 100
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total_to_evaluate - completed) / rate if rate > 0 else 0

            reduction = 100 * (1 - best_peak / default_peak) if default_peak else 0
            print(f"[{total_done:4d}/{total_combinations}] ({progress_pct:5.1f}%) "
                  f"Batch {batch_start//BATCH_SIZE + 1}/{(total_to_evaluate + BATCH_SIZE - 1)//BATCH_SIZE} | "
                  f"Best={best_peak:.6f} ({reduction:.1f}%) | "
                  f"Rate: {rate:.1f}/s | ETA: {remaining/60:.0f}m")

            # Save checkpoint
            cache['completed'] = total_done
            save_cache(cache)

    # Final save
    save_cache(cache)

    # Print results
    elapsed = time.time() - start_time
    successful = sum(1 for r in cache['results'].values() if r['success'])
    failed = len(cache['results']) - successful

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Evaluations: {len(cache['results'])}/{total_combinations}")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Average rate: {total_to_evaluate/elapsed:.1f} evaluations/second")
    print(f"Speedup vs sequential: ~{NUM_WORKERS:.1f}x")

    if best_params:
        best_result = cache['results'][best_params]
        reduction = 100 * (1 - best_peak / default_peak) if default_peak else 0
        print(f"\n=== OPTIMUM FOUND ===")
        print(f"k2 = {best_params[0]:.6f}")
        print(f"d2 = {best_params[1]:.6f}")
        print(f"alpha1 = {best_params[2]:.6f}")
        print(f"Peak amplitude = {best_peak:.6f}")
        print(f"Reduction from DEFAULT = {reduction:.1f}%")
        print("="*80)

        # Save optimum separately
        optimum = {
            'k2': best_params[0],
            'd2': best_params[1],
            'alpha1': best_params[2],
            'peak': best_peak,
            'reduction': reduction,
            'omega': best_result['omega'],
            'a_rms': best_result['a_rms']
        }
        with open('grid_search_final_optimum.pkl', 'wb') as f:
            pickle.dump(optimum, f)
        print(f"\n✓ Optimum saved to: grid_search_final_optimum.pkl")

    return cache

if __name__ == '__main__':
    cache = main()
