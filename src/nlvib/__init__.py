"""
NLvib — Python port of the NLvib MATLAB toolbox.

Nonlinear vibration analysis via harmonic balance, shooting method,
and arc-length continuation.

Original MATLAB toolbox by Malte Krack & Johann Gross (University of Stuttgart).
Python port — MIT.
"""

from importlib.metadata import PackageNotFoundError, version as _metadata_version

__version__: str = "0.1.0"

try:
    __version__ = _metadata_version("nlvib")
except PackageNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Top-level public API — re-export the most-used symbols so that users can
# write ``from nlvib import ContinuationSolver`` without importing sub-modules.
# ---------------------------------------------------------------------------

from nlvib.continuation import ContinuationOptions, ContinuationResult, ContinuationSolver
from nlvib.nonlinearities import (
    NonlinearElement,
    cubic_spring,
    elastic_dry_friction,
    polynomial_stiffness,
    quadratic_damper,
    tanh_dry_friction,
    unilateral_spring,
)
from nlvib.solvers import hb_residual, hb_residual_nma
from nlvib.systems import (
    ChainOfOscillators,
    FE_ElasticRod,
    FE_EulerBernoulliBeam,
    MechanicalSystem,
    SingleMassOscillator,
    System_with_PolynomialStiffness,
    build_beam_matrices,
    craig_bampton,
    rubin,
)
from nlvib.utils import aft_transform, arc_length, dynamic_scaling, freq_to_time, time_to_freq
from nlvib.visualization import (
    plot_backbone,
    plot_convergence,
    plot_floquet,
    plot_frf,
    plot_harmonic_content,
    plot_mode_shape,
    plot_phase_portrait,
    plot_time_series,
)

__all__ = [
    "__version__",
    # continuation
    "ContinuationOptions",
    "ContinuationResult",
    "ContinuationSolver",
    # nonlinearities
    "NonlinearElement",
    "cubic_spring",
    "elastic_dry_friction",
    "polynomial_stiffness",
    "quadratic_damper",
    "tanh_dry_friction",
    "unilateral_spring",
    # solvers
    "hb_residual",
    "hb_residual_nma",
    # systems
    "ChainOfOscillators",
    "FE_ElasticRod",
    "FE_EulerBernoulliBeam",
    "MechanicalSystem",
    "SingleMassOscillator",
    "System_with_PolynomialStiffness",
    "build_beam_matrices",
    "craig_bampton",
    "rubin",
    # utils
    "aft_transform",
    "arc_length",
    "dynamic_scaling",
    "freq_to_time",
    "time_to_freq",
    # visualization
    "plot_backbone",
    "plot_convergence",
    "plot_floquet",
    "plot_frf",
    "plot_harmonic_content",
    "plot_mode_shape",
    "plot_phase_portrait",
    "plot_time_series",
]
