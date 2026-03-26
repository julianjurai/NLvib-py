"""
nlvib.visualization — Matplotlib-based plotting utilities for NLvib results.

All public functions return a ``matplotlib.figure.Figure`` and never call
``plt.show()``.  An optional ``backend`` parameter enables plotly output
(plotly is an optional dependency).
"""

from nlvib.visualization.plots import (
    ContinuationResult,
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
    "ContinuationResult",
    "plot_backbone",
    "plot_convergence",
    "plot_floquet",
    "plot_frf",
    "plot_harmonic_content",
    "plot_mode_shape",
    "plot_phase_portrait",
    "plot_time_series",
]
