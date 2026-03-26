# Visualization

The `nlvib.visualization` module provides eight plotting functions covering all standard
NLvib result types. All functions:

- Return a `matplotlib.figure.Figure` — no `plt.show()` calls, no global state
- Accept an optional `ax=` parameter to plot into an existing axes object
- Accept an optional `backend="matplotlib"|"plotly"` parameter (plotly is an optional dependency)
- Drive stable/unstable branch coloring from a `stability` boolean array on the result object

---

## plot_frf

::: nlvib.visualization.plots.plot_frf

---

## plot_backbone

::: nlvib.visualization.plots.plot_backbone

---

## plot_time_series

::: nlvib.visualization.plots.plot_time_series

---

## plot_phase_portrait

::: nlvib.visualization.plots.plot_phase_portrait

---

## plot_floquet

::: nlvib.visualization.plots.plot_floquet

---

## plot_mode_shape

::: nlvib.visualization.plots.plot_mode_shape

---

## plot_harmonic_content

::: nlvib.visualization.plots.plot_harmonic_content

---

## plot_convergence

::: nlvib.visualization.plots.plot_convergence
