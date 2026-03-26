# CMS Model Reduction

The `nlvib.systems.cms` module provides Component Mode Synthesis (CMS) reduction
methods for assembling reduced-order models of large finite-element systems.

Two classical reduction bases are implemented:

- **Craig-Bampton** — constraint modes + fixed-interface normal modes
- **Rubin** — free-interface modes with residual flexibility correction

Equation references: Craig & Bampton (1968); Rubin (1975); Krack & Gross (2019) §5.

---

## craig_bampton

::: nlvib.systems.cms.craig_bampton

---

## rubin

::: nlvib.systems.cms.rubin
