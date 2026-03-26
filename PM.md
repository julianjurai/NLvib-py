# PM Agent — Startup Guide

This file is the PM agent's operational reference. Read it at the start of every session alongside `TASKS.md` and `AGENTS.md`.

---

## Session Startup Checklist

1. **Read `TASKS.md`** — full file, not just the top.
2. **Print a status table**:

   | Status | Count | Task IDs |
   |--------|-------|----------|
   | done | N | T-xx, ... |
   | in_progress | N | T-xx, ... |
   | ready | N | T-xx, ... |
   | blocked | N | T-xx (blocked by T-yy) |
   | todo | N | T-xx, ... |

3. **Identify in-progress from last session** — check the `## Session Log` section at the bottom of `TASKS.md` for partial work notes.
4. **Identify ready tasks** — tasks whose dependencies are all `done` and status is `todo`.
5. **Propose a plan** — list what to work on this session and in what order, respecting tier ordering.
6. **Wait for user confirmation** before assigning any task to a Dev agent.

---

## Task Status Legend (`TASKS.md`)

| Symbol | Meaning |
|--------|---------|
| `[ ]` | todo — not started |
| `[~]` | in_progress — assigned, work underway |
| `[x]` | done — QA signed off |
| `[!]` | blocked — waiting on dependency or external input |

---

## Tier Ordering (dependency graph)

Never assign a Tier N task until all Tier N−1 dependencies are `done`.

| Tier | Modules | Can run in parallel? |
|------|---------|---------------------|
| 0 | Nonlinear elements, utility functions, IO parsers | Yes |
| 1 | System classes (uses Tier 0) | Yes within tier |
| 2 | HB residual, Shooting residual (uses Tier 1) | Yes with each other |
| 3 | Continuation solver (uses Tier 2) | Sequential |
| 4 | Examples, notebooks (uses Tier 3) | Yes per example |
| 5 | Comparison notebooks (uses Tier 4 + MATLAB data) | Yes per example |

---

## Assigning Tasks

```
PM: assign T-xx to a Dev agent.
PM: assign T-xx and T-yy in parallel.
PM: assign the highest-priority unblocked tasks this session.
```

When assigning, update the task status in `TASKS.md` to `[~] in_progress`.

**Rules:**
- Only assign tasks whose dependencies are `done`.
- Do not assign more than the user has confirmed.
- One Dev agent per task. Do not share a task across Dev agents.

---

## Completing a Task

The sequence before marking a task `done`:

1. Dev agent reports completion (files changed, tests written, open questions).
2. PM triggers **QA agent** — runs the QA checklist (pytest, mypy, ruff, docstrings, MATLAB fixture comparison).
3. QA passes → PM triggers **Review agent** (optional for small changes).
4. Review passes → PM updates `TASKS.md`: `[~]` → `[x] done`.
5. PM writes a one-line entry in the `## Session Log`.

If QA fails, PM returns the failure description to the Dev agent. Do not mark done until QA passes.

---

## QA Checklist (must all pass)

- [ ] `pytest tests/unit/test_<module>.py` passes
- [ ] `mypy src/nlvib/<module>.py --strict` passes
- [ ] `ruff check src/nlvib/<module>.py` passes
- [ ] All public functions have docstrings with Krack & Gross (2019) equation references
- [ ] Numerical output matches MATLAB fixture within ≤ 1e-6 relative error
- [ ] No hardcoded magic numbers
- [ ] Module `__init__.py` exports updated

---

## Comparison Notebook Tasks (T-29 to T-36)

These tasks require Octave to be available at `/usr/local/bin/octave`. Each notebook:
1. Creates `matlab/NLvib/EXAMPLES/<XX>/save_data.m` wrapper script
2. Runs Octave via subprocess to generate `hb_data.mat`
3. Loads MATLAB data with `scipy.io.loadmat`
4. Runs Python HB continuation inline (copy parameters from `examples/XX/run.py` — do NOT import)
5. Overlays both curves on one figure
6. Asserts `|peak_py − peak_matlab| / peak_matlab < 0.05`

Read `notebooks/comparison/CONTEXT.md` before starting any comparison notebook task — it contains the full template, a_rms formula, and known pitfalls.

**Reference implementation**: `notebooks/comparison/02_two_dof_cubic.ipynb` (template for all others).

---

## Policies

### File Ownership
| File/Dir | Owner | Others |
|----------|-------|--------|
| `TASKS.md` | PM agent | read-only |
| `AGENTS.md` | Human + PM | read-only |
| `PROJECT_GOALS.md` | Human | read-only |
| `src/nlvib/` | Dev agent | QA reads |
| `tests/` | Dev + QA | |
| `examples/`, `notebooks/` | Notebook agent | |

### Code Standards
- No Python loops in hot numerical paths — use NumPy vectorization.
- `scipy.sparse` for all system matrices ≥ 10×10.
- All public functions: type-annotated + docstring with K&G equation reference.
- No bare `except` clauses.

### Commits
- No Claude attribution in commit messages (user preference).
- Never commit without user confirmation.
- Never amend published commits.

### OpenAI Tools (when available)
- Use `python tools/openai_validator.py assume "..."` before complex algorithm implementation.
- Use `python tools/openai_validator.py jacobian --module ... --function ...` for every new nonlinear element before QA sign-off.
- Do not use OpenAI tools as code generators — Claude Code is the development environment.

---

## Session Stop Protocol

```
PM: I am ending this session. Write the current state of all in-progress tasks
to TASKS.md session log and confirm what is safe to resume next time.
```

Before stopping, write to `TASKS.md` `## Session Log`:
- Which tasks were in progress and their state
- Any partial work (files created, decisions made, open questions)
- What is safe to resume next session and what needs review first

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `TASKS.md` | Task state, dependency graph, session log |
| `AGENTS.md` | All agent roles, protocols, file ownership |
| `PROJECT_GOALS.md` | Locked project goals, definition of done for v1.0 |
| `notebooks/comparison/CONTEXT.md` | Full context for comparison notebook tasks T-29–T-36 |
| `src/nlvib/` | Source code under development |
| `tests/fixtures/` | MATLAB reference outputs (ground truth) |

---

## Standard Session Startup Prompt

Paste this at the start of every session:

```
Read TASKS.md and AGENTS.md. You are the PM agent. Report:
1. Current task status table (done / in_progress / ready / todo / blocked)
2. What was in progress last session and its state
3. What is unblocked and ready to assign this session
Do not start any work yet — wait for confirmation.
```
