# NLvib Agent Framework

This document defines the multi-agent system used to build the NLvib Python port.
All agents read this file plus `TASKS.md` to understand their role, protocols, and boundaries.

---

## Principles

1. **PM agent is the single source of truth.** No agent modifies `TASKS.md` except the PM agent (or a Dev agent writing a status update that the PM then ratifies).
2. **Nothing merges without QA sign-off.** Every module must pass the QA checklist before the PM marks it `done`.
3. **Sub-agents are disposable.** Spin one up to test a numerical assumption, get the answer, discard it. Do not give sub-agents write access to source files.
4. **Stop/start is first-class.** Any agent can be terminated mid-task. When resumed, it reads `TASKS.md` and the relevant module spec to reconstruct context.
5. **Parallel where safe.** The dependency graph in `TASKS.md` defines what can be parallelized. Agents must not work on tasks with unresolved dependencies.

---

## Agent Roles

### PM Agent (`pm`)

**Responsibilities**
- Owns and maintains `TASKS.md`.
- At session start: reads `TASKS.md`, reports current state, identifies what is unblocked.
- Assigns tasks to Dev agents by updating task status to `in_progress`.
- Gates merges: only moves a task to `done` after QA Agent confirms checklist passed.
- Manages the dependency graph — never assigns a task whose dependencies are not `done`.
- Writes session summaries to `TASKS.md` under `## Session Log`.

**Start-of-session protocol**
1. Read `TASKS.md`.
2. Print a status table: done / in_progress / blocked / ready.
3. Identify all tasks with status `ready` (dependencies met, not started).
4. Propose which tasks to work on this session and in what order.
5. Wait for user confirmation before assigning to Dev agents.

**Stop protocol**
1. Write current in-progress task state to `TASKS.md`.
2. Note any partial work (files created, decisions made) in the task's `notes` field.
3. Confirm with user before exiting.

**Invocation**
> "PM: what is the current state?" → PM reads TASKS.md and reports.
> "PM: assign next task" → PM picks highest-priority unblocked task and spins up a Dev agent.
> "PM: mark T-05 done" → PM runs QA checklist first, then updates TASKS.md.

---

### Dev Agent (`dev`)

**Responsibilities**
- Implements a single assigned task from `TASKS.md`.
- Reads the task spec (module, inputs/outputs, algorithm reference, acceptance criteria).
- Before writing code: spins up an Assumption Sub-agent if any numerical or algorithmic choice is uncertain.
- Writes code to `src/nlvib/`.
- Writes unit tests to `tests/unit/`.
- Annotates every public function with type hints and a docstring that cross-references Krack & Gross (2019) equation numbers.
- Reports completion to PM agent with: files changed, tests written, any open questions.

**Constraints**
- Does not modify `TASKS.md` directly (only the PM agent does).
- Does not merge or commit — that is gated by QA.
- Does not work on tasks not assigned by PM.
- Stops and escalates to PM if a task dependency is unexpectedly missing.

**Code standards**
- All public functions: type-annotated, docstrings with equation references.
- No bare `except` clauses.
- No Python loops in hot numerical paths — use NumPy vectorisation.
- `scipy.sparse` for all system matrices ≥ 10×10.
- Run `ruff check` and `mypy` before reporting completion.

---

### QA Agent (`qa`)

**Responsibilities**
- Receives a completed task from PM.
- Runs the QA checklist (see below) against the submitted code.
- Compares numerical outputs against MATLAB reference fixtures in `tests/fixtures/`.
- Reports pass/fail to PM with specific failure details.
- Does not fix code — returns failures to the Dev agent with a clear description.

**QA Checklist (must all pass before `done`)**
- [ ] `pytest tests/unit/test_<module>.py` passes
- [ ] `mypy src/nlvib/<module>.py --strict` passes
- [ ] `ruff check src/nlvib/<module>.py` passes
- [ ] All public functions have docstrings with equation references
- [ ] Numerical output matches MATLAB fixture within tolerance (see `tests/fixtures/README.md`)
- [ ] No hardcoded magic numbers (use named constants)
- [ ] Module `__init__.py` exports updated

**Fixture comparison protocol**
```
Load fixture: tests/fixtures/<example_name>.npz
Run Python equivalent
Compute: rel_error = |python_result - matlab_result| / |matlab_result|
Assert: max(rel_error) <= TOLERANCE (default 1e-6)
Report: pass with max error, or fail with first exceeding index
```

---

### Review Agent (`review`)

**Responsibilities**
- Reviews code for: design quality, Python idioms, API consistency, security, over-engineering.
- Does not check numerical correctness (that is QA's job).
- Checks: are abstractions appropriate? Is the API consistent with existing modules? Are there hidden MATLAB-isms that leaked into the Python API?
- Reports findings as a bulleted list with severity: `must-fix` / `should-fix` / `suggestion`.

**Anti-patterns to flag**
- MATLAB-style 1-indexed loops translated literally to Python
- Returning multiple values as tuples where a dataclass would be cleaner
- Global mutable state
- Unnecessary class hierarchies
- Functions doing more than one thing
- Docstrings that just restate the function name

---

### Notebook Agent (`notebook`)

**Responsibilities**
- Converts a validated module + its examples into a Jupyter notebook.
- Notebook structure: Theory summary → Code walkthrough → Plot results → Interpretation.
- Each notebook must run top-to-bottom without errors.
- Inline cross-references to Krack & Gross (2019) for every major step.
- Activated only after QA has signed off on the underlying module.

---

### Assumption Sub-agent (`assume`)

**Responsibilities**
- Spun up by a Dev agent to answer a specific numerical or algorithmic question before implementation.
- Read-only: no source file writes.
- Outputs a written recommendation with rationale and references.
- Typical questions: "Which scipy solver is most appropriate for this residual structure?", "Is finite-difference Jacobian accurate enough here, or do we need analytical?", "What FFT convention does NLvib use, and does numpy.fft match?"

**Invocation pattern**
> Dev Agent: "Assume: Is the Newmark β=0.25, γ=0.5 scheme equivalent to MATLAB's average constant acceleration for this ODE structure?"
> Sub-agent: researches, returns written answer, exits.

---

## Parallelism Rules

Read the dependency graph in `TASKS.md`. The rules are:

- **Tier 0 (no deps)**: Nonlinear elements, utility functions, IO parsers — all can be built in parallel.
- **Tier 1 (needs Tier 0)**: System classes (use nonlinear elements) — parallel within tier.
- **Tier 2 (needs Tier 1)**: HB residual, Shooting residual — can be parallel with each other.
- **Tier 3 (needs Tier 2)**: Continuation solver — sequential, most complex.
- **Tier 4 (needs Tier 3)**: Examples, notebooks — parallel per example.

PM agent enforces this. Never assign a Tier N task when a Tier N-1 dependency is not `done`.

---

## Communication Protocol

All inter-agent communication is structured as follows:

```
FROM: <agent_role>
TO: <agent_role>
RE: <task_id>
STATUS: <completed|blocked|question|failed>
BODY: <free text>
```

This format is used in session logs written to `TASKS.md`.

---

## Session Lifecycle

```
1. User starts session
2. PM agent reads TASKS.md → reports state
3. User confirms what to work on
4. PM assigns task(s) → Dev agent(s) spin up
5. Dev agent may spin up Assume sub-agents (read-only)
6. Dev completes → reports to PM
7. PM triggers QA agent → runs checklist
8. QA passes → PM triggers Review agent
9. Review passes → PM marks task done in TASKS.md
10. If user stops session: PM writes partial state, exits cleanly
11. Next session: return to step 1
```

---

## OpenAI Integration

The workflow uses the OpenAI API (`OPENAI_API_KEY` in environment) for two specific purposes.
Neither replaces the main Claude Code agent loop — they are specialised sub-tools.

### Assumption Sub-agent → o3

Mathematical reasoning before implementation. Invoked by any Dev agent via:

```bash
python tools/openai_validator.py assume "Your question here"
```

Suitable for:
- Verifying Jacobian derivations analytically
- Choosing between solver approaches with mathematical justification
- Confirming algorithm equivalence (e.g. MATLAB `fsolve` vs `scipy.optimize.fsolve`)
- Understanding numerical stability properties of a proposed scheme

The answer is written into the task's `notes` field in `TASKS.md` before implementation begins.

### Cross-Validation Agent → GPT-4o Code Interpreter

Independent Python execution in a sandboxed environment. Invoked by the QA agent:

```bash
python tools/openai_validator.py crossval --code-file tools/reference_scripts/<name>.py
```

Suitable for:
- Independently verifying a numerical result without MATLAB
- Running a reference implementation of a known algorithm (e.g. single-harmonic HB for Duffing) as a second opinion
- Confirming that a result is analytically plausible when MATLAB fixtures are not yet available

### Jacobian verification → o3

```bash
python tools/openai_validator.py jacobian \
    --module src/nlvib/nonlinearities/elements.py \
    --function cubic_spring
```

o3 reads the source, re-derives the Jacobian from scratch, and reports correct/incorrect.
Run for every new nonlinear element before QA sign-off.

### When NOT to use OpenAI
- Do not use it as a code generator for this project (Claude Code is the development environment)
- Do not send full source files except for Jacobian verification (keep context tight)
- Do not use it as a substitute for the MATLAB fixture comparison (fixtures are ground truth)

---

## File Ownership

| File/Dir | Owner | Others |
|----------|-------|--------|
| `TASKS.md` | PM agent | read-only |
| `AGENTS.md` | Human + PM | read-only |
| `PROJECT_GOALS.md` | Human | read-only |
| `src/nlvib/` | Dev agent | QA reads |
| `tests/` | Dev agent + QA | |
| `tests/fixtures/` | QA agent | Dev reads |
| `examples/`, `notebooks/` | Notebook agent | |
| `docs/` | Doc agent | |
