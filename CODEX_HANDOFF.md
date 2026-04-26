# Codex handoff instructions

You are implementing a Python scaffold for the core reproduction path of a numerical PDE paper.
Do **not** redesign the architecture. Fill in the existing scaffold while preserving file responsibilities.

## Scope
Implement only the following in v1:

- Picard nonlinear solver
- NK2 = implicit midpoint Jacobian-free Newton–Krylov
- Picard-type multigrid preconditioning
- M1 / M2 / M3
- eta-based time step control with eta = 0.10 and 0.50
- Table-style summaries similar to Tables XIII–XVI

## Architecture constraints
Keep the project at exactly these top-level files unless absolutely necessary:

- config.py
- problem.py
- physics.py
- discretization.py
- solvers.py
- methods.py
- driver.py
- experiments.py

Do not create a large package tree in v1.

## Numerical design requirements
1. Arrays should be `numpy.ndarray`.
2. NK2 should use a Jacobian-free matvec of the form
   Jv ≈ [F(E + eps v) - F(E)] / eps
3. Newton-Krylov should use GMRES.
4. Preconditioning should be a Picard-type multigrid V-cycle.
5. Picard and NK2 should both return iteration stats for each step.
6. `driver.run_simulation()` must record:
   - time history
   - dt history
   - linear iterations per step
   - nonlinear iterations per step
   - eta per step
7. `experiments.run_table_xiii_to_xvi()` must batch over:
   - method in {picard, nk2}
   - model in {M1, M2, M3}
   - eta in {0.10, 0.50}
   - grid in {(32,32), (64,64), (128,128), (256,256)}

## Implementation strategy
- First make M1 work on a small grid.
- Then generalize to M2 and M3.
- Prefer correctness and clarity over premature optimization.
- Keep function signatures intact unless there is a strong reason to change them.

## If something is ambiguous
Prefer adding a short TODO comment rather than silently inventing a different algorithm.

