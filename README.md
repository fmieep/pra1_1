# Radiation Diffusion Reproduction Scaffold (v1)

This scaffold is designed for the **core reproduction path** of the paper:

- Picard nonlinear solver
- NK2 = implicit-midpoint Jacobian-free Newton–Krylov
- Picard-type multigrid preconditioning
- M1 / M2 / M3 models
- eta-based time step control with eta = 0.10 and 0.50
- Tables XIII–XVI style iteration statistics

## Current goal

This is **not** a full implementation yet. It is a structured scaffold for Codex (or a human) to fill in.

## File layout

- `config.py`: run and solver configuration dataclasses
- `problem.py`: grid, material map, boundary config, initial condition
- `physics.py`: M1/M2/M3 model definitions and diffusion coefficients
- `discretization.py`: gradients, face coefficients, diffusion operator
- `solvers.py`: GMRES, CG, multigrid V-cycle, Picard MG preconditioner
- `methods.py`: Picard and NK2 step solvers, residuals, JFNK matvec
- `driver.py`: eta-controlled time stepping loop
- `experiments.py`: batch runs and table-style summaries
- `CODEX_HANDOFF.md`: exact instructions to give Codex

## Minimal development order

1. Make `problem.py` + `physics.py` run for M1 on a tiny grid.
2. Implement `discretization.py` and verify the diffusion operator on a simple smooth field.
3. Implement `methods.residual_nk2()` and test it on a trivial steady state.
4. Wrap a working GMRES in `solvers.py`.
5. Implement `methods.nk2_step()` with identity preconditioning.
6. Add `PicardMGPreconditioner` and a minimal `mg_vcycle()`.
7. Implement `methods.picard_step()`.
8. Implement `driver.run_simulation()` and `experiments.run_table_xiii_to_xvi()`.

## Important paper-aligned choices

- Use `numpy.ndarray` everywhere in v1.
- Keep `methods.py` as the single place for time-step methods.
- Keep `solvers.py` as the single place for linear solvers and multigrid in v1.
- Only split files later if they become too large.

