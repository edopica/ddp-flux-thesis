# 2026-09-14 — Conditional-graph Optimisers notebook

## Goal

Add a hands-on, executable Pluto.jl tutorial that connects three layers:

1. Optimisers.jl in isolation, especially the skip-on-`nothing` semantics.
2. Optimisers inside Flux (`setup`, `withgradient`, `train!`).
3. The distributed case, where the current per-leaf `DistributedOptimizer`
   issues data-dependent collectives and breaks on conditional graphs.

## Deliverable

`notebook/conditional_graph_optimisers.jl` — a valid Pluto `.jl` notebook
(`# v0.20.0` header, 25 cells, 13 Markdown + 12 code, matching `Cell order`
block).

Sections:

1. Setup (`using Optimisers, Flux, Functors, Zygote`; library roles).
2. Optimisers.jl alone: nested `NamedTuple`, `setup`, `Leaf(rule, state)`,
   real-gradient `update!`, `nothing`-gradient skip, Adam `nothing` vs zero
   state, `trainable`/`isnumeric` and BatchNorm running stats.
3. Flux context: `Chain`, `Flux.setup(Adam, ...)`, `withgradient`, `train!`,
   and `Flux.gradient` returning `nothing` for an unused branch.
4. Distributed case: the real per-leaf `DistributedOptimizer.apply!`, an
   in-process rank-dependent `Branchy` gradient-presence table, the three
   outcomes (uniform / silent corruption / deadlock), a `RUN_MPI` toggle that
   either launches `scripts/checks/conditional_deadlock_check.jl` or prints the
   recorded traces, and the dropped `DistributedOptimizerState` solution.
5. Takeaways: scenario table and the required contract.

## Verification

- Structural check (`/tmp/opencode/check_notebook.jl`): header, UUID format,
  marker/`Cell order` equality, code-cell extraction.
  Result: `STRUCTURE_OK cells=25 code_cells=12`.
- Parse check (`/tmp/opencode/parse_cells.jl`): every cell body parses with
  `Meta.parseall`.
  Result: `PARSE_OK cells=25 markdown=13`.
- Execution check: extracted code cells (Markdown skipped) run under the thesis
  project.
  Command:
  `julia /tmp/opencode/check_notebook.jl notebook/conditional_graph_optimisers.jl /tmp/opencode/notebook_cells.jl`
  then
  `julia --project=. /tmp/opencode/notebook_cells.jl`
  Result: `EXIT=0`, no errors or warnings.
- Value check (`/tmp/opencode/verify_values.jl`): confirms the displayed values:
  `Leaf(Descent(0.1), nothing)`; `a = [0.95, 1.95, 2.95]`; `b` unchanged;
  Adam zero state `βt=(0.81, 0.998001)` vs nothing state `βt=(0.9, 0.999)`;
  BatchNorm trainable arrays `2` and running mean not trainable;
  deadlock `rank0=(trunk,a,b)` / `rank1=(trunk,c)`;
  corruption `rank0=(trunk,a)` / `rank1=(trunk,b)`;
  control all present.

`RUN_MPI` stayed `false` for notebook validation. The real 2-rank detector was
run by the orchestrator in the same session: 12/12 PASS, 1m20.0s, exit 0
(`artifacts/logs/conditional-deadlock/detector-2026-09-14.log`). See
`wiki/devlog/02-conditional-graph/2026-09-14-conditional-graph-onboarding.md`.

## Notes

- The notebook does not modify Flux or any other repo. Read-only references
  only.
- `reports/conditional-graph-support.typ` is referenced as the companion
  report; it now exists (18-page compiled PDF).
- Consistent with ADR-0007: the unused-parameter feature is deferred; the
  notebook documents the dropped whole-model design, not a shipped fix.
