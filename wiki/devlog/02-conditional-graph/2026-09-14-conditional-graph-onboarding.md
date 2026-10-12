# 2026-09-14 — Conditional-graph onboarding: report, detector, notebook

PR #2694 is pushed and its live CI is running. This session restarts the
conditional-graph (unused-parameter) work that was dropped from the PR, starting
with onboarding material so the author and the agent share the same mental model
of how Optimisers.jl and the current `DistributedOptimizer` interact.

No Flux source was modified. All work is in the thesis-control repo. The
companion notebook devlog is `2026-09-14-conditional-graph-optimisers-notebook.md`.

## 1. Typst report

`reports/conditional-graph-support.typ` (source, ~40 KB) compiled with
`typst compile reports/conditional-graph-support.typ` (Typst 0.15.1, exit 0) to
`reports/conditional-graph-support.pdf` (A4, 18 pages, 300,488 bytes).

Content: title page + outline; executive summary; the Optimisers.jl update
contract (`Leaf`, two-pass `_grads!`/`_update!`, `Zero` skip semantics); the
current `DistributedOptimizer.apply!`; a failure-scenario table; three
native-Typst figures (deadlock sequence, silent-corruption sequence,
presence-protocol flow + decision matrix); why `nothing -> zeros` breaks Adam
skip semantics; the M1.1 `DistributedOptimizerState` design; the compatibility
surface; evidence from the detector; open questions.

No external Typst packages (`cetz` is not cached and the package registry is not
reliable); all diagrams use builtins (`box`/`rect`/`line`/`place`/`grid`). The
report cites Optimisers.jl **0.4.7** (`~/.julia/packages/Optimisers/ZOK0f`, the
version pinned by this project's `Manifest.toml`): `subtract!` at
`interface.jl:104`, leaf `_grads!` at `108`, non-leaf `_grads!` at `119`. Later
releases (0.4.8/0.4.9 = `tMaaf`) shift these by +8; the report footnotes this.

## 2. Deadlock / corruption detector

Files:

- `scripts/checks/conditional_deadlock_probe.jl` — child process, runs under
  `mpiexec -n 2`. Three modes:
  - `deadlock`: rank 0 uses `(trunk, a, b)` (3 allreduces), rank 1 uses
    `(trunk, c)` (2 allreduces). Rank 0 blocks forever on its third collective.
  - `corruption`: rank 0 uses `(trunk, a)`, rank 1 uses `(trunk, b)`. Equal
    counts, so nothing hangs; the collective for `a` pairs with the collective
    for `b` and both get the wrong averaged gradient (silent).
  - `control`: every rank uses all four parameters; the update is correct.
  A `LoggingBackend <: Flux.AbstractFluxDistributedBackend` wraps
  `DistributedUtils.allreduce!` and prints `TRACE rank=<r> allreduce #<n>
  size=<s>` before each collective, which makes the missing collective
  diagnosable.
- `scripts/checks/conditional_deadlock_check.jl` — single-process driver. It
  launches the probe with `MPI.mpiexec()`, waits for both ranks to print
  `PROBE_READY` (so Julia startup / precompilation is not counted as a hang),
  then applies a `HANG_TIMEOUT` (default 20 s). It asserts the three modes'
  documented behavior and exits 0 when the current implementation is confirmed
  broken, 1 otherwise.
- `Makefile` target `verify-conditional-deadlock` (added to `.PHONY`).

### Bug found while writing the driver

The first driver run captured an empty trace. Cause: `pipeline(cmd,
stdout=log, stderr=log)` opens the same path twice; each handle starts at offset
0, so the large stderr stack trace overwrote the stdout markers. Fix: open one
`IO` handle and pass it as both `stdout` and `stderr`. Verified by the rerun.

### Result

Command:

```bash
make verify-conditional-deadlock
```

Result (2026-09-14): **12/12 PASS, exit 0, 1m20.0s**. Log:
`artifacts/logs/conditional-deadlock/detector-2026-09-14.log`.

Deadlock trace (rank 0 killed while blocked inside `Optimisers.update!`):

```
PROBE_READY rank=0
PROBE_READY rank=1
TRACE rank=0 allreduce #1 size=4
TRACE rank=1 allreduce #1 size=4
TRACE rank=0 allreduce #2 size=4
TRACE rank=1 allreduce #2 size=4
TRACE rank=0 allreduce #3 size=4
PROBE_UPDATE_DONE rank=1
[driver kills the job: rank 0 never finishes]
```

Corruption trace:

```
PROBE_VERDICT mode=corruption a_expected=1.1 a_actual=1.5 b_expected=1.4 b_actual=1.5 corrupted=true
```

Control trace:

```
PROBE_VERDICT mode=control correct=true a=1.2 b=1.8 c=1.4
```

The detector is a **characterization** test: it passes while the bug exists.
When the whole-model fix lands, `deadlock` mode will complete instead of
hanging, the expectations will no longer hold, and the detector must be updated
(then it becomes the RED test for the fix).

## 3. Pluto tutorial notebook

`notebook/conditional_graph_optimisers.jl` — valid Pluto `.jl` (`# v0.20.0`,
25 cells = 13 Markdown + 12 code, matching `Cell order`). Five sections:
Optimisers.jl alone (`setup`/`Leaf`, real-gradient update, skip-on-`nothing`,
Adam `nothing` vs zero, `trainable`/`isnumeric`); Optimisers inside Flux
(`setup`/`withgradient`/`train!`, `Flux.gradient` returning `nothing`); the
distributed case (real per-leaf `apply!`, rank-dependent presence table, the
three outcomes, a `RUN_MPI=false` toggle that either launches the detector or
prints the recorded traces, and the dropped `DistributedOptimizerState`
design); takeaways.

Validation (no MPI run from the notebook): structure check
`STRUCTURE_OK cells=25 code_cells=12`; parse check `PARSE_OK cells=25
markdown=13`; extracted code cells run under the thesis project with `EXIT=0`
and no warnings; value assertions on Leaf/Adam/BatchNorm/presence pass.

## Notes and limitations

- The detector targets the current per-leaf implementation in the thesis env's
  Flux (`ddp_flux` dirty master `b633cdc7`). It is not added to the Flux test
  suite and does not touch the pushed PR branch.
- The detector's `deadlock` mode relies on killing a hung MPI job after
  `HANG_TIMEOUT`; it is a bug-reproduction test, not a pass/fail regression test
  for CI.
- M1.1 correctness evidence remains CPU/MPI only; NCCL runtime is untested.
- The report's figures are native Typst, not sequence-diagram DSL output.

## Next action

Review the report and notebook with the user, then decide whether to start the
whole-model synchronization implementation as a new Flux branch/worktree (per
ADR-0007 phase 2), with the detector converted into a proper RED test.
