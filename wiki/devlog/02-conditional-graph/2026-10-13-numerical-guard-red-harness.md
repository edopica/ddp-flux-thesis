# DB-1/DB-4 numerical-guard RED harness and baseline run

Date: 2026-10-13

Built the RED-first success regression for the DB-1/DB-4 coordinated
non-finite-loss fix, as planned in
[`reports/conditional-graph-db1-db4-fix-plan.md`](../../../reports/conditional-graph-db1-db4-fix-plan.md)
sections 4 and 6. No Flux source change and no commit occurred.

## Files

- `scripts/checks/conditional_graph_numerical_guard_probe.jl` (child process, under mpiexec).
- `scripts/checks/conditional_graph_numerical_guard_check.jl` (single-process parent driver).

The historical scripts `conditional_graph_deferred_probe.jl` and
`conditional_graph_deferred_check.jl` remain unchanged.

## Harness summary

- `--group loss|gradient|all`; only `loss` is implemented. `gradient` and
  `all` print `GROUP_UNAVAILABLE` and exit 2, because DB-3 is a separate plan.
- Two- and four-rank case tables and both backends (`mpi`, `nccl`).
- Separate startup (420 s) and execution (120 s) watchdogs. A timeout always
  fails a case. Env overrides: `NUMERICAL_GUARD_STARTUP_TIMEOUT`,
  `NUMERICAL_GUARD_EXECUTION_TIMEOUT`, phase `NUMERICAL_GUARD_PHASE`.
- The parent validates each rank's collective trace by ordinal and metadata
  (`op`, eltype, length, buffer type category) against the scenario's explicit
  expected stage schedule. Children never write files.
- The optional `Duplicated` route runs in this environment. The UNTESTED path
  exists but is not exercised by this run.

## RED baseline run (unmodified Flux)

Exact command, from the thesis root:

```
julia --startup-file=no --project=../flux-integration/test scripts/checks/conditional_graph_numerical_guard_check.jl --group loss --ranks 2 --backend mpi --outdir artifacts/logs/conditional-wrapper/numerical-guards/red-local-loss-mpi2
```

Result: 13 cases, all FAIL, driver exit 1, no startup failure, no orphan
processes. Artifacts:
`artifacts/logs/conditional-wrapper/numerical-guards/red-local-loss-mpi2/`.

- Asymmetric cases (`train-one` fault rank 0 and 1, `train-one-inf`,
  `train-one-ninf`, `loop-one`, `loop-withgradient-one`, `aux-nan-first`,
  `recover-one`, `dup-train-one`) time out. Every raw log shows
  `GUARD_PROBE_READY` on both ranks, the fault rank's `LOSS_INVALID` plus
  `REJECTED caught=DomainError` (or `RETURNED_AT_FAULT` in the manual loops),
  and the finite rank's unmatched `COLLECTIVE_BEGIN`.
- `no-trainable`: the finite rank completes all five steps without catching.
  This is wrong participation.
- `train-all`, `healthy`, `aux-nonfinite-ok`: child-local checks pass, but the
  traces contain no len-1 Float32 loss vote; the parent fails the missing
  VOTE stages.
- Flux source md5s match `baseline-metadata.md`
  (`src/train.jl eea4e545d3eb2118c3a5b23653c7a71a`,
  `public_api.jl 9bfa981384eaf88750ec3d19e8bb76bd`), HEAD
  `55171e659b2f425da980cbc83177bf5c38f0a7a1`.

## Deviations and notes

- Barriers use the MPI communicator (for NCCL, `BACKEND.mpi_backend.comm`),
  because `BACKEND.comm` is an NCCL communicator on that backend.
- The driver's `metadata.md` records child argv only. Printing the Julia `Cmd`
  embeds the full parent environment, including secrets.
- The Duplicated availability probe and its compilation run in warmup before
  `GUARD_PROBE_READY`, so the execution watchdog covers only measured work.

## Exact next action

Implement the DB-1/DB-4 hook and loss vote in `../flux-integration`, then rerun
this harness for the GREEN gate. Keep the RED artifacts.
