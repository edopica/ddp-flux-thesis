# 2026-09-16 - Conditional-graph wrapper Phase A: integrated baseline

## Goal

Phase A of `temp/conditional_graph_wrapper_implementation_plan.md`: prove that the
implementation branch contains the exact PR #2694 head and no wrapper code, then
record the baseline test results.

## Branch move (plan deviation)

The plan states that `fee2c310` descends from `3370b910` and that
`git merge --ff-only ddp/windows-path-fix` moves the implementation branch.

The repository contradicts this. Both commits are siblings off `fa3bf228`:

```text
fa3bf228 -+- 04ddeaa0 - 3370b910   (ddp/integration-m1-spike, wrapper spike)
          +- 5c6ea561 - ... - fee2c310   (ddp/windows-path-fix = PR #2694 head)
```

`git merge-base --is-ancestor 3370b910 fee2c310` returned false, so a fast-forward
is impossible. The user approved a hard reset instead:

```bash
git -C /home/kurapica/Projects/ddp_flux/flux-integration reset --hard fee2c310
```

The reset moved only `ddp/integration-m1-spike`. It did not touch
`ddp/windows-path-fix` or `ddp/pr2694-salvage`. The old spike commits remain
preserved on `archive/pr2694-unused-gradient-spike` at `3370b910`.

## Baseline results

Working directory: `/home/kurapica/Projects/ddp_flux/flux-integration`
Flux HEAD: `fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`
Julia: 1.12.6. Host: omarchy.

| Check | Command | Result |
|---|---|---|
| Routing | `julia --startup-file=no --project=test test/distributed_routing.jl` | 17/17, exit 0, 2 s |
| 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 5/5, exit 0, 55 s |

The 2-rank suite covers the 5 child files: common, data, optimizer, reduce, and
synchronized. `conditional_distributedtest.jl` is absent after the reset, as
expected.

`DistributedOptimizerState` is absent from `src/` and `test/`. The worktree is
clean after both runs.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `baseline-routing-fee2c310-2026-09-16.log`,
- `baseline-mpi2-fee2c310-2026-09-16.log`,
- `baseline-metadata.md`.

## Next action

Start Phase B: add `test/ext_distributed/conditional_distributedtest.jl` with one
focused real-AD equal-shaped branch case. The current per-leaf implementation must
fail fast with wrong values, not hang. Commit only the RED test.
