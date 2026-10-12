# 2026-09-16 - Conditional-graph wrapper Phase B: RED corruption test

## Goal

Phase B of `temp/conditional_graph_wrapper_implementation_plan.md`: prove the
product error without a deliberate hang. Add one focused real-AD test that fails
fast on the current per-leaf implementation.

## Change

Added `test/ext_distributed/conditional_distributedtest.jl` (55 lines, one
testset). No production code changed.

The test model `TwoBranch` has two equal-shaped branches. Rank 0 trains `a`; every
other rank trains `b`. The inactive branch receives a `nothing` gradient. The test
computes the analytic global mean for both branches and asserts both updated
parameter values on every rank.

## Why the current code fails

`DistributedOptimizer.apply!` runs one allreduce per leaf, and Optimisers.jl calls
it only when the local gradient exists. Rank 0 issues one collective for `a`; rank
1 issues one collective for `b`. MPI pairs them by call order, so the two branches
mix. The reduced value is `avg(g_a, g_b)` instead of the per-branch global mean,
and each rank updates only its own branch.

## RED evidence

Command (run from `/home/kurapica/Projects/ddp_flux/flux-integration`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

Result: exit 1, 24 s, `0 passed, 2 failed` per rank. Fast numerical failure, no
timeout.

For `eta = 0.1`, `ta = zeros(4)`, `tb = fill(3.0, 4)`:

| Value | Expected | Observed |
|---|---|---|
| `a` on rank 0 | `0.9` | `1.1` |
| `b` on rank 0 | `1.2` | `1.0` |
| `a` on rank 1 | `0.9` | `1.0` |
| `b` on rank 1 | `1.2` | `1.1` |

## Commit

`6065ce8f` "Add conditional-graph distributed corruption regression (RED)", one
file, `+55`. Follow-up `8add0e42` "Strengthen conditional-graph RED test
preconditions" adds `@test tworkers >= 2` and `gs[1]` inactive-branch `nothing`
assertions (`+12`). Not pushed. Worktree clean after the commits.

The strengthened rerun still fails fast: exit 1, 24 s, `3 passed, 2 failed` per
rank. The three preconditions pass; the same two numerical assertions fail.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseB-RED-mpi2-2026-09-16.log`,
- `phaseB-RED-mpi2-strengthened-2026-09-16.log`,
- `phaseB-RED-metadata.md`.

## Next action

Start Phase C: move communication from per-leaf `apply!` to whole-model
`update`/`update!` methods on a `DistributedOptimizerState` wrapper. Run this
focused test again and store GREEN evidence.
