# Phase B RED metadata

Date: 2026-09-16
Host: omarchy
Julia: 1.12.6

## Flux repository state

- Worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Branch: `ddp/integration-m1-spike`
- HEAD: `fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`
- Change under test: new untracked file `test/ext_distributed/conditional_distributedtest.jl`
- Production code unchanged.

## Thesis repository state

- Branch: `main`
- HEAD: `7e098319cc7ed902c1ffcd084842799c03b11181`

## RED run: focused conditional file, two ranks

- Command:

  ```bash
  FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
    -n 2 julia --startup-file=no --project=test \
    test/ext_distributed/conditional_distributedtest.jl
  ```

- Working directory: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Exit code: 1
- Elapsed: 24 s (no timeout; the failure is a fast numerical mismatch)
- Test totals: `0 passed, 2 failed, 0 errored, 0 broken` on each rank
- Log: `phaseB-RED-mpi2-2026-09-16.log`

## Failure reason (expected)

The model has two equal-shaped branches. Rank 0 trains `a`; rank 1 trains `b`.
The inactive branch has a `nothing` gradient.

Current per-leaf `DistributedOptimizer.apply!` runs a collective only for the
active branch on each rank. MPI pairs the two single-rank allreduces by call
order, so `a`'s gradient is averaged with `b`'s gradient.

Analytic values for `eta = 0.1`, `ta = zeros(4)`, `tb = fill(3.0, 4)`:

| Value | Expected (global mean) | Observed |
|---|---|---|
| `a` on rank 0 | `0.9` | `1.1` (cross-reduced with `b`) |
| `b` on rank 0 | `1.2` | `1.0` (branch skipped) |
| `a` on rank 1 | `0.9` | `1.0` (branch skipped) |
| `b` on rank 1 | `1.2` | `1.1` (cross-reduced with `a`) |

Every rank fails both assertions. The corruption is silent; there is no hang.

## Follow-up: strengthened preconditions

Commit `8add0e42` "Strengthen conditional-graph RED test preconditions" adds two
guards so a one-rank run or changed AD behavior cannot produce a false green:

- `@test tworkers >= 2`,
- `@test gs[1].a !== nothing` / `@test gs[1].b === nothing` on rank 0, and the
  mirror on every other rank.

Rerun (same command as above):

- Exit code: 1
- Elapsed: 24 s
- Test totals: `3 passed, 2 failed` per rank (the 3 preconditions pass; the same
  2 numerical assertions fail)
- Log: `phaseB-RED-mpi2-strengthened-2026-09-16.log`

## Pass condition check (plan section 12, Phase B)

| Condition | Result |
|---|---|
| New assertion fails for the expected numerical reason | pass |
| Failure is fast, not a timeout | pass (24 s) |
| Production code untouched | pass |
| Only the RED test is added | pass |
