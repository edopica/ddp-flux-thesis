# Phase A baseline metadata

Date: 2026-09-16
Host: omarchy
Julia: 1.12.6

## Flux repository state

- Worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Branch: `ddp/integration-m1-spike`
- HEAD: `fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`
- PR #2694 branch `ddp/windows-path-fix`: `fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`
- Archive branch `archive/pr2694-unused-gradient-spike`: `3370b910f6e83466baafd8566b0a8ed8296b2d81`
- Worktree status after all runs: clean

## Thesis repository state

- Branch: `main`
- HEAD: `7e098319cc7ed902c1ffcd084842799c03b11181`
- Working tree: pre-existing unrelated modifications and untracked files (not touched by Phase A)

## Branch move note (deviation from plan section 11.3)

The plan states that `fee2c310` descends from `3370b910` and that
`git merge --ff-only ddp/windows-path-fix` moves the implementation branch.

This is not true in the repository. Both commits are siblings off `fa3bf228`:

```text
fa3bf228 -+- 04ddeaa0 - 3370b910   (ddp/integration-m1-spike, wrapper spike)
          +- 5c6ea561 - ... - fee2c310   (ddp/windows-path-fix = PR #2694 head)
```

A fast-forward is therefore impossible. The user approved a hard reset.
Phase A used:

```bash
git -C /home/kurapica/Projects/ddp_flux/flux-integration reset --hard fee2c310
```

The reset only moved `ddp/integration-m1-spike`. It did not touch
`ddp/windows-path-fix` or `ddp/pr2694-salvage`. The old spike commits remain
preserved on `archive/pr2694-unused-gradient-spike`.

## Baseline run 1: standalone routing test

- Command: `julia --startup-file=no --project=test test/distributed_routing.jl`
- Working directory: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Exit code: 0
- Elapsed: 2 s
- Result: `distributed test routing | 17 17`
- Log: `baseline-routing-fee2c310-2026-09-16.log`

## Baseline run 2: two-rank distributed suite

- Command: `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi`
- Working directory: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Exit code: 0
- Elapsed: 55 s
- Result: `Distributed | 5 5` (5 child files: common, data, optimizer, reduce, synchronized)
- Log: `baseline-mpi2-fee2c310-2026-09-16.log`

## Pass condition check (plan section 12, Phase A)

| Condition | Result |
|---|---|
| HEAD equals `fee2c310` | pass |
| `DistributedOptimizerState` absent | pass |
| Routing test passes 17/17 | pass |
| Current distributed suite passes 5/5 | pass |
| Worktree remains clean | pass |
