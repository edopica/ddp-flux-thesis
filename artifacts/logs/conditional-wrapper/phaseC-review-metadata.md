# Phase C review follow-up metadata: direct-nesting composition guard

Date: 2026-09-17
Host: omarchy
Julia: 1.12.6
Optimisers: 0.4.9

## Flux repository state

- Worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Branch: `ddp/integration-m1-spike`, HEAD `f5b87975`, worktree clean, not pushed
- Commits created in this follow-up:
  1. `109aad9c` `Synchronize whole-model gradients before optimizer updates` —
     Phase C checkpoint (2 files, +311/-13); not final correctness approval
  2. `a18392dd` `Add nested DistributedOptimizer composition regression (RED)` —
     tests only (1 file, +33)
  3. `f5b87975` `Reject directly nested DistributedOptimizer at setup` —
     production fix only (1 file, +15/-10)

## Defect

`Optimisers.setup(opt::DistributedOptimizer, model)` delegated straight to
`Optimisers.setup(opt.opt, model)`. A directly nested
`DistributedOptimizer(backend, DistributedOptimizer(backend, Adam()))`
therefore dispatched back to the specialized method, bypassing the per-leaf
`init`/`apply!` composition guards, and built nested `DistributedOptimizerState`
wrappers. Updates would have synchronized twice. Plan section 7.5 requires
`DistributedOptimizer` to be the outermost rule.

## Fix (`f5b87975`)

Both setup methods (plain model and `EnzymeCore.Duplicated`) reject
`opt.opt isa DistributedOptimizer` with `ArgumentError(_COMPOSE_ERROR)` before
the inner setup runs. `_COMPOSE_ERROR` moved above the setup methods; the
guarding comment was rewritten. No new methods were added (no ambiguity risk);
`DistributedOptimizer(backend, OptimiserChain(...))` remains valid; nesting
inside an `OptimiserChain` is still rejected by the per-leaf `init` guard.

## RED evidence (before the fix, commit `a18392dd`)

Command (from the flux-integration worktree; `MPIEXEC =
/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`):

```
FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl
```

- EXIT 1, 13 s
- `DistributedOptimizer composition guard`: 4 passed, 2 failed per rank
- The 2 failures are exactly the direct-nesting assertions (plain model and
  `EnzymeCore.Duplicated`): `Expected: ArgumentError`, `No exception thrown`,
  i.e. setup succeeded instead of rejecting the composition. No other errors
  or fixture/dependency failures.
- The chain-nesting rejection and the valid outermost-chain setup passed.
- Log: `phaseC-review-guard-RED-mpi2-2026-09-17.log`

## GREEN evidence (after the fix, commit `f5b87975`)

Commands run from `/home/kurapica/Projects/ddp_flux/flux-integration`.

| # | Command | Exit | Elapsed | Result | Log |
|---|---|---|---|---|---|
| 1 | focused `optimizer_distributedtest.jl`, 2 ranks (cold) | 0 | 15 s | composition testset 6 passed/6 per rank; one-time local `FluxMPIExt` precompile from the source change | `phaseC-review-guard-GREEN-mpi2-2026-09-17.log` |
| 2 | focused `optimizer_distributedtest.jl`, 2 ranks (warm) | 0 | 10 s | 6 passed/6 per rank, no precompile | `phaseC-review-guard-GREEN-mpi2-warm-2026-09-17.log` |
| 3 | focused `conditional_distributedtest.jl`, 2 ranks | 0 | 19 s | 5 passed/5 per rank | `phaseC-review-conditional-mpi2-2026-09-17.log` |
| 4 | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 56 s | `Distributed \| 6 6 54.4s`, all 6 children | `phaseC-review-suite-nprocs2-2026-09-17.log` |
| 5 | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 | 3 s | `AMBIGUITIES=0` | `phaseC-review-ambiguities-2026-09-17.log` |
| 6 | `git diff --check` | 0 | <1 s | clean (no output) | `phaseC-review-diff-check-2026-09-17.log` |

`using Flux` precompiled once after the fix: exit 0, 14 s (`Flux`, 154 already
precompiled).

## Scope notes

- Only the composition guard was changed. Traversal ordering, mixed tied/isbits
  occurrence handling, CPU presence metadata, and legacy tuple-`trainable`
  handling remain explicitly deferred to Phase D and were not touched.
- No remote, GPU/NCCL, or full main-suite runs were performed.
- Wiki and log changes live in the thesis repository; the three Flux commits
  contain only the files listed above.
