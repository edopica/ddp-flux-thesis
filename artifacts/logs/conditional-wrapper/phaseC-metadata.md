# Phase C GREEN metadata

Date: 2026-09-17
Host: omarchy
Julia: 1.12.6

## Flux repository state

- Worktree: `/home/kurapica/Projects/ddp_flux/flux-integration`
- Branch: `ddp/integration-m1-spike`
- HEAD: `8add0e42b7d9388ff365335152126e5b1e9de657` (unchanged from Phase B)
- Working tree: uncommitted modifications to
  - `src/distributed/public_api.jl` (wrapper boundary + synchronization)
  - `test/ext_distributed/optimizer_distributedtest.jl` (wrapper state shape)
- Working-tree diff SHA-256 (`git diff | sha256sum`):
  `21706710c51dd2d63dc6593d6c2348079909257277f146199ad1d4f16ed6418b`
- No commit was created for Phase C.

## Thesis repository state

- Branch `main`, HEAD `7e098319cc7ed902c1ffcd084842799c03b11181`
- Artifacts and wiki changes are working-tree changes (not committed).

## Change under test (plan Phase C)

Moved gradient communication out of the per-leaf
`Optimisers.apply!(::DistributedOptimizer, ...)` into whole-model
`Optimisers.update`/`update!` methods on a new `DistributedOptimizerState`
wrapper:

- `DistributedOptimizerState` + `Functors.@functor DistributedOptimizerState (tree,)`
- `Optimisers.setup(::DistributedOptimizer, model)` (and `EnzymeCore.Duplicated`)
- specialized `update`/`update!` (higher-order gradients rejected)
- composition errors in per-leaf `init`/`apply!` (`ArgumentError`)
- state delegation for `Optimisers.adjust`/`adjust!`/`freeze!`/`thaw!`
- `DistributedUtils.synchronize!!(::..., ::DistributedOptimizerState)`
- minimal whole-model sync: presence allreduce + per-buffer allreduce,
  rebuilt gradient tree (ported from the preserved `3370b910` reference)

`optimizer_distributedtest.jl` now asserts the wrapper type and reads the
inner state via `st_dopt.tree`.

## Commands and results

All commands run from `/home/kurapica/Projects/ddp_flux/flux-integration`.
`MPIEXEC = /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`

| # | Command | Exit | Elapsed | Result | Log |
|---|---|---|---|---|---|
| 1 | parse check (`Meta.parseall`) of the two edited files | 0 | <1 s | `PARSE_OK`, `TEST_PARSE_OK` | - |
| 2 | `julia --startup-file=no --project=test -e 'using Flux; ... DistributedOptimizerState'` | 0 | 17.2 s | `LOAD_OK`; precompiled `Flux` (source change) | - |
| 3 | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` | 0 | 28 s | `5 passed / 5 total` per rank (GREEN) | `phaseC-GREEN-mpi2-2026-09-17.log` |
| 4 | same as #3 (warm, no precompilation) | 0 | 21 s | `5 passed / 5 total` per rank | `phaseC-GREEN-mpi2-warm-2026-09-17.log` |
| 5 | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 9 s | no testset output (all `@test`s pass) | `phaseC-GREEN-optimizer-mpi2-2026-09-17.log` |
| 6 | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 61 s | `Distributed \| 6 6 58.0s` (6 child files) | `phaseC-suite-nprocs2-2026-09-17.log` |
| 7 | `julia --startup-file=no --project=test -e 'using Flux, Test; length(Test.detect_ambiguities(Flux.DistributedUtils; recursive=false))'` | 0 | ~15 s | `AMBIGUITIES=0` | `phaseC-ambiguities-2026-09-17.log` |
| 8 | `git diff --check` | 0 | <1 s | no output (clean) | `phaseC-diff-check-2026-09-17.log` |

## Notes

- Run #3 triggered one-time local precompilation of `FluxMPIExt` inside the
  test (`Precompiling packages...`). This is a local source-change artifact,
  not the HPC cache hazard; run #4 is the clean warm evidence.
- Four-rank runs are deliberately deferred to Phase F.
- `make verify-conditional-deadlock` is not rerun in Phase C; the detector
  conversion is Phase G.
- Known Phase C limitations carried into Phase D: rebuild maps isbits
  occurrences by a global counter (mixed tied+isbits order defect, plan
  section 9.2), presence vector is allocated from a parameter buffer (GPU
  host-control defect, plan section 9.3), and old tuple-returning `trainable`
  behavior is not yet proven (plan section 9.4).
