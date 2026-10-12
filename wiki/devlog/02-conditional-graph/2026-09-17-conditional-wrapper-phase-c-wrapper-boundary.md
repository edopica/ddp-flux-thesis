# 2026-09-17 - Conditional-graph wrapper Phase C: wrapper boundary

## Goal

Phase C of `temp/conditional_graph_wrapper_implementation_plan.md`: move gradient
communication out of per-leaf `Optimisers.apply!` into whole-model
`Optimisers.update`/`update!` methods on a `DistributedOptimizerState` wrapper,
then turn the Phase B RED corruption test GREEN.

## Change

`src/distributed/public_api.jl` (+~300 lines):

- New `DistributedOptimizerState{B,T}` wrapper (fields `backend`, `tree`) declared
  with `Functors.@functor DistributedOptimizerState (tree,)`.
- `Optimisers.setup(::DistributedOptimizer, model)` and an
  `EnzymeCore.Duplicated` variant return the wrapper.
- Specialized `Optimisers.update`/`update!` on the wrapper hold the communication
  boundary; higher-order gradients are rejected.
- Per-leaf `Optimisers.init` and `Optimisers.apply!` throw `ArgumentError` to
  reject composition (`DistributedOptimizer` cannot be nested in an
  `OptimiserChain`).
- `Optimisers.adjust`/`adjust!`/`freeze!`/`thaw!` delegate through the wrapper
  state and rule.
- `DistributedUtils.synchronize!!(backend, ::DistributedOptimizerState)` delegates
  to the inner tree.
- Minimal whole-model synchronization ported from the preserved M1.1 reference at
  `3370b910`: model-order walk via public `Optimisers.trainable`/`Optimisers.isnumeric`,
  structural gradient pairing, one presence-flag allreduce plus one allreduce per
  globally present buffer, and a rebuilt gradient tree with `ZeroTangent` for
  globally absent or tied-repeat occurrences. Public Optimisers APIs only.

`test/ext_distributed/optimizer_distributedtest.jl`: asserts
`st_dopt isa DistributedUtils.DistributedOptimizerState`, reads the inner state via
`st_dopt.tree.a.state`, and asserts the state returned by `Optimisers.update` is the
wrapper.

## GREEN evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration`; Julia 1.12.6 on
omarchy. `MPIEXEC = /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`.

| Check | Command | Result |
|---|---|---|
| Focused conditional test | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` | exit 0, 5 passed / 5 per rank, 28 s first run (one-time local `FluxMPIExt` precompile), 21 s warm rerun |
| Optimizer test | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 9 s |
| Full 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, `Distributed \| 6 6 58.0s`, 61 s wall |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 |
| Whitespace | `git diff --check` | clean (no output) |

The first focused run printed `Precompiling packages...` for `FluxMPIExt`; this is a
local source-change artifact, not the Bocconi cache hazard. The warm rerun is the clean
GREEN evidence.

## Known limitations deferred to Phase D

Carried deliberately from plan sections 9.2-9.4:

- the rebuilt gradient tree maps isbits occurrences by a global counter, so a model
  that mixes tied and isbits parameters can misorder them;
- the presence vector is allocated from a parameter buffer, so a device parameter
  puts host control flow on a GPU value;
- the old tuple-returning `Optimisers.trainable` behavior is not yet proven.

## Git state

- Flux HEAD is unchanged from Phase B: `8add0e42b7d9388ff365335152126e5b1e9de657`.
- Two modified, UNCOMMITTED files: `src/distributed/public_api.jl` (+~300 lines)
  and `test/ext_distributed/optimizer_distributedtest.jl` (wrapper state shape).
- Working-tree diff SHA-256 (`git diff | sha256sum`):
  `21706710c51dd2d63dc6593d6c2348079909257277f146199ad1d4f16ed6418b`.
- No Phase C commit was created.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseC-metadata.md`,
- `phaseC-GREEN-mpi2-2026-09-17.log`,
- `phaseC-GREEN-mpi2-warm-2026-09-17.log`,
- `phaseC-GREEN-optimizer-mpi2-2026-09-17.log`,
- `phaseC-suite-nprocs2-2026-09-17.log`,
- `phaseC-ambiguities-2026-09-17.log`,
- `phaseC-diff-check-2026-09-17.log`.

## Not done

Four-rank runs, the detector conversion, and the compatibility review are Phases F,
G, and H. `make verify-conditional-deadlock` was not rerun.

## Next action

Start Phase D (plan section 12): correct traversal and presence handling. Replace
the global isbits counter with an explicit occurrence plan, move presence metadata
to the CPU, and add focused tests for each requirement as it lands.
