# Phase D1 evidence metadata — canonical traversal order

Date: 2026-09-28

## Environment

| Item | Value |
|---|---|
| Host | omarchy (x86_64, local CPU/MPI) |
| Julia | 1.12.6 |
| Optimisers | 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`) |
| Flux worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Flux branch | `ddp/integration-m1-spike` |
| Flux HEAD | `f5b87975a4c62ce254736625b0a80015b6a1adb6` (unchanged; D1 changes uncommitted) |
| MPI launcher | `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec` |

## Working-tree state per run

| Run | `src/distributed/public_api.jl` | `test/ext_distributed/conditional_distributedtest.jl` |
|---|---|---|
| RED | unchanged (`f5b87975`) | modified (D1 testset added) |
| GREEN | modified (traversal normalization) | modified (D1 testset added) |

## RED (before production changes)

Command (from `flux-integration`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

- Exit code: 1
- Wall: 25 s
- Per rank: control testset `equal-shaped conditional branches do not cross-reduce` 5/5;
  new testset `canonical traversal order for reversed named trainable` 6 passed / 4 failed / 10.
- Intended defect (values, rank 0 from the log): reversed named `trainable`
  `(b=..., a=...)` made `_walk!` collect buffers in selection order while `_build!`
  rebuilds in functor order, so isbits parameters received each other's gradients:
  `model2.a = [0.45, ...]` vs reference `[0.85, ...]`, `model2.b = [0.85, ...]` vs
  reference `[0.45, ...]`. The recording backend also showed the gradient allreduces in
  reverse model order (`rec.log[2]` = local `gb`, `rec.log[3]` = local `ga`).
- Log: `phaseD-d1-RED-mpi2-2026-09-28.log`

## Production fix

`src/distributed/public_api.jl` (uncommitted):

- `_walk!` now iterates the model's functor children through `_foreach_child` and
  filters them with `_trainable_child`, the same normalized selection `_build!` uses.
- Added `_foreach_child` methods mirroring `_buildmap` for `NamedTuple`, `Tuple`,
  `AbstractArray`, `AbstractDict`.
- Comments document the canonical-order contract for both passes.

`test/ext_distributed/conditional_distributedtest.jl` (uncommitted): D1 regression with
`IsoVec` (isbits `AbstractArray`), `ReversedTrainable` (`trainable` in reverse field
order), a recording backend, native-Optimisers global-mean reference, value/order/count
assertions.

Diff: 2 files, +111/−1 (src +20/−1, test +92).

## GREEN (after the fix)

| Run | Command | Exit | Wall | Result (per rank / total) |
|---|---|---|---|---|
| Focused conditional, 2 ranks | same as RED | 0 | 37 s (cold, Flux + FluxMPIExt recompiled) | control 5/5; D1 10/10 |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 11 s | 6/6 |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 59 s | `Distributed | 6 6 55.5s` (6 child files) |
| Ambiguities | `julia --startup-file=no --project=test -e 'using Flux, Test; Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)'` | 0 | — | 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean |

Logs:

- `phaseD-d1-GREEN-mpi2-2026-09-28.log`
- `phaseD-d1-GREEN-optimizer-mpi2-2026-09-28.log`
- `phaseD-d1-GREEN-suite-nprocs2-2026-09-28.log`
- `phaseD-d1-ambiguities-2026-09-28.log`
- `phaseD-d1-diff-check-2026-09-28.log`

## Scope notes

- Four-rank execution remains Phase F. No GPU/NCCL, no four-rank run, no detector update
  (Phase G), no Phase E test battery, no commit.
- The D1 fix does not address the mixed tied/isbits counter defect (D2) or legacy tuple
  `trainable` selection (D4). At baseline `f5b87975` a legacy tuple `trainable` already
  fails with `MethodError: haskey(::Tuple, ::Symbol)` (verified 2026-09-28); D1 preserves
  that failure mode (the same `_trainable_child` fallback is now reached in `_walk!`).
- The recording backend records pre-allreduce buffer values; presence metadata is the
  host `Vector{Float32}` already used by Phase C for all-isbits plans.
