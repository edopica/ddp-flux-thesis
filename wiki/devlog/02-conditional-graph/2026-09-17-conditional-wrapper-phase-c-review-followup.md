# 2026-09-17 - Conditional-graph wrapper Phase C review follow-up: direct-nesting guard

## Goal

Close the composition-guard gap found by review of the Phase C wrapper boundary.
Plan section 7.5 requires `DistributedOptimizer` to be the outermost rule, but
direct nesting of `DistributedOptimizer` inside `DistributedOptimizer` was
accepted at `Optimisers.setup` time.

## Defect

`Optimisers.setup(opt::DistributedOptimizer, model)` delegated directly to
`Optimisers.setup(opt.opt, model)`. For
`DistributedOptimizer(backend, DistributedOptimizer(backend, Adam()))`, the
delegated call dispatched back to the specialized method, bypassed the per-leaf
`init`/`apply!` composition guards, and built nested `DistributedOptimizerState`
wrappers. Updates would have synchronized twice, once per wrapper level.

## Fix

Both setup methods (plain model and `EnzymeCore.Duplicated`) now reject
`opt.opt isa DistributedOptimizer` with `ArgumentError(_COMPOSE_ERROR)` before
the inner setup runs. `_COMPOSE_ERROR` was moved above the setup methods and the
guarding comment was rewritten. No new methods were added, so no method
ambiguity is introduced. `DistributedOptimizer(backend, OptimiserChain(...))`
remains valid with an ordinary inner state tree; nesting inside an
`OptimiserChain` is still rejected by the per-leaf `init` guard.

## Commits

Three commits on `ddp/integration-m1-spike` in
`/home/kurapica/Projects/ddp_flux/flux-integration`; worktree clean at
`f5b87975`; not pushed.

| Commit | Message | Purpose |
|---|---|---|
| `109aad9c` | "Synchronize whole-model gradients before optimizer updates" | Phase C checkpoint (2 files: `src/distributed/public_api.jl`, `test/ext_distributed/optimizer_distributedtest.jl`; +311/-13). Development checkpoint, not final correctness approval. |
| `a18392dd` | "Add nested DistributedOptimizer composition regression (RED)" | Tests only (1 file, +33): the direct-nesting assertions fail before the fix. |
| `f5b87975` | "Reject directly nested DistributedOptimizer at setup" | Production fix only (1 file, +15/-10). |

## RED evidence

Tests added to `test/ext_distributed/optimizer_distributedtest.jl` in
`@testset "DistributedOptimizer composition guard"`, model
`(a=zeros(4), b=zeros(4))`. The testset asserts:

- direct nesting throws `ArgumentError` for the plain-model and
  `EnzymeCore.Duplicated` `Optimisers.setup` paths;
- nesting inside an `OptimiserChain` still throws;
- a valid outermost `OptimiserChain` setups successfully with `Leaf`/tuple inner
  state.

Command (run from `/home/kurapica/Projects/ddp_flux/flux-integration`;
`MPIEXEC = /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl
```

Result at `a18392dd`: exit 1, 13 s, composition testset `4 passed / 2 failed`
per rank. The two failures are exactly the direct-nesting assertions
(`Expected: ArgumentError`, `No exception thrown`), not fixture or dependency
errors. The chain-nesting rejection and the valid outermost-chain setup passed.
Log: `phaseC-review-guard-RED-mpi2-2026-09-17.log`.

## GREEN evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration`; Julia 1.12.6 on
omarchy. `MPIEXEC` as above.

| Check | Command | Result |
|---|---|---|
| Focused optimizer test (cold) | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 15 s, 6/6 per rank; one-time local `FluxMPIExt` precompile from the source change |
| Focused optimizer test (warm) | same | exit 0, 10 s, 6/6 per rank, no precompile |
| Focused conditional test | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` | exit 0, 19 s, 5/5 per rank |
| Full 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 56 s, `Distributed \| 6 6 54.4s`, all 6 children |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 |
| Whitespace | `git diff --check` | clean (no output) |

`using Flux` after the fix: exit 0, 14 s.

## Scope notes

- Only the composition guard changed. Traversal ordering, mixed tied/isbits
  occurrence handling, CPU presence metadata, and legacy tuple-`trainable`
  handling remain deferred to Phase D and were not touched.
- No remote, GPU/NCCL, or full main-suite runs were performed.
- Wiki and logs stay in the thesis repo; the three Flux commits contain only the
  files listed above.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseC-review-metadata.md`,
- `phaseC-review-guard-RED-mpi2-2026-09-17.log`,
- `phaseC-review-guard-GREEN-mpi2-2026-09-17.log`,
- `phaseC-review-guard-GREEN-mpi2-warm-2026-09-17.log`,
- `phaseC-review-conditional-mpi2-2026-09-17.log`,
- `phaseC-review-suite-nprocs2-2026-09-17.log`,
- `phaseC-review-ambiguities-2026-09-17.log`,
- `phaseC-review-diff-check-2026-09-17.log`.

## Next action

Revise the Phase D plan (`temp/conditional_graph_wrapper_implementation_plan.md`
section 12) before implementing it, then implement Phase D: correct traversal
and presence handling.
