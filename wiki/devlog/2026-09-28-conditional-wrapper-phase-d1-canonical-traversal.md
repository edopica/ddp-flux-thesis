# 2026-09-28 - Conditional-graph wrapper Phase D1: canonical traversal order

## Goal

Phase D1 of `temp/conditional_graph_wrapper_implementation_plan.md`: make both
gradient-synchronization passes visit trainable occurrences in the same canonical
model order, so a named `trainable` selection in reverse field order cannot reorder
collectives or scramble the occurrence-to-buffer mapping.

## Defect

`_walk!` (collection pass) iterated `pairs(Optimisers.trainable(x))`, while
`_build!` (rebuild pass) traversed the model's functor children and filtered them
with `_trainable_child`. `Optimisers.setup` normalizes a named `trainable` selection
into model field order, so the leaves live in model order; the collection pass did
not. For a model with fields `(a, b)` and `trainable(m) = (b = m.b, a = m.a)`:

- buffers were allocated in selection order `b, a`;
- the collectives therefore ran in reverse model order;
- `_build!` consumed isbits buffers by position in functor order, so `a` received
  `b`'s global-mean gradient and vice versa — silently, with equal-shaped isbits
  leaves and no tied parameters.

## Fix

`src/distributed/public_api.jl` (uncommitted):

- `_walk!` now iterates `functor(x)[1]` through a new `_foreach_child` and selects
  children with `_trainable_child(Optimisers.trainable(x), k)` — the same normalized
  selection `_build!` uses.
- `_foreach_child` has methods for `NamedTuple`, `Tuple`, `AbstractArray`, and
  `AbstractDict`, mirroring `_buildmap` traversal order (comment ties the two
  traversals together).
- Comments on `_walk!` document the canonical-order contract.
- No other production behavior changed. Legacy tuple `trainable` still fails with
  the same `MethodError: haskey(::Tuple, ::Symbol)` (verified at baseline); that
  remains Phase D4.

`test/ext_distributed/conditional_distributedtest.jl` (uncommitted): new testset
`canonical traversal order for reversed named trainable`:

- `IsoVec{N}` — an isbits `AbstractArray` leaf (SVector-like, avoids a StaticArrays
  dependency); `Optimisers.isnumeric` accepts it and isbits leaves are never
  identity-cached.
- `ReversedTrainable(a, b)` with `Optimisers.trainable(m) = (b = m.b, a = m.a)`.
- `RecordingBackend` wrapping the MPI backend, recording `(size, eltype, values)` of
  every pre-allreduce buffer so equal-shaped buffers are distinguished by value, not
  shape.
- Local gradients differ per field and per rank (`ga = rank + 1`, `gb = rank + 5`).
- Wrapper update is compared against an independent native Optimisers reference:
  explicit `allreduce!(backend, ..., avg)` of gradient copies, then plain
  `Optimisers.update`.
- Assertions: reference values, one presence reduction (`Float32`, size 2) followed
  by gradient reductions in model order (`ga` then `gb`), and equal collective
  counts across ranks (`bcast!` of the count from rank 0).

Diff: 2 files, +111/−1 (`src/distributed/public_api.jl` +20/−1,
`test/ext_distributed/conditional_distributedtest.jl` +92). Uncommitted; Flux HEAD
remains `f5b87975`.

## RED evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration` with only the test
added (`MPIEXEC = /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl
```

Result: exit 1, 25 s. Per rank: control testset 5/5; D1 testset 6 passed / 4 failed.
Rank 0 values: `model2.a = [0.45, ...]` vs reference `[0.85, ...]` and
`model2.b = [0.85, ...]` vs reference `[0.45, ...]` — the global means were swapped.
The recording backend showed `rec.log[2]` = local `gb` and `rec.log[3]` = local `ga`
(reverse model order). The failures are the intended numerical and protocol defect,
not fixture errors.
Log: `phaseD-d1-RED-mpi2-2026-09-28.log`.

## GREEN evidence

Environment: Julia 1.12.6, Optimisers 0.4.9
(`/home/kurapica/.julia/packages/Optimisers/tMaaf`), host omarchy.

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks (cold) | same as RED | exit 0, 37 s, control 5/5 + D1 10/10 per rank; one Flux + FluxMPIExt recompile from the source change |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 11 s, 6/6 per rank |
| Full 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 59 s, `Distributed \| 6 6 55.5s`, all 6 children |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 |
| Whitespace | `git diff --check` | clean (no output) |

## Scope notes

- D1 only. The mixed tied/isbits counter defect (D2), CPU presence metadata (D3),
  legacy tuple `trainable` (D4), and the D5 regression gate remain open.
- Four-rank execution remains Phase F; no GPU/NCCL, detector, Bocconi, or main-suite
  runs.
- No commit. Tests and fix stay reviewable as separate changes in the worktree;
  commit/push only with explicit user authorization.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseD-d1-metadata.md`,
- `phaseD-d1-RED-mpi2-2026-09-28.log`,
- `phaseD-d1-GREEN-mpi2-2026-09-28.log`,
- `phaseD-d1-GREEN-optimizer-mpi2-2026-09-28.log`,
- `phaseD-d1-GREEN-suite-nprocs2-2026-09-28.log`,
- `phaseD-d1-ambiguities-2026-09-28.log`,
- `phaseD-d1-diff-check-2026-09-28.log`.

## Next action

Implement Phase D2: explicit occurrence mapping for the mixed tied/isbits counter
defect. Add the mixed model (mutable `p`, the same `p`, an isbits parameter,
distinct mutable `q`), record RED before production changes, then replace inferred
buffer indices with explicit occurrence-plan entries.

## Independent D1 review

Result: approved for the D1 scope, with no blocking code findings.

The review compared the uncommitted diff with plan section 12 and Optimisers 0.4.9 source.
Both synchronization passes now use the same child order and selection helper.
The regression distinguishes equal-shaped buffers by value and compares updates against native Optimisers.
The stored RED log contains four intended failures per rank, without fixture errors.
The stored full-suite GREEN log reports all six children passed at two ranks.

Independent commands ran from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl
git diff --check && julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers ", pkgversion(Optimisers), " ", pathof(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); @assert isempty(a)'
```

Both commands exited 0. The focused test passed control 5/5 and D1 10/10 on each rank.
The testset times were 13.3–13.4 seconds and 1.4 seconds, respectively.
The static command reported Optimisers 0.4.9 and zero ambiguities. The whitespace check passed.
The review did not repeat the full suite or the historical RED run.

Non-blocking documentation finding: plan section 22 still requests the completed D1 regression.
The quickstart header also describes the older clean Phase C checkpoint.
The plan header and current context correctly identify D2 as the next action.

D2–D5 remain open. The review changed only the context and this devlog.
