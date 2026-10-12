# 2026-09-30 - Conditional-graph wrapper Phase E: restore the M1.1 battery

## Goal

Phase E of `temp/conditional_graph_wrapper_implementation_plan.md` (section 12):
restore the proven M1.1 testsets from
`3370b910:test/ext_distributed/conditional_distributedtest.jl` without restoring
the known M1.1 defects; keep the Phase D regressions; add the two Phase E items
(partial freeze on all ranks with collective-count assertions, wrapper traversal
through device adaptation if the public API supports it); run the conditional
file after each group and finish with the complete dedicated two-rank suite.

## Policy for porting

Phase D had already replaced five M1.1 testsets with stronger testsets, so those
were not repeated:

- BatchNorm `μ`/`σ²` exclusion (D5, real AD, 2 steps),
- transpose/adjoint parent space (D5, real AD, bounded buffers),
- caller-gradient non-mutation (D5, rank-distinct gradients + read-only case),
- `update!` immutable model returns (D5),
- globally absent Adam state (D5, active-then-absent snapshots).

Everything else was ported, reusing existing fixtures (`TwoBranch`,
`RecordingBackend`, `_avg_sum`) where the M1.1 source and Phase D agreed, and
adapting where Phase D had changed the contract (e.g. wrapper state accessed as
`state.tree`, `Flux.setup` returning the wrapper).

## Added testsets (13, +83 assertions per rank)

| Testset | Source | Notes |
|---|---|---|
| `real AD conditional branches average over multiple steps` | M1.1 #1 | 4-step rank-gated forward, analytic global mean integrated alongside updates, cross-rank equality |
| `Adam matches an independent global-mean reference` | M1.1 #2 | 5 steps; parameters, both moments, and decayed `beta` compared with an in-process Adam on the analytic global mean |
| `tied parameters through real AD` | M1.1 #6 | Zygote returns distinct per-occurrence gradients; both `update` and `update!` match a native reference; `m.a === m.b` retained |
| `tied parameter through a transpose with real AD` | M1.1 #7 | parent-space buffer accumulates the direct and transposed occurrence; native reference; `m.a === m.b.parent` |
| `isbits parameters (FillArrays) sync independently` | M1.1 #8 | analytic `Fill` global mean plus native reference |
| `higher-order gradients rejected` | M1.1 #12 | both update paths throw `ArgumentError` |
| `collective count protocol with real AD` | M1.1 #15 | 1 presence + 2 gradient reductions per step; globally absent params excluded |
| `Flux.train! with DistributedOptimizerState` | M1.1 #17 | `Flux.setup` returns the wrapper; full `Flux.train!` loop |
| `nested model (Chain of custom layer and Dense)` | M1.1 #17 | `ScaleLayer` + `Dense`, `Flux.train!`, cross-rank parameter equality |
| `synchronize!! on DistributedOptimizerState` | M1.1 #17 | root state reaches the inner tree |
| `adjust / adjust! on wrapped state and rule` | M1.1 #17 | backend retained, adjusted rule used by the next update |
| `partial freeze on all ranks keeps the collective protocol` | Phase E | freeze/thaw via the inner tree; collectives unchanged (3 per step); frozen value held, live value updated, thaw resumes; same count on every rank |
| `Functor device adaptation keeps the backend` | Phase E | `fmap` over the wrapper adapts only `tree` and keeps the backend; adapted model still reduces and updates; `AdaptedBackend` delegates adapted buffers to the real backend on raw storage |

New fixtures: `ScaleLayer` (M1.1), `AdaptedArray` and `AdaptedBackend` (Phase E).
No mock or simulated result is used.

## Iterations (test defects, not production defects)

Five RED iterations were needed, all in the new testsets; no production source
was touched:

1. partial freeze: compared against the in-place-updated original model
   instead of a snapshot;
2. partial freeze: step 2 restarted from the original model instead of `m1`,
   so expectations were off by one update;
3. device adaptation: `_avg_sum` is length-4 only;
4. device adaptation: `AdaptedArray` is not an MPI datatype (`strides`
   missing) -> `AdaptedBackend` reduces the raw storage;
5. device adaptation: the inner update returns a plain `Vector` for this
   parameter, so the container-type assertion was relaxed to the
   `AbstractArray`/value contract.

The M1.1 ported testsets passed as written; the D1-D5 testsets were unaffected.

## GREEN evidence

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` | exit 0, 1 m 13.5 s: 315/315 per rank across 28 testsets |
| Focused conditional totals | same wrapped in one outer `@testset` | `conditional file \| 315 315 \| 1m19.4s`, exit 0 |
| Focused optimizer, 2 ranks | same with `optimizer_distributedtest.jl` | exit 0, 10.9 s: composition guard 6/6 per rank |
| Complete dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 1 m 55.8 s: `Distributed \| 6 6 1m52.7s` |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 (Optimisers 0.4.9) |
| Whitespace | `git diff --check` | clean |

Per-rank totals before Phase E: 232. Added: 83. Final: 315. New testsets:
11 + 9 + 9 + 5 + 5 + 2 + 4 + 3 + 2 + 3 + 11 + 11 + 8 = 83.

Tied real-AD gradients: both ranks return `gs.a = [0.0, 2.0]` and
`gs.b = [-2.0, 0.0]` as distinct objects; the ported test proves the wrapper's
local tied sum matches native Optimisers. Rank-distinct tied averaging remains
covered by the D2 testset.

## Scope notes

- Test-only change: `test/ext_distributed/conditional_distributedtest.jl`
  +411/-0. No production source edited. No commit.
- Two ranks only. Phase F (four-rank weighting), Phase G (detector conversion),
  and Phase H (compatibility review) remain.
- No four-rank run, no GPU/NCCL run, no detector update, no Bocconi run, no
  main-suite run.
- Exact commands, package versions, exit codes, elapsed times, and per-testset
  counts: `artifacts/logs/conditional-wrapper/phaseE-metadata.md`.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseE-metadata.md`,
- `phaseE-GREEN-mpi2-2026-09-30.log` (per-testset; attempts 1-5 kept),
- `phaseE-TOTALS-mpi2-2026-09-30.log`,
- `phaseE-GREEN-optimizer-mpi2-2026-09-30.log`,
- `phaseE-GREEN-suite-nprocs2-2026-09-30.log`,
- `phaseE-ambiguities-2026-09-30.log`,
- `phaseE-diff-check-2026-09-30.log`.

## Next action

Phase F: run the CPU/MPI gates at two and four ranks. The four-rank analytical
case must prove weighting (one rank branch A, three ranks branch B). Then
Phase G (convert the external detector) and Phase H (compatibility review).
Commit/push only with explicit user authorization.

## Independent review (2026-09-30)

Result: **Phase E reopened for two P2 coverage gaps**. No production defect was established. Flux source and tests remain unchanged by this review.

### P2: Device adaptation does not exercise state-array traversal

Location: `test/ext_distributed/conditional_distributedtest.jl:1417-1440`.

Descent stores `nothing` in its leaf state. An independent probe counted zero array transformations through `fmap` on this wrapper. The equality assertion therefore does not prove state-array adaptation. The update then uses a fresh `adstate`, rather than the `adapted` state from `fmap`.

Use Adam or Momentum with nonzero state arrays. Assert their adapted types and preserved values, backend identity, and unchanged source state. Run the update with the actual adapted state and compare values and optimizer state against a native reference.

### P2: The M1.1 port omits API checks without equivalents

The archive contains `Chain tuple-grad diagnostic` and `Enzyme Duplicated model setup and update`. Neither testset has an equivalent in the current distributed tests. The composition guard checks only rejected nested-rule setup for Duplicated models. It does not cover supported setup or rejected updates.

The archive also calls `freeze!(state)` and `thaw!(state)`. The new partial-freeze test calls only the inner leaf APIs (`:1362`, `:1369`). Thus, it bypasses the wrapper delegation methods. Restore these checks before declaring the battery complete. Also account for the archived shared-gradient-object tied-parameter subcase in the port inventory.

The partial-freeze test uses Descent and discards each returned state. A stateful version must retain returned states and assert frozen moment preservation and correct thaw behavior.

### Evidence

All commands ran from `/home/kurapica/Projects/ddp_flux/flux-integration`.

```bash
git show 3370b910:test/ext_distributed/conditional_distributedtest.jl
git status --short --branch && git diff --stat && git diff --check
time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test -e 'using Test; @testset "Phase E independent review" begin include("test/ext_distributed/conditional_distributedtest.jl") end'
julia --startup-file=no --project=test -e 'using Flux, Optimisers, Functors, Test; struct ReviewBackend <: Flux.AbstractFluxDistributedBackend end; model=(w=ones(2),); state=Optimisers.setup(Flux.DistributedUtils.DistributedOptimizer(ReviewBackend(), Descent(0.1)), model); n=Ref(0); adapted=Functors.fmap(x -> x isa AbstractArray ? (n[] += 1; copy(x)) : x, state); @test n[] == 0; @test state.tree == Optimisers.setup(Descent(0.1), model); println("array adaptation calls = ", n[], "; Descent leaf state = ", state.tree.w.state); a=Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); @test isempty(a); println("ambiguities = ", length(a))'
```

- All commands exited 0. The Flux diff contains one test file, +411/-0, with clean whitespace.
- Independent focused run: 315/315 per rank, 70.9 seconds in each test summary, 71.487 seconds wall time.
- Probe output: `array adaptation calls = 0; Descent leaf state = nothing` and `ambiguities = 0`.
- The stored complete-suite log reports `Distributed | 6 6 1m52.7s`. This review inspected that evidence but did not repeat the full suite.
- The review did not run four ranks or native GPU tests.

Next action: correct the two Phase E coverage gaps and repeat the focused two-rank gate before Phase F.

## Corrections applied (2026-09-30)

Both P2 gaps were corrected. Tests only; no production source changed.

### Device adaptation now transforms state arrays

The testset now uses Adam and warms `mt`/`vt` with one distributed step
before adapting. Assertions:

- exactly 2 arrays transformed by `Functors.fmap`, both `AdaptedArray`;
- adapted values equal the source arrays; `beta` tuple unchanged;
- `adapted.backend === state.backend === rec`;
- the source state keeps `Vector{Float64}` arrays with the original values;
- the update runs through the **adapted** state and matches a native Adam
  reference on parameters, both moments, and the decayed-beta tuple;
- the gradient collective really saw `AdaptedArray{Float64,1}`.

`Base.convert(::Type{AdaptedArray{T,N}}, v::AbstractArray)` was added so the
Adam state (whose `Leaf.state` type is fixed by the adapted tree) stays
adapted across updates. Same pattern as the `IsoVec` fixture.

### Restored M1.1 API checks

- `tied parameters through real AD` regains the shared-gradient-object
  subcase: `(a = gshared, b = gshared)` is counted once per occurrence and
  matches native Optimisers, with `m.a === m.b` after the update.
- `Chain tuple-gradient diagnostic`: warning text and equal step for both
  `update!` and `update`.
- `Enzyme Duplicated model setup and update`: supported setup,
  `Flux.setup`, rejection of both update paths. `import Enzyme` was added for
  `Enzyme.make_zero`; the wrapper dispatch is on `EnzymeCore.Duplicated`.
- `freeze! and thaw! preserve wrapper state and collectives` replaces the
  previous partial-freeze test: Adam, retained returned states, whole-wrapper
  `freeze!`/`thaw!` delegation, partial freeze through the inner tree, frozen
  moment preservation, unchanged collective counts (3 per step), and a native
  reference that freezes/thaws at the same steps.

### Correction gate (two MPI ranks)

| Check | Result |
|---|---|
| Focused conditional totals | exit 0, 1 m 22.0 s: `conditional file \| 367 367 \| 1m22.0s` per rank |
| Focused optimizer | exit 0, 10.3 s: composition guard 6/6 per rank |
| Complete dedicated suite | exit 0, 2 m 4.8 s: `Distributed \| 6 6 2m02.6s` |
| Ambiguities | 0 |
| `git diff --check` | clean |

Per-rank totals: 232 (pre-Phase E) + 135 (Phase E revised) = **367**. The
correction runs passed on the first attempt; no new RED iterations occurred.

New testsets/changes: +560/−0 total at final state (was +411 before the
correction).

Evidence: `artifacts/logs/conditional-wrapper/phaseE-correction-*` and the
updated `phaseE-metadata.md`.

Next action: Phase F (two- and four-rank CPU/MPI gates; four-rank weighting).

## Independent correction review (2026-09-30)

**Both P2 findings are closed. Phase E is approved within its CPU/MPI scope.** No new blocking finding arose from the correction review.

- The adaptation test transforms two nonzero Adam arrays and asserts their types, values, and backend identity. Its update uses the actual adapted state. The native reference checks parameters, both moments, and beta products. The recorded gradient buffer has the adapted type.
- The restored tests cover Chain tuple-gradient warnings, supported Duplicated setup, rejected Duplicated updates, and shared-gradient tied occurrences.
- The freeze test exercises whole-wrapper and partial delegation with Adam. It retains returned states and compares updates against a native reference.

Independent command, from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test -e 'using Test; @testset "Phase E correction independent review" begin include("test/ext_distributed/conditional_distributedtest.jl") end'
git status --short --branch && git diff --stat && git diff --check
```

Both commands exited 0. The focused run passed 367/367 assertions per rank, with 78.3 seconds per summary and 78.972 seconds wall time. The Flux diff contains one test file, +560/-0, with clean whitespace.

The review inspected the correction logs for the optimizer file (6/6), complete suite (6/6, 2m02.6s), and ambiguity check (0). These checks were not repeated independently. No Flux source or test edits occurred during this review.

Next action: run Phase F, focused child followed by the dedicated suite at two and four ranks. The four-rank analytical case must prove total-worker weighting.

## Commit

Committed in the Flux worktree as `55171e65` ("Add Phase E conditional wrapper
tests and restore M1.1 API checks"), branch `ddp/integration-m1-spike`, not
pushed. The commit contains only
`test/ext_distributed/conditional_distributedtest.jl` (+560/−0); no
production source is included. Checkpoint chain: D1-D3 at `cbdd3edc`, D4 + D5
at `d6046103`, Phase E at `55171e65`.

Post-commit verification: `git status --short` clean, `git diff --check`
clean, `git diff d6046103..55171e65 --stat` = 1 file changed, 560 insertions.
