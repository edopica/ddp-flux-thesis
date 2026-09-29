# 2026-09-28 - Conditional-graph wrapper Phase D2: explicit occurrence mapping

## Goal

Phase D2 of `temp/conditional_graph_wrapper_implementation_plan.md`: replace the
inferred occurrence-to-buffer mapping in the rebuild pass with an explicit
occurrence plan, so a model that mixes tied mutable parameters with isbits
parameters maps every occurrence to its own buffer.

## Defect

`_walk!` (collection pass) allocates one buffer per unique mutable parameter and one
buffer per isbits occurrence. `_build!` (rebuild pass) consumed buffers with a
`counter` that incremented for **every** numeric occurrence. A tied mutable parameter
contributes two occurrences but only one buffer, so the counter ran ahead of the
buffer list.

Fixture `MixedTiedIsbits(p, p2, c, q)` with `p2 === p`, isbits `c`, distinct mutable
`q`:

- buffers: `p` (index 1), `c` (index 2), `q` (index 3);
- occurrences consume counter values 1, 2, 3, 4: `p` -> buffer 1, `p2` -> `ZeroTangent`,
  `c` -> `bufs[3]` (**`q`'s buffer**), `q` -> `bufs[3]`.

`c` therefore received `q`'s global-mean gradient and `q`'s Adam update; the actual
`c` buffer was reduced but never emitted. With a trailing isbits parameter the
counter would also exceed the buffer list and raise a `BoundsError`.

## Fix

`src/distributed/public_api.jl` (uncommitted):

- New private `_SyncOccurrence(buffer_index::Int, emit::Bool)`.
- `_walk!` pushes one plan entry per numeric occurrence, in canonical model order:
  `emit = true` for the first occurrence of a mutable parameter and for each isbits
  occurrence, `emit = false` for later tied occurrences.
- `_build!` consumes plan entries through a cursor instead of inferring buffer
  indices; `seen`, `counter`, and `first_seen` are removed from the rebuild pass.
  `emit && present[occ.buffer_index] ? bufs[occ.buffer_index].buf : ZeroTangent()`
  reproduces the required semantics: first tied occurrence emits, tied repeats and
  globally absent parameters skip.
- `_sync_gradients` asserts complete plan consumption
  (`cursor[] == length(plan)`) and throws `ArgumentError` otherwise; a partially
  consumed plan means the two passes disagreed about the traversal.
- Collectives, presence protocol, buffer allocation, and device placement are
  unchanged.

`test/ext_distributed/conditional_distributedtest.jl` (uncommitted):

- New guard testset `equal-valued isbits occurrences stay independent` (`(c1 = v, c2 = v)`
  with `c1 === c2` as bit-equal isbits values, distinct gradients, 1 presence + 2
  gradient reductions, distinct results).
- New testset `explicit occurrence mapping for mixed tied/isbits parameters`:
  `MixedTiedIsbits(p, p2, c, q)` in the plan-specified order with explicit local
  gradients; three Adam steps (both tied occurrences present; first absent while the
  second contributes; second absent while the first contributes); native Optimisers
  Adam reference built from explicit `avg` reductions of the summed tied
  contributions; model-value and Adam-state comparisons after the steps; recording
  backend asserting the exact pre-reduce buffer values and `1 presence + 3 gradient`
  protocol; cross-rank count equality. The guard is placed before the regression so a
  RED failure cannot abort it.
- The mixed fixture is type-preserving: `c::IsoVec{4}` and `IsoVec` defines the
  `Base.convert(::Type{IsoVec{N}}, ::AbstractVector)` used on reconstruction, so
  `Optimisers.subtract!`'s `eltype(x).(x .- x̄)` result converts back to an `IsoVec`
  instead of degrading `c` to a `Vector` after step 1. `@test isbits(model.c)` runs
  before steps 2 and 3 and after step 3, so every step exercises the isbits path.

Combined uncommitted working-tree diff (D1+D2): 2 files, +355/−27.

## RED evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration` with only the test
changes (`MPIEXEC = /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl
```

Initial fixture: exit 1, 30 s. Per rank: control 5/5, D1 10/10, equal-valued guard
11/11, D2 testset **30 passed / 5 failed / 35**; all five failures on `c`
(`conditional_distributedtest.jl` lines 276, 289, 309, 332, 340). Log:
`phaseD-d2-RED-mpi2-2026-09-28.log`.

A first attempt was discarded before RED evidence existed: the new helper was written
as `_avg_sum(local...)` and `local` is a reserved word, giving a `ParseError` before
any test executed. Renamed the parameter to `gs` and reran. Per the plan, fixture
errors do not count as RED evidence.

Type-preserving fixture (strengthened rerun): the review finding was that after step 1
`Optimisers.subtract!` had degraded `c` from `IsoVec{4}` to `Vector{Float64}`, so only
step 1 exercised the isbits path. The fixture now types the field `c::IsoVec{4}` and
defines the reconstruction conversion, and asserts `isbits(model.c)` before steps 2-3
and after step 3. RED was re-recorded by temporarily reverting only the D2 production
hunks to the D1-only state (verified by diff against the saved fixed file): exit 1,
40 s, control 5/5, D1 10/10, guard 11/11, D2 **33 passed / 5 failed / 38**. All five
failures are again on `c` (lines 285, 298, 319, 343, 352):
`model.c ≈ ref_model.c` after steps 1-3 and `state_ok(state.tree.c.state, ...)` after
steps 1 and 3. Example (step 1): `[0.9000000002222222, ...] ≈
[0.9000000002857143, ...] (atol=1.0e-12)`; step 2:
`[0.8017208144373712, ...] ≈ [0.8027184078077062, ...]`. The `isbits(model.c)`
assertions pass, so the later failures are from the isbits path. Tied `p`/`p2` and `q`
assertions pass. Log: `phaseD-d2-RED-mpi2-typepreserving-2026-09-28.log`.

## GREEN evidence

Environment: Julia 1.12.6, Optimisers 0.4.9
(`/home/kurapica/.julia/packages/Optimisers/tMaaf`), host omarchy. Flux HEAD remains
`f5b87975`; D1+D2 changes uncommitted. The fixed production source was restored
byte-identical from a saved copy before this run.

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks | same as RED | exit 0, 40 s: control 5/5, D1 10/10, guard 11/11, D2 38/38 per rank |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi $MPIEXEC -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 10 s, 6/6 per rank |
| Full 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 59 s, `Distributed \| 6 6 57.4s`, all 6 children |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 (Optimisers 0.4.9) |
| Whitespace | `git diff --check` | clean (no output) |

## Scope notes

- D2 only. The mixed tied/isbits defect is corrected and guarded; equal-valued isbits
  occurrences stay independent.
- D3 (CPU presence metadata), D4 (legacy tuple `trainable`), and D5 (regression gate)
  remain open. Four-rank execution remains Phase F; no GPU/NCCL, detector, HPC, or
  main-suite runs.
- No commit. Tests and fix stay reviewable as separate changes in the worktree;
  commit/push only with explicit user authorization.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseD-d2-metadata.md`,
- `phaseD-d2-RED-mpi2-2026-09-28.log` (initial fixture),
- `phaseD-d2-RED-mpi2-typepreserving-2026-09-28.log` (strengthened fixture),
- `phaseD-d2-GREEN-mpi2-2026-09-28.log` (initial fixture),
- `phaseD-d2-GREEN-mpi2-typepreserving-2026-09-28.log` (strengthened fixture),
- `phaseD-d2-GREEN-optimizer-mpi2-2026-09-28.log`,
- `phaseD-d2-GREEN-suite-nprocs2-2026-09-28.log` (initial fixture),
- `phaseD-d2-GREEN-suite-nprocs2-typepreserving-2026-09-28.log` (strengthened fixture),
- `phaseD-d2-ambiguities-2026-09-28.log`,
- `phaseD-d2-diff-check-2026-09-28.log`.

## Next action

Implement Phase D3: CPU presence metadata. Add a device-like parameter fixture that
exposes device-derived metadata allocation through `_sync_gradients`, record the RED
allocation/access result, then allocate the presence vector directly as
`Vector{Float32}` on the host and rerun the focused MPI regressions. Four-rank gates
remain Phase F.

## Independent D2 review (2026-09-28)

Verdict: no blocking code findings. The explicit plan satisfies D2's mapping contract.
Collection records tied repeats even when their local gradient is absent.
Reconstruction emits each shared buffer once and checks complete plan consumption.
Both passes retain D1's canonical order.

One non-blocking test limitation remains at `conditional_distributedtest.jl:270`:
the first update replaces `IsoVec` with `Vector{Float64}`.
Only step 1 exercises mixed tied/isbits mapping.
Steps 2–3 still prove state continuity and the two absent-tied-occurrence cases.
The stored RED failures after step 1 reflect the earlier corrupted Adam state.
A type-preserving fixture and an `isbits(model.c)` assertion before each step can strengthen repeated mixed-model coverage.

The review inspected the stored RED log (30 pass / 5 fail per rank) and full-suite GREEN log (6/6).
Independent focused runs passed with Optimisers 0.4.9.
All commands below ran from `/home/kurapica/Projects/ddp_flux/flux-integration` and exited 0.
Wall times were not measured separately.

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test -e 'include("test/ext_distributed/conditional_distributedtest.jl"); let v = IsoVec{4}((1.0, 1.0, 1.0, 1.0)); s = Optimisers.setup(Optimisers.Adam(0.1), v); _, v2 = Optimisers.update!(s, v, ones(4)); println("fixture rank=", rank, " before=", typeof(v), " after=", typeof(v2), " isbits_after=", isbits(v2)); end'
FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl
git diff --check && julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers=", pkgversion(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); @assert isempty(a)'
```

Results:

- Conditional tests: control 5/5, D1 10/10, equal-valued guard 11/11, D2 35/35 per rank.
- Fixture probe on both ranks: `before=IsoVec{4} after=Vector{Float64} isbits_after=false`.
- Optimizer composition: 6/6 per rank.
- Ambiguities: 0. Flux whitespace check: clean.

The review changed only the thesis context and this devlog.
The next implementation action remains D3's device-like fixture and RED evidence for CPU presence metadata.
