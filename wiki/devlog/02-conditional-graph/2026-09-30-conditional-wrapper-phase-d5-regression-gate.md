# 2026-09-30 - Conditional-graph wrapper Phase D5: regression gate

## Goal

Phase D5 of `temp/conditional_graph_wrapper_implementation_plan.md`: retain or add
focused assertions for the wrapper contracts that D1-D4 do not already pin, run the
focused conditional and optimizer files at two MPI ranks, run the complete dedicated
two-rank suite, and rerun the ambiguity and whitespace checks. D5 closes Phase D.

## Contracts in the gate

The plan section 12 D5 list maps to the test matrix (section 13):

- BatchNorm running statistics remain excluded,
- Transpose and Adjoint gradients pair in parent space,
- caller gradients remain unchanged for both update paths,
- `update!` propagates immutable model returns and retains wrapper identity,
- `update` returns a new wrapper without mutating the input model or state,
- globally absent parameters and Adam state remain unchanged after active steps,
- locally absent parameters receive the global mean (already covered by the control
  testset; strengthened with a cross-rank equality check),
- empty trainable models perform no collectives,
- array containers retain their shape during gradient reconstruction,
- Phase C composition guards remain GREEN (focused optimizer file).

## Work

`test/ext_distributed/conditional_distributedtest.jl` gained eight focused testsets:

| Testset | Notes |
|---|---|
| `BatchNorm running statistics stay excluded` | Real-AD 2-step `Chain(Dense, BatchNorm)`; 8 collectives = 2 × (1 presence + weight + β + γ); `μ`/`σ²` unchanged; `γ` moved |
| `transpose and adjoint gradients pair in parent space` | Real AD; parent-space values and buffers checked; row-vector `transpose(v)` case included |
| `caller gradients stay unchanged on both update paths` | `update`/`update!` with rank-distinct gradients, per-rank snapshots, and an independent global mean; read-only gradient case (`ReadOnlyGradient`, `setindex!` throws). Corrected after review (was: identical gradients + writable `view`) |
| `update! propagates immutable model returns and keeps wrapper identity` | isbits `Fill` parameter proves the returned model carries the update; `state2 === state` |
| `update returns a fresh wrapper without mutating inputs` | `Adam` state snapshots; output equals a native Adam step on the global mean |
| `globally absent parameters keep Adam state after active steps` | 2 active steps first (nonzero moments, moved value), then 3 globally absent steps; value and full optimizer state exactly equal the snapshot; `snap_state != fresh.absent.state` asserted. Corrected after review (was: absent never active, compared with fresh setup) |
| `empty trainable model performs no collectives` | `_sync_gradients` returns the input tree identity; `update!`/`update` issue 0 collectives |
| `array containers keep their shape during reconstruction` | 2×2 `Matrix` container; shape, axes, values, protocol, and public update path |

The first seven testsets passed on the unchanged source. The array-container
testset failed: this was the intended D5 defect (see below).

## Defect corrected: array containers were flattened

`_buildmap(f, ch::AbstractArray)` mapped over `eachindex(ch)`. For an
`IndexLinear` container (`Matrix{Vector{Float64}}`), `eachindex` is a linear
`Base.OneTo`, so `map` rebuilt a length-4 `Vector` for a 2×2 model child. The
values still paired by column-major order, so the downstream update could
appear to succeed, but the rebuilt gradient did not mirror the model shape that
Optimisers consumes. Plan section 8.2 requires positional containers to keep
their shape and index order.

Fix in `src/distributed/public_api.jl` (uncommitted, D5 part):

```julia
_buildmap(f, ch::AbstractArray) = map(k -> f(k, ch[k]), CartesianIndices(ch))
_foreach_child(f, ch::AbstractArray) = (foreach(k -> f(k, ch[k]), CartesianIndices(ch)); nothing)
```

`CartesianIndices` allocates the rebuilt container with the model shape and
iterates in column-major order; both passes still consume the occurrence plan in
lockstep. `_child` already accepts `CartesianIndex`.

## RED evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration` before the array fix:

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

Exit 1, 1 m 4.3 s. Per rank: control 5/5, D1 10/10, guard 11/11, D2 38/38,
D3 16/16, D4 20/20, D4-review 30/30, and all seven new contract testsets 5-16/…
passed; `array containers keep their shape` **18 passed / 2 failed / 20**:

- `size(synced.w) == size(model.w)` → `(4,) == (2, 2)`,
- `axes(synced.w) == axes(model.w)` → `(Base.OneTo(4),) == (Base.OneTo(2), Base.OneTo(2))`.

Values, protocol count, and the end-to-end wrapper/reference comparison passed on
the flattened gradient: the defect is exactly the container shape. No fixture or
dependency error. Log: `phaseD-d5-RED-mpi2-2026-09-30.log`.

## GREEN evidence

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks | same as RED | exit 0, 1 m 18.0 s (one local Flux + FluxMPIExt recompile): 224/224 per rank across all 15 testsets |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 11.7 s, composition guard 6/6 per rank |
| Complete dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 1 m 41.4 s, `Distributed \| 6 6 1m38.2s` |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 (Optimisers 0.4.9) |
| Whitespace | `git diff --check` | clean |

Combined uncommitted D4 + D5 diff relative to `cbdd3edc`: 2 files, +581/−66
(`src` +67/−65, test +514/−1). D5 alone is `src` +5/−2 and test +332/−0.

## Scope notes

- Phase D (D1-D5) is complete after this gate. Phase E (restore and improve the
  M1.1 battery), Phase F (four ranks), Phase G (detector conversion), and Phase H
  (compatibility review) remain.
- No four-rank run, no GPU/NCCL run, no detector update, no Bocconi run, no main-suite
  run. No mock or simulated value is used in the D5 testsets.
- No commit: D4 + D5 remain reviewable as uncommitted changes; commit/push only
  with explicit user authorization.
- Exact commands, exit codes, elapsed times, and assertion totals:
  `artifacts/logs/conditional-wrapper/phaseD-d5-metadata.md`.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseD-d5-metadata.md`,
- `phaseD-d5-RED-mpi2-2026-09-30.log`,
- `phaseD-d5-GREEN-mpi2-2026-09-30.log`,
- `phaseD-d5-GREEN-optimizer-mpi2-2026-09-30.log`,
- `phaseD-d5-GREEN-suite-nprocs2-2026-09-30.log`,
- `phaseD-d5-ambiguities-2026-09-30.log`,
- `phaseD-d5-diff-check-2026-09-30.log`.

## Next action

Phase E: restore the M1.1 battery without its known defects, starting from
`3370b910:test/ext_distributed/conditional_distributedtest.jl` as reference, then
run the conditional file after each ported group. Request user authorization
before committing D4 + D5.

## Independent review (2026-09-30)

Result: **D5 reopened for two P2 coverage gaps**. The production shape fix has no identified defect in the reviewed scope.
The review changed no Flux source or tests.

### Findings

1. **P2: Add a prior active step for the absent Adam parameter.**
   At `conditional_distributedtest.jl:878-893`, `absent` never contributes to the loss.
   Its state is compared with fresh setup state after other parameters receive updates.
   This does not cover the D5 requirement for absence after a prior active step on that same parameter.
   A regression that resets absent state to its initial value can pass this test.
   First activate that parameter and establish nonzero Adam moments.
   Then snapshot its value and full state before a globally absent step.
   Assert exact preservation after that step.

2. **P2: Make caller-gradient mutation observable across ranks.**
   At `conditional_distributedtest.jl:765-790`, every rank supplies identical gradients on all three paths.
   An in-place mean reduction leaves these values unchanged at two ranks, even if the buffer aliases the caller gradient.
   Native Descent also leaves the gradient unchanged.
   Thus, these assertions do not detect the communication aliasing defect they claim to prevent.
   Use rank-distinct gradients on both update paths and in the view case.
   Compare the updated model with an independent global mean and the caller gradient with its local snapshot.
   The current view is writable through Julia indexing, despite the comment that calls it non-writable.

### Checks and evidence

Source state: Flux `cbdd3edc` plus the existing uncommitted D4 and D5 changes in two files.
Optimisers: 0.4.9, `/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`.

The independent checks ran from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
set -o pipefail && { time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl; } 2>&1 | tee /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/phaseD-d5-review-conditional-mpi2-2026-09-30.log
set -o pipefail && { time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl; } 2>&1 | tee /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/phaseD-d5-review-optimizer-mpi2-2026-09-30.log
julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers ", pkgversion(Optimisers)); println("Source: ", pathof(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("Ambiguities: ", length(a)); @assert isempty(a)'
git diff --check
```

| Check | Independent result |
|---|---|
| Conditional, two MPI ranks | Exit 0, 224/224 per rank, 56.748 s |
| Optimizer, two MPI ranks | Exit 0, 6/6 per rank, 11.805 s |
| Ambiguities | Exit 0, count 0 |
| Flux whitespace | Exit 0, clean |

The review also examined the stored array-shape RED log and complete two-rank suite GREEN log.
Those logs support the reported shape failure and suite result of 6/6.
The review did not repeat the complete suite or run four-rank, GPU, detector, or compatibility checks.
These results do not establish performance, higher-order gradient support, Enzyme updates, nested composition, or external state-consumer compatibility.

Exact next action: strengthen the two D5 regressions, then repeat the focused files and complete dedicated two-rank gate before Phase E.

## Review corrections (2026-09-30)

Both P2 gaps were closed. Only the test file changed relative to the D5-GREEN
state; the production source was not modified for the corrections and was
restored byte-for-byte after the two temporary RED injections
(`src/distributed/public_api.jl` md5 `06c2b464a5d9c1c784e07cb7752a0e7e`).

### Correction 1: prior active step before global absence

`globally absent parameters keep Adam state after active steps` now runs two
active steps on `absent` first (its gradient is nonzero on every rank), then
asserts that the value moved and that the state is not the fresh one:

- `model.absent != ones(3)`,
- `any(!iszero, snap_state[1])` and `any(!iszero, snap_state[2])` (nonzero
  moments),
- `snap_state[3] != fresh.absent.state[3]` (Adam's `βt` pair advanced; in
  Optimisers 0.4.9 `init(o::Adam, x) = (zero(x), zero(x), T.(o.beta))`, so
  `state[3]` is not a step counter),
- `snap_state != fresh.absent.state`.

`snap_value = copy(model.absent)` and `snap_state = deepcopy(state.tree.absent.state)`
are taken after the active steps. The three globally absent steps then assert
`model.absent == snap_value` and `state.tree.absent.state == snap_state`
exactly. Because the test also asserts `snap_state != fresh.absent.state`, a
regression that resets the absent leaf to its initial state fails the snapshot
comparison.

### Correction 2: observable caller-gradient mutation

`caller gradients stay unchanged on both update paths` now uses rank-distinct
gradients on the non-mutating `update` path, the `update!` path, and the
read-only case; each caller gradient is compared with its local pre-update
snapshot, and each updated model is compared with an independent global mean
computed by reducing the snapshot with the raw backend. The writable `view` was
replaced by `ReadOnlyGradient`, a test-local `AbstractVector` whose `setindex!`
throws, so a write into the caller gradient cannot pass silently. Each case also
asserts the precondition `!(mean ≈ local)` so identical gradients cannot make
the mutation unobservable.

### Targeted RED evidence

Two temporary production defects were injected one at a time, solely to show the
corrected testsets detect them. Both injections were reverted immediately and
the file was md5-verified; `git diff --check` is clean on the restored tree.

**RED A — communication buffer aliases the caller gradient.**
Injected in `_walk!` (scoped to the length-2 gradients of the corrected
testset, so earlier testsets keep their behavior and the run reaches it):

```diff
-        buf = isbits(x) ? Array(x) : similar(x)
+        buf = (g isa AbstractArray && length(g) == 2) ? g : (isbits(x) ? Array(x) : similar(x))
```

Result (`phaseD-d5-correction-RED-aliasing-mpi2-2026-09-30.log`): the nine
earlier testsets pass (5/10/11/38/16/20/30/6/16); the corrected testset fails
with `7 passed, 2 failed, 1 errored`:

- rank 0 line 788: `gs.w == snap` → `[0.4, -0.07500000000000001] == [0.3, -0.1]`,
- rank 1 line 788: `gs.w == snap` → `[0.4, -0.07500000000000001] == [0.5, -0.05]`,
- both ranks line 801 (second case) and an unaliasing error at line 812
  (`copyto!` on the aliased `ReadOnlyGradient`).

The caller gradients hold exactly the global mean at the failure: the aliasing
is detected. Exit 1, 1 m 10.8 s.

**RED B — absent leaves fed a zero gradient instead of `ZeroTangent`.**
Injected in `_build!`:

```diff
-        return occ.emit && present[occ.buffer_index] ?
-            bufs[occ.buffer_index].buf : CRC.ZeroTangent()
+        return occ.emit ?
+            bufs[occ.buffer_index].buf : CRC.ZeroTangent()
```

Result (`phaseD-d5-correction-RED-absent-zero-gradient-mpi2-2026-09-30.log`):
every earlier testset (including the caller-gradient testset, 11/11) passes; the
corrected absent testset reports `14 passed, 2 failed` on both ranks:

- `model.absent == snap_value` → `1.0039413571883287` (three zero-gradient Adam
  steps moved the value through the decaying first moment) vs the snapshot
  `1.0019999869276413`,
- `state.tree.absent.state == snap_state` → moments and `βt` decayed:
  `(0.5314410000000002, 0.994014980014994)` vs `(0.7290000000000001, 0.997002999)`.

The prior active steps make parameter movement and moment decay observable.
The old full-state comparison also detected zero-gradient updates because Adam's decay products advance even when its moments remain zero.
The corrected test additionally detects loss or reset of accumulated state. Exit 1, 1 m 14.3 s.

### Final GREEN gate (restored source)

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` | exit 0, 58.1 s: 232/232 per rank across 15 testsets |
| Focused optimizer, 2 ranks | same with `optimizer_distributedtest.jl` | exit 0, 10.7 s: composition guard 6/6 per rank |
| Complete dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 1 m 48.3 s: `Distributed \| 6 6 1m45.2s` |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 (Optimisers 0.4.9) |
| Whitespace | `git diff --check` | clean |

Final per-rank counts: control 5, D1 10, guard 11, D2 38, D3 16, D4 20,
D4-review 30, BatchNorm 6, transpose/adjoint 16, caller gradients 11, `update!`
immutable 9, `update` fresh 15, absent Adam 16, empty model 9, array containers
20 = **232/232**. Combined uncommitted D4 + D5 diff relative to `cbdd3edc`:
2 files, +633/−66 (`src` +67/−65, test +567/−1). No production change for the
corrections, no new defect shipped, no commit.

New evidence: `phaseD-d5-correction-GREEN-conditional-mpi2-2026-09-30.log`,
`phaseD-d5-correction-GREEN-optimizer-mpi2-2026-09-30.log`,
`phaseD-d5-correction-GREEN-suite-nprocs2-2026-09-30.log`,
`phaseD-d5-correction-RED-aliasing-mpi2-2026-09-30.log`,
`phaseD-d5-correction-RED-absent-zero-gradient-mpi2-2026-09-30.log`,
`phaseD-d5-correction-ambiguities-2026-09-30.log`,
`phaseD-d5-correction-diff-check-2026-09-30.log`.

Exact next action after the corrections: Phase E (restore the M1.1 battery,
starting from `3370b910:test/ext_distributed/conditional_distributedtest.jl`).
The D4 + D5 diff is committed at `d6046103` (see below).

## Commit (2026-09-30)

User authorization received; the Phase D4 + D5 diff was committed on branch
`ddp/integration-m1-spike`:

- `d6046103` "Read trainable selection from setup state and keep array
  containers shaped" — 2 files, +633/−66 (`src/distributed/public_api.jl`,
  `test/ext_distributed/conditional_distributedtest.jl`). Not pushed.

Previous commit on the branch: `cbdd3edc` (D1-D3). The Flux worktree is clean.

## Independent follow-up review (2026-09-30)

Verdict: **Both P2 findings are closed. D5 is approved within its CPU/MPI scope.**
The corrected test establishes nonzero Adam moments before global absence and compares independent snapshots afterward.
The caller-gradient cases now use rank-distinct values, local snapshots, independent means, and a read-only fixture.
No new blocking finding arose in this follow-up.

The temporary injections are mutation-test evidence for assertion sensitivity, not evidence of new production defects.
The stored logs show the intended failures and the final complete suite result of 6/6.
The independent follow-up repeated the focused conditional file, not the complete suite.

Command, from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
set -o pipefail && { time FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl; } 2>&1 | tee /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/conditional-wrapper/phaseD-d5-review2-conditional-mpi2-2026-09-30.log
git diff --check && git diff --stat && md5sum src/distributed/public_api.jl
```

Results: conditional 232/232 per rank, exit 0, 59.032 s. The whitespace check passed.
Production md5: `06c2b464a5d9c1c784e07cb7752a0e7e`, which matches the correction report.
The review changed no Flux source or tests.

Exact next action: Phase E, using `3370b910:test/ext_distributed/conditional_distributedtest.jl` as reference.
Commit authorization remains separate. Four-rank, GPU, detector, and compatibility evidence remain outside this review.
