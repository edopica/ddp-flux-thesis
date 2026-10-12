# Phase D5 evidence metadata — regression gate

Date: 2026-09-30 (runs finished ~14:5x local)

## Environment

| Item | Value |
|---|---|
| Host | omarchy (x86_64, local CPU/MPI) |
| Julia | 1.12.6 |
| Optimisers | 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`) |
| Flux worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Flux branch | `ddp/integration-m1-spike` |
| Flux HEAD | `cbdd3edcb5fdeaa791968768a5d7c660e5817244` (D1-D3 committed; D4 + D5 changes uncommitted) |
| MPI launcher | `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec` |
| Rank count | 2 (real MPI); no four-rank run (Phase F) and no GPU/NCCL run |

## Working-tree state per run

| Run | `src/distributed/public_api.jl` | `test/ext_distributed/conditional_distributedtest.jl` |
|---|---|---|
| RED | uncommitted D4 only (array `_buildmap` still `eachindex`) | modified (D4 testsets + all new D5 testsets) |
| GREEN | uncommitted D4 + D5 (`_buildmap`/`_foreach_child` use `CartesianIndices` for arrays) | modified (same testsets) |

Combined uncommitted diff relative to `cbdd3edc`: 2 files, **+581/−66**
(`src` +67/−65, test +514/−1). D5's own contribution relative to the D4
working tree is `src` +5/−2 and test +332/−0.

## Intended defect (D5 gate finding)

The D5 contract "array containers retain their shape during gradient
reconstruction" (plan section 8.2: "For positional containers, retain their
shape and index order. Do not flatten an array container during
reconstruction.") did not hold for multidimensional `Array` containers.

`_buildmap(f, ch::AbstractArray)` mapped over `eachindex(ch)`. For an
`IndexLinear` container such as a `Matrix{Vector{Float64}}`, `eachindex`
returns a `Base.OneTo` of linear indices, so `map` rebuilt a length-4
`Vector` for a 2×2 model child instead of a 2×2 `Matrix`. Values still paired
by column-major linear order (which is why a permissive downstream update can
appear to succeed), but the rebuilt gradient did not mirror the model shape
that Optimisers consumes.

The same change makes `_foreach_child` (the walk order) use the same
`CartesianIndices` domain, so both passes still traverse in canonical
column-major model order.

## RED (before the production change)

Command (from `flux-integration`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

- Exit code: 1
- Wall: 1 m 4.3 s
- Per rank: control 5/5; D1 10/10; isbits guard 11/11; D2 38/38; D3 16/16;
  D4 20/20; D4-review 30/30; BatchNorm 6/6; transpose/adjoint 16/16; caller
  gradients 8/8; `update!` immutable 9/9; `update` fresh wrapper 15/15;
  globally absent 11/11; empty model 9/9; array containers **18 passed / 2
  failed / 20**.
- The two failures are exactly the intended defect:
  - line 963 `size(synced.w) == size(model.w)` evaluated `(4,) == (2, 2)`;
  - line 964 `axes(synced.w) == axes(model.w)` evaluated
    `(Base.OneTo(4),) == (Base.OneTo(2), Base.OneTo(2))`.
- No fixture error and no dependency error: the value assertions, protocol
  count, and end-to-end wrapper-reference comparison all passed on the
  flattened gradient.
- Log: `phaseD-d5-RED-mpi2-2026-09-30.log`

## Production fix

`src/distributed/public_api.jl` (uncommitted, D5 part):

```julia
_buildmap(f, ch::AbstractArray) = map(k -> f(k, ch[k]), CartesianIndices(ch))
_foreach_child(f, ch::AbstractArray) = (foreach(k -> f(k, ch[k]), CartesianIndices(ch)); nothing)
```

- `CartesianIndices` allocates the rebuilt container with the model's shape
  and iterates in column-major index order.
- `_foreach_child` uses the same domain, so collection and reconstruction
  still consume the occurrence plan in lockstep.
- The `_child` helper already accepts `CartesianIndex` for array containers.

## GREEN (after the fix)

| Run | Command | Exit | Wall | Result (per rank / total) |
|---|---|---|---|---|
| Focused conditional, 2 ranks | same as RED | 0 | 1 m 18.0 s (local Flux + FluxMPIExt recompile after the source change) | control 5/5; D1 10/10; guard 11/11; D2 38/38; D3 16/16; D4 20/20; D4-review 30/30; BatchNorm 6/6; transpose/adjoint 16/16; caller 8/8; `update!` immutable 9/9; `update` fresh 15/15; absent 11/11; empty 9/9; containers 20/20 — **224/224 per rank** |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 11.7 s | composition guard 6/6 per rank |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 1 m 41.4 s | `Distributed \| 6 6 1m38.2s` (6 child files) |
| Ambiguities | `julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers=", pkgversion(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); @assert isempty(a)'` | 0 | — | Optimisers 0.4.9; 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean (no output); only the two expected files; HEAD `cbdd3edc` |

Logs:

- `phaseD-d5-RED-mpi2-2026-09-30.log`
- `phaseD-d5-GREEN-mpi2-2026-09-30.log`
- `phaseD-d5-GREEN-optimizer-mpi2-2026-09-30.log`
- `phaseD-d5-GREEN-suite-nprocs2-2026-09-30.log`
- `phaseD-d5-ambiguities-2026-09-30.log`
- `phaseD-d5-diff-check-2026-09-30.log`

## Regression contracts now pinned by D5 testsets

| Contract (plan section 13) | Testset | Proof |
|---|---|---|
| BatchNorm running statistics excluded | `BatchNorm running statistics stay excluded` | 2 real-AD steps; 8 collectives = 2 × (1 presence + weight + β + γ); `μ`/`σ²` byte-unchanged; `γ` moved |
| Transpose/Adjoint parent-space pairing | `transpose and adjoint gradients pair in parent space` | parent values match analytic global mean; buffer is the parent-space 2×2 (and 1×3 for `transpose(v)`); count = presence + one parent buffer |
| Caller gradients unchanged, both paths | `caller gradients stay unchanged on both update paths` | `update`, `update!`, and a non-writable `view` gradient; snapshots unchanged; model updated correctly |
| `update!` immutable return + wrapper identity | `update! propagates immutable model returns and keeps wrapper identity` | isbits `Fill` parameter: `state2 === state`, returned model is fresh, input `Fill` untouched; immutable struct through `update` |
| `update` fresh wrapper, no input mutation | `update returns a fresh wrapper without mutating inputs` | `state2 !== state`, `state2.tree !== state.tree`, `state2.backend === backend`, input model/state/gradients unchanged; output matches native Adam on the global mean (moments included) |
| Globally absent parameter + Adam state | `globally absent parameters keep Adam state after active steps` | 3 real-AD steps; `absent` value and full Adam state equal to a fresh setup; `used`/`local_only` moved; cross-rank equality for the locally absent `local_only` |
| Locally absent receives global mean | `equal-shaped conditional branches...` + `globally absent...` | control testset applies the rank-0-only branch on every rank; `local_only` cross-rank equality |
| Empty trainable model, no collectives | `empty trainable model performs no collectives` | `_sync_gradients` returns the input tree identity; `update!`/`update` complete with 0 recorded collectives; setup warning matched |
| Array containers keep shape/index order | `array containers keep their shape during reconstruction` | direct `_sync_gradients` on a 2×2 `Matrix` container: `size`/`axes` equal the model, elements equal explicit global means, protocol = 1 presence + 4 ordered gradients, public wrapper update matches native reference elementwise |
| Phase C composition guards | `optimizer_distributedtest.jl` composition testset | directly nested and chain-nested setups throw `ArgumentError`; valid outermost chain setups |

## Real and simulated evidence

- All D5 runs use the real MPI backend at two ranks; every reduction is a real
  allreduce and every reference value is either an explicit global mean or an
  independent native Optimisers run.
- No mock or simulated presence value is used in the D5 testsets (the D3 mock
  remains isolated to its own testset).
- The only production change in D5 is the array traversal/reconstruction domain.

## Scope notes

- D5 only. Phase D is complete with D1-D5 after this gate; Phase E (restore the
  M1.1 battery), Phase F (four ranks), Phase G (detector), and Phase H
  (compatibility review) remain.
- No GPU/NCCL, detector, HPC, or main-suite run. No commit: D4 + D5 stay
  uncommitted until the user authorizes a commit.
- `git status` shows only the two expected modified files.

## Review correction pass (2026-09-30)

Independent review reopened D5 for two P2 coverage gaps in the test file. Both
were corrected with test-only changes; the production source is unchanged
(`src/distributed/public_api.jl` md5 `06c2b464a5d9c1c784e07cb7752a0e7e`,
verified before and after both temporary RED injections). Combined uncommitted
D4 + D5 diff after the corrections: 2 files, **+633/−66** (`src` +67/−65
unchanged, test +567/−1).

### Test corrections

1. `globally absent parameters keep Adam state after active steps` (was lines
   875-903): `absent` now takes 2 active steps first (nonzero moments, moved
   value), then the test asserts `snap_state != fresh.absent.state` and nonzero
   moments, snapshots `model.absent` and the full state
   (`deepcopy(state.tree.absent.state)`), runs 3 globally absent steps, and
   asserts exact preservation (`model.absent == snap_value`,
   `state.tree.absent.state == snap_state`). `local_only` is active in phase 1
   too. Testset now 16 tests (was 11).
2. `caller gradients stay unchanged on both update paths` (was lines 757-791):
   rank-distinct gradients on both update paths and the read-only case; per-rank
   snapshots compared after update; updated models compared with an independent
   global mean (`global_mean(snap)`); precondition `!(mean ≈ snap)`;
   `ReadOnlyGradient` (new test-local type whose `setindex!` throws) replaces the
   writable `view`. Testset now 11 tests (was 8 before the correction).

### Targeted RED evidence (temporary defects, reverted)

| RED | Injected defect | Earlier testsets | Corrected testset | Exit / wall | Log |
|---|---|---|---|---|---|
| A | `_walk!`: `buf = (g isa AbstractArray && length(g) == 2) ? g : (isbits(x) ? Array(x) : similar(x))` (aliased communication buffer) | 5/10/11/38/16/20/30/6/16 pass | `7 passed, 2 failed, 1 errored / 10`: `gs.w == snap` → `[0.4, -0.075]` vs `[0.3, -0.1]` (rank 0) and `[0.5, -0.05]` (rank 1); unaliasing error on `ReadOnlyGradient` | 1 / 1 m 10.8 s | `phaseD-d5-correction-RED-aliasing-mpi2-2026-09-30.log` |
| B | `_build!`: `return occ.emit ? bufs[...].buf : CRC.ZeroTangent()` (absent leaf emits a zero gradient) | all pass, including caller 11/11 | `14 passed, 2 failed / 16`: `model.absent == snap_value` (`1.0039413571883287` vs `1.0019999869276413`); `state.tree.absent.state == snap_state` (`(0.5314410000000002, 0.994014980014994)` vs `(0.7290000000000001, 0.997002999)`) | 1 / 1 m 14.3 s | `phaseD-d5-correction-RED-absent-zero-gradient-mpi2-2026-09-30.log` |

The length-2 scope in RED A keeps earlier testsets on their original code path
so the run reaches the corrected testset; RED B only affects globally absent
leaves, which earlier testsets do not have.

### Final GREEN gate (restored source)

| Run | Command | Exit | Wall | Result |
|---|---|---|---|---|
| Focused conditional, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl` | 0 | 58.1 s | 232/232 per rank (15 testsets) |
| Focused optimizer, 2 ranks | same with `optimizer_distributedtest.jl` | 0 | 10.7 s | composition guard 6/6 per rank |
| Complete dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 1 m 48.3 s | `Distributed \| 6 6 1m45.2s` |
| Ambiguities | `julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; ... detect_ambiguities(Flux.DistributedUtils; recursive=false) ...'` | 0 | — | Optimisers 0.4.9; 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean |

Correction-pass per-rank testset counts (final): control 5, D1 10, guard 11,
D2 38, D3 16, D4 20, D4-review 30, BatchNorm 6, transpose/adjoint 16, caller
gradients 11, `update!` immutable 9, `update` fresh 15, absent Adam 16, empty
model 9, array containers 20 = **232**.

Correction-pass logs:

- `phaseD-d5-correction-GREEN-conditional-mpi2-2026-09-30.log`,
- `phaseD-d5-correction-GREEN-optimizer-mpi2-2026-09-30.log`,
- `phaseD-d5-correction-GREEN-suite-nprocs2-2026-09-30.log`,
- `phaseD-d5-correction-RED-aliasing-mpi2-2026-09-30.log`,
- `phaseD-d5-correction-RED-absent-zero-gradient-mpi2-2026-09-30.log`,
- `phaseD-d5-correction-ambiguities-2026-09-30.log`,
- `phaseD-d5-correction-diff-check-2026-09-30.log`.

The reviewer's note on the writable `view` is addressed by replacing it with the
read-only wrapper. No four-rank, GPU/NCCL, detector, HPC, or main-suite run was
added.

## Commit (2026-09-30)

The full Phase D4 + D5 diff was committed after the corrections were verified:

- `d6046103` "Read trainable selection from setup state and keep array containers
  shaped" — 2 files, +633/−66, on branch `ddp/integration-m1-spike`. Not pushed;
  the Flux worktree is clean.
