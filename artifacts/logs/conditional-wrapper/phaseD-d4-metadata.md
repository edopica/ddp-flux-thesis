# Phase D4 evidence metadata — legacy tuple-returning `trainable`

Date: 2026-09-29 (runs finished ~14:5x local)

## Environment

| Item | Value |
|---|---|
| Host | omarchy (x86_64, local CPU/MPI) |
| Julia | 1.12.6 |
| Optimisers | 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`) |
| Flux worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Flux branch | `ddp/integration-m1-spike` |
| Flux HEAD | `cbdd3edcb5fdeaa791968768a5d7c660e5817244` (D1-D3 committed; D4 changes uncommitted) |
| MPI launcher | `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec` |
| Rank count | 2 (real MPI); no four-rank run (Phase F) and no GPU/NCCL run |

## Working-tree state per run

| Run | `src/distributed/public_api.jl` | `test/ext_distributed/conditional_distributedtest.jl` |
|---|---|---|
| RED | committed `cbdd3edc` (2-argument `_trainable_child(tr, k)`) | modified (D4 fixture + testset added) |
| GREEN | modified (`_trainable_child(ch, tr, k)` predicate; legacy path `ch[k] in tr`) | modified (same D4 testsets) |

Final uncommitted D4 diff: 2 files, +119/−11 (`src` +37/−11, test +93/−0).
D4 step 7 (warning preservation) and steps 3-4 (native setup/update comparison)
are all inside the same testset.

## Intended defect

`Optimisers.setup` supports old Flux-style `trainable` methods that return a plain
`Tuple` of selected fields. Its normalization (`_trainable`, Optimisers 0.4.9
`src/interface.jl:193-196`) maps each functor child `c` to `c in tr ? c : nothing`,
i.e. selection is by **value membership**, not identity, so equal-valued distinct
fields are selected together. It also emits the legacy warning once (`maxlog=3`).

The synchronization walk only knew the named-selection container shapes. Its
`_trainable_child(tr, k)` fallback for `tr::Tuple` called `haskey(tr, k)` with a
field-name `Symbol`, so a legacy model failed at the first `Optimisers.update!`
with:

```text
MethodError: no method matching haskey(::Tuple{Vector{Float64}}, ::Symbol)
```

The same walk must select exactly the children native setup selects, in model field
order, and must not repeat the legacy warning on every update.

## RED (before production changes)

Command (from `flux-integration`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

- Exit code: 1
- Wall: 33 s
- Per rank: control 5/5; D1 10/10; equal-valued guard 11/11; D2 38/38; D3 16/16;
  D4 `legacy tuple trainable selection matches native setup` **12 passed / 3 failed
  / 5 errored / 20**.
- The first error is exactly the intended defect:
  `MethodError: no method matching haskey(::Tuple{Vector{Float64}}, ::Symbol)`
  thrown by the wrapper `Optimisers.update!` (the `@test_logs` no-update-warning
  assertion), stack through `_trainable_child`.
- The 3 fails and the other 4 errors are downstream of the same throw: the update
  never ran, so `model.a`/`model.b` stayed `[1.0, …]` (expected `[0.85, …]` /
  `[0.45, …]`), `length(rec.log) == 3` evaluated `0 == 3`, and the `rec.log`
  indexing assertions raised `BoundsError`. No fixture error, no dependency error.
- Log: `phaseD-d4-RED-mpi2-2026-09-29.log`

## Production fix

`src/distributed/public_api.jl` (uncommitted, D4 part):

- Both traversal passes now call `_trainable_child(ch, tr, k)`, i.e. the selection
  rule sees the parent's functor children as well as the public
  `Optimisers.trainable(x)` result.
- `_trainable_child(ch::NamedTuple, tr::Tuple, k) = ch[k] in tr` adds the legacy
  no-names tuple path with the exact native value-membership rule (same `in`
  expression as `Optimisers._trainable`, no private API call, no identity test).
- The named/container methods are unchanged, only reshaped to dispatch on `(ch, tr)`
  (`haskey` for named selections, position check for tuples/arrays, `haskey` for
  dicts, generic fallback kept).
- `_walk!` and `_build!` evaluate the predicate in model field order, so canonical
  order and occurrence-plan alignment are preserved for legacy models too.
- No warning is emitted in the synchronization walk: native setup already emits the
  legacy warning once; `_sync_gradients` must not repeat it on every update.

## GREEN (after the fix)

| Run | Command | Exit | Wall | Result (per rank / total) |
|---|---|---|---|---|
| Focused conditional, 2 ranks | same as RED | 0 | 44 s (one local Flux + FluxMPIExt recompile after the source change) | control 5/5; D1 10/10; guard 11/11; D2 38/38; D3 16/16; D4 20/20 |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 11 s | composition guard 6/6 |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 65 s | `Distributed \| 6 6 1m01.5s` (6 child files) |
| Ambiguities | `julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers=", pkgversion(Optimisers), " path=", pathof(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); foreach(println, a)'` | 0 | — | Optimisers 0.4.9; 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean (no output) |

Logs:

- `phaseD-d4-GREEN-mpi2-2026-09-29.log`
- `phaseD-d4-GREEN-optimizer-mpi2-2026-09-29.log`
- `phaseD-d4-GREEN-suite-nprocs2-2026-09-29.log`
- `phaseD-d4-ambiguities-2026-09-29.log`
- `phaseD-d4-diff-check-2026-09-29.log`

## Regression contract (what the D4 testset proves)

Fixture `LegacyTupleTrainable`: `a` and `b` are equal-valued, distinct mutable
arrays; `excluded` is a distinct array with a different value;
`Optimisers.trainable(m) = (m.a,)` (old no-names tuple).

- **Value-membership selection**: native `Optimisers.setup` gives leaves to `a`
  *and* `b` (both `==` to the tuple entry), none to `excluded`. The wrapper tree is
  `==` to the native tree, so the inner update sees the same selected leaves.
- **Native update comparison**: the wrapper update matches a plain
  `Optimisers.update` reference built from explicit `avg` allreduces of the same
  gradients; `excluded` is ignored by both.
- **Protocol**: exactly 3 collectives on the recording backend — one presence
  reduction (`(2,)`, `Vector{Float32}`, `avg`) then one gradient reduction per
  selected buffer in model order (`a`, then `b`), distinguished by value because
  their shapes are equal. Counts equal across ranks (`bcast!`).
- **Warning contract**: `@test_logs` captures the legacy warning around wrapper
  setup (native setup emits it) and asserts no warnings at all during the wrapper
  update.

## Real and simulated evidence

- All D4 runs use the real MPI backend at two ranks; the reductions are real
  allreduces and the reference values are explicit global means.
- The legacy warning assertion uses a fresh `Test.TestLogger` per `@test_logs`, so
  it is independent of how often the default logger already printed the
  `maxlog=3` warning.
- No mock or simulated value is used in the D4 testset.

## Scope notes

- D4 only. The D5 regression gate remains open; four-rank execution remains Phase F.
- No GPU/NCCL run, no detector update (Phase G), no Phase E test battery, no HPC
  run, no commit. `git status` shows only the two expected modified files.
- Optimisers 0.4.9 is the only version recorded; `_trainable` behavior for other
  versions was not verified in this phase.

## Correction 2026-09-29 (independent review, P1)

The review found that the `_trainable_child(ch::NamedTuple, tr::Tuple, k) = ch[k] in tr`
predicate reevaluated value membership on every update. Native `Optimisers.setup`
resolves the legacy selection once and retains the resulting state tree; the predicate
instead dropped `b` from step two on, as soon as the first update made `a` and `b`
differ, even though the `b` leaf stayed in the tree. Characterization (exit 0 because it
asserts the defect): `phaseD-d4-review-multistep.jl` and
`phaseD-d4-review-mpi2-2026-09-29.log`; both update paths showed step 2 wrapper
`b[1]=0.45` vs native `-0.10` and 2 collectives instead of 3.

Corrected design: the synchronization walk now reads the selection from the state tree
returned by `Optimisers.setup` (`state.tree`). That tree is the retained setup-time
selection and cannot drift when parameter values change. `_walk!`/`_build!` pair each
model child with the corresponding tree node through the shared structural `_child`
lookup; a numeric child is trainable iff its tree node is an `Optimisers.Leaf`, and an
empty tree node (`()`) skips the subtree. `_sync_gradients` now takes the tree
(`_sync_gradients(backend, tree, model, grads)`); the D3 test builds its tree with
`Optimisers.setup(Optimisers.Descent(1.0), model)`. `_trainable_child` and its
value-membership rule are removed; no warning is emitted in the walk. Selection still
follows the model's functor child order, so the D1-D3 ordering contracts are unchanged.

### RED2 (corrected test, unchanged buggy source)

Command: same focused 2-rank run. Exit 1, 37.8 s. Prior testsets unchanged (5/5, 10/10,
11/11, 38/38, 16/16, 20/20). New testset
`legacy tuple trainable selection persists across updates` **24 passed / 6 failed / 30**
per rank, i.e. 3 intended failures per update path at step 2:
`model.b = [0.55, …]` vs native `[-0.605, …]`;
`state.tree == ref_state` false (momentum state `[0.45, …]` vs `[1.155, …]`);
`length(rec.log) == 5` vs expected 6. Log:
`phaseD-d4-RED2-mpi2-2026-09-29.log`.

### GREEN2 (corrected)

| Run | Command | Exit | Wall | Result (per rank / total) |
|---|---|---|---|---|
| Focused conditional, 2 ranks | same as RED | 0 | 48 s (local Flux + FluxMPIExt recompile) | control 5/5; D1 10/10; guard 11/11; D2 38/38; D3 16/16; D4 20/20; D4-review 30/30 |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 10 s | composition guard 6/6 |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 74 s | `Distributed \| 6 6 1m12.0s` (6 child files) |
| Ambiguities | same command as the first run | 0 | — | Optimisers 0.4.9; 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean; only the two expected files; HEAD `cbdd3edc` |

Final uncommitted D4 diff relative to `cbdd3edc`: 2 files, +244/−64 (`src` +62/−63,
test +182/−1, including the D3 direct-call update).

Logs: `phaseD-d4-GREEN2-mpi2-2026-09-29.log`,
`phaseD-d4-GREEN2-optimizer-mpi2-2026-09-29.log`,
`phaseD-d4-GREEN2-suite-nprocs2-2026-09-29.log`, and the rerun
`phaseD-d4-ambiguities-2026-09-29.log` / `phaseD-d4-diff-check-2026-09-29.log`.

### Added regression contract

`legacy tuple trainable selection persists across updates`: two steps through both
`Optimisers.update!` and `Optimisers.update`, with distinct per-rank and per-step
gradients and a stateful `Momentum` rule, compared against a native reference built
from explicit `avg` allreduces. Every step asserts both parameter values, the untouched
`excluded` field, `!(model.a ≈ model.b)` (the trigger that broke value-based selection),
`state.tree == ref_state` (rule, momentum state, frozen flags), cumulative collective
counts (`3 * step`), and cross-rank count equality.
