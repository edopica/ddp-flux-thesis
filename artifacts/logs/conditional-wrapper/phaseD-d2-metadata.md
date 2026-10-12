# Phase D2 evidence metadata — explicit occurrence mapping

Date: 2026-09-28

## Environment

| Item | Value |
|---|---|
| Host | omarchy (x86_64, local CPU/MPI) |
| Julia | 1.12.6 |
| Optimisers | 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`) |
| Flux worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Flux branch | `ddp/integration-m1-spike` |
| Flux HEAD | `f5b87975a4c62ce254736625b0a80015b6a1adb6` (unchanged; D1+D2 changes uncommitted) |
| MPI launcher | `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec` |

## Working-tree state per run

| Run | `src/distributed/public_api.jl` | `test/ext_distributed/conditional_distributedtest.jl` |
|---|---|---|
| RED (initial fixture) | unchanged (D1 working-tree changes only) | modified (D2 testsets added) |
| RED (type-preserving) | temporarily reverted to the D1-only state | modified (typed `IsoVec` field + `convert`, `isbits` assertions) |
| GREEN | modified (occurrence plan, restored byte-identical from a saved copy) | modified (typed `IsoVec` field + `convert`, `isbits` assertions) |

Final uncommitted diff (D1+D2): 2 files, +355/−27
(`src` +63/−27, test +292/−0).

## Intended defect

`_walk!` allocates one buffer per unique mutable parameter and one per isbits
occurrence. `_build!` consumed buffers with a counter that incremented for every
numeric occurrence. A tied mutable parameter contributes two occurrences but one
buffer, so the counter ran ahead: the following isbits parameter `c` was mapped to
the distinct mutable parameter `q`'s buffer (wrong global-mean gradient and wrong Adam
state), and the actual `c` buffer was reduced but never emitted. Equal-valued isbits
occurrences must stay independent (identity caching would collide them).

The fixture is type-preserving: `c::IsoVec{4}` and `IsoVec` defines the
`convert(::Type{IsoVec{N}}, ::AbstractVector)` used on reconstruction, so `c` stays
isbits on all three steps instead of becoming a `Vector` after the first update.
`@test isbits(model.c)` runs before steps 2 and 3 and after step 3, so the later
steps also exercise the isbits path.

## RED (before production changes)

### Initial fixture

Command (from `flux-integration`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

- Exit code: 1
- Wall: 30 s
- Per rank: control 5/5; D1 10/10; guard 11/11; D2 **30 passed / 5 failed / 35**.
- All 5 failures on `c` (test lines 276, 289, 309, 332, 340).
- Log: `phaseD-d2-RED-mpi2-2026-09-28.log`

### Type-preserving fixture (strengthened rerun)

The production source was temporarily reverted to the D1-only state (only the D2
hunks removed), verified by diff against the saved fixed file; the test file carries
the typed field and `convert`.

- Exit code: 1
- Wall: 40 s
- Per rank: control 5/5; D1 10/10; guard 11/11; D2 **33 passed / 5 failed / 38**.
- All 5 failures on `c` (test lines 285, 298, 319, 343, 352): `model.c` diverges from
  the reference after steps 1-3 and Adam state for `c` after steps 1 and 3; the three
  `isbits(model.c)` assertions pass, so the later steps are known to exercise the
  isbits path. Tied `p`/`p2` and `q` assertions pass.
- Log: `phaseD-d2-RED-mpi2-typepreserving-2026-09-28.log`
- One discarded attempt: the first helper used the reserved word `local` as a vararg
  name (`ParseError` before any test ran); renamed to `gs`. Not counted as RED.

## Production fix

`src/distributed/public_api.jl` (uncommitted, D2 part):

- New `_SyncOccurrence(buffer_index::Int, emit::Bool)`.
- `_walk!` pushes one plan entry per numeric occurrence: `emit=true` for the first
  occurrence of a mutable parameter and for each isbits occurrence, `emit=false` for
  later tied occurrences.
- `_build!` consumes plan entries through a cursor instead of inferring buffer
  indices from an occurrence counter; `seen`, `counter`, and `first_seen` are gone
  from the rebuild pass.
- `_sync_gradients` asserts complete plan consumption
  (`cursor[] == length(plan)`), otherwise throws `ArgumentError`.
- No change to collective order, presence protocol, or buffer allocation.

`test/ext_distributed/conditional_distributedtest.jl` (uncommitted, D2 part):
`MixedTiedIsbits(p, p2, c, q)` with `c::IsoVec{4}` and the type-preserving conversion,
explicit local gradients, three steps (both tied occurrences present; first absent;
second absent), native Adam reference, Adam state comparison, recording-backend
protocol checks (1 presence + 3 gradient reductions), plan-order buffer values,
cross-rank count equality, plus the equal-valued isbits guard testset.

## GREEN (after the fix, type-preserving fixture)

| Run | Command | Exit | Wall | Result (per rank / total) |
|---|---|---|---|---|
| Focused conditional, 2 ranks | same as RED | 0 | 40 s (one recompile from restoring the source) | control 5/5; D1 10/10; guard 11/11; D2 38/38 |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 10 s | 6/6 |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 59 s | `Distributed \| 6 6 57.4s` (6 child files) |
| Ambiguities | `julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println(pkgversion(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a))'` | 0 | — | Optimisers 0.4.9; 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean (no output) |

Logs:

- `phaseD-d2-GREEN-mpi2-typepreserving-2026-09-28.log`
- `phaseD-d2-GREEN-optimizer-mpi2-2026-09-28.log`
- `phaseD-d2-GREEN-suite-nprocs2-typepreserving-2026-09-28.log`
- `phaseD-d2-ambiguities-2026-09-28.log`
- `phaseD-d2-diff-check-2026-09-28.log`

The first-fixture GREEN logs (`phaseD-d2-GREEN-mpi2-2026-09-28.log`,
`phaseD-d2-GREEN-suite-nprocs2-2026-09-28.log`) are retained as the initial evidence
for the unstrengthened fixture.

## Scope notes

- D2 only. CPU presence metadata (D3), legacy tuple `trainable` (D4), and the D5
  regression gate remain open.
- Four-rank execution remains Phase F. No GPU/NCCL, no four-rank run, no detector
  update (Phase G), no Phase E test battery, no commit.
- The equal-valued isbits guard passed before and after the fix by design: it is a
  regression guard, not a RED case.
- These are real 2-rank MPI runs; no mock or simulated presence results are involved.
