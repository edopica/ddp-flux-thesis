# 2026-09-29 - Conditional-graph wrapper Phase D3: CPU presence metadata

## Goal

Phase D3 of `temp/conditional_graph_wrapper_implementation_plan.md`: keep the
presence metadata of `_sync_gradients` on the host CPU, independent of the
parameter device, and prove the split with a device-like fixture that needs no GPU
hardware.

## Defect

`_sync_gradients` derived the presence-flag vector from a parameter buffer:

```julia
anchor = findfirst(b -> !isbits(b.p), bufs)
pres = anchor === nothing ? zeros(Float32, np) : similar(bufs[anchor].buf, Float32, np)
for i in 1:np
    pres[i] = bufs[i].present ? 1f0 : 0f0
end
```

With a GPU parameter, `similar(parameter_buffer, Float32, np)` allocates the flags
on the parameter device. The scalar flag writes and the globally-present decision
then touch device memory from host code. CUDA scalar-indexing checks reject that
under `allowscalar(false)`, and the boolean decision would need a device
synchronization. The plan (section 9.3) already listed this as an M1.1 defect.

## Fix

`src/distributed/public_api.jl` (uncommitted):

- `_SyncBuf` loses its `p::P` field; the parameter reference existed only to anchor
  the device-derived allocation. `buf` and `present` remain.
- `_sync_gradients` now allocates `pres = zeros(Float32, np)` unconditionally; the
  `anchor` lookup is deleted.
- The comment above the protocol documents the required split: presence metadata is
  host CPU metadata reduced once with `avg` (the NCCL backend already falls back to
  MPI for CPU buffers), while parameter gradient buffers keep their own device and
  the selected backend.
- The `> 0` threshold and the single `avg` presence reduction are unchanged.

`test/ext_distributed/conditional_distributedtest.jl` (uncommitted): new Phase D3
fixture and testset `CPU presence metadata for device-like parameters`:

- `DeviceLikeArray{T,N} <: AbstractArray` emulates device placement:
  `similar(x, T, dims)` preserves the wrapper type, so any metadata derived from a
  parameter buffer would also be device-like. `copyto!` and `fill!` operate on the
  raw storage (device-side copies, no host access); scalar `getindex`/`setindex!`
  increment `DEVICE_SCALAR_ACCESSES`, standing in for `CUDA.allowscalar(false)`
  raising.
- `DeviceMetadataBackend` records every collective `(buftype, eltype, ndims, size,
  avg)` and supplies a deterministic presence fraction `0.25f0` (one active rank
  out of four) instead of the rank-dependent reduction. Gradient reductions are
  delegated to the real backend on the raw storage, so values remain real 2-rank
  global means.
- The fixture calls `DistributedUtils._sync_gradients(mb, model, grads)` directly,
  so it observes the production allocation rather than a copy of the expression.
- Assertions: zero host scalar accesses; three collectives (1 presence + 2
  gradients); exactly one 1-D `Float32 avg` collective whose `buftype` is exactly
  `Vector{Float32}`; parameter collective buffers are `DeviceLikeArray{Float64,1}`;
  rebuilt gradients are device-like and equal explicit `avg` references; equal
  collective counts across ranks (`bcast!`).
- The `0.25f0` fraction makes the `> 0` threshold observable: with `> 0.5` neither
  parameter would be globally present, no gradient collective would run, and
  `length(mb.log) == 3` would fail.

## RED evidence

Run from `/home/kurapica/Projects/ddp_flux/flux-integration` against the D1+D2
working-tree source (old presence allocation):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

Exit 1, 29 s. Per rank: control 5/5, D1 10/10, equal-valued guard 11/11,
D2 38/38, D3 **14 passed / 2 failed / 16**. The two failures are exactly the
intended defect:

- line 445 `DEVICE_SCALAR_ACCESSES[] == 0` -> `4 == 0` (host scalar reads/writes of
  device presence metadata),
- line 455 `presence.buftype === Vector{Float32}` ->
  `DeviceLikeArray{Float32, 1} === Vector{Float32}` (device-derived allocation).

Log: `phaseD-d3-RED-mpi2-2026-09-29.log`.

## GREEN evidence

Environment: Julia 1.12.6, Optimisers 0.4.9
(`/home/kurapica/.julia/packages/Optimisers/tMaaf`), host omarchy. Flux HEAD remains
`f5b87975`; D1+D2+D3 changes uncommitted.

| Check | Command | Result |
|---|---|---|
| Focused conditional, 2 ranks | same as RED | exit 0, 44 s: control 5/5, D1 10/10, guard 11/11, D2 38/38, D3 16/16 per rank |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | exit 0, 12 s, 6/6 per rank |
| Full 2-rank suite | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | exit 0, 66 s, `Distributed \| 6 6 1m03.6s` |
| Ambiguities | `Test.detect_ambiguities(Flux.DistributedUtils; recursive=false)` | 0 (Optimisers 0.4.9) |
| Whitespace | `git diff --check` | clean (no output) |

## Scope notes

- D3 only. The `0.25f0` presence result is simulated by the mock backend and is
  labeled as such; the four-rank weighting gate remains Phase F. The fixture does
  not prove native CUDA or NCCL behavior.
- Legacy tuple `trainable` (D4) and the D5 regression gate remain open. No
  GPU/NCCL, detector, Bocconi, or main-suite runs.
- No commit. Tests and fix stay reviewable as separate changes in the worktree;
  commit/push only with explicit user authorization.

## Evidence

`artifacts/logs/conditional-wrapper/`:

- `phaseD-d3-metadata.md`,
- `phaseD-d3-RED-mpi2-2026-09-29.log`,
- `phaseD-d3-GREEN-mpi2-2026-09-29.log`,
- `phaseD-d3-GREEN-optimizer-mpi2-2026-09-29.log`,
- `phaseD-d3-GREEN-suite-nprocs2-2026-09-29.log`,
- `phaseD-d3-ambiguities-2026-09-29.log`,
- `phaseD-d3-diff-check-2026-09-29.log`.

## Independent D3 review (2026-09-29)

Verdict: no blocking code findings. The implementation meets the D3 requirements in the approved plan.

- Host allocation keeps both `pres` and the derived `present` decisions on the CPU.
- Parameter buffers still use `similar(x)`. Removal of `_SyncBuf.p` leaves no remaining references.
- The protocol retains one `avg` presence reduction, the `> 0` threshold, and ordered gradient reductions.
- The fixture calls the production `_sync_gradients` function and detects the old allocation defect.
- The stored RED log contains the two expected failures on each rank, without fixture errors.
- The stored full-suite GREEN log records six successful child files at two ranks.
- Source inspection supports the NCCL fallback comment (`ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:70-72`).

Independent commands, from `/home/kurapica/Projects/ddp_flux/flux-integration`:

```bash
time timeout 180s env FLUX_TEST_DISTRIBUTED_BACKEND=mpi /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec -n 2 julia --startup-file=no --project=test test/ext_distributed/conditional_distributedtest.jl
git diff --check && julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers=", pkgversion(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); @assert isempty(a)'
```

Both commands exited 0. The conditional run took 27.649 s.
Per-rank totals: control 5/5, D1 10/10, isbits guard 11/11, D2 38/38, D3 16/16.
Optimisers version: 0.4.9. Ambiguities: zero. The whitespace check produced no output.
These results came directly from the review tool output. The review did not rerun the full suite or the historical RED state.

The fixture proves the scoped allocation contract. Its presence fraction is simulated, and its gradient inputs are host arrays.
It does not establish native CUDA/NCCL safety or the four-rank weighting requirement.

One low-priority documentation finding remains: the plan header and wiki quickstart still show the pre-D3 checkpoint.
Flux source and tests remain unchanged by this review, at `f5b87975` with the existing D1+D2+D3 diff.

## Next action after review

Commit follow-up: the user authorized commits for the documentation and the reviewed Flux implementation.
Flux commit `cbdd3edc` contains D1-D3 (2 files, +480/−34). The Flux worktree is clean.
The plan header, wiki quickstart, README, and current context now identify D4 as the next phase.
The documentation finding is resolved. Earlier uncommitted-state descriptions record the state at test time.

Commit command (from `flux-integration`):

```bash
git add src/distributed/public_api.jl test/ext_distributed/conditional_distributedtest.jl && git diff --cached --check && git commit -m "Correct distributed gradient traversal and CPU presence metadata" && git status --short && git rev-parse --short HEAD
```

Result: exit 0. Commit `cbdd3edc`. No push.

Implement Phase D4: legacy tuple-returning `trainable` selection. Add a struct with
an old tuple `trainable` method and an excluded array field, plus equal-valued
distinct mutable fields to expose value-membership semantics; compare selected
leaves and updates against public native `Optimisers.setup` behavior, record the
legacy-selection RED result, then normalize legacy selection into model-field order
through public APIs and preserve the native setup warning. Four-rank gates remain
Phase F.
