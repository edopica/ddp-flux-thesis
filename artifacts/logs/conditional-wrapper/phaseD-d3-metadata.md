# Phase D3 evidence metadata — CPU presence metadata

Date: 2026-09-29 (runs started 2026-09-28 23:5x, finished 2026-09-29 00:1x local)

## Environment

| Item | Value |
|---|---|
| Host | omarchy (x86_64, local CPU/MPI) |
| Julia | 1.12.6 |
| Optimisers | 0.4.9 (`/home/kurapica/.julia/packages/Optimisers/tMaaf/src/Optimisers.jl`) |
| Flux worktree | `/home/kurapica/Projects/ddp_flux/flux-integration` |
| Flux branch | `ddp/integration-m1-spike` |
| Flux HEAD | `f5b87975a4c62ce254736625b0a80015b6a1adb6` (unchanged; D1+D2+D3 changes uncommitted) |
| MPI launcher | `/home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec` |
| Rank count | 2 (real MPI); no four-rank run (Phase F) and no GPU/NCCL run |

## Working-tree state per run

| Run | `src/distributed/public_api.jl` | `test/ext_distributed/conditional_distributedtest.jl` |
|---|---|---|
| RED | D1+D2 working-tree state, old presence allocation (`similar(parameter_buffer, Float32, np)`) | modified (D3 testsets added) |
| GREEN | modified (presence vector allocated with `zeros(Float32, np)`; `_SyncBuf.p` removed) | modified (same D3 testsets) |

Final uncommitted diff (D1+D2+D3): 2 files, +480/−34
(`src` +74/−34, test +406/−0). D3 alone adds +114 test lines and reworks the
presence allocation in `_sync_gradients`.

## Intended defect

`_sync_gradients` derived the presence-flag vector from a parameter buffer:

```julia
anchor = findfirst(b -> !isbits(b.p), bufs)
pres = anchor === nothing ? zeros(Float32, np) : similar(bufs[anchor].buf, Float32, np)
```

With a GPU parameter this allocates the flags on the parameter device. The scalar
flag writes (`pres[i] = ...`) and the globally-present decision then touch device
memory from host code, rejected by CUDA scalar-indexing checks, and force a device
synchronization for control flow. Presence metadata is host metadata and must be a
`Vector{Float32}`; parameter gradient buffers must keep their own device.

## RED (before production changes)

Command (from `flux-integration`):

```bash
FLUX_TEST_DISTRIBUTED_BACKEND=mpi \
  /home/kurapica/.julia/artifacts/8bd6881e43b97a8d20af0db9dbda166281c02323/bin/mpiexec \
  -n 2 julia --startup-file=no --project=test \
  test/ext_distributed/conditional_distributedtest.jl
```

- Exit code: 1
- Wall: 29 s
- Per rank: control 5/5; D1 10/10; equal-valued guard 11/11; D2 38/38;
  D3 `CPU presence metadata for device-like parameters` **14 passed / 2 failed / 16**.
- The 2 failures are exactly the intended defect, not fixture or dependency errors:
  - `conditional_distributedtest.jl:445`: `DEVICE_SCALAR_ACCESSES[] == 0`
    evaluated `4 == 0` (host scalar reads/writes of device presence metadata);
  - `conditional_distributedtest.jl:455`: `presence.buftype === Vector{Float32}`
    evaluated `DeviceLikeArray{Float32, 1} === Vector{Float32}` (device-derived
    metadata allocation).
- Log: `phaseD-d3-RED-mpi2-2026-09-29.log`

The fixture (`DeviceLikeArray`, `DeviceMetadataBackend`, testset
`CPU presence metadata for device-like parameters`) is described in the devlog. It
calls `DistributedUtils._sync_gradients` itself, so the RED observes the real
allocation expression, not a copy of it.

## Production fix

`src/distributed/public_api.jl` (uncommitted, D3 part):

- `_SyncBuf` loses its `p::P` field. The parameter reference existed only to anchor
  the old device-derived presence allocation; buffers and `present` are the only
  state the synchronization walk needs.
- `_sync_gradients` allocates the flags with `pres = zeros(Float32, np)` instead of
  `similar(bufs[anchor].buf, Float32, np)`. The `anchor` lookup is deleted.
- The presence comment now documents the split: presence metadata is host CPU/flags
  reduced with `avg` (NCCL falls back to MPI for CPU buffers); parameter gradients
  stay on their own device and use the selected backend for their allreduces.
- One presence allreduce with `avg` and the `> 0` threshold are unchanged.

## GREEN (after the fix)

| Run | Command | Exit | Wall | Result (per rank / total) |
|---|---|---|---|---|
| Focused conditional, 2 ranks | same as RED | 0 | 44 s (one local Flux + FluxMPIExt recompile after the source change) | control 5/5; D1 10/10; guard 11/11; D2 38/38; D3 16/16 |
| Focused optimizer, 2 ranks | `FLUX_TEST_DISTRIBUTED_BACKEND=mpi <mpiexec> -n 2 julia --startup-file=no --project=test test/ext_distributed/optimizer_distributedtest.jl` | 0 | 12 s | composition guard 6/6 |
| Full dedicated suite, 2 ranks | `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` | 0 | 66 s | `Distributed \| 6 6 1m03.6s` (6 child files) |
| Ambiguities | `julia --startup-file=no --project=test -e 'using Flux, Test, Optimisers; println("Optimisers=", pkgversion(Optimisers)); a = Test.detect_ambiguities(Flux.DistributedUtils; recursive=false); println("ambiguities=", length(a)); foreach(println, a)'` | 0 | — | Optimisers 0.4.9; 0 ambiguities |
| Whitespace | `git diff --check` | 0 | — | clean (no output) |

Logs:

- `phaseD-d3-GREEN-mpi2-2026-09-29.log`
- `phaseD-d3-GREEN-optimizer-mpi2-2026-09-29.log`
- `phaseD-d3-GREEN-suite-nprocs2-2026-09-29.log`
- `phaseD-d3-ambiguities-2026-09-29.log`
- `phaseD-d3-diff-check-2026-09-29.log`

## Simulated and real evidence

- The `0.25f0` presence fraction is **simulated** by `DeviceMetadataBackend`: one
  active rank out of four, with no collective data movement for the flag vector. It
  exercises the `> 0` threshold (`> 0.5` would produce zero gradient collectives and
  fail `length(mb.log) == 3`). It does not replace the four-rank weighting gate in
  Phase F and does not prove native CUDA or NCCL behavior.
- Gradient reductions in the fixture are **real** 2-rank MPI allreduces (delegated
  through the mock to the configured backend on the raw storage) and are compared
  against explicit `avg` reference reductions.
- `DeviceLikeArray` is a CPU array that only emulates device placement semantics:
  `similar` preserves the wrapper type, host scalar access is counted, and
  device-side copies (`copyto!`, `fill!`) are defined so buffer preparation does not
  count as host access.

## Scope notes

- D3 only. Legacy tuple `trainable` (D4) and the D5 regression gate remain open.
- Four-rank execution remains Phase F. No GPU/NCCL run, no detector update
  (Phase G), no Phase E test battery, no HPC run, no commit.
- `_SyncBuf.p` removal is part of the D3 correction: the field had no remaining use
  once presence metadata stopped being derived from a parameter buffer.
