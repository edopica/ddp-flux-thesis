# Flux.jl DDP Thesis Dashboard

Current checkpoint: C3 - Model broadcast and parameter verification (Done)  
Active Flux branch: `ddp/sync-model`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux upstream commit: `57e29baf48edf50c0dd4fc9f027a8900ce3cef66`  
Last update: `2026-07-11`  
Last reproducible command: `make verify-sync`  
Current blocker: `none`  
Next action: Implement C4 — Distributed data sharding

## Goal

Stabilize, test, document, and evaluate Distributed Data Parallel training support in Flux.jl. The immediate objective is not performance work; the immediate objective is correctness, reproducibility, and understanding the current implementation.

## Repository layout

- `../Flux.jl`: local fork of Flux.jl used for source inspection and later implementation work.
- `.`: thesis-control workspace for progress tracking, scripts, audit notes, artifacts, logs, and agent context.
  - `scripts/setup/`: Environment and auditing utilities.
  - `scripts/checks/`: MPI health and smoke tests.
  - `scripts/remote/`: Cluster deployment and synchronization.
  - `scripts/reference/`: C1 single-process deterministic baseline.
  - `scripts/launch/`: C2 distributed launch utilities.
  - `scripts/sync/`: C3 model broadcast verification.

## Progress

| ID | Status | Branch | Deliverable | Pass condition | Evidence |
|---|---|---|---|---|---|
| C0 | Done | `ddp/audit` | `openwiki/audits/000-current-flux-distributed.md` | One-page audit of API, missing pieces, risky areas, and chosen baseline path | `artifacts/logs/audit_flux_distributed_20260702_172217.txt` |
| C1 | Done | `ddp/reference-loop` | Deterministic single-process reference loop | Fixed loss, gradients, and updates stored as baseline tests | `artifacts/baselines/reference_loop_baseline.jld2` |
| C2 | Done | `ddp/launch` | Distributed launch skeleton | 2-process and 4-process launches complete without deadlock | `openwiki/devlog/20260709-c2-launch.md` |
| C3 | Done | `ddp/sync-model` | Model broadcast and parameter verification | All ranks start from identical parameters | `openwiki/devlog/20260711-c3-sync-model.md` |
| C4 | Not started | `ddp/data` | Distributed data sharding | Dataset coverage and duplication policy are documented and tested | - |
| C5 | Not started | `ddp/grad-sync` | Gradient synchronization minimal case | DDP gradient matches single-process global-batch gradient within tolerance | - |
| C6 | Not started | `ddp/optimizer` | Optimisers.jl integration | Parameters remain synchronized after multiple updates | - |
| C7 | Not started | `ddp/docs-examples` | End-to-end two-GPU example | One documented command reproduces training on 2 GPUs | - |
| C8 | Not started | `ddp/tests` | Correctness battery | Tests catch known failure modes and pass locally or in hardware-enabled CI | - |
| C9 | Not started | `ddp/perf` | Profiling and bottleneck report | Timeline plus bottleneck analysis exists | - |
| C10 | Not started | `ddp/perf` | Performance improvement pass | Throughput improves or bottleneck is explained with evidence | - |
| C11 | Not started | `ddp/docs-examples` | Documentation and thesis-ready examples | New user can reproduce examples and understand limitations | - |
| C12 | Not started | `final/evaluation` | Final evaluation package | Correctness, throughput, speedup, memory, limitations, and future work are reported | - |

## C0 goals

- [x] Verify local Flux checkout and remotes.
- [x] Create local Julia environment that develops Flux from the local path.
- [x] Install `mpiexecjl`.
- [x] Run CPU/MPI smoke test.
- [x] Audit `src/distributed`, distributed extensions, tests, and docs.
- [x] Update `openwiki/context/current.md`.
- [x] Write first devlog entry.

## Cluster & MPI Guardrails

When launching on a Slurm cluster, it's crucial to match the launcher's PMI version with the MPI library's expected PMI protocol. 
Flux.jl DDP is tested exclusively with Julia's default `MPICH_jll`, which uses the **PMI2** wire protocol.
If you use a system MPI (`JULIA_MPI_BINARY=system`), you must match your `srun` configuration accordingly. 

**Guardrails implemented:**
- **Initialization checks**: `DistributedUtils.initialize(MPIBackend)` now inspects `ENV` for `PMIX_RANK` and `OMPI_COMM_WORLD_RANK`, warning or throwing errors if unsupported environments are detected.
- **Health Check**: Run `make health-check` to isolate MPI/PMI mismatches before attempting to train. This script prints the MPI version, PMI environment variables, and verifies process synchronization.
- **Launch Defaults**: Our `make` targets (`make launch-2`, `make launch-4`, `make health-check`) and remote scripts default to `srun --mpi=pmi2` within Slurm allocations via the `SLURM_MPI_TYPE` environment variable.

## Notes for future checkpoints

- **MLUtils DataLoader overhaul** (2026-07): The MLUtils DataLoader now supports `parallel=true` (multithreading) and `num_workers=N` (multi-process, PyTorch-style). Relevant for C4 (distributed data sharding) and C7+ (end-to-end examples). Source: https://github.com/JuliaML/MLUtils.jl
- **HuggingFaceDatasets.jl** is the recommended path for loading real datasets. Use as reference when moving beyond synthetic data in later checkpoints. Source: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl

## Reproducible commands

Include these commands, with `<FLUX_REPO_PATH>` replaced by the actual path:

```bash
make env
make install-mpiexec
make check
make health-check
make smoke-cpu
make audit
make reference
make reference-verify
make launch-2
make launch-4
make sync-model
make verify-sync
```

## C3 goals

- [x] Broadcast model parameters from rank 0 to all ranks via `synchronize!!`.
- [x] Verify all ranks hold bit-identical parameters (max deviation = 0.0).
- [x] Verify rank 0's model is unchanged after broadcast.
- [x] Broadcast optimizer state via `synchronize!!`.
- [x] Test with a larger model (3-layer MLP, 9729 params).
- [x] Verify synced model matches rank 0's original seed-42 build.
- [x] Update devlog and context.

