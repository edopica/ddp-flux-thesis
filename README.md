# Flux.jl DDP Thesis Dashboard

Current checkpoint: C8 - Correctness battery (IN PROGRESS)
Active Flux branch: `master`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux base commit: `d583411f` (merge of `ddp/docs-examples` + `upsteam-pr` into `master`)  
Base upstream commit: `404ff37d` (upstream/master, fetched 2026-08-17)  
PR #2694 head: `bc40df25` (rebased `upsteam-pr`, 2 commits, mergeable)  
Preserved branches: `upsteam-pr` (= PR head)  
Last update: `2026-08-17`  
Last reproducible command: `FLUX_REPO_PATH=../ddp_flux make c8-mpi-4` (deathstar, Julia 1.12, 4 MPI ranks, 7/7 PASS)  
Current blocker: `none — CI on PR #2694 running`  
Next action: monitor PR #2694 CI, then address review comments

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
| C0 | Done | `ddp/audit` | `wiki/audits/000-current-flux-distributed.md` | One-page audit of API, missing pieces, risky areas, and chosen baseline path | `artifacts/logs/audit_flux_distributed_20260702_172217.txt` |
| C1 | Done | `ddp/reference-loop` | Deterministic single-process reference loop | Fixed loss, gradients, and updates stored as baseline tests | `artifacts/baselines/reference_loop_baseline.jld2` |
| C2 | Done | `ddp/launch` | Distributed launch skeleton | 2-process and 4-process launches complete without deadlock | `wiki/devlog/20260709-c2-launch.md` |
| C3 | Done | `ddp/sync-model` | Model broadcast and parameter verification | All ranks start from identical parameters | `wiki/devlog/20260711-c3-sync-model.md` |
| C4 | Done | `ddp/data` | Distributed data sharding | Dataset coverage and duplication policy are documented and tested | `wiki/devlog/20260713-c4-data-sharding.md` |
| C5 | Done | `ddp/grad-sync` | Gradient synchronization minimal case | DDP gradient matches single-process global-batch gradient within tolerance | `scripts/sync/verify_gradients.jl` |
| C6 | Done | `ddp/optimizer` | Optimisers.jl integration | Parameters remain synchronized after multiple updates | `scripts/sync/verify_gradients.jl` |
| C6b | Done | `ddp/upstream-pr` | Upstream PR preparation | Commits polished, tests added/updated, tests passing | HPC: 20/20 tests pass, `wiki/devlog/20260716-c6b-hpc-tests.md` |
| C7 | Done | `ddp/docs-examples` | End-to-end two-GPU example | One documented command reproduces training on 2 GPUs | `make example-ddp` (HPC test devlog) |
| C8 | In progress | `master` | Correctness battery | Tests catch known failure modes and pass locally or in hardware-enabled CI | `make c8-mpi` (deathstar, 6/6 PASS) |
| C9 | Not started | `ddp/perf` | Profiling and bottleneck report | Timeline plus bottleneck analysis exists | - |
| C10 | Not started | `ddp/perf` | Performance improvement pass | Throughput improves or bottleneck is explained with evidence | - |
| C11 | Not started | `ddp/docs-examples` | Documentation and thesis-ready examples | New user can reproduce examples and understand limitations | - |
| C12 | Not started | `final/evaluation` | Final evaluation package | Correctness, throughput, speedup, memory, limitations, and future work are reported | - |

## Cluster & MPI Guardrails

When launching on a Slurm cluster, it's crucial to match the launcher's PMI version with the MPI library's expected PMI protocol. 
Flux.jl DDP is tested exclusively with Julia's default `MPICH_jll`, which uses the **PMI2** wire protocol.
If you use a system MPI (`JULIA_MPI_BINARY=system`), you must match your `srun` configuration accordingly. 

**Precompilation guardrail (Julia 1.12, measured 2026-09-03):** precompilation runs only on compute nodes. Login-node compiles target the wrong CPU (`graniterapids` vs `icelake-server`) and leave tests cold anyway. After every `sync_code.sh` or Flux source change, run `scripts/remote/precompile.sh hpc` (precompiles thesis env + Flux test env under `srun`). `make env` no longer precompiles and `JULIA_PKG_PRECOMPILE_AUTO=0` keeps the login node compile-free during Pkg operations; a stale cache will still recompile silently at first load, so never skip the precompile step after a sync. Details and measurements: `wiki/context/remote-nodes.md`, `wiki/devlog/2026-09-03-precompile-experiment.md`, ADR-0005.

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
make verify-data
make verify-gradients
```
