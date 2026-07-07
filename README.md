# Flux.jl DDP Thesis Dashboard

Current checkpoint: C1 - Deterministic single-process reference loop (Done)  
Active Flux branch: `ddp/reference-loop`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux upstream commit: `57e29baf48edf50c0dd4fc9f027a8900ce3cef66`  
Last update: `2026-07-07`  
Last reproducible command: `make reference-verify`  
Current blocker: `none`  
Next action: Implement C2 — distributed launch skeleton

## Goal

Stabilize, test, document, and evaluate Distributed Data Parallel training support in Flux.jl. The immediate objective is not performance work; the immediate objective is correctness, reproducibility, and understanding the current implementation.

## Repository layout

- `../Flux.jl`: local fork of Flux.jl used for source inspection and later implementation work.
- `.`: thesis-control workspace for progress tracking, scripts, audit notes, artifacts, logs, and agent context.

## Progress

| ID | Status | Branch | Deliverable | Pass condition | Evidence |
|---|---|---|---|---|---|
| C0 | Done | `ddp/audit` | `openwiki/audits/000-current-flux-distributed.md` | One-page audit of API, missing pieces, risky areas, and chosen baseline path | `artifacts/logs/audit_flux_distributed_20260702_172217.txt` |
| C1 | Done | `ddp/reference-loop` | Deterministic single-process reference loop | Fixed loss, gradients, and updates stored as baseline tests | `artifacts/baselines/reference_loop_baseline.jld2` |
| C2 | Not started | `ddp/launch` | Distributed launch skeleton | 2-process and 4-process launches complete without deadlock | - |
| C3 | Not started | `ddp/sync-model` | Model broadcast and parameter verification | All ranks start from identical parameters | - |
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

## Notes for future checkpoints

- **MLUtils DataLoader overhaul** (2026-07): The MLUtils DataLoader now supports `parallel=true` (multithreading) and `num_workers=N` (multi-process, PyTorch-style). Relevant for C4 (distributed data sharding) and C7+ (end-to-end examples). Source: https://github.com/JuliaML/MLUtils.jl
- **HuggingFaceDatasets.jl** is the recommended path for loading real datasets. Use as reference when moving beyond synthetic data in later checkpoints. Source: https://github.com/JuliaGenAI/HuggingFaceDatasets.jl

## Reproducible commands

Include these commands, with `<FLUX_REPO_PATH>` replaced by the actual path:

```bash
make env
make install-mpiexec
make check
make smoke-cpu
make audit
make reference
make reference-verify
```

