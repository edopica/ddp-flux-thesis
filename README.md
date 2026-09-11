# Flux.jl DDP Thesis Dashboard

Current checkpoint: PR #2694 review fixes committed and the final OpenMPI topology-emulation gate passed. The branch is not pushed.
Active Flux branch: `ddp/pr2694-salvage` in the clean `flux-pr2694-salvage` worktree at `c64f695f857a4248846163b0455b1a002978f869`.
Archived research branch: `archive/pr2694-unused-gradient-spike` at `3370b910`. Dirty `master` at `b633cdc7` remains frozen.
Flux repo paths: `/home/kurapica/Projects/ddp_flux/ddp_flux` (C8 dirty work, frozen) and `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` (PR #2694 salvage worktree, work only here)
Flux base commit: `b633cdc7` (dirty master, C8.2 WIP, snapshot `archive/dirty-master-c8.2-wip` at `b0034f55`)
Base upstream commit: `404ff37d` (upstream/master and PR #2694 merge base)
PR #2694 head: `08f468e3`. Candidate integration base: `fa3bf228`. Salvage head: `c64f695f`. Parent review-fix commit: `15ca2d9c`.
The salvage head descends from `fa3bf228`. A push to `ddp/upstream-pr` is fast-forward only.
Preserved branches: `upsteam-pr` (old PR head `bc40df25`), `archive/dirty-master-c8.2-wip`, `archive/pr2694-unused-gradient-spike`
Last update: `2026-09-11`
Last reproducible command: exact workflow `timeout=900` suite reruns in rootless Podman (2 ranks 5/5, 4 ranks 5/5, plain MPI preferences restored)
Current blocker: none. Final diff/status review and approval are required before push. The full main-suite opt-in route remains unverified end to end locally; routing checks cover test selection, and dedicated CI covers the direct runner.
Next action: Review the final diff and status. When approved, fast-forward push to `ddp/upstream-pr`. Then monitor live CI.

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
| M0 | Done | `ddp/integration-m1-spike` | ADR-0006 integration baseline: isolated worktree, metadata, port inventory, checklist | Dirty master preserved identical (verified), worktree clean at `fa3bf228`, scope approved | `wiki/devlog/2026-09-08-m0-m1-integration-spike.md` |
| M1 | Done (M1.1 corrective pass) | `ddp/integration-m1-spike` | Automatic unused-gradient architecture: `DistributedOptimizerState` wrapper @ `3370b910` (unpushed). M1 gate reopened by 2026-09-08 review (7 blocking findings); M1.1 is the corrective spike | 6/6 ext_distributed MPI suite at `JULIA_MPI_TEST_NPROCS=2` and `4` (direct runner) | `artifacts/logs/m1.1/`; `wiki/devlog/2026-09-08-m1.1-corrective-spike.md` |
| PS1 | Done | `ddp/pr2694-salvage` | Salvage scope reset: drop unused-parameter API/test/docs + NCCL CI placeholder from PR #2694 | Pure-deletion commit `5c6ea561`, 6 files / 267 lines | `wiki/devlog/2026-09-09-pr2694-salvage-phases-1-2.md` |
| PS2 | Done | `ddp/pr2694-salvage` | PMI guardrail narrowed to the known unsafe MPICH/PMIx case, tests-first | Checker tests RED `af728837` → implementation GREEN `cd8de233`, 169/169 | `wiki/devlog/2026-09-09-pr2694-salvage-phase3a/b-*.md` |
| PS3 | Done | `ddp/pr2694-salvage` | Complete cyclic data padding: N==0 ArgumentError, `cld`, cyclic `mod1` indices, docstring | Matrix RED `e64165d3` → GREEN `4eb5708e`; 45/45 per rank at `-n 2` and `-n 4`, EXIT 0 | `artifacts/logs/pr2694-salvage/phase4-data-GREEN-*.log`; `wiki/devlog/2026-09-09-pr2694-salvage-phase4-cyclic-data-padding.md` |
| PS4 | Done | `ddp/pr2694-salvage` | Retain & polish Carlo's test changes: rename `mpi_edge` to "MPI DDP Correctness (4 Ranks)", drop unused `FLUX_TEST_DISTRIBUTED_MPI` from both MPI jobs, rephrase stream-inheritance comment (comment/CI-label only, no behavioral change) | Commit `23cae338` (1 on top of `4eb5708e`), worktree clean; independent verifier APPROVE on 10-point checklist; `git diff --check` clean, YAML parses, `Meta.parseall` ok | `wiki/devlog/2026-09-09-pr2694-salvage-phase5-polish-carlo-tests.md` |
| PS5 | Done | `ddp/pr2694-salvage` | Phase 6 docs & consistency sweep: `gpu.md` launcher order + equal-length/cyclic sharding wording; `NEWS.md` exactly two user-facing bullets with `#2694` links; `AbstractFluxDistributedBacked` typo; whitespace/final-newline hygiene | Committed as part of `caf47145` (8 files, `+23/-19`); independent verifier APPROVE; `git diff --check` clean; no stale terms; no child test reads `ARGS` | `wiki/devlog/2026-09-10-pr2694-salvage-phase6-docs-consistency.md` |
| PS6 | Done (all gates PASS) | `ddp/pr2694-salvage` | Phase 7 verification gates for PR #2694 salvage | Gates 1-6 PASS (scope clean; guard 169/169; MPI suite 5/5 at nprocs 2 and 4; gate 4 OpenMPI topology emulation PASS via rootless podman, 2 ranks 5/5 + 4 ranks 5/5, OpenMPI 4.1.6; docs build exit 0; CI final review); commit `caf47145`, unpushed | `wiki/devlog/2026-09-10-pr2694-salvage-phase7-verification.md`; `artifacts/logs/pr2694-salvage/phase7/`; `temp/docker_ci_fix/podman/podman-phase7-SUMMARY.md` |
| PS7 | Done (committed, not pushed) | `ddp/pr2694-salvage` | Phases 2-7 code-review fixes: NCCL `force` guard bypass, main-suite distributed opt-in routing (sole `ext_distributed` entry point), training-bias padding wording, plus in-scope corrections (internal checker rename, `"MPIwrapper"`, `MPI.Initialized()` reuse, dead-sentinel removal, watchdog 1200→900, whitespace) | Review fixes committed as `15ca2d9c`. Final `c64f695f` corrects only stale routing-test comments. Worktree clean. Repository-root `Manifest.toml` unchanged. Rootless-Podman topology emulation passed 5/5 at 2 and 4 ranks in the original gate and exact `timeout=900` reruns. | `wiki/devlog/2026-09-10-pr2694-review-fixes.md`, `artifacts/logs/pr2694-salvage/review-fixes/final/`, `temp/docker_ci_fix/podman/podman-c64f695f-*.log`, `temp/docker_ci_fix/podman/podman-c64f695f-timeout900-*.log` |
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
