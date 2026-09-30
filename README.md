# Flux.jl DDP Thesis Dashboard

Phase E review correction (2026-09-30): **both P2 gaps corrected; correction gate GREEN**. The device-adaptation test now uses Adam, transforms exactly the two nonzero state arrays, asserts adapted types, preserved values, backend identity, and source immutability, then updates through the adapted state against a native reference. The M1.1 API checks are restored (`Chain tuple-gradient diagnostic`, `Enzyme Duplicated model setup and update`, shared-gradient tied subcase, whole-wrapper `freeze!`/`thaw!`); the freeze test uses Adam with retained states and frozen-moment assertions. Two-rank GREEN: focused conditional **367/367 per rank** across 30 testsets, optimizer 6/6, full dedicated suite `Distributed | 6 6 2m02.6s`, 0 ambiguities, clean `git diff --check`. Next action: Phase F (two- and four-rank gates). See the Phase E devlog "Corrections applied" section and `phaseE-correction-*` logs.

Phase E (restore and improve the M1.1 test battery) complete and GREEN, committed at Flux `55171e65` (not pushed). `test/ext_distributed/conditional_distributedtest.jl` gained 15 testsets (+560 lines, tests only): the M1.1 real-AD and public-API cases with no stronger Phase D equivalent (multi-step averaging, Adam stateful reference, tied through real AD and transpose, isbits Fill, higher-order rejection, collective protocol, `Flux.train!`, nested model, `synchronize!!`, `adjust`/`adjust!`), the two Phase E items (stateful `freeze! and thaw! preserve wrapper state and collectives`, `Functor device adaptation keeps the backend`), and the restored M1.1 API checks (`Chain tuple-gradient diagnostic`, `Enzyme Duplicated model setup and update`). Two-rank GREEN: focused conditional 367/367 per rank across 30 testsets, focused optimizer 6/6, full dedicated suite `Distributed | 6 6 2m02.6s`, 0 ambiguities, clean `git diff --check`. All RED iterations were new-test defects; no production change. Next action: Phase F (two- and four-rank CPU/MPI gates; four-rank must prove weighting). See the Phase E devlog and metadata.

Independent review (2026-09-30): **D5 reopened for two P2 test gaps, then corrected**. The absent Adam parameter now takes prior active steps (nonzero moments) before becoming absent, and the caller-gradient tests use rank-distinct gradients with snapshots plus a genuinely read-only gradient. Targeted RED runs show each corrected testset fails under the defect it targets: an aliased communication buffer mutates the caller gradient to the global mean (snapshot fails), and an absent leaf fed a zero gradient changes both value and full Adam state. Final two-rank gate GREEN: focused conditional 232/232 per rank, optimizer 6/6, full dedicated suite 6/6, zero ambiguities, clean `git diff --check`. Next action: Phase E (restore the M1.1 battery).

Review update (2026-09-30): **Phase D5 complete after review corrections — committed at Flux `d6046103`**. Eight focused testsets pin the retained contracts; the gate found and fixed one real defect (multidimensional array containers were flattened to a vector; `CartesianIndices` now preserves shape/index order), and the review's two P2 gaps are closed (active-to-absent Adam snapshot with nonzero moments; rank-distinct caller gradients, per-rank snapshots, independent global mean, and a read-only gradient case). Final two-rank gate: focused conditional 232/232 per rank, focused optimizer 6/6, full dedicated suite 6/6, 0 ambiguities, `git diff --check` clean. Next action: Phase E (restore the M1.1 battery). See the D5 devlog.

Current checkpoint: Conditional-graph wrapper Phase E complete, review-corrected, and GREEN, committed at Flux `55171e65` (Phase D at `d6046103`: D1-D3 at `cbdd3edc`; not pushed). Phase E is tests-only, +560/−0. GREEN: 367/367 per rank across the focused conditional file (30 testsets), full 2-rank suite 6/6. Phase F (two- and four-rank gates) is next.
Active implementation branch: `ddp/integration-m1-spike` at `55171e65` in `/home/kurapica/Projects/ddp_flux/flux-integration` (Phase E committed, not pushed).
Archived research branch: `archive/pr2694-unused-gradient-spike` at `3370b910`. Dirty `master` at `b633cdc7` remains frozen.
Flux repo paths: `/home/kurapica/Projects/ddp_flux/ddp_flux` (C8 dirty work, frozen) and `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` (PR #2694 salvage worktree, work only here)
Flux base commit: `b633cdc7` (dirty master, C8.2 WIP, snapshot `archive/dirty-master-c8.2-wip` at `b0034f55`)
Base upstream commit: `404ff37d` (upstream/master and PR #2694 merge base)
PR #2694 head: `fee2c310`. Candidate integration base: `fa3bf228`. Windows-path fix commits: `69703025`, `fee2c310`.
The latest push fast-forwarded `ddp/upstream-pr` from `c64f695f` to `fee2c310`.
Preserved branches: `upsteam-pr` (old PR head `bc40df25`), `archive/dirty-master-c8.2-wip`, `archive/pr2694-unused-gradient-spike`
Last update: `2026-09-30`
Last reproducible command: `JULIA_MPI_TEST_NPROCS=2 julia --startup-file=no --project=test test/ext_distributed/runtests.jl mpi` (Phase E correction gate GREEN: `Distributed | 6 6 2m02.6s`, exit 0)
Current blocker: None for local wrapper development. Native NCCL work remains blocked until CPU/MPI correctness passes.
Next action: Phase F from `temp/conditional_graph_wrapper_implementation_plan.md` — run the CPU/MPI gates at two and four ranks; the four-rank analytical case must prove weighting (one rank branch A, three ranks branch B).

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
| CW-E-review | Reopened (two P2 gaps) | `ddp/integration-m1-spike` | Non-vacuous state adaptation and omitted M1.1 API checks | Adapt actual state arrays and update that state; restore missing diagnostic and delegation checks | Phase E devlog, independent review section |
| C0 | Done | `ddp/audit` | `wiki/audits/000-current-flux-distributed.md` | One-page audit of API, missing pieces, risky areas, and chosen baseline path | `artifacts/logs/audit_flux_distributed_20260702_172217.txt` |
| C1 | Done | `ddp/reference-loop` | Deterministic single-process reference loop | Fixed loss, gradients, and updates stored as baseline tests | `artifacts/baselines/reference_loop_baseline.jld2` |
| C2 | Done | `ddp/launch` | Distributed launch skeleton | 2-process and 4-process launches complete without deadlock | `wiki/devlog/20260709-c2-launch.md` |
| C3 | Done | `ddp/sync-model` | Model broadcast and parameter verification | All ranks start from identical parameters | `wiki/devlog/20260711-c3-sync-model.md` |
| C4 | Done | `ddp/data` | Distributed data sharding | Dataset coverage and duplication policy are documented and tested | `wiki/devlog/20260713-c4-data-sharding.md` |
| C5 | Done | `ddp/grad-sync` | Gradient synchronization minimal case | DDP gradient matches single-process global-batch gradient within tolerance | `scripts/sync/verify_gradients.jl` |
| C6 | Done | `ddp/optimizer` | Optimisers.jl integration | Parameters remain synchronized after multiple updates | `scripts/sync/verify_gradients.jl` |
| C6b | Done | `ddp/upstream-pr` | Upstream PR preparation | Commits polished, tests added/updated, tests passing | Bocconi: 20/20 tests pass, `wiki/devlog/20260716-c6b-hpc-tests.md` |
| C7 | Done | `ddp/docs-examples` | End-to-end two-GPU example | One documented command reproduces training on 2 GPUs | `make example-ddp` (Bocconi test devlog) |
| C8 | In progress | `master` | Correctness battery | Tests catch known failure modes and pass locally or in hardware-enabled CI | `make c8-mpi` (deathstar, 6/6 PASS) |
| M0 | Done | `ddp/integration-m1-spike` | ADR-0006 integration baseline: isolated worktree, metadata, port inventory, checklist | Dirty master preserved identical (verified), worktree clean at `fa3bf228`, scope approved | `wiki/devlog/2026-09-08-m0-m1-integration-spike.md` |
| M1 | Done (M1.1 corrective pass) | `ddp/integration-m1-spike` | Automatic unused-gradient architecture: `DistributedOptimizerState` wrapper @ `3370b910` (unpushed). M1 gate reopened by 2026-09-08 review (7 blocking findings); M1.1 is the corrective spike | 6/6 ext_distributed MPI suite at `JULIA_MPI_TEST_NPROCS=2` and `4` (direct runner) | `artifacts/logs/m1.1/`; `wiki/devlog/2026-09-08-m1.1-corrective-spike.md` |
| PS1 | Done | `ddp/pr2694-salvage` | Salvage scope reset: drop unused-parameter API/test/docs + NCCL CI placeholder from PR #2694 | Pure-deletion commit `5c6ea561`, 6 files / 267 lines | `wiki/devlog/2026-09-09-pr2694-salvage-phases-1-2.md` |
| PS2 | Done | `ddp/pr2694-salvage` | PMI guardrail narrowed to the known unsafe MPICH/PMIx case, tests-first | Checker tests RED `af728837` → implementation GREEN `cd8de233`, 169/169 | `wiki/devlog/2026-09-09-pr2694-salvage-phase3a/b-*.md` |
| PS3 | Done | `ddp/pr2694-salvage` | Complete cyclic data padding: N==0 ArgumentError, `cld`, cyclic `mod1` indices, docstring | Matrix RED `e64165d3` → GREEN `4eb5708e`; 45/45 per rank at `-n 2` and `-n 4`, EXIT 0 | `artifacts/logs/pr2694-salvage/phase4-data-GREEN-*.log`; `wiki/devlog/2026-09-09-pr2694-salvage-phase4-cyclic-data-padding.md` |
| PS4 | Done | `ddp/pr2694-salvage` | Retain & polish Carlo's test changes: rename `mpi_edge` to "MPI DDP Correctness (4 Ranks)", drop unused `FLUX_TEST_DISTRIBUTED_MPI` from both MPI jobs, rephrase stream-inheritance comment (comment/CI-label only, no behavioral change) | Commit `23cae338` (1 on top of `4eb5708e`), worktree clean; independent verifier APPROVE on 10-point checklist; `git diff --check` clean, YAML parses, `Meta.parseall` ok | `wiki/devlog/2026-09-09-pr2694-salvage-phase5-polish-carlo-tests.md` |
| PS5 | Done | `ddp/pr2694-salvage` | Phase 6 docs & consistency sweep: `gpu.md` launcher order + equal-length/cyclic sharding wording; `NEWS.md` exactly two user-facing bullets with `#2694` links; `AbstractFluxDistributedBacked` typo; whitespace/final-newline hygiene | Committed as part of `caf47145` (8 files, `+23/-19`); independent verifier APPROVE; `git diff --check` clean; no stale terms; no child test reads `ARGS` | `wiki/devlog/2026-09-10-pr2694-salvage-phase6-docs-consistency.md` |
| PS6 | Done (all gates PASS) | `ddp/pr2694-salvage` | Phase 7 verification gates for PR #2694 salvage | Gates 1-6 PASS (scope clean; guard 169/169; MPI suite 5/5 at nprocs 2 and 4; gate 4 OpenMPI topology emulation PASS via rootless podman, 2 ranks 5/5 + 4 ranks 5/5, OpenMPI 4.1.6; docs build exit 0; CI final review); commit `caf47145`, unpushed | `wiki/devlog/2026-09-10-pr2694-salvage-phase7-verification.md`; `artifacts/logs/pr2694-salvage/phase7/`; `temp/docker_ci_fix/podman/podman-phase7-SUMMARY.md` |
| PS7 | Done (pushed) | `ddp/pr2694-salvage` | Phases 2-7 code-review fixes: NCCL `force` guard bypass, main-suite distributed opt-in routing (sole `ext_distributed` entry point), training-bias padding wording, plus in-scope corrections (internal checker rename, `"MPIwrapper"`, `MPI.Initialized()` reuse, dead-sentinel removal, watchdog 1200→900, whitespace) | Review fixes committed as `15ca2d9c`. Final `c64f695f` corrects only stale routing-test comments. PR #2694 now points to `c64f695f`. Rootless-Podman topology emulation passed 5/5 at 2 and 4 ranks in the original gate and exact `timeout=900` reruns. | `wiki/devlog/2026-09-10-pr2694-review-fixes.md`, `artifacts/logs/pr2694-salvage/review-fixes/final/`, `temp/docker_ci_fix/podman/podman-c64f695f-*.log`, `temp/docker_ci_fix/podman/podman-c64f695f-timeout900-*.log` |
| PS8 | Done (pushed; CI running) | `ddp/windows-path-fix` | Fix the Windows-only routing-test failure without redundant platform fixtures | Replaced rendered-expression substring matching with direct AST argument equality; focused test 17/17 PASS; commits `69703025`, `fee2c310`; PR head updated | `temp/2026-09-15-pr2694-windows-distributed-routing-failure.md`; `wiki/devlog/2026-09-16-pr2694-windows-routing-fix.md` |
| M2 | Onboarding done | `-` | Conditional-graph onboarding artifacts (pre-implementation) | Report compiles, detector reproduces deadlock+corruption+control, notebook cells execute | `reports/conditional-graph-support.pdf`; `make verify-conditional-deadlock` 12/12 exit 0; `notebook/conditional_graph_optimisers.jl`; `wiki/devlog/2026-09-14-conditional-graph-onboarding.md` |
| CW-A | Done | `ddp/integration-m1-spike` | Phase A: integrated PR #2694 head into the implementation branch (reset to `fee2c310`, wrapper absent, worktree clean) | HEAD == `fee2c310`; routing 17/17; 2-rank suite 5/5; worktree clean | `artifacts/logs/conditional-wrapper/`; `wiki/devlog/2026-09-16-conditional-wrapper-phase-a-baseline.md` |
| CW-B | Done (RED) | `ddp/integration-m1-spike` | Phase B: focused real-AD equal-shaped conditional-branch corruption regression (`test/ext_distributed/conditional_distributedtest.jl`), commits `6065ce8f`, `8add0e42` | Fails fast with the expected numerical corruption: exit 1, 24 s, 3 passed / 2 failed per rank; no production code changed | `artifacts/logs/conditional-wrapper/phaseB-RED-*`; `wiki/devlog/2026-09-16-conditional-wrapper-phase-b-red-test.md` |
| CW-C | Done (GREEN, checkpoint `109aad9c`) | `ddp/integration-m1-spike` | Phase C: wrapper boundary — `DistributedOptimizerState` + specialized `setup`/`update`/`update!`, composition and higher-order-gradient errors, `adjust`/`freeze` delegation, minimal whole-model sync ported from the preserved `3370b910` reference | Focused conditional test 5/5 per rank at 2 ranks, exit 0 (28 s cold / 21 s warm); full 2-rank suite 6/6, 58.0 s; 0 ambiguities; `git diff --check` clean; two files committed as `109aad9c` | `artifacts/logs/conditional-wrapper/phaseC-*`; `wiki/devlog/2026-09-17-conditional-wrapper-phase-c-wrapper-boundary.md` |
| CW-C2 | Done | `ddp/integration-m1-spike` | Phase C review follow-up: direct-nesting composition guard — both `Optimisers.setup` paths reject `DistributedOptimizer` inside `DistributedOptimizer` with `ArgumentError`; checkpoint `109aad9c`, RED `a18392dd`, fix `f5b87975` | Focused optimizer test 6/6 per rank at 2 ranks, exit 0 (15 s cold / 10 s warm); focused conditional test 5/5, 19 s; full 2-rank suite 6/6, 54.4 s; 0 ambiguities; `git diff --check` clean | `artifacts/logs/conditional-wrapper/phaseC-review-*`; `wiki/devlog/2026-09-17-conditional-wrapper-phase-c-review-followup.md` |
| CW-D1 | Done (committed `cbdd3edc`) | `ddp/integration-m1-spike` | Canonical traversal order for reversed named `trainable` | Two-rank RED 6 passed / 4 failed → GREEN 10/10 per rank. Full suite 6/6. | `wiki/devlog/2026-09-28-conditional-wrapper-phase-d1-canonical-traversal.md` |
| CW-D2 | Done (committed `cbdd3edc`) | `ddp/integration-m1-spike` | Explicit occurrence mapping for mixed tied/isbits parameters | Two-rank RED 33 passed / 5 failed → GREEN 38/38 per rank. Full suite 6/6. | `wiki/devlog/2026-09-28-conditional-wrapper-phase-d2-occurrence-plan.md` |
| CW-D3 | Done (reviewed, committed `cbdd3edc`) | `ddp/integration-m1-spike` | CPU presence metadata with device-like parameter buffers | Two-rank RED 14 passed / 2 failed → GREEN 16/16 per rank. Independent review run passed. Full suite 6/6. Zero ambiguities. | `wiki/devlog/2026-09-29-conditional-wrapper-phase-d3-cpu-presence-metadata.md` |
| CW-D4 | Done (P1 corrected, committed `d6046103`) | `ddp/integration-m1-spike` | Legacy tuple selection read from the setup state tree (stable selection across updates) | RED2 (new testset on buggy source) 24 passed / 6 failed at step 2 for both update paths → GREEN2 D4 20/20 + multi-step 30/30 per rank; total 5/5+10/10+11/11+38/38+16/16+20/20+30/30. Full suite 6/6. Zero ambiguities. | `artifacts/logs/conditional-wrapper/phaseD-d4-metadata.md`; `wiki/devlog/2026-09-29-conditional-wrapper-phase-d4-legacy-trainable.md` |
| CW-D5 | Done (review corrections applied, committed `d6046103`) | `ddp/integration-m1-spike` | Phase D5 regression gate: eight testsets for the retained contracts, plus the array-container shape fix (`_buildmap`/`_foreach_child` use `CartesianIndices`). Review corrections: active-to-absent Adam snapshot (nonzero moments) and rank-distinct caller gradients with a read-only gradient case | Gate RED: array-container `(4,) == (2, 2)`. Correction RED: aliased length-2 buffer → caller snapshot fails with the global mean; absent leaf fed zero gradient → value and full Adam state change. Final GREEN: focused conditional 232/232 per rank, optimizer 6/6, full 2-rank suite `Distributed \| 6 6 1m45.2s`, 0 ambiguities, `git diff --check` clean | `artifacts/logs/conditional-wrapper/phaseD-d5-metadata.md` (+ review/correction logs); `wiki/devlog/2026-09-30-conditional-wrapper-phase-d5-regression-gate.md` |
| CW-E | Done (GREEN, review-corrected, committed `55171e65`) | `ddp/integration-m1-spike` | Phase E: restore the M1.1 battery without its defects. 15 testsets (+560/−0, tests only): multi-step real-AD averaging, stateful Adam reference, tied through real AD (incl. shared-gradient subcase) and transpose, isbits `Fill`, higher-order rejection, real-AD collective protocol, `Flux.train!`, nested `Chain`, `synchronize!!`, `adjust`/`adjust!`, whole-wrapper + partial freeze with Adam state and collective counts, Functor device adaptation (Adam state arrays transformed, adapted state reused), `Chain` tuple-gradient diagnostic, `Enzyme Duplicated` setup/update | Review correction: adaptation used stateless Descent and a fresh state; missing M1.1 API checks; freeze test bypassed wrapper delegation. Corrected: Adam state-array adaptation vs native reference; restored Chain/Enzyme/shared-gradient/whole-wrapper-freeze checks. Two-rank GREEN: focused conditional **367/367 per rank** across 30 testsets (1 m 22.0 s; aggregate `conditional file \| 367 367 \| 1m22.0s`), optimizer 6/6 (10.3 s), full suite `Distributed \| 6 6 2m02.6s`, 0 ambiguities, `git diff --check` clean. Correction runs passed on first attempt; no production change | `artifacts/logs/conditional-wrapper/phaseE-metadata.md`; `phaseE-GREEN-*`, `phaseE-correction-GREEN-*` logs; `wiki/devlog/2026-09-30-conditional-wrapper-phase-e-m11-battery.md` |
| C9 | Not started | `ddp/perf` | Profiling and bottleneck report | Timeline plus bottleneck analysis exists | - |
| C10 | Not started | `ddp/perf` | Performance improvement pass | Throughput improves or bottleneck is explained with evidence | - |
| C11 | Not started | `ddp/docs-examples` | Documentation and thesis-ready examples | New user can reproduce examples and understand limitations | - |
| C12 | Not started | `final/evaluation` | Final evaluation package | Correctness, throughput, speedup, memory, limitations, and future work are reported | - |

## Cluster & MPI Guardrails

When launching on a Slurm cluster, it's crucial to match the launcher's PMI version with the MPI library's expected PMI protocol. 
Flux.jl DDP is tested exclusively with Julia's default `MPICH_jll`, which uses the **PMI2** wire protocol.
If you use a system MPI (`JULIA_MPI_BINARY=system`), you must match your `srun` configuration accordingly. 

**Precompilation guardrail (Julia 1.12, measured 2026-09-03):** precompilation runs only on compute nodes. Login-node compiles target the wrong CPU (`graniterapids` vs `icelake-server`) and leave tests cold anyway. After every `sync_code.sh` or Flux source change, run `scripts/remote/precompile.sh bocconi` (precompiles thesis env + Flux test env under `srun`). `make env` no longer precompiles and `JULIA_PKG_PRECOMPILE_AUTO=0` keeps the login node compile-free during Pkg operations; a stale cache will still recompile silently at first load, so never skip the precompile step after a sync. Details and measurements: `wiki/context/remote-nodes.md`, `wiki/devlog/2026-09-03-precompile-experiment.md`, ADR-0005.

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
