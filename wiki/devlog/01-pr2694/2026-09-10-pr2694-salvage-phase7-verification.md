# 2026-09-10 — PR #2694 salvage phase 7: verification gates

Follows phases 1-2 (`...-phases-1-2.md`), phase 3a (`...-phase3a-launcher-guard-tests.md`), phase 3b (`...-phase3b-pmi-guardrail.md`), phase 4 (`...-phase4-cyclic-data-padding.md`), phase 5 (`...-phase5-polish-carlo-tests.md`), and phase 6 (`2026-09-10-pr2694-salvage-phase6-docs-consistency.md`). Phase 7 of `temp/pr_salvage_plan.md` (lines 268-290) is now complete.

Method: the orchestrator launched a swarm of subagents — gate checks plus independent audits. All subagents completed; no subagent failed.

Salvage head is now `caf47145`, worktree clean. The commit is **NOT pushed**.

## Gate 1 — Static scope checks: PASS (after fix)

- Scope content clean on `404ff37d...HEAD`:
  - no `resolve_unused_parameters`, no `DistributedOptimizerState`, no conditional-gradient API / `conditional_distributedtest.jl`;
  - `docs/src/guide/distributed.md` absent and no `docs/make.jl` registration;
  - no native-NCCL-validation claim;
  - no child test reads `ARGS`;
  - scope-reset commit `5c6ea561` present.
- Initial FAIL: the committed `test/ext_distributed/runtests.jl` had 5 trailing-whitespace lines (33, 38, 42, 58, 63) and 3 committed test files lacked final newlines.
- Fixed and committed as `caf47145` "Document distributed changes and fix whitespace hygiene (phase 6)" — 8 files, `+23/-19` (the 4 phase-6 docs/consistency files plus 4 distributed test files). NOT pushed.
- `git diff --check 404ff37d...HEAD` now clean (exit 0); worktree clean.

## Gate 2 — PMI guard tests: PASS (system-OpenMPI reproduction not possible locally)

- `julia --project=test test/distributed_launcher_guard.jl` → **169/169 pass** (`guard-focused.log`).
- MPICH + `PMIX_RANK`/`OMPI_COMM_WORLD_RANK` → actionable `ErrorException` thrown BEFORE `MPI.Init()`; `MPI.Initialized() == false` after the throw (`guard-repro.log`).
- `force=true`: the pure checker returns `nothing`; a real `initialize(MPIBackend; force=true)` proceeds to native `MPI_Init` and aborts (exit 15) because it was not launched under MPI — expected and reported, not a defect.
- System-OpenMPI reproduction NOT POSSIBLE LOCALLY: OpenMPI is not installed; the local MPI is `MPICH_jll` 5.0.1 and `MPI.MPI_LIBRARY = "MPICH"` (`mpi-config.log`). Deferred to the Docker gate (gate 4).

## Gate 3 — MPI suite: PASS

- `JULIA_MPI_TEST_NPROCS=2 julia --project=test test/ext_distributed/runtests.jl` → exit 0, 5/5 child files (49s) (`mpi-nprocs2.log`).
- `JULIA_MPI_TEST_NPROCS=4 julia --project=test test/ext_distributed/runtests.jl` → exit 0, 5/5 child files (54s) (`mpi-nprocs4.log`).
- Child output visible; no timeout/hang.

## Gate 4 — OpenMPI parity (rootless podman): PASS

- Executed with the real runner `scripts/podman/run-ci-parity.sh` only, per `scripts/podman/README.md` — rootless podman, no docker, no sudo, no `newgrp`, no `--cpuset-cpus`.
- Staged content: salvage worktree `caf47145ba1ea2ca06370a2b3422df0b7ddf4a0f`, rsynced to `/tmp/opencode/docker_ci/repo-salvage/`. The Flux worktree was not modified.
- All 9 steps exit 0: rsync; `mpi_fast` setup (`Pkg.add("MPIPreferences")` + `use_system_binary()`); instantiate; `install_mpiexecjl(force=true)`; 2-rank suite; `mpi_edge` setup (`use_system_binary(mpiexec="mpiexec --use-hwthread-cpus")`); probe; 4-rank suite; restore `use_system_binary()`.
- `mpi_fast` (2 ranks): exit 0, `Distributed   |    5      5  35.8s` (5/5).
- `mpi_edge` (4 ranks): exit 0, `Distributed   |    5      5  1m01.4s` (5/5) — this is the four-rank OpenMPI parity run.
- Probe: `mpiexec (OpenRTE) 4.1.6`; `nproc` = 4; affinity `0-3`; `MPI.mpiexec()` = `mpiexec --use-hwthread-cpus`; `MPI.MPI_LIBRARY` = `OpenMPI`; `4-rank launch OK`. No `not enough slots` in any log.
- Anomaly (non-blocking): the 4-rank suite log contains `155 dependencies precompiled but different versions are currently loaded (...). Restart julia to access the new versions.`; the suite still exited 0 and passed 5/5.
- Logs: `temp/docker_ci_fix/podman/podman-phase7-00..08-*.log`; summary `temp/docker_ci_fix/podman/podman-phase7-SUMMARY.md`.
- No podman runner problems occurred; nothing needed fixing.
- Supersedes the earlier "NOT RUN" status (rootful docker socket access was denied). The runner reproduces the OpenMPI slot condition through hwloc topology emulation; see `wiki/devlog/01-pr2694/2026-09-10-rootless-podman-ci-parity.md` and `wiki/failures/2026-09-10-podman-taskset-cpuset.md`.

## Gate 5 — Docs build: PASS

- `julia --project=docs docs/make.jl` → exit 0 (`docs-build.log`).
- `docs/build/guide/gpu.html` rendered with the phase-6 prose; no `distributed.html`; no stray references.
- Warnings are pre-existing cross-reference warnings in untouched pages. `deploydocs` auto-skipped (CI unset).

## Gate 6 — Final review: PASS

- CI workflow has exactly two jobs: "MPI DDP Correctness (2 Ranks)" and "(4 Ranks)".
- `--use-hwthread-cpus` only in the 4-rank job; no commented NCCL placeholder; no `FLUX_TEST_DISTRIBUTED_MPI`.
- Runner deterministic (`sort!`), `addenv` backend propagation, inherited child streams, real exit-code checks.
- `data_distributedtest.jl` exercises N=0,1,2,7,10 at both rank counts with the required invariants.

## Resolved conflict: `mpiexecjl` and `--project` ordering

One auditor claimed that moving `--project` off the `mpiexecjl` wrapper breaks it. Verified false: `MPI.jl`'s `bin/mpiexecjl` loops over ALL arguments (lines 29-45) and extracts `--project`/`--project=*` regardless of position, setting `JULIA_PROJECT` for the children. The phase-6 ordering `mpiexecjl -n <np> julia --project=. <file>.jl` is correct.

## Audit findings (in-scope, NOT yet fixed — for a future decision)

1. Guard documented as MPI-only (`src/distributed/public_api.jl:36`, `NEWS.md:9`) but also runs for `NCCLBackend`; `initialize(NCCLBackend; force=true)` is a `MethodError` because `__initialize(::Type{NCCLBackend})` has no `force` kwarg (`ext/FluxMPINCCLExt/FluxMPINCCLExt.jl:10-14`).
2. `"MPItrampoline"` is not a valid `MPI.MPI_LIBRARY` value in the `check_launcher_compat` docstring example (`src/distributed/public_api.jl:62-63`; also `test/distributed_launcher_guard.jl:91,126`).
3. Wording gap: duplicated observations bias training gradients (the optimizer all-reduces per-rank mean gradients over padded shards), not only reported metrics.
4. `test/ext_distributed/runtests.jl:28` uses `CUDA.devices()` with no `using CUDA` (latent, pre-existing; only under `FLUX_TEST_DISTRIBUTED_NCCL=true`).
5. Dead `try/catch BoundsError → NaN32` in `test/ext_distributed/data_distributedtest.jl:47-51`.
6. `force` labelled "Undocumented internal kwargs" but is documented public API (`ext/FluxMPIExt/FluxMPIExt.jl:12-13`).
7. Redundant outer guard condition double-calls `MPI.Initialized()` (`ext/FluxMPIExt/FluxMPIExt.jl:15-19`).
8. Stale, out-of-scope `gpu.md` examples: single-rank `idxs` printout (`:353`), `MPIBackend` shown under an `NCCLBackend` example, pre-M1 `Leaf` state (`:371-377`).

## Logs

`artifacts/logs/pr2694-salvage/phase7/`: `guard-focused.log`, `guard-repro.log`, `mpi-nprocs2.log`, `mpi-nprocs4.log`, `mpi-config.log`, `docs-build.log`, `docs-instantiate.log`, `instantiate.log`.

Gate 4 (rootless podman): `temp/docker_ci_fix/podman/podman-phase7-00..08-*.log` plus `temp/docker_ci_fix/podman/podman-phase7-SUMMARY.md`.

## Checks that passed / what is NOT verified

Passed: gates 1, 2, 3, 4, 5, 6 — all six gates.

NOT verified: native NCCL runtime (out of scope for this PR). Real cgroup-cpuset parity is not reproducible rootless and was not attempted; gate 4 reproduces the OpenMPI slot condition via hwloc topology-file emulation, not a real cgroup cpuset. Do not claim native NCCL or real-cpuset parity.

## State after session

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` @ `caf47145` [`ddp/pr2694-salvage`], clean. The commit is **NOT pushed**. `ddp/upstream-pr` can still be fast-forwarded.

## Next exact action

1. Decide whether to fix the in-scope audit findings (NCCL/guard doc wording, `"MPItrampoline"` example, training-bias wording) or defer them.
2. Decide when/whether to push `ddp/pr2694-salvage` (currently `caf47145`, unpushed; the final branch can fast-forward `ddp/upstream-pr`).

Gate 4 is resolved (OpenMPI parity PASS via rootless podman) and no longer blocks.
