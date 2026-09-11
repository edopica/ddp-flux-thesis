# 2026-09-10 — PR #2694 salvage: review-fix corrections (phases 2-7 code review)

Follows phases 1-2 (`...-phases-1-2.md`), phase 3a (`...-phase3a-launcher-guard-tests.md`), phase 3b (`...-phase3b-pmi-guardrail.md`), phase 4 (`...-phase4-cyclic-data-padding.md`), phase 5 (`...-phase5-polish-carlo-tests.md`), phase 6 (`2026-09-10-pr2694-salvage-phase6-docs-consistency.md`), and phase 7 (`2026-09-10-pr2694-salvage-phase7-verification.md`). Plan: `temp/pr_salvage_plan.md`.

A six-agent review of PR #2694 salvage phases 2-7 produced findings. The orchestrator fixed all in-scope findings with RED/GREEN subagents. Commit `15ca2d9c` contains the review fixes. Final commit `c64f695f857a4248846163b0455b1a002978f869` only corrects stale routing-test comments. The worktree is clean. Nothing was pushed.

## What the review found

### Push blockers (all fixed)

1. **NCCL could not use the guard bypass.** `initialize(NCCLBackend; force=true)` raised `MethodError` because `__initialize(::Type{NCCLBackend}; ...)` had no `force` keyword, so the documented expert bypass did not work for the NCCL path (which bootstraps through `MPI.Init()`).
2. **Main-suite distributed opt-in was broken.** `ParallelTestRunner` discovered the `ext_distributed/*` child files individually, so workers ran without `FLUX_TEST_DISTRIBUTED_BACKEND` and failed. The dedicated runner `test/ext_distributed/runtests.jl` must be the sole entry point.
3. **Padding docs omitted the training effect.** `DistributedDataContainer` docs described only epoch/metric accounting, not that duplicated observations enter training batches, get extra weight in the averaged gradients, and change the effective training objective.

### Additional in-scope corrections (all fixed)

- `"MPItrampoline"` → `"MPIwrapper"` in the checker docstring and guard-test fixtures (MPI.jl reports `"MPIwrapper"`).
- `check_launcher_compat` renamed to `__check_launcher_compat` (internal; users call `initialize`) across `src/distributed/public_api.jl`, `ext/FluxMPIExt/FluxMPIExt.jl`, and `test/distributed_launcher_guard.jl`.
- `force` moved out of the "Undocumented internal kwargs" comment in `FluxMPIExt.jl` (it is public API).
- `MPI.Initialized()` captured once and reused in `FluxMPIExt.jl` (was called twice in the outer guard).
- `test/ext_distributed/data_distributedtest.jl`: removed the dead `BoundsError → NaN32` sentinel (it now records the failure but still reaches the collective, preserving collective order); the empty-data test now asserts the error message (`"empty"`, `"numobs(data) == 0"`) as well as `ArgumentError`.
- `test/ext_distributed/runtests.jl`: removed the false "children MUST call `exit(1)`" comment.
- `.github/workflows/distributed_ci.yml`: child watchdog `FLUX_TEST_DISTRIBUTED_TIMEOUT` `"1200"` → `"900"` in both jobs (the step timeout stays 20 min), so the child can emit diagnostics before GitHub terminates the step.
- Whitespace hygiene: removed trailing spaces in `public_api.jl`; added the missing final newline to `FluxMPINCCLExt.jl`.

Findings **not** fixed in this set: phase-7 finding 4 (latent `CUDA.devices()` without `using CUDA` in `ext_distributed/runtests.jl`, only reachable when `FLUX_TEST_DISTRIBUTED_NCCL=true`) remains; phase-7 finding 8 (stale out-of-scope `gpu.md` examples) remains deferred.

## Method (RED then GREEN)

Each blocker was reproduced by a RED subagent before any implementation, then fixed by a GREEN subagent. The routing regression additionally used a real-discovery check (see below).

- **RED blocker 1** — new static-contract test for the NCCL `force` keyword: `4 passed / 2 failed` (`:force in declared`, `:force in forwarded`). Log: `artifacts/logs/pr2694-salvage/review-fixes/red-nccl-force.log`.
- **RED blocker 2** — new `test/distributed_routing.jl` failed at include time because `test/test_utils_distributed.jl` did not exist. Log: `artifacts/logs/pr2694-salvage/review-fixes/red-routing.log`.

## Exact review-fix file changes (commit `15ca2d9c`)

| File | Change |
|---|---|
| `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` | Added `force::Bool=false` to `__initialize(::Type{NCCLBackend}; ...)`; forwarded `force` to the MPI initializer; added final newline |
| `src/distributed/public_api.jl` | Docstring: guardrail protects `MPI.Init()` including the NCCL bootstrap; `force=true` to the same `initialize` call; `__check_launcher_compat` rename + internal note; `"MPIwrapper"`; error prefix `"Distributed initialization failed:"`; trailing whitespace removed |
| `NEWS.md` | PMI-guardrail bullet (NCCL bootstrap, same `initialize` call) and data-sharding bullet (training-batch/gradient weighting) updated |
| `docs/src/guide/gpu.md` | Padding wording now states duplicated observations enter training batches, weight averaged gradients, and change the effective training objective |
| `ext/FluxMPIExt/FluxMPIExt.jl` | `force` removed from "Undocumented internal kwargs"; `MPI.Initialized()` captured once; calls `__check_launcher_compat` |
| `test/distributed_launcher_guard.jl` | Checker renamed; `"MPIwrapper"` fixtures; new NCCL static-contract testset + AST helpers |
| `test/ext_distributed/data_distributedtest.jl` | Dead `BoundsError → NaN32` sentinel removed; empty-data test asserts the message |
| `test/ext_distributed/runtests.jl` | Removed the false child-`exit(1)` comment |
| `.github/workflows/distributed_ci.yml` | Child watchdog `1200` → `900` in both jobs |
| `test/runtests.jl` | Includes the helper; always strips `ext_distributed` from discovery; deletes the helper from discovery; conditionally adds one `"ext_distributed"` runner entry when `FLUX_TEST_DIST_MPI || FLUX_TEST_DIST_NCCL` |
| `test/test_utils_distributed.jl` (new) | `strip_distributed_tests!`, `add_distributed_runner!` routing helpers |
| `test/distributed_routing.jl` (new) | 18-test routing regression for the helpers |

The distributed runner is now the only distributed entry point: `test/runtests.jl` strips every `ext_distributed*` key from `find_tests` discovery, then re-adds a single `"ext_distributed"` key mapped to `include(ext_distributed/runtests.jl)` only when a distributed flag is set. Child files are never discovered individually.

## Verification results

All logs under `artifacts/logs/pr2694-salvage/review-fixes/` (incremental) and `.../review-fixes/final/` (final).

- **Guard tests**: `__check_launcher_compat decision matrix` **169/169** + NCCL static contract **6/6**, exit 0 — `final/guard-final.log`.
- **Routing regression**: `distributed test routing` **18/18**, exit 0 — `final/routing-final.log`.
- **Routing against REAL `ParallelTestRunner.find_tests("test")`** — `final/routing-real-discovery.log`: BEFORE = 6 `ext_distributed/*` child keys; AFTER_STRIP = none; AFTER_ADD = `["ext_distributed"]`; `ROUTING_OK`. This substitutes for running the full main suite locally (see the failure below).
- **2-rank dedicated runner**: 5/5, exit 0 — `final/mpi-nprocs2.log` (33.8s).
- **4-rank dedicated runner**: 5/5, exit 0 — `final/mpi-nprocs4.log` (40.9s).
- **Docs build**: `julia --project=docs docs/make.jl` exit 0, Doctests 1/1; rendered `docs/build/guide/gpu.html` contains the new training-objective wording — `final/docs-build.log`, `final/docs-build.meta`.
- **Static scope checks clean** — `final/scope-checks.log`: no `resolve_unused_parameters`, no `DistributedOptimizerState`, no `docs/src/guide/distributed.md`, no native-NCCL-validation claim, no child `ARGS` read, no `MPItrampoline`, no old `check_launcher_compat`; `git diff --check` clean.
- **Repository-root `Manifest.toml` unchanged** — sha256 `3b86244bc58065456acd34494db9d936953678e3824b107f8fdb4f68ecbb0046` (`final/manifest-before.sha256`, `final/manifest-after.sha256`).

Incremental GREEN logs (`review-fixes/green-*.log`) cover the individual corrections: `green-nccl-ext.log`, `green-core-api.log`, `green-guard-test.log`, `green-routing.log`, `green-data-n2.log`, `green-news.log`, `green-docs.log`, `green-workflow.log`, `green-mpi-ext.log`.

## Local main-suite / Reactant failure (important)

Running the main suite `test/runtests.jl` locally is **NOT possible on this host** and twice crashed the desktop (opencode + herdr). Attempted command (metadata `final/main-suite-route.meta`):

```bash
FLUX_TEST_DISTRIBUTED_MPI=true FLUX_TEST_REACTANT=false JULIA_MPI_TEST_NPROCS=2 \
  julia --project=test test/runtests.jl ext_distributed
```

The route log is empty (`final/main-suite-route.log`, 0 bytes) because the process died before output.

Root cause: `test/runtests.jl` calls `Pkg.add("MPI")` unconditionally when distributed flags are set, which instantiates the test project. At the time of the crash `test/Project.toml` listed `Reactant` in `[deps]` while the resolved `Manifest.toml` did not contain it, so Pkg instantiation pulled in Reactant/XLA. In addition, `test/test_module.jl` runs `using` on any package where `Base.find_package(pkg) !== nothing`, and `find_package("Reactant")` resolves, so every `ParallelTestRunner` worker loaded Reactant. On a 14 GiB host this exhausted memory and killed opencode/herdr.

`FLUX_TEST_REACTANT=false` does **NOT** help: `Pkg.add("MPI")` still instantiates the project (and the `Reactant` dependency line was still present in the working tree). Side effect: the aborted first run's `Pkg.add("Reactant")` appended a `Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"` line to `test/Project.toml`; the orchestrator reverted it with `git restore test/Project.toml`. No tracked environment changes remain; the gitignored root `Manifest.toml` hash is unchanged (see above).

The main-suite route is therefore verified locally via the real-discovery routing check + routing unit test + direct dedicated runner. The full route remains a CI gate. Full write-up: `wiki/failures/2026-09-10-main-suite-reactant-instantiation.md`.

## Final commit and OpenMPI topology-emulation gate rerun (2026-09-11)

Commit `c64f695f857a4248846163b0455b1a002978f869` follows review-fix commit `15ca2d9c`. It only corrects stale comments in `test/distributed_routing.jl`.

The rootless Podman/OpenMPI gate reran against exact `c64f695f`. All original 9 steps exited 0. The 2-rank and 4-rank suites each passed 5/5.

Then the exact workflow `timeout=900` suites reran. Both the 2-rank and 4-rank suites passed 5/5. The final step restored plain MPI preferences.

Evidence: `temp/docker_ci_fix/podman/podman-c64f695f-*.log` and `temp/docker_ci_fix/podman/podman-c64f695f-timeout900-*.log`.

This gate uses hwloc topology emulation. It does not provide real cgroup-cpuset parity. See `wiki/devlog/2026-09-10-rootless-podman-ci-parity.md` and `wiki/failures/2026-09-10-podman-taskset-cpuset.md`.

## Git state after session

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` is clean on `ddp/pr2694-salvage` at exact `c64f695f857a4248846163b0455b1a002978f869`. Nothing was pushed. `ddp/upstream-pr` can still be fast-forwarded.

## Next exact action

1. Review the final diff and status.
2. When approved, fast-forward push to `ddp/upstream-pr`.
3. Then monitor live CI.
