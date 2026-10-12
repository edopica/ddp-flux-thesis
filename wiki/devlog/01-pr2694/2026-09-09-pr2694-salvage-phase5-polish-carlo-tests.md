# 2026-09-09 — PR #2694 salvage phase 5: retained & polished Carlo's test changes

Follows phases 1-2 (`...-phases-1-2.md`), phase 3a (`...-phase3a-launcher-guard-tests.md`), phase 3b (`...-phase3b-pmi-guardrail.md`), and phase 4 (`...-phase4-cyclic-data-padding.md`). Phase 5 of `temp/pr_salvage_plan.md` (lines 228-251) is now complete. Delegated to two subagents — an implementation subagent applied the edits, then an INDEPENDENT verification subagent audited the result and returned APPROVE; the orchestrator reviewed the end state.

## What was done

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`, branch `ddp/pr2694-salvage`:

- Commit `23cae338` "Polish distributed CI labels and harness wording (phase 5)" — one commit on top of `4eb5708e` (phase 4 GREEN). Working tree clean.
- Exactly 2 files changed; no behavioral change (comment/CI-label polish only).

### `.github/workflows/distributed_ci.yml`

- Job `mpi_edge` renamed from "MPI DDP Sharding & Asymmetry (4 Ranks)" to "MPI DDP Correctness (4 Ranks)"; both jobs now read "MPI DDP Correctness (2 Ranks)" / "(4 Ranks)".
- Removed the `FLUX_TEST_DISTRIBUTED_MPI: "true"` env line from BOTH `mpi_fast` and `mpi_edge`. That variable is only read by the main suite `test/runtests.jl`, NOT by the standalone distributed runner `test/ext_distributed/runtests.jl`, which both jobs invoke directly; the runner reads `JULIA_MPI_TEST_NPROCS`, `FLUX_TEST_DISTRIBUTED_TIMEOUT`, `FLUX_TEST_DISTRIBUTED_NCCL`.
- Run step "Run Distributed Tests (Edge cases)" renamed to "Run Distributed Tests".
- Verified no commented-out NCCL CI job block remains (already removed in phase 2 scope-reset commit `5c6ea561`).
- `--use-hwthread-cpus` still configured ONLY in the four-rank job (comment + `MPIPreferences.use_system_binary` + probe step all inside `mpi_edge`); the 8-line comment block explaining the GitHub ubuntu-latest slots rationale kept verbatim.

### `test/ext_distributed/runtests.jl`

- The comment above `proc = run(pipeline(cmd, stdout=stdout, stderr=stderr), wait=false)` rephrased from "Inherit streams: child errors must show in CI. A bare run(cmd, wait=false) would hide them (devnull)." to "Explicitly inherit the child streams so the child's output and errors are visible in the CI logs."
- The `proc = run(...)` code line is byte-identical.
- NCCL route untouched (the `FLUX_TEST_DISTRIBUTED_NCCL` backends line and the `nccl` child branch); no claim was added that CPU-buffer fallback tests validate native NCCL.

## Method (delegated implement → verify)

- Implementation subagent applied the edits in the `flux-pr2694-salvage` worktree (no commit).
- INDEPENDENT verification subagent audited the result against a 10-point checklist distilled from Phase 5 of `temp/pr_salvage_plan.md` (read lines 228-251 of that plan for the authoritative spec) and returned APPROVE.
- No RED/GREEN cycle was used for phase 5 because it is a wording/CI-label polish with no semantic change — that decision was the orchestrator's.

## Checks that passed (do not claim more)

- `git diff --check` clean.
- Workflow YAML parses (jobs `mpi_fast`, `mpi_edge`).
- `Meta.parseall` on `runtests.jl` returns ok.
- Grep confirms no `FLUX_TEST_DISTRIBUTED_MPI`, no "Sharding & Asymmetry", no "Edge cases", no commented NCCL block, no `supervisor`/`native`/`CPU-buffer`/`fallback` claims in the changed files or `test/ext_distributed/`.

The full MPI suite was NOT re-run (comment-only change); full two-rank/four-rank MPI runs remain a Phase 7 verification gate.

## Retained-from-fa3bf228 items re-confirmed

Present and untouched during phase 5:

- `test/ext_distributed/distributed_setup.jl`;
- child test files do not read `ARGS`;
- backend selected from `FLUX_TEST_DISTRIBUTED_BACKEND` env in `distributed_setup.jl`;
- deterministic `sort!(distributedtestfiles)`;
- `MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"` in `test/Project.toml`.

## Pre-existing out-of-scope observation (note)

`test/ext_distributed/runtests.jl` around line 28 references `CUDA.devices()` with no `using CUDA` visible at file top. It would only error if `FLUX_TEST_DISTRIBUTED_NCCL=true` were set, which the current CI never does. Predates phase 5 and untouched. Recorded as a note, NOT a phase-5 defect.

## Scope boundaries honored

- Comment/CI-label polish only: exactly 2 files, no behavioral change.
- No RED/GREEN cycle for phase 5 (orchestrator decision — no semantic change to test).
- Retained-from-`fa3bf228` harness structure untouched (see above).

## State after session

`flux-pr2694-salvage` @ `23cae338` [`ddp/pr2694-salvage`], clean. Logs in `ddp-flux-thesis/artifacts/logs/pr2694-salvage/`.

## Next exact action

Phase 6 of `temp/pr_salvage_plan.md` — documentation & consistency sweep on the flux worktree: make only narrow edits to the existing distributed section of `docs/src/guide/gpu.md` — correct launcher ordering to `mpiexecjl -n <np> julia --project=. <filename>.jl`, replace "split uniformly" with precise equal-length/cyclic-duplication wording, warn that padded observations affect epoch accounting and metrics (no unused-gradient instructions, no native-NCCL-testing claims) — and update `NEWS.md` to contain only the two user-facing changes (known MPICH launcher mismatch diagnostic + `force=true`; equal-length cyclic data sharding and empty-data rejection). Then Phase 7 verification gates (static scope checks, PMI guard tests, full `JULIA_MPI_TEST_NPROCS={2,4}` MPI suite runs, Docker OpenMPI parity, docs build). When replying to Carlo, explain that `distributed_setup.jl` centralizes the setup instead of `test/test_module.jl` because each MPI rank is a fresh standalone Julia process; importing the full general test module into every rank is unnecessary.
