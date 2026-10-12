# 2026-09-09 — PR #2694 salvage phase 3a: launcher-guard tests designed first (RED baseline)

Plan: `temp/pr_salvage_plan.md`. This entry covers the tests-first step of phase 3 (narrow the PMI guardrail). The implementation commit is intentionally NOT done yet — tests are committed first so the before/after change is visible.

## What was done

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`, branch `ddp/pr2694-salvage`:

- Commit `af728837`: new file `test/distributed_launcher_guard.jl` — decision-matrix contract tests for a pure, MPI-free checker that does not exist yet: `DistributedUtils.check_launcher_compat(env; library, force=false, mpi_initialized=false) -> nothing | String`.
- RED baseline run: `julia --project=test test/distributed_launcher_guard.jl` → **52 errored / 0 passed / EXIT=1** (every row `UndefVarError: check_launcher_compat`). Log: `artifacts/logs/pr2694-salvage/phase3-launcher-guard-RED.log`.

## Test design rationale (anti-green-seeking)

Rows encode the target semantics from the plan, including cases that current inline guard logic would get WRONG if ported naively:

- MPICH + `PMIX_RANK` / `OMPI_COMM_WORLD_RANK` (also empty-value, SLURM-combined, both) → non-empty `String` error mentioning `MPICH`, the trigger variable, and `force=true`.
- Same envs + `force=true` → `nothing`.
- Same envs + `mpi_initialized=true` → `nothing` (guard only pre-`MPI.Init()`).
- OpenMPI + PMIx/OpenMPI vars → `nothing` (system OpenMPI allowed).
- MPICH + clean env / PMI2-only / SLURM-only / PATH-only / `PMI_RANK` only → `nothing` (no speculative warnings, no PMI2-claim).
- Non-MPICH non-OpenMPI libraries (`MPItrampoline`, `MicrosoftMPI`, `IBMSpectrumMPI`, `IntelMPI`, `MVAPICH`, `FujitsuMPI`, `unknown`) + PMIx/OpenMPI vars → `nothing` — this pins the "detect only the known unsafe MPICH case" requirement that the current guard violates (it errors for every non-OpenMPI library).
- Contract sweep: always `nothing` or non-empty `String`, never throws.

Design decisions locked by the tests (implementation must follow):
- Checker is a **pure function in core `DistributedUtils`** (MPI-free, takes `library` string + `env` dict), so it runs in the main suite without MPI; `FluxMPIExt.__initialize` calls it with `MPI.MPI_LIBRARY`/`MPI.Initialized()`.
- Message must name the library, name the trigger variable, and mention `force=true`.

## Verification

- Worktree clean; only the new test file committed.
- Run from salvage worktree: `julia --project=test test/distributed_launcher_guard.jl` (no MPI needed).

## Next exact action

Implement phase 3 code so the committed tests go GREEN: add `DistributedUtils.check_launcher_compat` (pure, in `src/distributed/public_api.jl`) per the matrix, rewrite the guard in `ext/FluxMPIExt/FluxMPIExt.jl` to delegate to it before `MPI.Init()`, remove the broad non-OpenMPI claim and the speculative Slurm warning, update the `initialize` docstring + NEWS wording; then rerun the test file (expect 52 pass) and re-run the RED log command for the before/after record.
