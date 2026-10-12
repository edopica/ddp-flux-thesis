# 2026-09-09 — PR #2694 salvage phase 3b: PMI guardrail implemented (GREEN)

Follow-up to `2026-09-09-pr2694-salvage-phase3a-launcher-guard-tests.md`. Phase 3 of `temp/pr_salvage_plan.md` is now complete.

## What was done

Worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`, branch `ddp/pr2694-salvage`:

- Commit `cd8de233`: 3 files, +53/-41.
  1. `src/distributed/public_api.jl`: new pure, MPI-free `DistributedUtils.check_launcher_compat(env; library, force=false, mpi_initialized=false) -> nothing | String` (docstring documents the contract). It returns `nothing` unless `library == "MPICH"` AND `PMIX_RANK`/`OMPI_COMM_WORLD_RANK` is present AND neither `force` nor `mpi_initialized` is set; the returned message names the library, the trigger variable(s), and the `force=true` bypass, plus launch alternatives (`mpiexecjl`, `srun --mpi=pmi2`, `MPIPreferences.use_system_binary()`).
  2. `ext/FluxMPIExt/FluxMPIExt.jl`: the 40-line inline guard (broad "non-OpenMPI ⇒ PMI2" errors + speculative SLURM no-PMI2 `@warn`) replaced by a 4-line delegation to the checker, still gated on `!force && !MPI.Initialized()`.
  3. `initialize` docstring and `NEWS.md`: rewritten to describe only the limited behavior (MPICH under a PMIx/OpenMPI launcher is the only judged-unsafe case; system OpenMPI allowed; `force=true` expert bypass).

## Verification (all on the salvage worktree, local MPICH_jll)

- `julia --project=test test/distributed_launcher_guard.jl`: **169/169 pass** (GREEN vs 52 errored RED at `af728837`). Log: `artifacts/logs/pr2694-salvage/phase3-launcher-guard-GREEN.log`.
- Integration smoke 1 — real `FluxMPIExt` with `PMIX_RANK=0` set and MPICH loaded: `initialize(MPIBackend)` throws the checker message before `MPI.Init()`; `MPI.Initialized()` stays `false`.
- Integration smoke 2 — clean env: `initialize(MPIBackend)` succeeds and backend reports initialized.
- `grep` across `src/`, `ext/`, `docs/`, `NEWS.md`: no stale references to the removed claims ("expects PMI2", "is not OpenMPI", SLURM-no-PMI2 warning).

## Notes / scope boundaries honored

- No broad PMI claims: non-MPICH non-OpenMPI libraries are never judged.
- Communication stays outside autodiff; no optimiser changes; nothing else touched.
- Real-launch verification of the guardrail (OpenMPI CI workflow, Docker parity, mismatch reproduction) remains a phase-7 gate, not covered by these unit tests.

## Next exact action

Phase 4 of `temp/pr_salvage_plan.md` — complete cyclic data padding in `DistributedDataContainer` (N==0 → ArgumentError; `cld`; cyclic `mod1` indices; five dataset sizes × two/four ranks), tests-first again.
