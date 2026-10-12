# 2026-09-09 — PR #2694 salvage: phases 1 and 2 (worktrees + scope reset)

Plan: `temp/pr_salvage_plan.md`. Executed phases 1 and 2 only. Delegated to subagents; orchestrator verified the end state.

## Phase 1 — worktrees in order (DONE)

- Created permanent archive branch `archive/pr2694-unused-gradient-spike` @ `3370b910` (preserves the M1/M1.1 unused-gradient research; unpushed, local only, like `archive/dirty-master-c8.2-wip`).
- Retired the temporary worktree `/tmp/opencode/pr2694-ci-fix` (was clean; branch `ddp/pr2694-ci-fix` kept @ `fa3bf228`).
- Created permanent worktree `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage` on new branch `ddp/pr2694-salvage` starting at `fa3bf228`.
- Untouched and verified unchanged: dirty `master` (`b633cdc7`, identical dirty file list), `flux-integration` @ `3370b910` on `ddp/integration-m1-spike`, `archive/dirty-master-c8.2-wip`.

## Phase 2 — scope-reset commit (DONE)

Commit `5c6ea561` "Salvage PR #2694 scope reset: drop unused-parameter API, test, docs guide, and NCCL CI placeholder" on `ddp/pr2694-salvage`:

- `src/distributed/public_api.jl`: removed `resolve_unused_parameters!!` utility (-27; no export existed).
- `test/ext_distributed/unused_parameters_distributedtest.jl`: deleted (-65).
- `NEWS.md`: removed all unused-parameter text (-4).
- `docs/src/guide/distributed.md`: deleted (-160) + its `docs/make.jl` registration (-1).
- `.github/workflows/distributed_ci.yml`: removed commented NCCL CI placeholder and NCCL-validation claims (-10).
- Pure deletion commit: 6 files, 267 deletions, 0 insertions. Nothing ported from `04ddeaa0`/`3370b910`. `gpu.md` untouched (its fa3bf228 cleanup retained).

## State after session

Worktrees: `ddp_flux` @ `master` (dirty, frozen) · `flux-integration` @ `3370b910` [`ddp/integration-m1-spike`] · `flux-pr2694-salvage` @ `5c6ea561` [`ddp/pr2694-salvage`], clean. `ddp/pr2694-salvage` descends from `fa3bf228`; future push to `ddp/upstream-pr` is fast-forward only.

## Next exact action

Phase 3 — narrow the PMI guardrail in `ext/FluxMPIExt/FluxMPIExt.jl` (work ONLY in `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`): targeted MPICH-mismatch detection before `MPI.Init()`, keep system-OpenMPI allowance and `force=true`, drop broad PMI2 claim and speculative Slurm warning, actionable message; then add focused compatibility-checker tests per the plan decision matrix.
