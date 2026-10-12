# ADR-0006 - PR #2694 integration baseline (M0)

Date: 2026-09-08
Status: Accepted

## Context

PR #2694 review continues, and the remaining agreed scope (unused-parameter handling in the
normal training path) requires a new architecture (whole-model gradient synchronization with a
wrapper optimizer state). Dirty `master` (`b633cdc7`) carries in-progress C8.2 test expansion
plus an early, unvalidated `no_sync`/`sync_gradients!` experiment. Starting from dirty master
would mix C8 work with the integration change and make the diff unreviewable. The plan
(implementation strategy, 2026-09-08) therefore requires an isolated workspace built from the
PR branch, with dirty master preserved untouched.

## Facts recorded at M0

- **PR #2694 head**: `08f468e3` ("Bound Install MPI step with timeout").
- **Candidate base**: `fa3bf228` ("Address DDP review: env-based backend, centralized setup,
  visible child output, CI slots fix") — tip of `ddp/pr2694-ci-fix`, one commit ahead of the PR
  head.
- **Upstream merge base**: `404ff37d` (`merge-base ddp/pr2694-ci-fix upstream/master`).
- **Integration worktree**: `/home/kurapica/Projects/ddp_flux/flux-integration`
  (branch `ddp/integration-m1-spike`), HEAD = `fa3bf228`, tree clean.
- **Dirty master**: untouched in the `ddp_flux` working tree (8 modified + 1 untracked file,
  verified identical before/after snapshot) and snapshotted on
  `archive/dirty-master-c8.2-wip` (`b0034f55`) from `b633cdc7`.

## Dirty-master inventory (port candidates, NOT yet ported)

| File | Change | Port? |
|---|---|---|
| `src/distributed/public_api.jl` | +31: `no_sync` flag, `sync_gradients!`, `set_nosync!` (early whole-model sync attempt) | Only if M1 design keeps them; otherwise superseded by wrapper-state design |
| `test/ext_distributed/common_distributedtest.jl` | +101 collective-invariant tests | Port after M1 proof, separate commit |
| `test/ext_distributed/data_distributedtest.jl` | +248 dtype matrix etc. | Port after M1 proof |
| `test/ext_distributed/helper.jl` | +133 watchdog / child-exit hardening | Port after M1 proof |
| `test/ext_distributed/optimizer_distributedtest.jl` | +135 | Port after M1 proof |
| `test/ext_distributed/synchronized_distributedtest.jl` | +129 | Port after M1 proof |
| `test/ext_distributed/runtests.jl` | +24 | Port after M1 proof |
| `test/ext_distributed/unused_parameters_distributedtest.jl` | +1 | Port after M1 proof |
| `test/ext_distributed/end_to_end_distributedtest.jl` | new (untracked, 329 lines) | Port after M1 proof |

## Decision

1. Do all integration work in the `flux-integration` worktree on `ddp/integration-m1-spike`
   based at `fa3bf228`.
2. Never commit to `ddp/pr2694-ci-fix`, `master`, or any archive branch from the integration
   worktree.
3. Port dirty-master work only as reviewed, separately-committed changes, and only after the
   M1 correctness proof passes. Do not merge dirty master wholesale.
4. No push anywhere until all local checkpoints pass and the final diff is reviewed.

## M0 blocker-to-file checklist

- [x] Dirty master preserved: working tree state verified identical before/after snapshot.
- [x] Snapshot branch exists: `archive/dirty-master-c8.2-wip` @ `b0034f55`.
- [x] Integration worktree exists: `../flux-integration` @ `fa3bf228`, branch `ddp/integration-m1-spike`.
- [x] Baseline clean: `git status --porcelain` empty; `no_sync` and `end_to_end_distributedtest.jl` absent.
- [x] Metadata recorded: PR head `08f468e3`, base `fa3bf228`, merge base `404ff37d`.
- [x] Dirty-master port inventory recorded.
- [x] User reviews the exact scope before M1 implementation begins (approved 2026-09-08).

## Consequences

- Two Flux checkouts coexist (`ddp_flux` for C8/dirty work, `flux-integration` for the PR
  integration); `FLUX_REPO_PATH` must be set per run to avoid cross-contamination.
- The M1 spike lands as a narrow architecture diff on `ddp/integration-m1-spike`, reviewed
  before any unrelated fixes are added.
