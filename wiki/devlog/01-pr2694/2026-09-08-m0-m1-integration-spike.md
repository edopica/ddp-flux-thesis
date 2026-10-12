# 2026-09-08 — M0/M1: PR #2694 integration workspace + automatic unused-gradient spike

## Context

PR #2694 review continues; the remaining agreed scope (unused-parameter handling in the
normal training path) needs a new architecture (whole-model gradient synchronization with a
wrapper optimizer state). Per ADR-0006, all integration work happens in an isolated worktree
built from the PR branch; dirty master is preserved untouched.

- M0 = isolate the integration workspace and record the baseline (ADR-0006).
- M1 = spike the automatic unused-gradient architecture and prove it on CPU/MPI.

## M0 — Integration workspace isolated (facts)

- Dirty master (`ddp_flux` @ `b633cdc7`) preserved untouched (working tree verified
  identical before/after) and snapshotted on `archive/dirty-master-c8.2-wip` @ `b0034f55`.
- New git worktree `/home/kurapica/Projects/ddp_flux/flux-integration`, branch
  `ddp/integration-m1-spike`, HEAD `fa3bf228` (tip of `ddp/pr2694-ci-fix`), tree clean, no
  C8 content leaked.
- Metadata: PR #2694 head `08f468e3`; candidate base `fa3bf228`; upstream merge base
  `404ff37d`.
- Metadata + port inventory + checklist: `wiki/decisions/ADR-0006-integration-baseline.md`
  (user approved scope via checkpoint question).
- Policy: dirty master not merged wholesale; C8 test expansions port later as separate
  commits only after M1 proof.

## M1 — Automatic unused-gradient architecture spike (commit `04ddeaa0`, NOT pushed)

### Design

- `DistributedOptimizer` stays `<: Optimisers.AbstractRule` but per-leaf `apply!` no longer
  communicates.
- New `DistributedUtils.DistributedOptimizerState` wrapper returned by specialized
  `Optimisers.setup(::DistributedOptimizer, model)`.
- `Optimisers.update`/`update!` specialized on it:
  1. fill missing (`nothing`/AbstractZero) gradients with device-matched zeros
     (`similar(p)`),
  2. one in-place allreduce per unique trainable parameter in model-structure order (tied
     params deduped by identity, reduced exactly once),
  3. delegate to the native inner update.
- Communication stays outside AD.
- Also added: state-level adjust/adjust!/freeze!/thaw!, `DistributedUtils.synchronize!!` on
  the wrapped state, updated docstrings, `resolve_unused_parameters!!` kept as legacy
  fallback (users must not call it manually).
- Public APIs only: Optimisers setup/update/update!/adjust/adjust!/freeze!/thaw!/apply!/
  init/Leaf/isnumeric/maywrite, Functors fmapstructure_with_path/isleaf/KeyPath/getkeypath/
  haskeypath. No private Optimisers internals.

### Tests

- Fixed `optimizer_distributedtest.jl` wrapper access.
- New `test/ext_distributed/conditional_distributedtest.jl` (225 lines): rank-gated
  conditional branches, analytic global-mean gradient checks (Descent), Flux.setup +
  Optimisers.setup, Flux.train! + explicit withgradient/update!/update, Adam, adjust/adjust!
  (state and rule level), synchronize!! on wrapped state, nested model (Chain of custom
  struct + Dense), tied parameter reduced exactly once (analytic), collective-count check via
  CountingBackend (== unique trainable params, equal on both ranks), frozen params untouched.

### Results

- Full ext_distributed MPI suite green at 2 ranks (children ran 4-rank):
  `make c8-mpi FLUX_REPO_PATH=../flux-integration` -> "Distributed | 7 7", EXIT=0, 2m15s
  warm; orchestrator independently re-ran and confirmed.
- New test file alone: 13 testsets, EXIT=0.
- Diff stat vs `fa3bf228`: 3 files changed, 337 insertions(+), 10 deletions(-).
- Go/no-go gate: PASSED (public interfaces, normal training syntax preserved, one
  deterministic collective per unique trainable param, no custom optimizer, no silent
  GPU->CPU transfer).

### Known limitations (accepted for CPU/MPI baseline)

- `higher...` grads pass through unsynced (same as pre-M1 behavior).
- Whole-subtree ZeroTangent grads on non-leaf array wrappers: pre-existing unexercised edge.
- NCCL path API-compatible but untested (CPU/MPI only per agreed scope).

## Exact commands

```bash
cd /home/kurapica/Projects/ddp_flux/ddp-flux-thesis
make c8-mpi FLUX_REPO_PATH=../flux-integration   # local, 2 ranks (children 4-rank), 7/7 PASS, EXIT=0
git -C ../flux-integration diff fa3bf228        # review the M1 architecture diff
```

## Next action

- Review the M1 architecture diff (`git -C ../flux-integration diff fa3bf228`) and the
  conditional proof; then decide whether to extend the spike (port dirty-master C8 tests,
  NCCL compatibility, or upstream PR updates) or fold M1 into the PR branch.
