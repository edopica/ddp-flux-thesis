# 2026-07-29: Branch Cleanup — Merged into master

## Summary

Cleaned up the Flux fork: merged all DDP work into `master`, deleted stale branches, preserved only `upsteam-pr` for reference.

## Actions Taken

1. **Fast-forwarded `master` to `upstream/master`** (`780af631`).
2. **Merged `ddp/docs-examples` into `master`** (fast-forward).
   - `df95f602`: Fix three DDP correctness issues (original)
   - `f01e126a`: docs: add distributed training guide
3. **Merged `upsteam-pr` into `master`** (1 conflict in `docs/src/guide/distributed.md`, resolved by taking `upsteam-pr` version).
   - `426132de`: Fix three DDP correctness issues (revised)
   - `504e816e`: docs: remove explicit CUDA.device! from DDP guide
4. **Deleted old branches**: `ddp/docs-examples`, `ddp/upstream-pr`, `ddp/data`, `ddp/audit`, `ddp/tests`.
5. **Preserved**: `upsteam-pr` for reference.
6. **Updated README.md and wiki/context/current.md** to point to `master`.

## Current State

- Flux fork: `master` at `d583411f`, `upsteam-pr` preserved
- Base upstream: `780af631`

## Next Action

Fix launcher `@test true` silent-pass bug in `test/ext_distributed/runtests.jl`.
