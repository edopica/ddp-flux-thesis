# Current context

Date: 2026-07-29
Active checkpoint: C8 - Correctness battery
Active Flux branch: `master` (at `d583411f`, merging `ddp/docs-examples` + `upsteam-pr`)
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Preserved reference branch: `upsteam-pr`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority was understanding the current distributed code (C0-C6b done).
- CPU/MPI verification and upstream PR prep is mostly done.
- C7 is complete: an end-to-end example `scripts/examples/train_ddp.jl` was successfully created, executed on HPC using `make example-ddp`, and added as a documentation guide (`distributed.md`) to the Flux upstream docs.
- The example explicitly handles a conditional graph (using `resolve_unused_parameters!!`) and operates conditionally on NCCL if CUDA is present.
- The `resolve_unused_parameters!` API in Flux was updated to `!!` in recent upstream PR work, and our example scripts are aligned with this.
- HPC precompilation works smoothly with `scripts/remote/precompile.sh hpc`.

## What was done last

- **Branch cleanup**: Merged all DDP work (`ddp/docs-examples` + `upsteam-pr`) into `master` at `d583411f`. Deleted stale branches (`ddp/docs-examples`, `ddp/upstream-pr`, `ddp/data`, `ddp/audit`, `ddp/tests`). Preserved `upsteam-pr` for reference.
- Merge conflict in `docs/src/guide/distributed.md` resolved by taking `upsteam-pr` version (superset: validation loop, checkpointing, `Random.seed!`, no explicit `CUDA.device!`).
- Updated README.md and wiki/context/current.md to point to `master`.
- Conducted roundtable review of C8 plan: launcher `@test true` bug identified as P0 fix.

## Commands that pass

- `make example-ddp` (HPC and local)
- `make precompile`
- (All previous C1-C6b checks)

## Open questions

- (Resolved by C8.md) Known failure modes to test: tiny-dataset sharding (N=1, W=4), NCCL silent fallback, tautological optimizer test, missing per-test timeouts, CI not enabling distributed flags.

## Next exact action

P0: Fix the launcher's `@test true` silent-pass bug in `test/ext_distributed/runtests.jl`.
