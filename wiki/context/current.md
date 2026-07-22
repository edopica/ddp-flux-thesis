# Current context

Date: 2026-07-22
Active checkpoint: C8 - Correctness battery
Active Flux branch: `ddp/docs-examples`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Thesis repo commit: `pending`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority was understanding the current distributed code (C0-C6b done).
- CPU/MPI verification and upstream PR prep is mostly done.
- C7 is complete: an end-to-end example `scripts/examples/train_ddp.jl` was successfully created, executed on HPC using `make example-ddp`, and added as a documentation guide (`distributed.md`) to the Flux upstream docs.
- The example explicitly handles a conditional graph (using `resolve_unused_parameters!!`) and operates conditionally on NCCL if CUDA is present.
- The `resolve_unused_parameters!` API in Flux was updated to `!!` in recent upstream PR work, and our example scripts are aligned with this.
- HPC precompilation works smoothly with `scripts/remote/precompile.sh hpc`.

## What was done last

- Executed C7: End-to-end examples.
- Created `scripts/examples/train_ddp.jl` demonstrating conditional graphs, data sharding, and synchronization.
- Ran test on HPC (`NTASKS=2 ./scripts/remote/run.sh hpc make example-ddp`) successfully.
- Drafted `docs/src/guide/distributed.md` in `ddp_flux` and added to `make.jl`.

## Commands that pass

- `make example-ddp` (HPC and local)
- `make precompile`
- (All previous C1-C6b checks)

## Open questions

- For C8, what specific known failure modes should be tested that aren't already covered by `reduce_distributedtest`, `unused_parameters_distributedtest`, and `data_distributedtest`?

## Next exact action

Begin C8: Correctness battery.
