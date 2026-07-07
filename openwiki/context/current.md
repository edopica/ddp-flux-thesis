# Current context

Date: 2026-07-05
Active checkpoint: C0 - Repository and API audit (Completed) -> C1 - Deterministic single-process reference loop
Active Flux branch: `ddp/audit`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commit: `57e29baf48edf50c0dd4fc9f027a8900ce3cef66`  
Thesis repo commit: `not committed yet`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority is understanding the current distributed code.
- CPU/MPI verification comes before GPU/NCCL work.
- The existing Flux distributed implementation (`DistributedUtils`) has a good foundation (MPI/NCCL abstractions, `DistributedOptimizer`, `DistributedDataContainer`) but lacks documentation, end-to-end examples, and some tests aren't being run by the test runner.

## What was done last

- Local Julia environment created pointing to the local fork.
- `mpiexecjl` installed and CPU/MPI smoke tests passed (`make smoke-cpu`).
- Completed C0 audit by writing `000-current-flux-distributed.md`.

## Commands that pass

- `make env`
- `make install-mpiexec`
- `make check`
- `make smoke-cpu`
- `make audit`

## Commands that fail

- None recorded yet.

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Start checkpoint C1: Create a deterministic single-process reference loop to use as a baseline for future DDP implementation.
