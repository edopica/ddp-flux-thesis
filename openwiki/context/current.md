# Current context

Date: 2026-07-09
Active checkpoint: C1 - Deterministic single-process reference loop (Done) -> C2 - Distributed launch skeleton
Active Flux branch: `ddp/reference-loop`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commit: `57e29baf48edf50c0dd4fc9f027a8900ce3cef66`  
Thesis repo commit: `pending commit`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority is understanding the current distributed code.
- CPU/MPI verification comes before GPU/NCCL work.
- The existing Flux distributed implementation (`DistributedUtils`) has a good foundation (MPI/NCCL abstractions, `DistributedOptimizer`, `DistributedDataContainer`) but lacks documentation, end-to-end examples, and some tests aren't being run by the test runner.
- A deterministic single-process reference loop exists (`scripts/ReferenceLoop.jl`) producing bit-identical results across runs (seed=42, 20 steps, Chain(Dense(1=>256,tanh), Dense(256=>1)), y=x³ data).
- Baseline is saved at `artifacts/baselines/reference_loop_baseline.jld2`.
- Loss converges from 0.240 to 0.034 over 20 steps (MSE, Adam 0.001).

## What was done last

- Successfully configured the `deathstar` remote node environment via SSH using `sync_code.sh deathstar` and `setup_node.sh deathstar` (after fixing `setup_node.sh` to use `bash -l -c` to invoke the correct Julia version).
- Added documentation for `deathstar` remote node (`openwiki/context/deathstar.md`), emphasizing cautious usage since it's a shared personal PC without SLURM.
- Implemented Remote Execution & HPC Integration Plan (created `sync_code.sh`, `setup_node.sh`, `run_remote.sh` and documented in `remote-nodes.md`).
- Implemented C1: deterministic reference loop module (`scripts/ReferenceLoop.jl`).
- Created `scripts/save_reference.jl` (baseline generator) and `scripts/verify_reference.jl` (determinism verifier).
- Added JLD2 dependency for serialization.
- Added `make reference` and `make reference-verify` targets.
- Created `artifacts/baselines/` with `.gitignore` for JLD2 files.
- Both `make reference` and `make reference-verify` pass — PASS: all 20 steps are bit-identical.

## Commands that pass

- `make env`
- `make install-mpiexec`
- `make check`
- `make smoke-cpu`
- `make audit`
- `make reference`
- `make reference-verify`

## Commands that fail

- None recorded yet.

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Start checkpoint C2: Create a distributed launch skeleton (2-process and 4-process MPI launches without deadlock).
