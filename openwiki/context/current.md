# Current context

Date: 2026-07-09
Active checkpoint: C2 - Distributed launch skeleton (Done) -> C3 - Model broadcast and parameter verification
Active Flux branch: `ddp/launch`  
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

- Implemented C2: Created `scripts/launch_skeleton.jl` that successfully launches via MPI.
- Resolved HPC execution blocker: `run_remote.sh` now provisions SLURM interactive allocations properly via `salloc`.
- Both local (`mpiexecjl`) and remote (`srun`) runs finish without deadlock.
- Refactored SSH workflow: Extracted HPC slurm logic into `scripts/remotes/hpc.conf` and updated `run_remote.sh` to dynamically source configs.
- Fixed `Makefile` and `setup_node.sh` to robustly handle `mpiexecjl` paths.
- Verified `make launch-4` on `deathstar` successfully without SLURM.
- **Implemented PMI Mismatch Guardrails**: Added strict environment checks in `FluxMPIExt.jl`, added a `make health-check` target, formalized `SLURM_MPI_TYPE=pmi2`, and documented HPC launch requirements in the README.

## Commands that pass

- `make env`
- `make install-mpiexec`
- `make check`
- `make health-check`
- `make smoke-cpu`
- `make audit`
- `make reference`
- `make reference-verify`
- `make launch-2`
- `make launch-4`

## Commands that fail

- None recorded yet.

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Start checkpoint C3: Model broadcast and parameter verification. Implement syncing model parameters so that all ranks start from an identical state.
