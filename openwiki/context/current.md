# Current context

Date: 2026-07-14
Active checkpoint: C6 - Optimisers.jl integration (Done) -> C7 - End-to-end two-GPU example
Active Flux branch: `ddp/optimizer`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commit: `d6ff1bba08f39eb866ec5747bc23c1777cc88c6f`  
Thesis repo commit: `ddbe0e1`

## What is known

- The local Flux.jl fork exists.
- The first implementation priority is understanding the current distributed code.
- CPU/MPI verification comes before GPU/NCCL work.
- The existing Flux distributed implementation (`DistributedUtils`) has a good foundation (MPI/NCCL abstractions, `DistributedOptimizer`, `DistributedDataContainer`) but lacks documentation, end-to-end examples, and some tests aren't being run by the test runner.
- A deterministic single-process reference loop exists (`scripts/reference/ReferenceLoop.jl`) producing bit-identical results across runs (seed=42, 20 steps, Chain(Dense(1=>256,tanh), Dense(256=>1)), y=x³ data).
- Baseline is saved at `artifacts/baselines/reference_loop_baseline.jld2`.
- Loss converges from 0.240 to 0.034 over 20 steps (MSE, Adam 0.001).
- **Model broadcast works**: `synchronize!!` with `FluxDistributedModel` wrapper broadcasts all parameter tensors from rank 0 to all ranks with 0.0 max deviation.
- **Optimizer state broadcast works**: `synchronize!!` handles `Optimisers.Leaf` state correctly.
- Both the reference 2-layer MLP (769 params) and a 3-layer MLP (9729 params) sync correctly.

## What was done last

- Implemented C5 & C6: Validated that gradient accumulation over mini-batches matches global batch gradients within floating point precision limits.
- Found that `Adam` amplifies small gradient floating-point noise `O(eps(Float32))` into macroscopic parameter differences `O(0.0005)`. Used `Descent` for strictly deterministic parameter trajectory matching.
- Implemented `DistributedUtils.resolve_unused_parameters!` in `Flux.jl` to fix conditional graph deadlocks by replacing `nothing` gradients with zero-filled arrays.
- Created `scripts/sync/verify_gradients.jl` and verified cross-rank parameter synchronization over 20 steps.
- Updated `make verify-gradients` to run tests on HPC cluster via PMI2.

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
- `make sync-model`
- `make verify-sync`
- `make verify-data`
- `make verify-gradients`

## Commands that fail

- None recorded yet.

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Start checkpoint C7: End-to-end two-GPU example. Implement a complete example using real data (e.g. HuggingFaceDatasets.jl) that can be run on 2 GPUs to ensure NCCL communication paths are working.
