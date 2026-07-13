# Current context

Date: 2026-07-13
Active checkpoint: C4 - Distributed data sharding (Done) -> C5 - Gradient synchronization minimal case
Active Flux branch: `ddp/data`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commit: `d6ff1bba08f39eb866ec5747bc23c1777cc88c6f`  
Thesis repo commit: `pending commit`

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

- Implemented C4: Fixed mathematical flaw in `DistributedDataContainer` that caused bounds errors and DDP deadlocks when distributing unaligned dataset sizes.
- Replicated PyTorch's `DistributedSampler` default behavior by padding datasets.
- Created `scripts/data/verify_sharding.jl` and `notebook/C4_Data_Sharding.ipynb`.
- Added `make verify-data`.
- All ranks now process exactly the same number of items and batches per epoch.

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

## Commands that fail

- None recorded yet.

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Start checkpoint C5: Gradient synchronization minimal case. Ensure DDP gradients match single-process global-batch gradients within numerical tolerance.
