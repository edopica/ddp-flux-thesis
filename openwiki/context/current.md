# Current context

Date: 2026-07-11
Active checkpoint: C3 - Model broadcast and parameter verification (Done) -> C4 - Distributed data sharding
Active Flux branch: `ddp/sync-model`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commit: `57e29baf48edf50c0dd4fc9f027a8900ce3cef66`  
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

- Implemented C3: Created `scripts/sync/broadcast.jl` and `scripts/sync/verify.jl`.
- `broadcast.jl`: builds models with different seeds per rank, broadcasts from rank 0, verifies bit-identical parameters via allreduce max-deviation check.
- `verify.jl`: 4 deep verification tests (per-tensor check, optimizer state, reference consistency, larger model).
- Added `make sync-model` and `make verify-sync` targets.
- Both pass on 2 processes locally with 0.0 max deviation.

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

## Commands that fail

- None recorded yet.

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Start checkpoint C4: Distributed data sharding. Implement `DistributedDataContainer` usage and verify dataset coverage and duplication policy are documented and tested.
