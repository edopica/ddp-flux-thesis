# Current context

Date: 2026-07-16
Active checkpoint: C7 - End-to-end examples
Active Flux branch: `ddp/upstream-pr`  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commits: `60272e41` (squashed fixes), `d00c5af6` (tests), `dc73fc75` (import fix)  
Thesis repo commit: `6a176ca`

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
- **PR branch ready**: `ddp/upstream-pr` on fork `edopica/ddp_flux` has 3 commits above upstream/master.
- **All tests pass**: reduce_distributedtest + unused_parameters_distributedtest (20/20) on HPC with 2 MPI ranks.

## What was done last

- Executed C6b: Prepared upstream PR branch with squashed commits, created test files, fixed imports, verified on HPC.
- All 3 fix areas validated: PMI guardrails, data padding, resolve_unused_parameters! (nested).

## Commands that pass

- `make env`
- `make precompile`
- `make install-mpiexec`
- `make check`
- `make health-check`
- `make smoke-cpu`
- `make audit`
- `make status`
- `make reference`
- `make reference-verify`
- `make launch-2`
- `make launch-4`
- `make sync-model`
- `make verify-sync`
- `make verify-data`
- `make verify-gradients`
- `make verify-conditional`

## Commands that fail

- Flux full test suite locally (OOM during Enzyme precompilation)

## Open questions

- What GPU/NCCL hardware is available locally?

## Next exact action

Begin C7: End-to-end examples. Build a documented DDP training script that uses the current MPI backend, reference model, and deterministic baseline to verify end-to-end correctness across 2 ranks.
