# 2026-09-16 - Conditional-graph wrapper implementation plan

## Decision

The user approved completion of the `DistributedOptimizerState` wrapper. A possible Optimisers.jl API change will follow the wrapper evaluation.

Development will continue in `/home/kurapica/Projects/ddp_flux/flux-integration` on `ddp/integration-m1-spike`.

The implementation branch is at preserved M1.1 commit `3370b910`. It will first incorporate PR #2694 head `fee2c310`.

The feature remains separate from PR #2694.

## Handoff

Added `temp/conditional_graph_wrapper_implementation_plan.md`.

The plan includes:

- the Optimisers.jl skip contract,
- the current deadlock and silent-corruption mechanisms,
- the required three-state gradient semantics,
- the wrapper architecture,
- the whole-model synchronization algorithm,
- the branch integration procedure,
- phased RED and GREEN work,
- a complete CPU/MPI test matrix,
- evidence and stop conditions,
- the later Flux, Lux, and Optimisers.jl review questions.

## New technical findings

The M1.1 implementation is not ready to restore unchanged.

Its rebuild counter can diverge from unique buffer indices in a model that mixes tied and isbits parameters.

Its presence vector can live on a GPU. Host control flow then reads a GPU boolean value through scalar indexing.

The new implementation will use an explicit occurrence plan and CPU presence metadata. Parameter gradients will remain on their parameter devices.

The implementation will also add coverage for old tuple-returning `Optimisers.trainable` methods.

## Verification

`git diff --check -- temp/conditional_graph_wrapper_implementation_plan.md` passed.

No Flux source changed. No distributed test ran because this session produced the implementation handoff only.

## Next action

Make sure that both Flux worktrees are clean. Then record the commit IDs and fast-forward `ddp/integration-m1-spike` to `fee2c310`.

Run the routing and two-rank MPI baselines before production edits.
