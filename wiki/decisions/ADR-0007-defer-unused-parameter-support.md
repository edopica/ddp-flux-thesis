ADR-0007 — Defer automatic unused-parameter support

Date: 2026-09-09  
Status: Proposed  
Related PR: FluxML/Flux.jl #2694

## Context
PR #2694 started with three goals:
1. Report known MPI and PMI configuration mismatches before MPI.Init().
2. Give every distributed worker the same number of observations.
3. Prevent deadlocks when different ranks use different model parameters.

The first two goals have limited and well-defined solutions. The third goal requires a change to the optimizer update process.
The first implementation added resolve_unused_parameters!!. This function replaced missing gradients with zero arrays. Users had to call the function manually before the optimizer update.
This approach was not sufficient for normal Flux training.

## Problem with the first approach
A normal Flux training step uses this sequence:
1. Flux.withgradient calculates the gradient tree.
2. Optimisers.update or Optimisers.update! traverses that tree.
3. DistributedOptimizer averages each gradient before the local optimizer update.

Optimisers.jl skips a parameter when its gradient is nothing. This behavior is correct for single-process training.
The behavior becomes unsafe when ranks use different conditional branches. One rank can skip a parameter while another rank calls allreduce! for that parameter. The ranks then call different collective operations and can deadlock.
The manual helper did not change this update sequence. Flux.train!, Flux.withgradient, and Optimisers.update did not call the helper automatically.
The associated test used a manually constructed gradient tree. It did not run a rank-dependent forward pass. It also did not perform a real distributed optimizer update.
Therefore, the test did not prove that normal conditional training was safe. The NEWS statement about deadlock prevention was stronger than the available evidence.

## Optimizer integration problem

A per-parameter optimizer rule cannot solve this problem by itself. The rule only receives one parameter and one gradient at a time.
Safe synchronization requires access to the complete model and gradient trees before any parameter update. Every rank must derive the same parameter order from the model.
The implementation must also distinguish two cases:
- A gradient is missing on one rank but present on another rank.
- A gradient is missing on every rank.
The first case requires a zero contribution from the rank with the missing gradient. The second case must preserve Optimisers.jl skip-step behavior.
This distinction affects stateful optimizers such as Adam. A zero gradient and an absent gradient do not always produce the same optimizer state.

## Architecture experiments
The ddp/integration-m1-spike branch tested a whole-model synchronization design.
Commit 04ddeaa0 introduced DistributedOptimizerState. This wrapper intercepted the complete optimizer update before Optimisers.jl traversed individual parameters.
Review found several correctness and compatibility problems. These problems included:
- incorrect treatment of non-trainable arrays;
- incorrect handling of immutable model values;
- differences between model and gradient structures;
- loss of globally absent gradient semantics;
- unsupported optimizer-rule composition;
- mutation of caller-owned gradients;
- device-placement risks.
Commit 3370b910 corrected most of these problems. It also added real automatic-differentiation tests at two and four MPI ranks.
Further review found additional cases. These included mixed tied and isbits parameters, GPU-safe presence handling, and compatibility with older trainable definitions.
These experiments proved that unused-parameter support is not a small helper function. It is an optimizer architecture change with a larger compatibility surface.

# Decision
PR #2694 will not add automatic unused-parameter support.
The PR will remove:
- resolve_unused_parameters!!;
- its distributed test file;
- its user documentation;
- its NEWS claim.
The PR title and description will no longer claim that it fixes conditional unused-parameter deadlocks.
The experimental commits will remain available on an archive or research branch. They will not enter the current PR.
This decision defers the feature. It does not reject the feature.
Reduced scope of PR #2694
PR #2694 will contain two correctness fixes.
MPI and PMI mismatch reporting
The initialization guard will report only recognized unsafe launcher and library combinations.
The implementation will not assume that every non-OpenMPI library requires PMI2. Correct system OpenMPI configurations must remain valid.
Equal-length data sharding
DistributedDataContainer will cyclically repeat valid observation indices when padding is necessary.
Empty input will produce a clear error. Tests will cover datasets smaller than the worker count.
The documentation will state that padding duplicates observations.

## Distributed CI support
The PR will retain the test infrastructure that directly supports these fixes:
- a shared setup file for distributed child processes;
- backend selection through ENV;
- visible child output;
- dedicated MPI jobs;
- the OpenMPI hardware-thread slot configuration for four-rank CI.
Native NCCL evidence is outside the scope of this PR.

### Rationale
This reduced scope makes every PR claim supported by a direct test.
The MPI guard has a specific failure mode. The data-padding fix also has a specific failure mode. Both fixes can use focused tests without changing the optimizer architecture.
Automatic unused-parameter handling needs a separate design review. Combining that work with the two smaller fixes increases review cost and merge risk.
A smaller PR also gives maintainers a clear choice. They can review the known fixes without accepting a new optimizer-state abstraction.

## Consequences
PR #2694 will no longer fix conditional models that use different parameters on different ranks.
Users must keep the same trainable parameter path active on every rank. The documentation must state this limitation.
The current DistributedOptimizer behavior remains unchanged. This avoids an unreviewed change to optimizer-state structure and public API behavior.
The M1 and M1.1 work remains useful. It records failure modes, design constraints, and required tests for future work.

## Future work
Unused-parameter support will use a separate issue and pull request.
Phase 1 — Define the contract
The design work will define these requirements with Flux and Optimisers maintainers:
- the feature works through Flux.setup and Flux.train!;
- users do not call a manual gradient-repair function;
- communication stays outside automatic differentiation;
- every rank uses the same collective order;
- globally absent gradients preserve skip-step behavior;
- the implementation uses public Optimisers.jl interfaces;
- the design keeps existing optimizer-state workflows where possible.
This phase will also determine whether Optimisers.jl needs a public whole-gradient-tree hook.
Phase 2 — CPU and MPI implementation
The first implementation will target CPU arrays and MPI.
The evidence will include:
- a real rank-dependent conditional model;
- two-rank and four-rank execution;
- analytic global-gradient comparisons;
- Descent and Adam state comparisons;
- globally absent parameters;
- nested models and non-trainable fields;
- tied parameters and wrapped arrays;
- immutable model values;
- explicit collective-count and order tests.
The implementation will not proceed to GPU support until these tests pass.
Phase 3 — GPU and NCCL support
A later phase will test the same behavior with CUDA arrays and NCCL.
The GPU work will disable scalar indexing. It will also make sure that gradient data does not move silently to CPU memory.
Hardware evidence will require at least two GPUs and two ranks.
Preserved research
The following local commits contain the optimizer experiments:
- 04ddeaa0 — first whole-model optimizer-state spike;
- 3370b910 — corrective CPU/MPI spike.
These commits are research evidence. They are not part of the reduced PR scope.
Their tests and review findings will guide the future unused-parameter proposal.

## Final outcome
PR #2694 will deliver two focused correctness fixes and reliable MPI CI.
Automatic unused-parameter support will continue as separate work after this PR is complete. This separation keeps the current PR understandable, testable, and reviewable.
