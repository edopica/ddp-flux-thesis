# ADR-0002 - Baseline DDP path

Date: 2026-07-02
Status: Proposed

## Context

The thesis deliverable is a correct and usable DDP path for Flux.jl, not a new training algorithm.

## Decision

Start from the existing Flux DistributedUtils path where possible. Keep the first baseline boring and testable:

- CPU/MPI smoke tests first.
- GPU/NCCL after CPU behavior is understood.
- Local gradients computed with Zygote.
- Communication outside AD for the first baseline.
- Optimisers.jl for optimizer setup and updates.
- Functors-compatible traversal for nested models.
- Correctness before performance.

## Consequences

- Easier debugging and validation.
- Less risk of diverging from Flux idioms.
- Performance features such as bucketing, overlap, mixed precision, or custom kernels are deferred until correctness tests pass.
