# ADR 0003: MPI Environment Strategy and PMI Protocol Guardrails

## Date
2026-07-10

## Status
Accepted

## Context
During the testing of the distributed launch skeleton (C2) on HPC clusters running SLURM, we encountered cryptic native library aborts during `MPI.Init()`. Investigation revealed a mismatch between the cluster's default Process Management Interface (PMI) protocol (PMIx) and the protocol expected by Julia's default MPI binary, `MPICH_jll` (PMI2 on Linux). 

While PMIx is the modern standard for exascale computing and offers significantly faster bootstrapping at scale, adopting it immediately would require users and CI to configure `MPI.jl` to use a system-provided MPI library (`MPIPreferences.use_system_binary()`). This breaks the "zero-configuration" reproducible default that is critical for our initial correctness and baseline checkpoints (C0-C8).

Furthermore, we need to ensure that our current architectural choices for Distributed Data Parallel (DDP) do not block future scalability goals, such as Fully Sharded Data Parallel (FSDP) on thousands of GPUs, which heavily relies on the performance characteristics of PMIx/system MPI implementations.

## Decision
1. **Implement PMI2 Guardrails (Soft Lock):** We have added strict environment checks in `FluxMPIExt.jl` that explicitly require a PMI2 environment (or warn/abort if PMIx is detected) to protect the default `MPICH_jll` binary from obscure segmentation faults.
2. **Maintain Agnostic Core Logic:** The core DDP (and future FSDP) communication logic will remain strictly agnostic to the underlying PMI wire protocol. We will rely exclusively on standard MPI collectives (`Bcast`, `Allreduce`, `Allgather`, `Reduce_scatter`).
3. **Deferred Transition to System MPI (PMIx):** We will retain the `MPICH_jll` + PMI2 setup for all correctness baselines. We will smoothly drop the guardrails and transition to system MPI binaries (enabling PMIx) during the performance and GPU-scaling checkpoints (C9/C10), where hardware-aware MPI and NCCL integration become necessary.

## Rationale
- **Reproducibility:** This strategy preserves out-of-the-box reproducibility for local development and CPU smoke tests without requiring users to compile OpenMPI or MPICH.
- **Safety:** The guardrails provide clear, actionable Julia-level errors instead of uncatchable native C aborts when environment mismatches occur.
- **Future-proofing:** Because FSDP and DDP algorithms fundamentally rely on standard MPI collectives rather than dynamic process management features specific to PMIx, our code remains natively compatible with massive scale. When the time comes to scale to 1000+ GPUs, the transition to PMIx will be a configuration change (`JULIA_MPI_BINARY=system`), not a code rewrite, leaving the door fully open for the Julia exascale community.
