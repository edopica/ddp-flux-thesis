# C8 Correctness Matrix

**Last verified against implementation: 2026-07-30**
_Sync: Replaced `reduce_distributedtest.jl` with `common_distributedtest.jl` in collective rows; added Comm:Dtype coverage matrix row; filled Source Guard/CI columns for implemented tests (N=1 sharding, Empty dataset, Uneven batches, Nested Models); marked unimplemented rows as Planned._

This document maps each invariant in Flux Distributed Data Parallel (DDP) training to its test file, expected behavior, and the historical regression it prevents.

## Public Behavior Decisions (C8.0)

Before adding tests, the following conventions are explicitly defined:

1. **Empty datasets**: Launching training with an empty dataset (`dataset_size = 0`) must throw an `ArgumentError` before distributed execution starts.
2. **Datasets smaller than world size**: For `0 < dataset_size < world_size`, valid samples are cyclically repeated so every rank receives the same number of elements (padding).
3. **Completely unused parameters**: For parameters unused by **all** ranks in a conditional model, their gradient leaf is `nothing`, and the optimizer state for that parameter must remain **frozen** (no update to momentum buffers).
4. **Loss reduction convention**:
    * Reference (single-process): `loss = sum(losses) / global_batch_size`
    * Each DDP rank: `local_loss = sum(losses_i) / local_batch_size`
    * Distributed gradient avg: `avg = sum(gradients) / world_size`
5. **Testing seeding convention**: Use rank-dependent seeding: `Random.seed!(base_seed + MPI.Comm_rank(comm))` after synchronization. Deterministic fixed tensors are strongly preferred for mathematical equivalence tests to rule out PRNG differences.

---

## The Matrix

| Invariant / Behavior | Test File | Ranks | Backend | Device | Expected Result | Protected Historical Failure | Source Guard Location | CI / Execution Command |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Data: N=1 sharding** | `data_distributedtest.jl` | 4 | MPI | CPU | Cyclic pad emits `1,1,1,1`. No out-of-bounds error. | Uncovered tiny-dataset padding bug causing OOB index. | `src/distributed/public_api.jl:276` | `make c8-mpi-4` |
| **Data: Empty dataset** | `data_distributedtest.jl` | 4 | MPI | CPU | `ArgumentError` thrown. | Silent execution with no steps. | `src/distributed/public_api.jl:266` | `make c8-mpi-4` |
| **Data: Shuffle determinism** | `data_distributedtest.jl` | 2/4 | MPI | CPU | Mutually exclusive shards across epochs (Planned). | Different ranks overlapping data randomly. | TBD | `make c8-mpi` / `c8-mpi-4` |
| **Data: Uneven batches** | `data_distributedtest.jl` | 4 | MPI | CPU | Identical local batch counts, partial last batch OK. | Deadlock due to differing loop iterations. | `src/distributed/public_api.jl:273-276` | `make c8-mpi-4` |
| **Comm: Sum vs Average** | `common_distributedtest.jl` | 2 | MPI | CPU | `avg` returns average, `+` returns sum. | Accidentally skipping division in all-reduce. | TBD | `make c8-mpi` |
| **Comm: Buffer reuse** | `common_distributedtest.jl` | 2 | MPI | CPU | Second all-reduce independent of first. | Stale buffer leakage between steps. | TBD | `make c8-mpi` |
| **Comm: Functors traversal** | `synchronized_distributedtest.jl` | 2 | MPI | CPU | Identical `collect_arrays` field order across ranks. | Silent message mismatching due to struct ordering. | TBD | `make c8-mpi` |
| **Comm: NCCL Native** | `common_distributedtest.jl` | 2 | NCCL| GPU | Operands are `CuArray`, results match MPI baseline. | Silent host-fallback treating MPI as NCCL pass. | TBD | `make c8-nccl` |
| **Comm: Dtype coverage matrix + nonzero root broadcast** | `common_distributedtest.jl` | 2 | MPI + NCCL | CPU + GPU | Float32/Float64 bcast/reduce/allreduce pass on MPI; Float16/Float32 on NCCL; bcast with root=1 works | Implicit MPI fallback masking NCCL test gaps; untested nonzero-root broadcast | `common_distributedtest.jl` (test is self-guarding) | `make c8-mpi` / `make c8-nccl` |
| **Sync: Nested Models** | `synchronized_distributedtest.jl` | 2 | MPI | CPU | All ranks hold identical nested parameters after sync. | Unsynchronized initial parameters. | `synchronized_distributedtest.jl:46-50` | `make c8-mpi` |
| **Sync: Multi-LR Opt** | `optimizer_distributedtest.jl` | 2 | MPI | CPU | State structure, hyperparameters match across ranks (Planned). | Divergent optimizer setup across ranks. | TBD | `make c8-mpi` |
| **Equiv: Multi-step Descent**| `optimizer_distributedtest.jl` | 2 | MPI | CPU | Gradients & params match 1-proc global-batch ref (Planned). | Silent lack of comms (tautological optimizer test). | TBD | `make c8-mpi` |
| **Equiv: Multi-step Adam** | `optimizer_distributedtest.jl` | 2 | MPI | CPU | Stateful optimizer momentum matches ref (Planned). | State initialized/advanced differently on ranks. | TBD | `make c8-mpi` |
| **Equiv: Grad Accumulation**| `optimizer_distributedtest.jl` | 2 | MPI | CPU | Equivalent to 1 macro-batch; comms only on final step (Planned).| Communication on every micro-batch (sync boundary bug). | TBD | `make c8-mpi` |
| **Cond: 2-rank disjoint** | `unused_parameters_distributedtest.jl` | 2 | MPI | CPU | Shared params avg, disjoint params scaled by world_size (Planned). | Divergent loss/grad on conditional subgraphs. | TBD | `make c8-mpi` |
| **Cond: 4-rank asymmetric** | `unused_parameters_distributedtest.jl` | 4 | MPI | CPU | Works with layers entirely unused on some ranks (Planned). | Deadlock on uneven parameter usage. | TBD | `make c8-mpi-4` |
| **Cond: resolve! ordering** | `unused_parameters_distributedtest.jl` | 2 | MPI | CPU | Missing resolve yields error, swapped yields error or pass (Planned). | Missing resolver deadlock/silent divergence. | TBD | `make c8-mpi` |
| **Cond: All-unused param** | `unused_parameters_distributedtest.jl` | 2 | MPI | CPU | Optimizer state remains frozen (no momentum update) (Planned). | Updating momentum for inactive parameters. | TBD | `make c8-mpi` |
