# Fix three DDP deadlock/crash scenarios in DistributedUtils

This PR addresses three independent correctness issues in Flux's distributed
training path (MPI backend).  Each bug causes either a deadlock or a hard crash
when running DDP with non‑trivial models or launch configurations.

## 1. PMI version mismatch guardrails

**Problem:** Julia's default `MPICH_jll` speaks PMI2.  When launched under a
PMIx launcher (e.g. `srun` without `--mpi=pmi2`) or under OpenMPI's `mpirun`,
`MPI.Init()` aborts with an uninformative message, making diagnosis difficult.
Even inside a Slurm allocation with `mpiexecjl`, the missing PMI2 variables
produce a confusing hang.

**Fix:** Before calling `MPI.Init()`, check for environment markers:

- `PMIX_RANK` → error with PMIx mismatch message  
- `OMPI_COMM_WORLD_RANK` → error with OpenMPI mismatch message  
- `SLURM_JOB_ID` but no `PMI2_*` → warn about missing `--mpi=pmi2`

A `force=true` keyword allows advanced users to bypass the check (e.g. when
using a custom MPI build).  The logic is entirely inside `FluxMPIExt` and
runs before any MPI symbols are touched.

## 2. Padding in DistributedDataContainer to prevent allreduce deadlocks

**Problem:** When `n_samples % n_workers ≠ 0`, `DistributedDataContainer`
assigned fewer batches to the last rank.  That rank would exit the training
loop early while the other ranks blocked inside an `allreduce` call, causing a
permanent deadlock.

**Fix:** Pad the index sequence so every rank receives the same number of
elements.  Surplus slots are filled by wrapping around from the start
(`[1, 2, …, n_samples, 1, 2, …]`).  The data‑loading logic is unchanged;
only the index array passed to `DistributedDataContainer` is padded.

## 3. `resolve_unused_parameters!!` for conditional computation graphs

**Problem:** A model with conditional branches (e.g. `if`/`else` selecting
different layers) can produce `nothing` gradients on ranks that did not execute
a particular parameter.  When `DistributedOptimizer` later calls `allreduce`
over all gradients, the mismatched set of buffers deadlocks — different ranks
participate in different collectives.

**Fix:** New public function `resolve_unused_parameters!!(backend, gs, model)`
replaces every `nothing` gradient with a zero‑filled array matching the
parameter's shape, element type, and device.  It walks nested structures
(NamedTuples, etc.) and works with arbitrary tree depths thanks to Functors'
`fmap`.

## Tests added / updated

| File | Tests |
|------|-------|
| `test/ext_distributed/data_distributedtest.jl` | Renamed from `data.jl` to run with the test suite. 3 test groups: evenly-divisible partition, non-divisible (padded) partition with correct padded sum, and the original 10‑element case with expected padded sum. |
| `test/ext_distributed/unused_parameters_distributedtest.jl` (new) | 4 test groups (20 tests): replaces‑nothing‑with‑zeros, nested NamedTuples, no‑op when all present, preserves exact shapes and element types (Float32/Float64). Runs on both MPI and NCCL backends. |
| `test/ext_distributed/reduce_distributedtest.jl` | Fixed pre-existing upstream breakage in `reduce_distributedtest.jl` (added missing imports required for the test to run at all). |
| Pre-existing Orphaned Tests | Renamed `common.jl`, `optimizer.jl`, and `synchronized.jl` to `_distributedtest.jl` suffix so they are picked up by the test runner. Also fixed pre-existing upstream breakage (missing imports). |

All tests pass with `mpiexecjl -n 2` on CPU/MPI.

## Files changed

```
 NEWS.md                                            | 15 ++++++++++++
 docs/src/guide/gpu.md                              | 18 ++++++++++++++
 ext/FluxMPIExt/FluxMPIExt.jl                       | 37 +++++++++++++++++-
 src/distributed/public_api.jl                      | 38 ++++++++++++++++++-
 .../{common.jl => common_distributedtest.jl}       | 18 +++++++++-
 test/ext_distributed/data.jl                       | 24 ---------------
 test/ext_distributed/data_distributedtest.jl       | 62 +++++++++++++++++++++++++
 .../{optimizer.jl => optimizer_distributedtest.jl} | 19 +++++++++-
 test/ext_distributed/reduce_distributedtest.jl     |  4 +++
 ...hronized.jl => synchronized_distributedtest.jl} | 19 +++++++++-
 .../unused_parameters_distributedtest.jl           | 80 ++++++++++++++++++++++++++++++
 11 files changed, 310 insertions(+), 39 deletions(-)
```

## Backward compatibility

All changes are additive or replace a broken code path with a working one.
`resolve_unused_parameters!!` is a new, optional public API — existing
training loops that don't use conditional computation are unaffected.
The PMI check can be bypassed with `force=true`.
