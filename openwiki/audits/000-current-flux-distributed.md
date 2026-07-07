# C0 - Audit current Flux distributed implementation

Date: 2026-07-05  
Flux repo path: `/home/kurapica/Projects/ddp_flux/ddp_flux`  
Flux commit: `57e29baf48edf50c0dd4fc9f027a8900ce3cef66`  
Branch: `ddp/audit`

## Pass condition

One-page audit note covering existing API, missing pieces, risky areas, and chosen baseline path.

## Files inspected

- [x] `src/distributed/backend.jl` — Backend types and GPU-awareness preferences
- [x] `src/distributed/public_api.jl` — `DistributedUtils` module (308 lines)
- [x] `ext/FluxMPIExt/FluxMPIExt.jl` — MPI backend implementation (181 lines)
- [x] `ext/FluxMPINCCLExt/FluxMPINCCLExt.jl` — NCCL backend implementation (110 lines)
- [x] `test/ext_distributed/` — 6 test files (common, data, optimizer, synchronized, reduce, runtests)
- [x] Flux docs mentioning distributed training — **none found**
- [x] Open issues or TODOs found locally in relevant files — **none found**

## Current API surface

### Backend initialization

- `DistributedUtils.initialize(MPIBackend)` / `DistributedUtils.initialize(NCCLBackend)` — calls `MPI.Init()`, assigns GPU devices round-robin via `MLDataDevices.set_device!`.
- `DistributedUtils.get_distributed_backend(T)` — returns a backend struct wrapping the communicator. Requires prior `initialize` call.
- `NCCLBackend` wraps an `MPIBackend` internally — MPI is always initialized first, then NCCL unique-ID is broadcast via MPI.
- GPU-aware MPI controlled by `@load_preference` constants (`MPI_CUDA_AWARE`, `MPI_ROCM_AWARE`); when false, data is copied to CPU before MPI calls.

### Rank, world size, and device handling

- `DistributedUtils.local_rank(backend)` → `MPI.Comm_rank` / `NCCL.rank`
- `DistributedUtils.total_workers(backend)` → `MPI.Comm_size` / `NCCL.size`
- All three are marked `@non_differentiable`.
- Device assignment happens at init time only, round-robin by local rank.

### Collectives

| Collective | In-place variant | Two-buffer variant | `avg` support | Non-differentiable |
|---|---|---|---|---|
| `bcast!` | ✅ | ✅ | N/A | ✅ |
| `allreduce!` | ✅ | ✅ | ✅ (sum then ÷ nworkers) | ✅ |
| `reduce!` | ✅ | ✅ | ✅ (sum then ÷ nworkers) | ✅ |

- Device mismatch between `sendbuf` / `recvbuf` is handled with a warning and copy.
- For non-CUDA-aware MPI, GPU arrays are copied to CPU → MPI call → `copyto!` back.
- NCCL extension falls back to MPI for non-CuArray arguments.

### Model synchronization

- `synchronize!!(backend, obj; root=0)` — recursive `fmap`-based broadcast from root to all ranks.
- Requires wrapping the model in `FluxDistributedModel(model)` to trigger `fmap`.
- Traverses `NamedTuple`, `Tuple`, `AbstractArray` recursively; scalar `isbitstype` values are broadcast via `bcast!([val])[]`.
- Non-bitstype, non-container values are silently returned unchanged (potential correctness gap).
- Also handles `Optimisers.Leaf` — synchronizes optimizer state.

### Data sharding

- `DistributedDataContainer(backend, data)` — partitions observations into `ceil(N / nworkers)` chunks per rank.
- Uses `MLUtils.numobs` and `MLUtils.getobs` interface.
- Implements `Base.length` and `Base.getindex`.
- **Last rank may receive fewer observations** (remainder partition). No duplication, no shuffling.

### Optimizer integration

- `DistributedOptimizer(backend, opt) <: AbstractRule` — wraps any `Optimisers.jl` rule.
- `Optimisers.apply!` calls `allreduce!(backend, gradient, avg)` **before** delegating to the inner optimizer.
- `Optimisers.init` and `Optimisers._adjust` delegate to inner optimizer.
- Gradient averaging convention: **mean** (divide by `total_workers`).

### Tests

| File | What it tests | Notes |
|---|---|---|
| `common.jl` | `bcast!`, `reduce!`, `allreduce!` with sum and avg | Tests both `Array` and device-specific array types |
| `data.jl` | `DistributedDataContainer` partition sizes, sum coverage | Verifies global sum matches via allreduce |
| `optimizer.jl` | `DistributedOptimizer` setup, synchronize, update | Compares against single-process optimizer |
| `synchronized.jl` | `synchronize!!` for NamedTuple, Tuple, Leaf, nothing, Symbol, Int | Most comprehensive test file |
| `reduce_distributedtest.jl` | `reduce!` with sum, rank-based values | Standalone MPI test |
| `runtests.jl` | Runner: discovers `*_distributedtest.jl`, runs via `mpiexec` | Only `reduce_distributedtest.jl` matches the pattern |

**Key observation**: `common.jl`, `data.jl`, `optimizer.jl`, `synchronized.jl` are NOT discovered by `runtests.jl` because they don't end in `_distributedtest.jl`. They appear to be intended for inclusion but are **not run by the current test harness**.

## What can be reused

1. **Backend abstraction** — clean MPI / NCCL split via extensions; device fallback logic works.
2. **Collective primitives** — `bcast!`, `allreduce!`, `reduce!` with device-aware fallback.
3. **`DistributedOptimizer`** — sound `Optimisers.jl` integration; gradient averaging before update.
4. **`synchronize!!`** — recursive broadcast for model params and optimizer state.
5. **`DistributedDataContainer`** — basic partition-based sharding.
6. **Test patterns** — the test files exercise the API well even if the runner doesn't pick them all up.

## Missing pieces

1. **No Flux documentation** — zero mentions of distributed training in `openwiki/`. No user-facing guide exists.
2. **No end-to-end training loop example** — only primitive tests, no complete DDP training script.
3. **Test runner gap** — most test files (`common.jl`, `data.jl`, `optimizer.jl`, `synchronized.jl`) are not run by `runtests.jl`.
4. **No gradient correctness test** — no test that verifies DDP gradients match a single-process global-batch gradient.
5. **No multi-step parameter synchronization test** — optimizer test does only 1 step; no check that params stay in sync over multiple steps.
6. **No data shuffling / epoch logic** — `DistributedDataContainer` is a static partition, no support for reshuffling across epochs.
7. **No loss scaling convention** — no guidance on whether to average or sum loss before backward pass.
8. **No `FluxDistributedModel` documentation** — the wrapper requirement is not documented; easy to miss.
9. **No profiling infrastructure**.

## Risky areas

1. **`synchronize!!` silently returns non-bitstype non-container values unchanged** — if a model contains a custom struct leaf (e.g., a scale factor wrapped in a struct), it will NOT be broadcast. This is a silent correctness bug.
2. **`DistributedOptimizer.apply!` calls `allreduce!` which is `@non_differentiable`** — gradient averaging is outside AD, which is correct for DDP, but this means second-order gradient methods would silently drop the collective. This is fine for the thesis scope but should be documented.
3. **`avg` in MPI: sum then divide** — this is numerically different from a true average for large worker counts due to floating point; unlikely to matter in practice but worth noting.
4. **Non-CUDA-aware MPI path allocates temporaries** — every collective creates CPU copies; could be a perf bottleneck but acceptable for correctness-first work.
5. **`DistributedDataContainer` last-rank imbalance** — the last rank can have significantly fewer observations. With 2 ranks and odd N, this is 1 observation difference; with many ranks, the imbalance grows.
6. **`reduce!` avg on non-root ranks** — the division `sendrecvbuf ./= total_workers` happens on all ranks, but `MPI.Reduce!` only stores the result at root. Non-root ranks get corrupted data. (Not an issue if they don't use the buffer, but the API doesn't warn about it.)

## CPU verification status

Command:

```bash
make smoke-cpu MPIEXECJL=~/.julia/bin/mpiexecjl
```

Result: **PASS** — `rank=0 world=2 allreduce_sum=3`, `rank=1 world=2 allreduce_sum=3`

## GPU/NCCL status

Not tested (no CUDA hardware available on this machine).

## Chosen baseline path

1. **CPU-only, MPI backend** for initial development (C1–C6).
2. Use `FluxMPIExt` as-is; do NOT modify collective implementations.
3. Build a deterministic single-process reference loop first (C1), then replicate it with DDP.
4. Keep gradient communication **outside AD** (use `@non_differentiable` collectives).
5. Use `DistributedOptimizer` with gradient averaging for the main path.
6. Wrap model in `FluxDistributedModel` for `synchronize!!`.
7. GPU/NCCL testing deferred to C7+ when hardware is available.

## Next checkpoint

C1 — Single-process deterministic reference loop with fixed loss, gradients, and parameter updates stored as baseline.
