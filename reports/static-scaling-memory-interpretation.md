# GPU memory: capacity evidence and cache interpretation

Date: 2026-10-25. Scope: completed Leonardo jobs 62195500 and 62241616.

## Finding

The 390-update campaign establishes the observed operational process-memory
footprint and OD-8 headroom. The four-run W=1 diagnostic shows that most of
Flux's end-of-window GPU residency can be returned to the driver while
retaining the model and optimizer. These answer different questions.

| Counter/boundary | Current Flux, W=1 | PyTorch, W=1 |
|---|---:|---:|
| Full campaign worst sampled process peak | 49,856 MiB | 10,894 MiB |
| Diagnostic process residency after GC, two repeats | 49,792 / 49,760 MiB | 10,894 / 10,894 MiB |
| Diagnostic process residency after cache release | 1,270 / 1,334 MiB | 1,786 / 1,786 MiB |
| Process residency released at that boundary | 48,522 / 48,426 MiB | 9,108 / 9,108 MiB |

The diagnostic passed remotely and independently locally, with unchanged
model/optimizer hashes in every run and 118/118 evidence checksums matching.
Flux pool use dropped from 21,533.365 MiB at the raw post-training boundary
to 128.991 MiB after cleanup/GC, while the pool reservation stayed high.
Reclaim subsequently reduced reservation and process residency without
changing the remaining pool use. Host state hashing occurs between the raw
snapshot and explicit GC and may itself trigger collection.

## Why cache release is not the default after each update

A caching allocator retains freed blocks for reuse by later allocations.
Returning them to the driver after every update would discard that reuse;
subsequent updates may have to allocate the blocks again. CUDA.jl documents
that cached memory is automatically released under pressure, and that normal
high-level array operations should not require a manual reclaim to avoid OOM.
An OOM prevented only by manual reclaim is documented as a bug condition,
not expected routine operation.
[CUDA.jl memory management](https://cuda.juliagpu.org/stable/usage/memory/).

PyTorch makes the same tradeoff: its caching allocator speeds allocation and
allows deallocation without device synchronization; emptying the cache
returns unused blocks for other applications without freeing live tensors.
[PyTorch 2.9 CUDA memory management](https://docs.pytorch.org/docs/2.9/notes/cuda.html#memory-management).

Our experiment released cache after the measured updates. It did not test
training with reclaim after every update, a smaller GPU, a larger batch, or
memory pressure from another application. A lower post-release nvidia-smi
reading therefore establishes reclaimability at that boundary, not a 97%
reduction in the memory required for forward/backward or a measured speedup.
Potential reallocation/collection costs in this workload remain unmeasured.

This also leaves a useful engineering question: why does this workload build
such a large pool reservation and temporarily retain so many collectible
allocations? Reclaim diagnoses those categories but does not identify the
allocation sites or explain their individual sizes. Allocator reuse and GC
policy may matter; the current evidence does not attribute the whole amount
to cuDNN workspaces, a leak, DDP, or a single defective allocation.

## Reporting and implementation policy

Keep the original OD-8 sampled process peaks and margins in the capacity
report. Label post-GC/post-release snapshots separately. Report pool-used,
pool-reserved and process counters with their actual meanings; the final
process residual is not pure live-tensor memory. The Flux/PyTorch used-byte
comparison also reflects different retained gradient buffers.

W=1 reproduces this behavior without DDP. Multi-rank DDP overhead remains a
separate quantity. Similar 20- versus 390-update sampled peaks support no
large observed footprint increase over that horizon, not proof of no leaks.
The first-repeat timing outlier does not establish its compilation cause.

Keep the baseline training path unchanged. Explicit release can be useful at
a deliberate phase boundary when returning memory to another workload; its
cost and subsequent allocation behavior would need measurement before use
as a training optimization. CUDA.jl documents an advisory
`JULIA_CUDA_SOFT_MEMORY_LIMIT` for sharing a GPU; this has not been tested in
our campaign and is not a new default selected here.
[CUDA.jl memory limits](https://cuda.juliagpu.org/stable/usage/memory/#Memory-limits).

## Evidence and next action

- [390-update capacity campaign](../artifacts/logs/static-scaling/2026-10-25-campaign-62195500/RESULTS.md), job 62195500: 27 runs, all nine configurations under OD-8.
- [W=1 cache diagnostic](../artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616/RESULTS.md), job 62241616: four fresh-process runs, two repeats per framework, six snapshots each.
- [Diagnostic devlog and exact commands](../wiki/devlog/03-static-scaling/2026-10-25-cache-diagnostic.md).
- [OD-14](../experiments/static-scaling/OD-14-benchmark-acceptance.md): deterministic-mode sensitivity remains a separate final-report requirement.

**Exact next action:** integrate the operational-peak and cache-reclaimability
findings into the supervisor presentation with separate counter labels.
