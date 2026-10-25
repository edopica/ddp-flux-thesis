# W=1 cache diagnostic — job 62241616

**PASS:** all four fresh-process runs completed; remote and independent local
`CACHE_DIAGNOSTIC_CHECK PASS`; all 118/118 fetched evidence checksums match.
Slurm reports `COMPLETED 0:0`, elapsed `00:14:00`, node `lrdn3215`.

## Scope and controls

- Current Flux and PyTorch, W=1, batch 256, epoch 0; two repeats each,
  interleaved Flux/PyTorch, using the full-dataset mmap loader.
- 20 discarded warmup updates, reset from the common initial bundle, then 20
  measured updates. Six synchronized lifecycle snapshots per process.
- One task, four allocated CPUs, one A100-SXM-64GB; logical device 0,
  `CUDA_VISIBLE_DEVICES=0`. This exposes one GPU rather than the capacity
  campaign's four; W=1 training and CPU/thread policies are retained.
- Compute-node current-Flux precompile completed before measurements.
- GC and cache release occur after the measured updates. Retain the same
  model, optimizer, dataset maps and final loss during the final snapshots.
  PyTorch retains parameter `.grad` buffers; Flux does not retain equivalent
  buffers. State hashes before/after cleanup exactly match in every run.

## Observed memory

All table values are **MiB**, at synchronized point samples, except the
explicitly named sampled trace peaks below. Pool used is measured after GC
and remains unchanged after cache release. Reserved includes used bytes.

| Framework | Repeat | Post-training process | Post-GC process | Post-release process | Released process | Pool used after GC | Pool reserved, GC → release |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flux-current` | 1 | 49792 | 49792 | 1270 | 48522 | 128.991 | 49248 → 736 |
| `flux-current` | 2 | 49760 | 49760 | 1334 | 48426 | 128.991 | 49216 → 800 |
| `pytorch` | 1 | 10894 | 10894 | 1786 | 9108 | 192.375 | 10310 → 1202 |
| `pytorch` | 2 | 10894 | 10894 | 1786 | 9108 | 192.375 | 10310 → 1202 |

Flux initialization is 690 MiB process / 160 MiB pool reserved in both repeats.
After warmup, the process is 45,504 MiB with 44,960 MiB pool reserved.
After the model/optimizer reset and baseline GC, pool used is 128.991 MiB,
while process residency and pool reservation remain at those warmup levels.
At the raw post-training sample, Flux pool used is 21,533.365 MiB in both
repeats. Explicit GC reduces this to 128.991 MiB without reducing reserved
or process bytes. Cache release subsequently removes 48,426–48,522 MiB of
process residency (97.32–97.45% of post-GC residency), with unchanged pool
used and training-state hashes.

PyTorch initialization is 646 MiB process / 86 MiB reserved. Both repeats
reach 10,894 MiB process / 10,310 MiB reserved. GC does not change process
residency; `empty_cache()` releases 9,108 MiB (83.61%), leaving 1,786 MiB
process and 1,202 MiB reserved, with 192.375 MiB allocator-used unchanged.

Raw trace training peaks are 49,792 / 49,760 MiB for Flux and 10,894 MiB for
both PyTorch runs. Training samples are respectively 18 / 39 and 19 / 18;
all exceed density requirements. Maximum trace gaps are below 0.29 s.

## Interpretation and limits

The large end-of-window Flux process footprint is predominantly reclaimable
pool reservation, with substantial collectible allocations still counted in
pool use until GC. It is not approximately 50 GiB of retained model state.
W=1 reproduces the phenomenon: DDP is unnecessary for it to occur. This does
not quantify or exclude extra DDP memory at W=2/4.

Both frameworks cache memory. CUDA.jl `used_memory()` reports application-in-
use pool bytes; `cached_memory()` reports total pool backing memory, including
used bytes. Runtime documentation and the loaded CUDACore source excerpt
verify `USED_MEM_CURRENT` versus `RESERVED_MEM_CURRENT`. PyTorch counters have
their own documented tensor/allocator meanings. The 129 versus 192 MiB values
are not an intrinsic framework tensor-efficiency comparison, especially given
the different retained gradient buffers. Post-release residual process memory
includes runtime/library overhead and remaining reservations; it is not pure
live-tensor memory.

State hashing copies tensors to the CPU between the raw post-training sample
and explicit GC, and may itself trigger ordinary Julia GC. Therefore the
post-training-to-post-GC transition measures cleanup across that interval,
not an isolated attribution to the explicit GC call alone.

These are boundary snapshots and sampled traces, not exact transient
forward/backward peaks. The short horizon does not prove absence of leaks.
Reclaiming after training does not reduce the original measured operational
peak retroactively, establish a lower capacity requirement, or justify adding
reclaim to every training step. Diagnostic timings are not throughput evidence.
The accepted 390-update OD-8 process-memory baseline remains unchanged.

## Why caching remains the default

Pool reuse avoids returning and reacquiring blocks for successive updates.
CUDA.jl documents automatic reclamation under memory pressure; our release
occurs after training and does not establish a cheaper training step or a
smaller peak requirement. The exact allocation sites producing the large
reservation remain unisolated. See the
[interpretation and default caching rationale](../../../../reports/static-scaling-memory-interpretation.md)
for official references, evidence limits and the unchanged baseline policy.

## Provenance and verification

Julia 1.12.6, CUDA.jl 6.2.2; Python 3.11.17, torch 2.9.1+cu126, NumPy 2.4.6,
PyTorch CUDA 12.6 / cuDNN 91002. Loaded allocator documentation, source excerpts
and source SHA are embedded in each record. CUDACore memory source SHA is
`07219ff3fc77a712a18e958e68d7c5a44aa2faa0b7ed6533e350cb703a7b9ed4`.

Thesis Git parent: `976db1ffbc63472d306412b620d856a0cfd9cbc9`;
uncommitted deployed state:
`1c8f61ce2ac67e2be9429460d48ac1a0303e094aad8f3d902f689d4206ac73f9`.
Current Flux expected commit:
`f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`; Git-free commit/state values are
explicitly expected metadata, not an observed remote Git HEAD. Archived Flux
`src` (26 files) and `ext` (10 files) independently match the completed capacity
campaign snapshot byte for byte. No Flux source changed.

Portable `SHA256SUMS` covers the 118 original evidence files. Local additions
(this report, verification notes and scheduler record) are outside that original
manifest. Source-file checksums are independently validated by the checker.

```bash
(cd artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616 && sha256sum -c SHA256SUMS)
python3 scripts/static_scaling/check_cache_diagnostic.py \
  --root artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616
```

Output: `CACHE_DIAGNOSTIC_CHECK PASS`, four runs, exact cleanup state invariants.

The Slurm log is
`artifacts/logs/leonardo/2026-10-25-cache-diagnostic-62241616.log`;
its remote/local SHA-256 matches:
`3a196a0cd2055092463a497faa554c78e46f485efc3afe0c3f6d00d4aa124da5`.

Deployment, exact submission and fetch commands, local tests, and failures
are recorded in
`wiki/devlog/03-static-scaling/2026-10-25-cache-diagnostic.md` and
`wiki/failures/2026-10-25-cache-diagnostic-preparation.md`.

## Next action

Integrate the capacity and cache findings into the presentation with separate
labels for operational sampled peak and post-GC/post-release boundary counters.
No additional general memory campaign is needed for this question. The OD-14
deterministic-mode sensitivity remains a separate final-report requirement.
