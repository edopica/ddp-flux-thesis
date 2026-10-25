# Static-full memory campaign results — job 62195500

**Status: PASS (complete 390-update campaign, 3 repeats).** All nine
batch-256 framework/world configurations stayed strictly below the OD-8
whole-process 80% device budget across all 390 measured updates.

## Job specification

| Item | Value |
|---|---|
| Job ID | `62195500` (`static-full-memory`) |
| Cluster | Leonardo (`login01.leonardo.local` -> compute `lrdn2909.leonardo.local`) |
| Partition / QoS | `boost_usr_prod` / `boost_qos_lprod` |
| Resources | 1 node, 1 task, **4 CPUs**, 4×A100-SXM-64GB |
| Walltime / Elapsed | `08:00:00` / `01:51:47` |
| Driver Start (UTC) | `2026-10-24T23:43:29Z` |
| Campaign Start (UTC) | `2026-10-24T23:48:23Z` |
| Driver End (UTC) | `2026-10-25T01:35:08Z` |
| Slurm status | `COMPLETED`, exit `0:0` |
| Markers | `FULL_CAMPAIGN_READY PASS batch=256 updates=390`, `FULL_CAMPAIGN_preflight PASS`, `FULL_CAMPAIGN_CAMPAIGN START`, `FULL_CAMPAIGN_ARTIFACT_CHECK PASS`, `TIMING_DRIVER_EXIT=0`, `FULL_CAMPAIGN PASS` |
| Local root | `artifacts/logs/static-scaling/2026-10-25-campaign-62195500` |
| Slurm log | `artifacts/logs/leonardo/2026-10-25-static-full-memory-62195500.log` |
| Driver log | `artifacts/logs/static-scaling/2026-10-25-campaign-62195500/driver.log` |

## Memory evaluation (OD-8 whole-process sampled peak)

Device total: `64,944.8 MiB` (`68,099,571,712` bytes).
80% OD-8 limit: `51,955.8 MiB` (`54,479,657,369` bytes).
Target sampling interval: 0.25 s (`nvidia-smi` per-process sampler).

| Framework | W | Worst-rank peak (MiB) | % Device | Margin to 80% (MiB) | Startup peak (MiB) | Repeat spread (MiB) | Min train samples | Max sample gap (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flux-current` | 1 | 49,856.0 | 76.77% | **2,099.8** | 45,504.0 | 96.0 | 320 | 0.362 |
| `flux-current` | 2 | 37,338.0 | 57.49% | 14,617.8 | 35,482.0 | 32.0 | 184 | 0.313 |
| `flux-current` | 4 | 34,282.0 | 52.79% | 17,673.8 | 34,282.0 | 32.0 | 140 | 0.388 |
| `flux-historical` | 1 | 49,760.0 | 76.62% | **2,195.8** | 45,504.0 | 0.0 | 320 | 0.363 |
| `flux-historical` | 2 | 41,978.0 | 64.64% | 9,977.8 | 35,354.0 | 32.0 | 167 | 0.344 |
| `flux-historical` | 4 | 39,370.0 | 60.62% | 12,585.8 | 36,458.0 | 64.0 | 109 | 0.407 |
| `pytorch` | 1 | 10,894.0 | 16.77% | 41,061.8 | 10,894.0 | 0.0 | 338 | 0.335 |
| `pytorch` | 2 | 6,138.0 | 9.45% | 45,817.8 | 6,138.0 | 0.0 | 176 | 0.344 |
| `pytorch` | 4 | 3,994.0 | 6.15% | 47,961.8 | 3,992.0 | 0.0 | 99 | 0.381 |

### Preflight (20 updates) vs campaign (390 updates) stability

The primary question before launch was whether expanding from 20 updates
(preflight job `62167742`) to 390 updates (campaign job `62195500`) would cause
memory accumulation that exceeds the 80% budget, given the narrow ~2.1 GiB margin
at W=1.

Comparing worst-rank peak memory:

| Configuration | 20-update preflight | 390-update campaign | Delta |
|---|---:|---:|---:|
| `flux-current` W=1 | 49,792.0 MiB | 49,856.0 MiB | **+64.0 MiB** (+0.13%) |
| `flux-current` W=2 | 37,274.0 MiB | 37,338.0 MiB | **+64.0 MiB** (+0.17%) |
| `flux-current` W=4 | 34,250.0 MiB | 34,282.0 MiB | **+32.0 MiB** (+0.09%) |
| `flux-historical` W=1 | 49,728.0 MiB | 49,760.0 MiB | **+32.0 MiB** (+0.06%) |
| `flux-historical` W=2 | 41,914.0 MiB | 41,978.0 MiB | **+64.0 MiB** (+0.15%) |
| `flux-historical` W=4 | 39,242.0 MiB | 39,370.0 MiB | **+128.0 MiB** (+0.33%) |
| `pytorch` W=1 | 10,894.0 MiB | 10,894.0 MiB | **0.0 MiB** |
| `pytorch` W=2 | 6,138.0 MiB | 6,138.0 MiB | **0.0 MiB** |
| `pytorch` W=4 | 3,994.0 MiB | 3,994.0 MiB | **0.0 MiB** |

The worst sampled Flux peaks differ by only 32–128 MiB between the 20-update
preflight and the 390-update campaign. This does not establish absence of
leaks or quantify every transient allocation.

### Memory composition caveat

As established during preflight:
1. At `flux-current` W=1, `45,504 MiB` of the `49,856 MiB` peak is already present
   before the measured training window (91.3% of the total). The 390 updates
   only account for `4,352 MiB` of additional allocation.
2. At `flux-current` W=4, the peak was `34,282 MiB` at startup and remained
   `34,282 MiB` during training (0 MiB growth).
3. The benchmark harness calls `GC.gc()` but does not call `CUDA.reclaim()`, so
   `nvidia-smi` samples process memory including CUDA.jl's internal memory pool
   blocks and cuDNN workspace. While the 4.6× W=1 footprint difference between
   Flux and PyTorch is operational reality on the cluster, it reflects cached
   process footprint rather than live tensor memory.

The subsequent [W=1 cache diagnostic](../2026-10-25-cache-diagnostic-62241616/RESULTS.md)
measures reclaimability directly. See the [memory interpretation](../../../../reports/static-scaling-memory-interpretation.md)
for why caching remains the default and why the post-release footprint does
not replace the operational peak above.

## Timing summary (4 CPUs per task)

The campaign used `--cpus-per-task=4` specifically to measure memory without
spurious CPU memory scaling. Dataloading throughput is constrained relative to the
32-CPU benchmark.

| Framework | W | Median (s) | Min (s) | Max (s) | Spread (s) | Images/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flux-current` | 1 | 91.000 | 90.957 | 610.827 | 519.870 | 1,097.1 | 1.0000 | 1.0000 |
| `flux-current` | 2 | 52.337 | 52.316 | 52.539 | 0.222 | 1,907.6 | 1.7387 | 0.8694 |
| `flux-current` | 4 | 41.288 | 40.685 | 111.286 | 70.601 | 2,418.1 | 2.2040 | 0.5510 |
| `flux-historical` | 1 | 90.970 | 90.953 | 90.973 | 0.020 | 1,097.5 | 1.0000 | 1.0000 |
| `flux-historical` | 2 | 47.529 | 47.502 | 47.577 | 0.075 | 2,100.6 | 1.9140 | 0.9570 |
| `flux-historical` | 4 | 31.650 | 31.641 | 32.177 | 0.537 | 3,154.5 | 2.8743 | 0.7186 |
| `pytorch` | 1 | 95.671 | 95.654 | 95.689 | 0.035 | 1,043.6 | 1.0000 | 1.0000 |
| `pytorch` | 2 | 49.765 | 49.759 | 49.795 | 0.036 | 2,006.2 | 1.9225 | 0.9612 |
| `pytorch` | 4 | 28.695 | 28.693 | 28.713 | 0.020 | 3,479.3 | 3.3341 | 0.8335 |

*Note on Flux-current Repeat 1:* the first repeat has timing outliers at
W=1 (`610.8 s`) and W=4 (`111.3 s`), while later minima are `90.96 s` and
`40.68 s`. These timing records do not establish compilation as the cause.
The medians remain descriptive under the recorded four-CPU policy.

## State diagnostics (OD-14)

All 15 state diagnostics comparisons completed and passed schema and integrity
checks:
- `NUMERICAL_EQUIVALENCE_CLAIM: none`.
- Observed parameter `rel_l2` differences are approximately 0.019–0.022.
  OD-14 assigns no numerical-equivalence threshold to these differences.
- Optimizer moments `m` and `v` were tracked and finite; `beta_power` was exactly
  `62/62` across all comparisons.
