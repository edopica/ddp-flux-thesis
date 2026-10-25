# Memory campaign completed — job 62195500 — 2026-10-25

**Status: PASS (complete 390-update campaign, 3 repeats).** All nine batch-256
framework/world configurations remained below the OD-8 whole-process sampled
peak limit (`51,955.8 MiB`, 80% of `64,944.8 MiB`) across all 390 measured
updates. Independent local artifact check passed: `FULL_CAMPAIGN_ARTIFACT_CHECK PASS`.

## Job summary

- **Job ID:** `62195500` (`static-full-memory`)
- **Node:** `lrdn2909.leonardo.local`
- **Allocation:** `boost_usr_prod`, `boost_qos_lprod`, 1 node, 1 task, 4 CPUs, 4×A100-SXM-64GB
- **Runtime:** `01:51:47` elapsed (`2026-10-24T23:43:22Z` to `2026-10-25T01:35:09Z`)
- **Exit code:** `COMPLETED 0:0`
- **Scope:** 3 repeats × 9 configurations (27 runs) of 390 measured updates (and 20 warmup) at batch 256.
- **Evidence root:** `artifacts/logs/static-scaling/2026-10-25-campaign-62195500`
- **Slurm log:** `artifacts/logs/leonardo/2026-10-25-static-full-memory-62195500.log`

## Memory performance findings

Device total: `64,944.8 MiB`. 80% OD-8 limit: `51,955.8 MiB`.
Sampled with `nvidia-smi` at 0.25 s target interval.

| Framework | W | Worst-rank peak (MiB) | % Device | Margin to 80% (MiB) | Startup peak (MiB) | Repeat spread (MiB) | Min train samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flux-current` | 1 | 49,856.0 | 76.77% | **2,099.8** | 45,504.0 | 96.0 | 320 |
| `flux-current` | 2 | 37,338.0 | 57.49% | 14,617.8 | 35,482.0 | 32.0 | 184 |
| `flux-current` | 4 | 34,282.0 | 52.79% | 17,673.8 | 34,282.0 | 32.0 | 140 |
| `flux-historical` | 1 | 49,760.0 | 76.62% | **2,195.8** | 45,504.0 | 0.0 | 320 |
| `flux-historical` | 2 | 41,978.0 | 64.64% | 9,977.8 | 35,354.0 | 32.0 | 167 |
| `flux-historical` | 4 | 39,370.0 | 60.62% | 12,585.8 | 36,458.0 | 64.0 | 109 |
| `pytorch` | 1 | 10,894.0 | 16.77% | 41,061.8 | 10,894.0 | 0.0 | 338 |
| `pytorch` | 2 | 6,138.0 | 9.45% | 45,817.8 | 6,138.0 | 0.0 | 176 |
| `pytorch` | 4 | 3,994.0 | 6.15% | 47,961.8 | 3,992.0 | 0.0 | 99 |

### 1. Stability across campaign length (20 vs 390 updates)

The primary question before launch was whether memory would accumulate over 390 updates
and exhaust the ~2.1 GiB margin observed in the 20-update preflight (`62167742`).
Comparing the worst-rank peak memory:
- `flux-current` W=1: `49,792 MiB` (preflight) -> `49,856 MiB` (campaign), delta **+64 MiB** (+0.13%).
- `flux-current` W=2: `37,274 MiB` (preflight) -> `37,338 MiB` (campaign), delta **+64 MiB** (+0.17%).
- `flux-current` W=4: `34,250 MiB` (preflight) -> `34,282 MiB` (campaign), delta **+32 MiB** (+0.09%).
- `flux-historical` W=1: `49,728 MiB` (preflight) -> `49,760 MiB` (campaign), delta **+32 MiB** (+0.06%).
- `flux-historical` W=2: `41,914 MiB` (preflight) -> `41,978 MiB` (campaign), delta **+64 MiB** (+0.15%).
- `flux-historical` W=4: `39,242 MiB` (preflight) -> `39,370 MiB` (campaign), delta **+128 MiB** (+0.33%).
- `pytorch` W=1, 2, 4: byte-identical peaks between preflight and campaign (`10,894`, `6,138`, `3,994 MiB`).

The campaign sampled peaks are close to those of the 20-update preflight.
This supports no large observed footprint increase over this horizon; it does
not prove absence of leaks.

### 2. Peak memory composition

- At `flux-current` W=1, `45,504 MiB` (91.3% of the `49,856 MiB` peak) was already
  allocated before the measured window began. The 390 measured updates added only `4,352 MiB`.
- At `flux-current` W=4, the peak was `34,282 MiB` at startup and remained `34,282 MiB`
  throughout all 390 updates (zero increase during training).
- The harness calls `GC.gc()` but does not call `CUDA.reclaim()`, so `nvidia-smi`
  process memory includes CUDA.jl's memory pool and cuDNN workspace. While the 4.6×
  process footprint difference between Flux and PyTorch at W=1 is an operational reality
  on the GPU cluster, it is a process footprint measurement, not a live-tensor footprint.

### 3. Throughput and scaling under 4 CPUs

Dataloading was constrained by design (`--cpus-per-task=4`) to prevent CPU-side memory scaling:
- `flux-current`: W=1 median `91.00 s` (1,097.1 img/s), W=2 median `52.34 s` (speedup `1.74x`, 86.9% eff), W=4 median `41.29 s` (speedup `2.20x`, 55.1% eff). Repeat 1 has timing outliers at W=1 and W=4; their cause is not established by the timing records.
- `flux-historical`: W=1 median `90.97 s` (1,097.5 img/s), W=2 median `47.53 s` (speedup `1.91x`, 95.7% eff), W=4 median `31.65 s` (speedup `2.87x`, 71.9% eff).
- `pytorch`: W=1 median `95.67 s` (1,043.6 img/s), W=2 median `49.76 s` (speedup `1.92x`, 96.1% eff), W=4 median `28.70 s` (speedup `3.33x`, 83.4% eff).

## Verification

The campaign was copied locally and verified independently with:
```bash
python3 scripts/static_scaling/check_full_campaign.py campaign \
  --root artifacts/logs/static-scaling/2026-10-25-campaign-62195500
```
Output:
- `FULL_TIMING ... PASS`
- `FULL_MEMORY ... PASS`
- `FULL_DIAGNOSTICS ... PASS` (15 comparisons, `NUMERICAL_EQUIVALENCE_CLAIM: none`)
- `FULL_CAMPAIGN_ARTIFACT_CHECK PASS`
- All 27 records and raw sample CSV traces verified.

## Follow-up advisory review

Reviewed the campaign report, current context, Phase 5 memory probe source,
and `experiments/static-scaling/OD-14-benchmark-acceptance.md`. No remote
command or new measurement ran in this review; the existing independent check
was read, not rerun. Recommendation: preserve this baseline and prepare a
small full-dataset W=1 current-Flux/PyTorch cache diagnostic before considering
a larger profiling campaign. Record synchronized allocator and process counters
before/after GC and cache release with consistent retained objects. Validate
the loaded allocator API semantics; the fixture probe uses a different path.
Cache release must stay outside timing and use its own evidence identity.

Claim qualifications for presentation: startup residency is not a quantitative
cache attribution; similar 20/390-update sampled peaks do not prove absence of
leaks; the timing outlier alone does not establish compilation as its cause.
The OD-14 deterministic-mode sensitivity in both frameworks remains required
for the final cross-framework report. General profiling is optional and should
answer a specific unresolved question.

Read-only review used `sed -n '1,180p' scripts/static_scaling/memory_flux.jl`,
`sed -n '1,110p' scripts/static_scaling/memory_torch.py`, and
`cat experiments/static-scaling/OD-14-benchmark-acceptance.md`.
CUDA.jl memory-pool behavior was checked against the
[official documentation](https://cuda.juliagpu.org/stable/usage/memory/).
Initial path lookup errors are recorded in
`wiki/failures/2026-10-25-advisory-review-paths.md`.

**Exact next action:** prepare and locally review the matched W=1 full-dataset
cache diagnostic; no additional campaign is launched by this recommendation.
