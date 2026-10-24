# Memory preflight — job 62167742

**Status: PASS (preflight only).** All nine batch-256 framework/world
configurations are below the OD-8 whole-process sampled peak limit on one
A100. The 390-update campaign was **not** run. This is not a memory-over-time
or throughput result.

## Job

| Item | Value |
|---|---|
| Job | `62167742` (`static-full-preflight`) |
| Node | `lrdn3250.leonardo.local` |
| Allocation | `boost_qos_lprod`, 1 node, 1 task, **4 CPUs**, 4×A100-SXM-64GB, `02:00:00` |
| CPU policy | 4 CPUs per task, 1 Julia thread per rank; PyTorch 4 processes on 4 CPUs with one thread each |
| Slurm state | `COMPLETED`, exit `0:0`, elapsed `00:23:27` |
| Start/end (local) | `2026-10-24T21:19:25` / `2026-10-24T21:42:52` |
| Driver start/end (UTC) | `2026-10-24T19:19:31Z` / `2026-10-24T19:42:52Z` |
| Precompile | `FLUX_PRECOMPILE_OK` for current and historical Flux |
| Driver markers | `FULL_CAMPAIGN_READY PASS batch=256 updates=390`, `FULL_CAMPAIGN_preflight PASS`, `TIMING_DRIVER_EXIT=0` |
| Slurm log | `artifacts/logs/leonardo/2026-10-24-static-full-preflight-62167742.log` |

The allocation was deliberately reduced from 32 to 4 CPUs. CPU-side memory
and thread-pool usage scale with the core count, and throughput is not the
goal of this check. Timings are therefore not comparable to the 32-CPU
campaign.

## OD-8 result

`limit = 0.8 × 68,099,571,712 B = 51,955.8 MiB`; device total `64,944.8 MiB`.
`limit margin` is the headroom to the 80% gate, not to the whole device.

| Framework | W | Peaked process | Peak MiB | % of device | Limit margin MiB | Min training samples |
|---|---:|---|---:|---:|---:|---:|
| flux-current | 1 | rank 0 | 49,792.0 | 76.67% | **2,163.8** | 16 |
| flux-current | 2 | worst rank | 37,274.0 | 57.39% | 14,681.8 | 9 |
| flux-current | 4 | worst rank | 34,250.0 | 52.74% | 17,705.8 | 7 |
| flux-historical | 1 | rank 0 | 49,728.0 | 76.57% | **2,227.8** | 17 |
| flux-historical | 2 | worst rank | 41,914.0 | 64.54% | 10,041.8 | 8 |
| flux-historical | 4 | worst rank | 39,242.0 | 60.42% | 12,713.8 | 5 |
| pytorch | 1 | rank 0 | 10,894.0 | 16.77% | 41,061.8 | 18 |
| pytorch | 2 | worst rank | 6,138.0 | 9.45% | 45,817.8 | 9 |
| pytorch | 4 | worst rank | 3,994.0 | 6.15% | 47,961.8 | 5 |

The whole-process peak equals the training-window peak in every row. The
tightest cases are Flux at W=1: 49,792 MiB (current) and 49,728 MiB
(historical), about 76.6% of the device and only **2.1–2.2 GiB below the 80%
gate**.

## Peak composition — read this before comparing frameworks

The peak is dominated by memory already resident **before** the measured
training window:

| Framework | W | Last sample before window | Training peak | Increase during 20 updates |
|---|---:|---:|---:|---:|
| flux-current | 1 | 45,504 MiB | 49,792 MiB | +4,288 MiB |
| flux-current | 2 | 35,482 MiB | 37,274 MiB | +1,792 MiB |
| flux-current | 4 | 34,250 MiB | 34,250 MiB | 0 MiB |
| flux-historical | 1 | 45,504 MiB | 49,728 MiB | +4,224 MiB |
| flux-historical | 2 | 35,354 MiB | 41,914 MiB | +6,560 MiB |
| flux-historical | 4 | 36,426 MiB | 39,242 MiB | +2,816 MiB |
| pytorch | 1 | 10,894 MiB | 10,894 MiB | 0 MiB |
| pytorch | 2 | 6,138 MiB | 6,138 MiB | 0 MiB |
| pytorch | 4 | 3,992 MiB | 3,994 MiB | +2 MiB |

At Flux W=1, **91% of the 49.8 GiB peak is already present before the measured
window**. The measured 20-update increment is only 4.2 GiB. The harness calls
`GC.gc()` before timing but deliberately does **not** call `CUDA.reclaim()`, so
`nvidia-smi` also counts CUDA.jl's cached free blocks from setup and warmup.
PyTorch's process footprint is 4.6× smaller at W=1, which is a real operational
difference, but this table is **not** a clean live-tensor comparison of one
training step.

A short diagnostic is still required to separate live from cached memory:
after warmup, `GC.gc(); CUDA.synchronize(); CUDA.reclaim(); CUDA.synchronize()`,
record `nvidia-smi` and `CUDA.used_memory`/`cached_memory`, then run one and
twenty updates. If the footprint drops sharply, the 49.8 GiB is mostly cache;
if it stays high or grows with updates, Flux has a genuine memory problem for
this workload.

The simultaneous aggregate for W=2/W=4 sums memory across GPUs (for example
`flux-current W=4` is 136,872 MiB) and is not comparable to the single-device
80% limit; the gate is per process.

## Sampling coverage

Every rank has repeated whole-process samples and at least one training-window
sample; preflight requires one. Actual maximum trace gaps were 0.29–0.44 s
against the 0.25 s target. Training-window samples per rank were 5–18. The
training-window peaks are reported alongside the whole-process peaks; no
startup-only value exceeded the gate.

Sampled `nvidia-smi` process used-memory is not a continuous high-water mark.
It includes framework allocator caches and runtime/communication overhead.
The preflight window is 20 warmup + 20 measured updates; the campaign window is
20 warmup + 390 updates. A campaign run can therefore reach a higher peak than
this preflight, and this PASS does not establish that the campaign will stay
below 80%.

## Source identity

| Item | Value |
|---|---|
| Thesis commit | `660e363e6de5b6e4b398e542d79b6ce1adb381da` (Git-free copy; `expected-no-git`) |
| Thesis synced state SHA-256 | `1ddd2d20a3cdb83c6418807ddbebd7f7abc8a3952950d12990cdafc7c5698ecf` |
| Flux current commit / source tree | `f7b4520c…` / `83da2929…` |
| Flux historical commit / source tree | `8add0e42…` / `ad114ed3…` |
| Runtime | Julia `1.12.6`; Python `3.11.17`; torch `2.9.1+cu126`; NumPy `2.4.6`; driver `535.274.02` |
| Resources | `cpus_per_task=4`, `ntasks=1`, `gres=gpu:4` |

The source snapshot, dataset identity, runtime, resources, and launcher/checker
hashes are under this directory. The checker was re-run locally on this copy and printed the same nine PASS rows, `PREFLIGHT_SELECTED 256`, and `FULL_CAMPAIGN_READY PASS batch=256 updates=390`.

## Artifacts

- `preflight/decision.json` — selected batch 256, nine PASS rows.
- `preflight/b256/records/*.json` — nine timing records.
- `preflight/b256/samples/*.csv` and `.ranks/` — raw traces and per-rank
  PID/GPU registration.
- `driver.log`, `source-identity.txt`, `source-sha256.txt`, `runtime.json`,
  `resources.json`, `dataset-identity.json`, `provenance.json`.
- `source/` — source snapshot used by the checker.
- `recheck-decision.json` — independent local checker decision.
- `SHA256SUMS` — hashes of this local evidence directory.

## Next step

Do not treat this as campaign evidence. The next decision is whether to run
the 390-update memory campaign with the same 4-CPU policy and `REPEATS=3`; the
tight Flux W=1 margin is the main risk to watch.

Do not sync documentation-only edits before reusing the decision. The decision
binds `THESIS_STATE_SHA256=1ddd2d20…` to the synced remote tree; a later sync
changes that hash and the campaign readiness check rejects the decision. Keep
the remote tree unchanged or run a fresh preflight.
