# Timing review response — campaign 61314790

Date: 2026-10-19. Campaign: `61314790`, Leonardo, `lrdn3064`.
Scope: response to the four P2 findings in the
[fixture timing correctness review](../../../../../reports/static-scaling-fixture-timing-review.md).
The review accepted the reported medians, throughput, speedups, and efficiencies
and found no timing-boundary, sample-count, or gradient-averaging defect.

**Verdict retained: qualified timing PASS.** The evidence is bounded fixture
training-only timing plus a descriptive 300-update final-state snapshot
diagnostic. No numerical-equivalence rule is applied; the diagnostic does not
establish equivalence.

## Disposition

| Finding | Status | Summary |
|---|---|---|
| R1 — checker accepted incomplete correctness/provenance evidence | Fixed | The checker now enforces schema, precision, job/node, fixture/bundle, state hashes, all 36 source hashes, driver lines, and `provenance.json`. |
| R2 — full-length numerical agreement unmeasured | Measured; descriptive only | Job `61366015` compared final-state arrays after 300 updates. Differences are descriptive; no equivalence rule is applied. |
| R3 — execution policies differ | Documented; not eliminated | Determinism policy and timed gradient checks differ. The cross-framework comparison is descriptive, not a communication-only isolation. |
| R4 — provenance incomplete | Partially fixed | `provenance.json` now records source hashes, policy, and drift; the launcher captures `runtime-env.txt` for future runs. The two loaded shared helpers differ from the Phase 4 accepted snapshot. |

## R1 — checker accepted incomplete evidence (P2)

The checker now requires and validates:

- exact schema `static-timing-pilot-v1`;
- exact Flux precision `FP32 strict FMA, TF32 off` and the exact PyTorch
  precision flag dictionary (`cuda_matmul_allow_tf32=false`,
  `cudnn_allow_tf32=false`, `float32_matmul_precision="highest"`,
  `cudnn_deterministic=true`, `cudnn_benchmark=false`);
- non-empty `job` and `node`;
- fixture SHA-256
  `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3`
  and bundle SHA-256
  `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`;
- complete state hashes: Flux keys `param`, `m`, `v`, `beta_power`; PyTorch
  rank hashes `param`, `m`, `v`, `step` with `step=300`;
- all 36 source-snapshot hashes against `source-sha256.txt`;
- `driver.log` lines `FLUX_PRECOMPILE_OK`, `STATIC_TIMING_campaign PASS`, and
  `TIMING_DRIVER_EXIT=0`;
- `provenance.json` schema `static-timing-provenance-v1`, source hashes,
  execution policy, and qualifications.

The self-test covers one positive case and 12 mutations: removed Flux hashes,
wrong schema, wrong Flux precision, wrong PyTorch precision, wrong job, missing
`pass_checks`, wrong fixture hash, wrong duration, missing rank, tampered
source, missing provenance, and missing driver line.

```bash
python3 scripts/static_scaling/check_timing.py --self-test
# TIMING_ARTIFACT_SELFTEST PASS checks=13
```

## R2 — full-length numerical agreement measured descriptively (P2)

Job `61366015` (`static-timing-numerics`, `lrdn2256.leonardo.local`) produced
6/6 JSON records and 6/6 `state.npz` snapshots after 300 measured updates
(20 warmup, global batch 12, one repeat, strict FP32 FMA/TF32 off, W=1,2,4).
The driver printed `STATIC_TIMING_numerics PASS` and `TIMING_DRIVER_EXIT=0`;
Slurm reports `COMPLETED` exit `0:0`.

The comparator reports `TIMING_NUMERICS updates=300 policy=descriptive` and
`NUMERICAL_EQUIVALENCE_CLAIM none`. All arrays are finite. `beta_power` is
exactly equal (62/62) because the schedule is identical. No `param`, `m`, or
`v` array is exactly equal.

Measured envelope across all seven comparisons:

| Category | Range |
|---|---|
| `param` | `max_abs` up to `0.0333`; `rel_l2` up to `0.00224` |
| `m` | `rel_l2` up to `0.0485` |
| `v` | `rel_l2` up to `0.000333` |
| `beta_power` | exact |

These differences are descriptive only. Do not claim correctness, equivalence,
or an error bound. The envelope is similar within-framework and
cross-framework; no framework-specific anomaly is demonstrated. Flux produced
five distinct final hash sets per configuration across fresh campaign repeats;
PyTorch produced one. That is consistent with Flux's non-deterministic
FMA-compatible algorithm selection, not evidence of an incorrect result.

A numerical acceptance threshold remains an open decision (OD-14/Phase 6) and
is not set by this run. Two earlier attempts, `61362264` and `61363645`, failed
before producing accepted evidence; they are recorded in the
[failure record](../../../../../wiki/failures/2026-10-19-timing-numerics-attempts.md).

## R3 — execution policies differ (P2)

PyTorch sets `torch.backends.cudnn.deterministic=True`,
`torch.backends.cudnn.benchmark=False`, TF32 off, and
`float32_matmul_precision="highest"`. Flux uses strict FMA candidate filtering
with no determinism filter. Flux DDP includes synchronized gradient-finiteness
checks inside the timed window; PyTorch has no such guard inside the timed
loop.

The checker now requires the `execution_policy` block in `provenance.json`.
The measurements remain valid for the recorded implementations, but the
cross-framework timing comparison is descriptive, not a communication-only
isolation.

## R4 — provenance incomplete (P2)

New file:
`artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/provenance.json`.
It records all 36 timing source hashes, the Phase 4 accepted hashes it covers,
the shared-helper drift, the execution policy, and the qualifications. The two
loaded shared helpers are not byte-identical to the Phase 4 accepted snapshot:

| File | Phase 4 accepted SHA-256 | Timing snapshot SHA-256 | Loaded by timing |
|---|---|---|---|
| `phase4_common.jl` | `32d85954fdc7eba2b0073106faddde195adb73819c4b85855204bf08cc6351e6` | `30892f68450bcc1cd963b4778b38cba0e893cd28cdf5d0333ad693efdb0826e9` | yes |
| `phase4_torch.py` | `75c1905118b0d53b96f48efa1c32fc4cfbe09782aee605f125013391923d2712` | `7620d9d1fbef9b714e3afa2ddcb14f98802d7fdae8eed4be1160d4f1368826bf` | yes |
| `check_phase4.py` | `51633091cf0ebcbf4a0504911d76f31e1d547ef028f89dd3cb9876760180fac8` | `66f48c6eda6bf63011fe1803580fb5a2e22ca759a4515bef960cff60748cd4a6` | no |

The drift is recorded, not asserted equivalent. Four timing-specific snapshot
files match smoke `61305319`: `timing_flux.jl` `53d65a82...`, `timing_torch.py`
`8c711812...`, `check_timing.py` `7e71a7fe...`, and `precision.jl`
`3efd8f4a...` (full values in [RESULTS.md](RESULTS.md)). The smoke and the
timing campaign have 31 snapshot paths in common; three of those differ
(`check_phase4.py`, `phase4_common.jl`, `phase4_torch.py`). The timing
campaign additionally
snapshots the Phase 5 files `check_memory.py`, `check_phase5.py`,
`make_phase5_fixture.py`, `memory_flux.jl`, and `memory_torch.py`; the timing
launcher does not load them. Not all snapshot hashes match the smoke.

The launcher now captures `runtime-env.txt` for future runs: UTC date,
hostname, Julia depot/load path, `julia --version`, `Pkg.status()`,
`Base.load_path()`, installed Python distribution metadata, and Python/torch/numpy paths. Campaign
`61314790` predates this capture and has only the copied Julia/Python manifests
and requirements freeze. The launcher still loads the live environments; the
capture documents them but does not pin them.

## Open items

1. Decide the Phase 6 numerical-acceptance policy for 300-update differences
   (OD-14). No rule is applied by the descriptive diagnostic.
2. Treat `phase4_common.jl` and `phase4_torch.py` as drifted shared helpers
   until a comparison justifies otherwise.
3. Use `runtime-env.txt` for runs after campaign `61314790`; job `61366015`
   captured it, and campaign `61314790` retains only the copied manifests and
   requirements freeze.
4. Run the Phase 5 larger-batch memory pilot before the full-dataset campaign.

**Exact next action:** decide the Phase 6 numerical-acceptance policy for
300-update differences (OD-14), then run the Phase 5 larger-batch memory pilot
before the full-dataset campaign.

## Evidence

```bash
python3 scripts/static_scaling/check_timing.py --self-test
# TIMING_ARTIFACT_SELFTEST PASS checks=13

python3 scripts/static_scaling/check_timing.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790
# Fixture training-only pilot: batch=12, updates=300, warmup=20
# TIMING_PROVENANCE job=61314790 node=lrdn3064.leonardo.local source_files=36 drift=phase4_common.jl,phase4_torch.py
# framework W median_s min_s max_s images/s update_ms speedup efficiency
# flux 1 5.483059 5.457258 5.501638 656.568 18.277 1.0000 1.0000
# flux 2 5.714868 5.682742 5.743046 629.936 19.050 0.9594 0.4797
# flux 4 5.664201 5.589812 5.759220 635.571 18.881 0.9680 0.2420
# pytorch 1 8.203725 8.200385 8.219535 438.825 27.346 1.0000 1.0000
# pytorch 2 5.258007 5.256101 5.260617 684.670 17.527 1.5602 0.7801
# pytorch 4 4.542426 4.540239 4.548241 792.528 15.141 1.8060 0.4515
# TIMING_ARTIFACT_CHECK PASS

python3 scripts/static_scaling/compare_timing_snapshots.py --self-test
# TIMING_NUMERICS_SELFTEST PASS checks=7

python3 scripts/static_scaling/compare_timing_snapshots.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015
# TIMING_NUMERICS updates=300 policy=descriptive
# ... seven comparison blocks ...
# TIMING_NUMERICS_REPORT COMPLETE report=artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/numerics-report.json
# NUMERICAL_EQUIVALENCE_CLAIM none

wc -l artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/source-sha256.txt
# 36 artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/source-sha256.txt
```

## Links

- [RESULTS.md](RESULTS.md)
- [300-update snapshot results](../numerics-61366015/RESULTS.md)
- [provenance.json](provenance.json)
- [CAMPAIGN.md](../CAMPAIGN.md)
- [Review report](../../../../../reports/static-scaling-fixture-timing-review.md)
- [Failure record](../../../../../wiki/failures/2026-10-19-timing-numerics-attempts.md)
- [Devlog](../../../../../wiki/devlog/03-static-scaling/2026-10-19-timing-review-response.md)
