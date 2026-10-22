# Timing review response — 2026-10-19

## Summary

The fixture timing review accepted the reported medians, throughput, speedups,
and efficiencies and found no timing-boundary, sample-count, or
gradient-averaging defect. It raised four P2 findings against campaign
`61314790`: incomplete checker acceptance, unmeasured 300-update numerical
agreement, differing execution policies, and incomplete provenance.

The response is recorded in
[REVIEW-RESPONSE.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md).
The checker was hardened, `provenance.json` was added, and future runs now
capture `runtime-env.txt`. The campaign remains a **qualified timing PASS**.
The 300-update final-state snapshot diagnostic subsequently ran as job
`61366015` and is complete. It is descriptive only; no equivalence rule is
applied.

## Findings and disposition

| Finding | Status | Response |
|---|---|---|
| R1 — checker accepted incomplete correctness/provenance evidence | Fixed | Enforces exact schema, precision, job/node, fixture/bundle hashes, complete state hashes, all 36 source hashes, driver lines, and provenance/policy metadata. |
| R2 — 300-update numerical agreement unmeasured | Measured; descriptive only | Job `61366015` compared final-state arrays after 300 updates. No equivalence rule is applied. |
| R3 — execution policies differ | Documented; not eliminated | PyTorch uses deterministic cuDNN; Flux uses strict FMA candidate filtering without a determinism filter. Flux DDP has synchronized gradient-finiteness checks inside the timed window. |
| R4 — provenance incomplete | Partially fixed | `provenance.json` records source hashes, Phase 4 hashes, drift, policy, and qualifications; `runtime-env.txt` capture is added for future runs. |

## 300-update numerical snapshot (job 61366015)

Job `61366015` (`static-timing-numerics`, `lrdn2256.leonardo.local`,
`boost_qos_lprod`, 1 node / 1 task / 32 CPUs / 4× A100-SXM-64GB, `01:00:00`)
started `2026-10-19T08:53:56Z` and completed `COMPLETED` exit `0:0` in
`00:13:26` (end approx. `2026-10-19T09:07:22Z`). The precompile printed
`FLUX_PRECOMPILE_OK`; all six configurations printed `TIMING_PASS`; the driver
printed `STATIC_TIMING_numerics PASS` and `TIMING_DRIVER_EXIT=0`.

The run produced 6/6 JSON records and 6/6 `state.npz` snapshots for one repeat
of 300 measured updates and 20 warmup updates, global batch 12, alternating
the two saved Phase 2 batches, strict FP32 FMA/TF32 off, W=1,2,4.
`runtime-env.txt` records Julia 1.12.6, Flux 0.16.11, CUDA.jl 6.2.2,
cuDNN.jl 6.2.2, Python 3.11.17, torch 2.9.1+cu126, and numpy 2.4.6, with
executable and module paths; no `RUNTIME_ENV_CAPTURE_FAILED` line is present.

The comparator reports `TIMING_NUMERICS updates=300 policy=descriptive` and
`NUMERICAL_EQUIVALENCE_CLAIM none`. All arrays are finite. `beta_power` is
exactly equal (62/62) because the schedule is identical. No `param`, `m`, or
`v` array is exactly equal.

| Category | Measured envelope |
|---|---|
| `param` | `max_abs` up to `0.0333`; `rel_l2` up to `0.00224` |
| `m` | `rel_l2` up to `0.0485` |
| `v` | `rel_l2` up to `0.000333` |
| `beta_power` | exact |

The differences are descriptive only. Do not claim correctness, equivalence, or
an error bound. The envelope is similar within-framework and cross-framework;
no framework-specific anomaly is demonstrated. The numerical acceptance
threshold remains an open decision (OD-14/Phase 6) and is not set by this run.
Results: [numerics RESULTS.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md).

Two earlier attempts did not produce accepted evidence. `61362264` failed
because `python -m pip` does not exist in the uv-managed venv
(`No module named pip`) and `set -e` aborted before training. `61363645` failed
after Flux W1 because `timing_torch.py` called
`mt.param_cpu_np`/`mt.optimizer_leaf_state_np`, which live in
`phase4_torch.py`. The [failure record](../../failures/2026-10-19-timing-numerics-attempts.md)
documents the fixes.

## Hardened checker

`scripts/static_scaling/check_timing.py` now requires and validates:

- exact schema `static-timing-pilot-v1`;
- exact Flux precision `FP32 strict FMA, TF32 off` and the exact PyTorch
  precision flag dictionary;
- non-empty job and node;
- fixture SHA-256
  `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3`
  and bundle SHA-256
  `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`;
- complete state hashes: Flux `param`/`m`/`v`/`beta_power`; PyTorch rank
  `param`/`m`/`v`/`step` with `step=300`;
- all 36 source-snapshot hashes against `source-sha256.txt`;
- `driver.log` lines `FLUX_PRECOMPILE_OK`, `STATIC_TIMING_campaign PASS`, and
  `TIMING_DRIVER_EXIT=0`;
- `provenance.json` schema, source hashes, policy, and qualifications.

The self-test has one positive case and 12 mutations. It prints:

```text
TIMING_ARTIFACT_SELFTEST PASS checks=13
```

The full checker output now includes:

```text
TIMING_PROVENANCE job=61314790 node=lrdn3064.leonardo.local source_files=36 drift=phase4_common.jl,phase4_torch.py
```

and still ends `TIMING_ARTIFACT_CHECK PASS`.

## Provenance and hash drift

`provenance.json` records all 36 timing source hashes, the covered Phase 4
accepted hashes, the shared-helper drift, the execution policy, and the
qualifications. The two loaded shared helpers are not byte-identical to the
Phase 4 accepted snapshot:

| File | Phase 4 accepted | Timing snapshot |
|---|---|---|
| `phase4_common.jl` | `32d85954fdc7eba2b0073106faddde195adb73819c4b85855204bf08cc6351e6` | `30892f68450bcc1cd963b4778b38cba0e893cd28cdf5d0333ad693efdb0826e9` |
| `phase4_torch.py` | `75c1905118b0d53b96f48efa1c32fc4cfbe09782aee605f125013391923d2712` | `7620d9d1fbef9b714e3afa2ddcb14f98802d7fdae8eed4be1160d4f1368826bf` |

`check_phase4.py` also differs (`51633091...` accepted, `66f48c6e...` timing)
but the timing launcher does not load it. The drift is recorded, not asserted
equivalent.

Four timing-specific snapshot files match smoke `61305319`: `timing_flux.jl`
`53d65a82...`, `timing_torch.py` `8c711812...`, `check_timing.py`
`7e71a7fe...`, and `precision.jl` `3efd8f4a...`. However, not all snapshot
hashes match the smoke. The smoke and campaign have 31 snapshot paths in
common; three of those differ (`check_phase4.py`, `phase4_common.jl`,
`phase4_torch.py`). The campaign adds five Phase 5 snapshot files that timing
does not load.

The launcher now captures `runtime-env.txt` for future runs: UTC date,
hostname, Julia depot/load path, `julia --version`, `Pkg.status()`,
`Base.load_path()`, `pip freeze`, and Python/torch/numpy paths. Campaign
`61314790` predates this capture and retains only the copied Julia/Python
manifests and requirements freeze. The launcher still loads live environments;
the capture documents them but does not pin them.

## Policy qualifications

- PyTorch sets `cudnn.deterministic=True`, `cudnn.benchmark=False`, TF32 off,
  and `float32_matmul_precision="highest"`.
- Flux uses strict FMA candidate filtering with no determinism filter.
- Flux DDP includes synchronized gradient-finiteness checks inside the timed
  window; PyTorch has no equivalent guard in the timed loop.
- The cross-framework timing comparison is descriptive, not a
  communication-only isolation.
- Flux produced five distinct final hash sets per configuration across fresh
  repeats; PyTorch produced one. This is consistent with Flux's
  non-deterministic FMA-compatible algorithm selection and is not evidence of
  an incorrect result.

## Remaining open numerical-policy item

The descriptive snapshot shows finite differences after 300 updates, but it
does not establish numerical equivalence. No acceptance rule is applied. The
Phase 3/4 comparisons cover three updates; this diagnostic does not replace
them.

The remaining decision is the Phase 6 numerical-acceptance policy for
300-update differences (OD-14). The measured envelope is recorded above; the
policy is not set by this run. The earlier Leonardo SSH publickey rejection is
retained as history
([failure](../../failures/2026-10-17-leonardo-ssh-publickey.md)); job `61366015`
subsequently ran and completed.

## Evidence

```bash
python3 scripts/static_scaling/check_timing.py --self-test
# TIMING_ARTIFACT_SELFTEST PASS checks=13

python3 scripts/static_scaling/check_timing.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790
# Fixture training-only pilot: batch=12, updates=300, warmup=20
# TIMING_PROVENANCE job=61314790 node=lrdn3064.leonardo.local source_files=36 drift=phase4_common.jl,phase4_torch.py
# ... six configuration rows ...
# TIMING_ARTIFACT_CHECK PASS

python3 scripts/static_scaling/compare_timing_snapshots.py --self-test
# TIMING_NUMERICS_SELFTEST PASS checks=7

python3 scripts/static_scaling/compare_timing_snapshots.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015
# TIMING_NUMERICS updates=300 policy=descriptive
# ... seven comparison blocks ...
# NUMERICAL_EQUIVALENCE_CLAIM none
```

Links:

- [RESULTS.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/RESULTS.md)
- [300-update snapshot results](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md)
- [provenance.json](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/provenance.json)
- [REVIEW-RESPONSE.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md)
- [CAMPAIGN.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/CAMPAIGN.md)
- [Failure record](../../failures/2026-10-19-timing-numerics-attempts.md)

## Next action

**Exact next action:** decide the Phase 6 numerical-acceptance policy for
300-update differences (OD-14), then run the Phase 5 larger-batch memory pilot
before the full-dataset campaign.
