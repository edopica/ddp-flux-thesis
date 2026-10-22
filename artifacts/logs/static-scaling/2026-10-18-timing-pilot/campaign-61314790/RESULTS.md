# GPU fixture timing campaign — 2026-10-18

**Status: COMPLETED and PASS (qualified by the 2026-10-19 review).** This
campaign supersedes the canceled partial campaign `61305371` (23 of 30
records), which remains non-evidence. Smoke `61305319` remains the preflight.
The [review response](REVIEW-RESPONSE.md) records the four P2 findings and the
checker/provenance fixes.

## Job record

| Item | Value |
|---|---|
| Job ID | `61314790` |
| Name | `static-timing-campaign` |
| Node | `lrdn3064.leonardo.local` |
| QoS | `boost_qos_lprod` |
| Allocation | 1 node, 1 task, 32 CPUs, 4× A100-SXM-64GB |
| Time limit | `02:00:00` |
| Start | `2026-10-18T23:06:02Z` (`2026-10-19 01:06:02` local) |
| End | `2026-10-18T23:48:09Z` (approx.) |
| Slurm state | `COMPLETED`, exit `0:0` |
| Elapsed | `00:42:07` |
| Compute-node precompile | `FLUX_PRECOMPILE_OK` |
| Driver summaries | `STATIC_TIMING_campaign PASS`, `TIMING_DRIVER_EXIT=0` |
| Records | 30/30 JSON, 5 repeats × 6 configurations; no failed runs |

## Workload

- Fixture: global batch 12, alternating the two saved Phase 2 batches.
- Updates: 20 warmup and 300 measured per repeat.
- View: training-only; input staging and host-to-device transfer are outside
  the measured window.
- Precision: strict FP32 FMA with TF32 off.
- Distributed: native NCCL DDP at W=2,4; ordinary single-GPU training at W=1.
- Repeats: five fresh processes per configuration, 30 records total.

## Analysis command and output

Command, from the repository root:

```bash
python3 scripts/static_scaling/check_timing.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790
```

Output:

```text
Fixture training-only pilot: batch=12, updates=300, warmup=20
TIMING_PROVENANCE job=61314790 node=lrdn3064.leonardo.local source_files=36 drift=phase4_common.jl,phase4_torch.py
framework W median_s min_s max_s images/s update_ms speedup efficiency
flux 1 5.483059 5.457258 5.501638 656.568 18.277 1.0000 1.0000
flux 2 5.714868 5.682742 5.743046 629.936 19.050 0.9594 0.4797
flux 4 5.664201 5.589812 5.759220 635.571 18.881 0.9680 0.2420
pytorch 1 8.203725 8.200385 8.219535 438.825 27.346 1.0000 1.0000
pytorch 2 5.258007 5.256101 5.260617 684.670 17.527 1.5602 0.7801
pytorch 4 4.542426 4.540239 4.548241 792.528 15.141 1.8060 0.4515
TIMING_ARTIFACT_CHECK PASS
```

Checker self-test:

```bash
python3 scripts/static_scaling/check_timing.py --self-test
# TIMING_ARTIFACT_SELFTEST PASS checks=13
```

The self-test covers one positive case and 12 mutations (removed Flux hashes,
wrong schema, wrong Flux/PyTorch precision, wrong job, missing `pass_checks`,
wrong fixture hash, wrong duration, missing rank, tampered source, missing
provenance, missing driver line).

## 300-update numerical snapshot

Job `61366015` (`static-timing-numerics`, `lrdn2256.leonardo.local`,
`boost_qos_lprod`, 1 node / 1 task / 32 CPUs / 4× A100-SXM-64GB, `01:00:00`)
produced 6/6 JSON records and 6/6 `state.npz` snapshots after one repeat of 300
measured updates (20 warmup, global batch 12, strict FP32 FMA/TF32 off,
W=1,2,4). The driver printed `STATIC_TIMING_numerics PASS` and
`TIMING_DRIVER_EXIT=0`; Slurm reports `COMPLETED` exit `0:0` with elapsed
`00:13:26`.

The comparator reports `TIMING_NUMERICS updates=300 policy=descriptive` and
`NUMERICAL_EQUIVALENCE_CLAIM none`. All arrays are finite; `beta_power` is
exactly equal (62/62). No `param`, `m`, or `v` array is exactly equal. The
measured envelope is `param` `max_abs` up to `0.0333` and `rel_l2` up to
`0.00224`, `m` `rel_l2` up to `0.0485`, and `v` `rel_l2` up to `0.000333`.

These differences are descriptive only. No correctness, equivalence, or error
bound is claimed. The numerical acceptance threshold remains open (OD-14/Phase
6). [Results](../numerics-61366015/RESULTS.md); failed attempts
`61362264`/`61363645` produced no accepted evidence
([record](../../../../../wiki/failures/2026-10-19-timing-numerics-attempts.md)).

## Source hashes and provenance

All 36 source-snapshot hashes match `source-sha256.txt`; the checker enforces
them. The run snapshot is under `source/`. The four timing-specific files are:

| Source | SHA-256 | Matches smoke `61305319` |
|---|---|---|
| `timing_flux.jl` | `53d65a823baad610dfdbc50b316a2fdb4d932d5e4c64164a49bc1909213d9624` | yes |
| `timing_torch.py` | `8c7118127f1adbe110f99f420a49cebe43dd85c5213520afe0aa17a8375238ca` | yes |
| `check_timing.py` | `7e71a7fe2264231a658281ad52c190c1e28467767bb5d142143487b3d45459f7` | yes |
| `precision.jl` | `3efd8f4a1b37da847a7bd456138dff86a8dfb927e9cdfb0ebfb64011564c2b81` | yes |

The `check_timing.py` entry is the campaign snapshot from before the
2026-10-19 hardening; the hardened checker validates that snapshot but is not
part of it.

Not all snapshot hashes match the smoke. The smoke and the timing campaign
have 31 snapshot paths in common; three of those differ: `check_phase4.py`,
`phase4_common.jl`, and `phase4_torch.py`. The timing campaign additionally
snapshots the Phase 5 files `check_memory.py`, `check_phase5.py`,
`make_phase5_fixture.py`, `memory_flux.jl`, and `memory_torch.py`; the timing
launcher does not load them.

The two loaded shared helpers are not byte-identical to the Phase 4 accepted
snapshot:

| File | Phase 4 accepted SHA-256 | Timing snapshot SHA-256 | Loaded by timing |
|---|---|---|---|
| `phase4_common.jl` | `32d85954fdc7eba2b0073106faddde195adb73819c4b85855204bf08cc6351e6` | `30892f68450bcc1cd963b4778b38cba0e893cd28cdf5d0333ad693efdb0826e9` | yes |
| `phase4_torch.py` | `75c1905118b0d53b96f48efa1c32fc4cfbe09782aee605f125013391923d2712` | `7620d9d1fbef9b714e3afa2ddcb14f98802d7fdae8eed4be1160d4f1368826bf` | yes |
| `check_phase4.py` | `51633091cf0ebcbf4a0504911d76f31e1d547ef028f89dd3cb9876760180fac8` | `66f48c6eda6bf63011fe1803580fb5a2e22ca759a4515bef960cff60748cd4a6` | no |

The drift is recorded in [provenance.json](provenance.json), not asserted
equivalent. `provenance.json` (schema `static-timing-provenance-v1`) also
records all 36 timing source hashes, the covered Phase 4 accepted hashes, the
execution policy, and the qualifications.

## Policy qualifications

- PyTorch uses deterministic cuDNN (`cudnn.deterministic=True`,
  `cudnn.benchmark=False`), TF32 off, and
  `float32_matmul_precision="highest"`.
- Flux uses strict FMA candidate filtering with no determinism filter.
- Flux DDP includes synchronized gradient-finiteness checks inside the timed
  window; PyTorch has no such guard inside the timed loop.
- The cross-framework timing comparison is therefore descriptive, not a
  communication-only isolation.
- Flux produced five distinct final hash sets per configuration across fresh
  repeats; PyTorch produced one. This is consistent with Flux's
  non-deterministic FMA-compatible algorithm selection and is not evidence of
  an incorrect result.

## What this does not establish

- No numerical equivalence through all 300 measured updates. A descriptive
  final-state snapshot diagnostic (job `61366015`) now measures the
  differences, but no equivalence rule is applied.
- No W=1 versus W>1 or Flux versus PyTorch acceptance test after 300 updates.
  The snapshot comparison is descriptive; the Phase 3/4 comparisons cover
  three updates.
- No communication-only isolation of NCCL efficiency or of the cause of flat
  Flux scaling.
- No full-dataset claim, no end-to-end timing claim, no Phase 5
  larger-batch-memory-pilot claim, and no learning-curve claim.
- No complete runtime identity for this campaign: the launcher loads live
  environments, and `runtime-env.txt` capture was added after `61314790`. The
  saved copy retains the Julia/Python manifests and requirements freeze only.

## Artifact paths

- Local copy: `artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/`
  (30 JSON records, `driver.log`, manifests, source snapshot, `provenance.json`; 2.4 MB).
- Slurm log: `artifacts/logs/leonardo/2026-10-19-static-timing-campaign-61314790.log`.
- Remote root: `$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/campaign-61314790/`.
- [provenance.json](provenance.json)
- [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md)
- [300-update numerical snapshot](../numerics-61366015/RESULTS.md)

## Scope limits

This is a bounded fixture training-only pilot. It is **not** the full-dataset
campaign, **not** end-to-end timing (no input staging or H2D in the window),
**not** the Phase 5 larger-batch memory pilot, and **not** learning curves.
GPU timing was previously deferred; the user explicitly requested it on
2026-10-19. Do not claim more than measured.

**Next action:** decide the Phase 6 numerical-acceptance policy for 300-update
differences (OD-14), then run the Phase 5 larger-batch memory pilot before the
full-dataset campaign.
