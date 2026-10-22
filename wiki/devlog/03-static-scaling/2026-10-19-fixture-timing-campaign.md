# Fixture GPU timing campaign PASS — 2026-10-19

## Summary

Job `61314790` (`static-timing-campaign`) completed on Leonardo and passed.
The campaign produced 30/30 JSON records: five fresh-process repeats for each
of the six Flux/PyTorch × W=1,2,4 configurations. No run failed.
This campaign supersedes the canceled partial campaign `61305371` (23/30
records), which remains non-evidence. Smoke `61305319` remains the preflight.
GPU timing was previously deferred; the user explicitly requested it on
2026-10-19.

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
| Precompile | `FLUX_PRECOMPILE_OK` on the compute node |
| Driver | `STATIC_TIMING_campaign PASS`, `TIMING_DRIVER_EXIT=0` |

## Fixture and workload

- Global batch 12, alternating the two saved Phase 2 batches.
- 20 warmup and 300 measured updates per repeat.
- Training-only view; input staging and host-to-device transfer are outside
  the measured window.
- Strict FP32 FMA with TF32 off.
- Native NCCL DDP at W=2,4; ordinary single-GPU training at W=1.
- Five fresh-process repeats per configuration, 30 records total.

## Checker output

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

All 36 source-snapshot hashes match `source-sha256.txt`. Four timing-specific
snapshot files match smoke `61305319`: `timing_flux.jl` `53d65a82...`,
`timing_torch.py` `8c711812...`, `check_timing.py` `7e71a7fe...`, and
`precision.jl` `3efd8f4a...` (full values in the results file). Not all
snapshot hashes match the smoke: the smoke and campaign have 31 snapshot
paths in common, and three of those differ (`check_phase4.py`,
`phase4_common.jl`, `phase4_torch.py`). The campaign additionally snapshots
five Phase 5 files that the timing launcher does not load.

## Interpretation

The fixture workload is flat for Flux across W=1/2/4: median duration is
`5.483 s` at W=1, `5.715 s` at W=2 (speedup `0.9594`), and `5.664 s` at W=4
(speedup `0.9680`; efficiency `0.4797` and `0.2420`). PyTorch scales on the
same fixture: `8.204 s` at W=1, `5.258 s` at W=2 (`1.5602x`), and `4.542 s`
at W=4 (`1.8060x`; efficiency `0.7801` and `0.4515`). These are measured
fixture values only; they do not establish full-dataset scaling or a cause
for the Flux behavior.

## Limitations and artifacts

This is a bounded fixture training-only pilot. It is **not** the full-dataset
campaign, **not** end-to-end timing (no input staging or H2D in the window),
**not** the Phase 5 larger-batch memory pilot, and **not** learning curves.
Do not claim more than measured.

- Results: [campaign-61314790/RESULTS.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/RESULTS.md).
- Campaign record: [CAMPAIGN.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/CAMPAIGN.md).
- Local artifact copy: `artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/`
  (30 JSON, `driver.log`, manifests, source snapshot; 2.4 MB).
- Slurm log: `artifacts/logs/leonardo/2026-10-19-static-timing-campaign-61314790.log`.
- Remote root: `$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/campaign-61314790/`.

## Review follow-up (2026-10-19)

The [review report](../../../reports/static-scaling-fixture-timing-review.md)
accepted the throughput/scaling figures and found no timing-boundary,
sample-count, or gradient-averaging defect. It raised four P2 findings.

1. **The checker accepted incomplete correctness/provenance evidence.** Fixed.
   `check_timing.py` now requires the exact schema and precision metadata,
   non-empty job/node, exact fixture and bundle hashes, complete state hashes
   (Flux `param`/`m`/`v`/`beta_power`; PyTorch rank `param`/`m`/`v`/`step`
   with `step=300`), all 36 source-snapshot hashes, the three required driver
   lines, and `provenance.json` schema/source/policy/qualifications.
2. **Numerical agreement through all 300 updates is measured descriptively.**
   Job `61366015` produced 6/6 final-state snapshots after 300 updates. All
   arrays are finite; `beta_power` is exact (62/62) and no `param`, `m`, or `v`
   array is exactly equal. The envelope is `param` `max_abs` up to `0.0333`
   and `rel_l2` up to `0.00224`, `m` `rel_l2` up to `0.0485`, and `v` `rel_l2`
   up to `0.000333`. No equivalence rule is applied. Flux still produced five
   distinct final hash sets per configuration across campaign repeats; PyTorch
   produced one. That is consistent with Flux's non-deterministic
   FMA-compatible algorithm selection, not evidence of an incorrect result.
3. **Execution policies differ.** Documented, not eliminated. PyTorch uses
   deterministic cuDNN (`benchmark=False`), TF32 off, and
   `float32_matmul_precision="highest"`; Flux uses strict FMA candidate
   filtering without a determinism filter. Flux DDP includes synchronized
   gradient-finiteness checks inside the timed window; PyTorch has no such
   guard in the timed loop. The cross-framework timing comparison is
   descriptive, not a communication-only isolation.
4. **Provenance was incomplete.** Partially fixed. A new
   [provenance.json](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/provenance.json)
   records all 36 timing hashes, the covered Phase 4 accepted hashes, the
   shared-helper drift, and the execution policy. The launcher now writes `runtime-env.txt`
   for future runs; `61314790` predates that capture and retains only the copied
   Julia/Python manifests and requirements freeze.

### Hash drift

The two loaded shared helpers are not byte-identical to the Phase 4 accepted
snapshot:

| File | Phase 4 accepted | Timing snapshot |
|---|---|---|
| `phase4_common.jl` | `32d85954...` | `30892f68...` |
| `phase4_torch.py` | `75c19051...` | `7620d9d1...` |

`check_phase4.py` also differs (`51633091...` accepted, `66f48c6e...` timing)
but the timing launcher does not load it. The drift is recorded, not asserted
equivalent. Further, the smoke `61305319` and this campaign share 31 snapshot
paths; three differ (`check_phase4.py`, `phase4_common.jl`, `phase4_torch.py`),
and the campaign adds five Phase 5 files that timing does not load. Therefore
not all snapshot hashes match the smoke.

### Checker self-test

```bash
python3 scripts/static_scaling/check_timing.py --self-test
# TIMING_ARTIFACT_SELFTEST PASS checks=13
```

The self-test covers one positive case and 12 mutations, including removed Flux
hashes, wrong schema, wrong Flux/PyTorch precision, wrong job, missing
`pass_checks`, wrong fixture hash, wrong duration, missing rank, tampered
source, missing provenance, and missing driver line.

### Completed numerical diagnostic

The 300-update final-state snapshot diagnostic ran as job `61366015`
(`lrdn2256.leonardo.local`) and completed `0:0` in `00:13:26`. It produced
6/6 JSON records and 6/6 snapshots with `STATIC_TIMING_numerics PASS` and
`TIMING_DRIVER_EXIT=0`. The comparison is descriptive;
`NUMERICAL_EQUIVALENCE_CLAIM none`. Two earlier attempts (`61362264`,
`61363645`) produced no accepted evidence
([failure](../../failures/2026-10-19-timing-numerics-attempts.md)).

The remaining policy decision is the Phase 6 numerical-acceptance rule for
300-update differences (OD-14).
[Results](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md).

Full response: [REVIEW-RESPONSE.md](../../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md).

## Next action

**Exact next action:** decide the Phase 6 numerical-acceptance policy for
300-update differences (OD-14), then run the Phase 5 larger-batch memory pilot
before the full-dataset campaign.

**Previous next action:** run the Phase 5 larger-batch memory pilot before the
full-dataset campaign, and decide whether to add the end-to-end timing view.
