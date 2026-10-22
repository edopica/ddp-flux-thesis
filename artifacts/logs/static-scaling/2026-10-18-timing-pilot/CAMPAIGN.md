# GPU fixture timing pilot — 2026-10-18

The pilot compares Flux and eager PyTorch at W=1,2,4 on Leonardo.
Global batch: 12. Warmup: 20 updates. Measured work: 300 updates per repeat.
Repeats: five per configuration. Timing: training-only, synchronized CUDA boundaries, maximum rank duration.
The same two saved Phase 2 batches alternate in all configurations.
This pilot does not measure input preparation or full-dataset training.

## Jobs

| Job | Purpose | Status |
|---|---|---|
| 61305319 | Six-configuration smoke, two warmup updates and three measured updates | COMPLETED 0:0 on `lrdn1233`; all six configurations `TIMING_PASS`; preflight |
| 61305371 | Five-repeat campaign, two-hour wall limit | CANCELED at user request after 23 of 30 records; non-evidence |
| 61314790 | Five-repeat campaign, 30 records | COMPLETED 0:0 on `lrdn3064`; `STATIC_TIMING_campaign PASS`, 30/30 records; [RESULTS.md](campaign-61314790/RESULTS.md) |
| 61366015 | 300-update final-state snapshot diagnostic, one repeat | COMPLETED 0:0 on `lrdn2256`; `STATIC_TIMING_numerics PASS`, 6/6 records and 6/6 snapshots; descriptive only; [RESULTS.md](numerics-61366015/RESULTS.md) |

Campaign `61314790` is complete and PASS (qualified by the 2026-10-19 review).
It supersedes the canceled partial campaign; the smoke remains the preflight.
Job `61366015` completed the separate 300-update final-state snapshot
diagnostic. It is descriptive only and sets no equivalence rule.
Evidence: [RESULTS.md](campaign-61314790/RESULTS.md),
[provenance.json](campaign-61314790/provenance.json),
[REVIEW-RESPONSE.md](campaign-61314790/REVIEW-RESPONSE.md),
[numerics RESULTS.md](numerics-61366015/RESULTS.md).

Earlier smoke attempts: `61304553` and `61304730` failed during provenance capture.
Job `61304948` passed precompile, then the operator canceled it after a Slurm allocation warning.
The [failure record](../../../../wiki/failures/2026-10-18-static-timing-pilot.md) explains the corrections.

Remote root:
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/`.

The `smoke-JOBID/`, `campaign-JOBID/`, and `numerics-JOBID/` directories preserve raw JSON, driver logs, source snapshots, environment records, and final-state snapshots.
Slurm logs are one level above this root, named `static-timing-{smoke,campaign,numerics}-JOBID.log`.

The [session devlog](../../../../wiki/devlog/03-static-scaling/2026-10-18-phase4-distributed-correctness.md) records exact submission commands.

## Analysis command

The complete campaign passes the artifact reader:

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

The checker now validates `provenance.json`, the recorded execution policy, and
all 36 source-snapshot hashes. Its self-test prints
`TIMING_ARTIFACT_SELFTEST PASS checks=13` (one positive and 12 mutations):

```bash
python3 scripts/static_scaling/check_timing.py --self-test
```

The reader rejects incomplete runs, unequal work, missing ranks, inconsistent
duration calculations, wrong schema/precision/job/node identity, missing
fixture/bundle hashes, incomplete state hashes, tampered source snapshots,
missing driver markers, and missing provenance/policy metadata.
It prints median images/s, update time, speedup, efficiency, and repeat ranges.
The exact output is in [RESULTS.md](campaign-61314790/RESULTS.md).
