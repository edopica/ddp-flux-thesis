# 300-update final-state snapshot diagnostic — 2026-10-19

**Status: COMPLETED and PASS (descriptive only).** Job `61366015`
(`static-timing-numerics`) produced one final-state snapshot per configuration
after 300 measured updates. The comparison is descriptive. **No numerical
equivalence rule is applied and no equivalence, correctness, or error bound is
claimed.**

## Job record

| Item | Value |
|---|---|
| Job ID | `61366015` |
| Name | `static-timing-numerics` |
| Node | `lrdn2256.leonardo.local` |
| QoS | `boost_qos_lprod` |
| Allocation | 1 node, 1 task, 32 CPUs, 4× A100-SXM-64GB |
| Time limit | `01:00:00` |
| Driver start | `2026-10-19T08:53:56Z` |
| End | approx. `2026-10-19T09:07:22Z` |
| Slurm state | `COMPLETED`, exit `0:0` |
| Elapsed | `00:13:26` |
| Compute-node precompile | `FLUX_PRECOMPILE_OK` |
| Configuration summaries | all six printed `TIMING_PASS` |
| Driver summaries | `STATIC_TIMING_numerics PASS`, `TIMING_DRIVER_EXIT=0` |
| Records | 6/6 JSON; 6/6 `state.npz` snapshots |

## Workload

- One repeat, 300 measured updates and 20 warmup updates.
- Global batch 12, alternating the two saved Phase 2 batches.
- Strict FP32 FMA with TF32 off.
- W=1,2,4 for Flux and PyTorch.

## Runtime environment

Captured in `runtime-env.txt`; no `RUNTIME_ENV_CAPTURE_FAILED` line is present.

| Component | Version |
|---|---|
| Julia | 1.12.6 |
| Flux | 0.16.11 |
| CUDA.jl | 6.2.2 |
| cuDNN.jl | 6.2.2 |
| Python | 3.11.17 |
| torch | 2.9.1+cu126 |
| numpy | 2.4.6 |

The capture also records executable and module paths.

## Comparator

Command, from the repository root:

```bash
python3 scripts/static_scaling/compare_timing_snapshots.py artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015
```

Exact output:

```text
TIMING_NUMERICS updates=300 policy=descriptive
comparison category arrays exact max_abs rel_l2 worst_leaf
flux-w2-vs-w1 param 62 0 0.0267725405 0.00176360589 param.stage1.block0.gn2.bias
flux-w2-vs-w1 m 62 0 8.77366983e-06 0.0447234165 m.stem.gn.weight
flux-w2-vs-w1 v 62 0 1.34306028e-05 0.000323226536 v.stage1.block1.conv1.weight
flux-w2-vs-w1 beta_power 62 62 0 0 beta_power.classifier.bias
flux-w4-vs-w1 param 62 0 0.0301878736 0.0022350052 param.stage1.block0.gn1.bias
flux-w4-vs-w1 m 62 0 1.05162204e-05 0.048496192 m.stem.gn.weight
flux-w4-vs-w1 v 62 0 1.12820417e-05 0.000247342901 v.stem.conv.weight
flux-w4-vs-w1 beta_power 62 62 0 0 beta_power.classifier.bias
pytorch-w2-vs-w1 param 62 0 0.0244703284 0.00223943871 param.stage1.block0.gn1.bias
pytorch-w2-vs-w1 m 62 0 1.05037479e-05 0.0471111959 m.stem.gn.weight
pytorch-w2-vs-w1 v 62 0 5.74439764e-06 0.000166062783 v.stage1.block1.conv1.weight
pytorch-w4-vs-w1 param 62 0 0.0332728997 0.00167197836 param.stage2.block1.gn1.bias
pytorch-w4-vs-w1 m 62 0 1.49296666e-05 0.041057524 m.stem.conv.weight
pytorch-w4-vs-w1 v 62 0 9.59821045e-06 0.000217657532 v.stem.conv.weight
flux-vs-pytorch-w1 param 62 0 0.0296371677 0.00176734019 param.stage1.block0.gn2.bias
flux-vs-pytorch-w1 m 62 0 7.16865804e-06 0.043020317 m.stem.gn.weight
flux-vs-pytorch-w1 v 62 0 8.18353146e-06 0.000195990249 v.stem.conv.weight
flux-vs-pytorch-w2 param 62 0 0.0233304985 0.00225564012 param.stage1.block0.gn2.bias
flux-vs-pytorch-w2 m 62 0 9.4129864e-06 0.0483030193 m.stem.conv.weight
flux-vs-pytorch-w2 v 62 0 1.59721822e-05 0.000332742395 v.stage1.block0.gn1.bias
flux-vs-pytorch-w4 param 62 0 0.0266558528 0.00175652978 param.stage2.block1.gn1.bias
flux-vs-pytorch-w4 m 62 0 1.05161225e-05 0.0416729123 m.stem.gn.weight
flux-vs-pytorch-w4 v 62 0 5.74067235e-06 0.000154264521 v.stage1.block0.conv1.weight
TIMING_NUMERICS_REPORT COMPLETE report=artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/numerics-report.json
NUMERICAL_EQUIVALENCE_CLAIM none
```

Comparator self-test:

```bash
python3 scripts/static_scaling/compare_timing_snapshots.py --self-test
# TIMING_NUMERICS_SELFTEST PASS checks=7
```

`numerics-report.json` SHA-256:
`c40c67b6125c3bee301258ab49067702848421fb6c5078a128fa719c7836a58b`.

## Measured envelope

Across all seven comparisons:

| Category | Largest `max_abs` | Largest `rel_l2` | Exact arrays |
|---|---:|---:|---:|
| `param` | `0.0333` | `0.00224` | none |
| `m` | `1.49296666e-05` | `0.0485` | none |
| `v` | `1.59721822e-05` | `0.000333` | none |
| `beta_power` | `0` | `0` | 62/62 per comparison |

All arrays are finite. `beta_power` is exactly equal because the schedule is
identical. No `param`, `m`, or `v` array is exactly equal. The differences are
similar within-framework and cross-framework; no framework-specific anomaly is
demonstrated.

## Source hashes

| Source | SHA-256 |
|---|---|
| `timing_flux.jl` | `7c3834613757ace22fbd2c6c99ecd3348c9c7dd5ac0cd94684b90d468a6924e1` |
| `timing_torch.py` | `f4d13cf8560cf04d35cebd6cbf8ee4ee763c15711c6b732e027393972f9055e8` |
| `compare_timing_snapshots.py` | `8084d6d818e2935aabb5d5763dfe68639421981edf4bc5dd04df3a4db85c4548` |
| `check_timing.py` | `e10bf4569fed4fa227deed3ad85e44f1ced6fa16f21bf0ccf4360390ee01289f` |
| `phase4_common.jl` | `30892f68450bcc1cd963b4778b38cba0e893cd28cdf5d0333ad693efdb0826e9` |
| `phase4_torch.py` | `7620d9d1fbef9b714e3afa2ddcb14f98802d7fdae8eed4be1160d4f1368826bf` |
| `precision.jl` | `3efd8f4a1b37da847a7bd456138dff86a8dfb927e9cdfb0ebfb64011564c2b81` |

## Scope limits

This is a bounded fixture diagnostic, not an acceptance test. Do not claim
correctness, equivalence, or an error bound from these differences. The
measured envelope does not demonstrate a framework-specific anomaly. A
numerical acceptance threshold remains an open decision (OD-14/Phase 6 policy)
and is not set by this run. Do not claim full-dataset, end-to-end, Phase 5
memory-pilot, or learning-curve evidence.

## Failed attempts

Neither attempt produced accepted evidence:

- `61362264` failed during runtime capture because `python -m pip` does not
  exist in the uv-managed venv (`No module named pip`); `set -e` aborted before
  training. No timing or snapshot record was produced.
- `61363645` failed after Flux W1 because `timing_torch.py` called
  `mt.param_cpu_np`/`mt.optimizer_leaf_state_np`, which live in
  `phase4_torch.py`. One Flux W1 JSON and snapshot were produced, but the run
  is not accepted.

Record: [2026-10-19 timing numerics failed attempts](../../../../../wiki/failures/2026-10-19-timing-numerics-attempts.md).

## Artifact paths

- Local: `artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/`
  (777 MB raw; six 130 MB snapshots).
- Slurm log: `artifacts/logs/leonardo/2026-10-19-static-timing-numerics-61366015.log`.
- Remote root: `$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/numerics-61366015/`.

## Links

- [Campaign record](../CAMPAIGN.md)
- [Timing results](../campaign-61314790/RESULTS.md)
- [Review response](../campaign-61314790/REVIEW-RESPONSE.md)
- [Devlog](../../../../../wiki/devlog/03-static-scaling/2026-10-19-timing-review-response.md)
- [Failure record](../../../../../wiki/failures/2026-10-19-timing-numerics-attempts.md)
