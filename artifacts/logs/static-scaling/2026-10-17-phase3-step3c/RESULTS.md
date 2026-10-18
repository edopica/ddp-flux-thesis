# Step 3c results — OD-10 revision applied, full three-update comparison PASS

Date: 2026-10-17 (UTC run 2026-10-18T00:08:54Z start).
Source: `2026-10-17-phase3-step3c.log` (driver job `60942933`, node
`lrdn0129`; precompile job `60941770`, node `lrdn2500`).
Status: **PASS**. The user approved the OD-10 revision before this run.
This run applies only the three approved backward-category floors; no
other tolerance, model-math, optimizer, precision-policy, Flux source, or
reference change.

## Frozen rule

Base rule (unchanged): `|a - b| <= atol + rtol * max(|a|, |b|)` with
`atol = 1e-5`, `rtol = 1e-4`.

Approved category overrides (only these three):

| Category | atol | rtol |
|---|---:|---:|
| `phase3.grads.u1` | 1e-3 | 1e-4 |
| `phase3.grads.u3` | 2e-3 | 1e-4 |
| `phase3.m.u3` | 2e-4 | 1e-4 |

All other categories use the base `1e-5` / `1e-4`. The checker printed:

```text
PHASE3_FLUX_CHECK_CONFIG device=cuda atol=1e-05 rtol=0.0001 grads_u1_atol=0.001 grads_u1_rtol=0.0001 grads_u3_atol=0.002 grads_u3_rtol=0.0001 m_u3_atol=0.0002 m_u3_rtol=0.0001
PHASE3_ACCEPTANCE_RULE base="|a - b| <= 1.0e-5 + 0.0001 * max(|a|, |b|)" phase3.grads.u1="|a - b| <= 0.001 + 0.0001 * max(|a|, |b|)" phase3.grads.u3="|a - b| <= 0.002 + 0.0001 * max(|a|, |b|)" phase3.m.u3="|a - b| <= 0.0002 + 0.0001 * max(|a|, |b|)" other_categories=base
```

The driver passed no `ATOL` or `RTOL`; no category override was passed by
environment or CLI. The frozen checker defaults supplied the values above.

## Per-category checks (full model)

| Check line | Status | max_abs | violations |
|---|---|---:|---:|
| `phase3.eta` | PASS | 0 | 0 |
| `phase3.activations` | PASS | 1.56164169e-5 | 0 |
| `phase3.logits` | PASS | 1.66893005e-6 | 0 |
| `phase3.loss` | PASS | 4.76837158e-7 | 0 |
| `phase3.grads.u1` | PASS | 3.02597880e-4 | 0 |
| `phase3.grads.u3` | PASS | 5.43192029e-4 | 0 |
| `phase3.params.u3` | PASS | 4.77582216e-6 | 0 |
| `phase3.m.u3` | PASS | 6.515719e-5 | 0 |
| `phase3.v.u3` | PASS | 2.32830644e-7 | 0 |
| `phase3.beta_power.u3` | PASS | 0 | 0 |
| `phase3.step` | PASS | reference 3 = Flux 3 | 0 |

Every category passes. The three overridden categories are 3.30x,
3.68x, and 3.07x below their new absolute floors respectively; no
category needed a widened `rtol`.

## Per-leaf summary (supplementary)

| Category | Leaves | max rel L2 | Worst leaf | zero_ref_nonzero_actual | nonzero_ref_zero_actual | below_floor |
|---|---:|---:|---|---:|---:|---:|
| `grads.u1` | 62 | 1.998759e-3 | `grad.u1.stage1.block0.conv1.weight` | 0 | 0 | 0 |
| `grads.u3` | 62 | 4.288785e-3 | `grad.u3.stage1.block1.conv1.weight` | 0 | 0 | 0 |
| `params.u3` | 62 | 4.336577e-3 | `param.u3.stem.gn.bias` | 0 | 0 | 0 |
| `param_increment.u3` | 62 | 1.258135e-2 | `delta.u3.stage1.block1.conv1.weight` | 0 | 0 | 0 |
| `m.u3` | 62 | 2.571253e-3 | `m.u3.stage1.block0.conv1.weight` | 0 | 0 | 0 |
| `v.u3` | 62 | 2.787131e-3 | `v.u3.stage1.block1.conv1.weight` | 0 | 0 | 0 |

There are no zero/nonzero gradient disagreements. The per-leaf relative
L2 values stay supplementary; they do not replace the elementwise rule.

## Algorithm-choice evidence

`PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0 enabled=true`.
All 32 choices report `CUDNN_FMA_MATH(3)` and `fallback=false`. All 225
recorded candidate math types are `3`; no tensor-op/TF32 candidate was
present.

| Kind | Chosen algorithms (count) |
|---|---|
| forward | algo 0 (7), algo 6 (4) |
| backward data | algo 0 (7), algo 4 (3) |
| backward filter | algo 0 (2), algo 3 (6), algo 5 (3) |

The chosen families are the same FP32 FMA families as Step 3b
(legacy direct, Winograd, FFT-tiling). Full per-choice lines, candidate
lists, and counts are in `algorithm-choices.txt`.
`CHECK cudnn.fma_algo_selection PASS` is recorded in the log.

## Reference equality and independent check

| Item | SHA-256 |
|---|---|
| `phase3_reference.npz` | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| `phase3_reference_manifest.json` | `221d185968ed3fb30a7d3a33fa8eb1afabad3719d259a4876765b73d1c78260d` |
| `logs/remote-check.log` | `0e3caeddca3f807908eda4367109dbf51a9685d424091fec070d1d0473e3bc73` |

The new reference `.npz` is byte-identical to the preserved attempt-3
reference. Local comparison against
`artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz`:
337 of 337 arrays exact-equal, 0 differing, maximum absolute difference
`0.0` (`reference-comparison.txt`).

The independent NumPy checker ran locally on the copied reference and
manifest: `PHASE3_REFERENCE_CHECK PASS arrays=337 atol=1e-05 rtol=0.0001`,
exit 0 (`local-reference-check.log`).

The manifest records the run-time provenance:
`source_commit=3088f61020137912af32d9c969e87ea9ae2a71c3`,
`source_state_sha256=e1970ef82a09e7864ec13e453afab727486b658954c46c842e28cbcd39f9e7e5`,
`flux_commit=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`,
`flux_state_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
`device=cuda:0`.

## Frozen-threshold change and relation to Step 3b

This run applies the OD-10 revision that the user approved after the
Step 3b calibration. The revision is not a silent tolerance change:

- the rule form and `rtol = 1e-4` are unchanged;
- only `grads.u1`, `grads.u3`, and `m.u3` receive new absolute floors;
- the numerical basis is the same-framework PyTorch CPU-vs-GPU control
  (job `60912525`) and the four-comparison envelope recorded in
  `artifacts/logs/static-scaling/2026-10-17-phase3-numerics/RESULTS.md`;
- each new floor has at least 2x margin over the observed envelope
  (2.42x, 2.67x, 2.07x);
- `specification.md`, `experiment.toml`, `open-decisions.md`, and
  `check_phase3_flux.jl` were updated consistently before this run.

Step 3b remains a valid failed acceptance attempt under the old rule and
is superseded by the approved calibration, not by an unrecorded tolerance
change. The Step 3b artifacts are retained unchanged.

## Outcome

`PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK PASS`, `PHASE3_REMOTE_CHECK PASS`; driver exit 0.
Phase 3 acceptance passes under the recorded OD-10 rule.
