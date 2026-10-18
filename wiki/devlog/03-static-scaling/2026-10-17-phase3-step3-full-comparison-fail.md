# Phase 3 Step 3: full comparison FAIL after FMA configuration

Date: 2026-10-17
Status: Step 3 executed and FAILED. Acceptance not met; Step 4 not started.

## What ran

The per-leaf diagnostic helper was added to
`scripts/static_scaling/check_phase3_flux.jl`
(`leaf_l2_report`; supplementary `PHASE3_LEAF_L2*` lines, parameter
increments, and `PHASE3_PARAM_INCREMENT_SUMMARY`). The existing
`CHECK <category>`, `PHASE3_MAXABS`, `PHASE3_FLUX_CHECK`, and exit
semantics were left unchanged. A local parse check and a small
formatting test passed.

The full driver then ran on Leonardo in a new evidence root:

- sync: explicit rsync with the two preserved-evidence excludes;
- precompile job `60904055`, node `lrdn3183`, `FLUX_PRECOMPILE_OK`;
- driver job `60904928`, node `lrdn0125`, one A100, `boost_qos_dbg`;
- new reference and logs under
  `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3/`;
- thesis HEAD `3088f610`, state `49e4f70b…`;
  Flux `f7b4520c`, clean (`e3b0c442…`);
- tolerances unchanged (`atol=1e-5`, `rtol=1e-4`).

## Result

`PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK FAIL`, `PHASE3_REMOTE_CHECK FAIL`, driver exit 1.

| Category | Result | max_abs | violations |
|---|---|---:|---:|
| eta | PASS | 0 | 0 |
| activations | FAIL | 7.64757395e-3 | 2,705,938 |
| logits | FAIL | 5.64932823e-4 | 4,981 |
| loss | PASS | 1.31607056e-4 | 0 |
| grads.u1 | FAIL | 5.24188578e-3 | 3,842,324 |
| grads.u3 | FAIL | 7.48962909e-3 | 3,906,482 |
| params.u3 | FAIL | 1.1857599e-5 | 5 |
| m.u3 | FAIL | 1.47970638e-3 | 721,437 |
| v.u3 | PASS | 3.09478492e-6 | 0 |
| beta_power.u3 | PASS | 0 | 0 |
| step | PASS | — | — |

Per-leaf relative-L2 summaries (supplementary, not acceptance):

| Category | max_rel_l2 | worst leaf | zero/nonzero |
|---|---:|---|---:|
| grads.u1 | 4.897306e-2 | grad.u1.stage1.block1.conv1.weight | 0 / 0 |
| grads.u3 | 5.608752e-2 | grad.u3.stage1.block1.conv1.weight | 0 / 0 |
| params.u3 | 1.148905e-1 | param.u3.stage1.block1.gn1.bias | 0 / 0 |
| m.u3 | 3.943915e-2 | m.u3.stage1.block1.conv1.weight | 0 / 0 |
| v.u3 | 5.166543e-2 | v.u3.stem.conv.weight | 0 / 0 |
| param_increment.u3 | 1.148905e-1 | delta.u3.stage1.block1.gn1.bias | 0 / 0 |

## Reference equality

The new reference is array-for-array identical to the preserved
attempt-3 reference: 337/337 exact, zero differing arrays, maximum
absolute difference 0.0. The new `.npz` hash is `97e5007c…` (same as
preserved). The new manifest differs only in creation/provenance fields
and is accepted by the independent NumPy checker locally
(`PHASE3_REFERENCE_CHECK PASS`, exit 0).

## Interpretation

The checker printed requested and effective `CUDNN_FMA_MATH(3)`, but the
full-model forward output is exactly the pre-FMA, TF32-enabled output:
the per-array activation violation counts and maximum match
`2026-10-17-phase3-diagnose-gpu.log` (job 60887065). The FMA descriptor
therefore did not change the executed kernels.

The evidence points to the known algorithm-selection gap: cuDNN.jl
6.2.2 sets `skipMathTypeCheck = (dtype == Float32)` in
`cudnnConvolutionAlgoPerfChoose`, so the Float32 autotuner can pick a
TF32-capable algorithm despite the FMA descriptor. The isolated stage-2
experiment happened to pick FMA-compatible algorithms; the full run did
not. This is not a residual FP32 reduction-order effect and not an
optimizer/model-math defect. See the Step 2 algorithm-selection caveat.

## Evidence

- `artifacts/logs/leonardo/2026-10-17-phase3-step3.log`
- `artifacts/logs/static-scaling/2026-10-17-phase3-step3/`
  (`PROVENANCE.md`, `COMMANDS.md`, `RESULTS.md`, `remote-check.log`,
  `reference-comparison.txt`, `local-reference-check.log`, SHA256SUMS)
- `artifacts/baselines/static-scaling/2026-10-17-phase3-step3/`
- Failure record:
  [2026-10-17-phase3-step3-fma-full-model](../../failures/2026-10-17-phase3-step3-fma-full-model.md)

## Exact next action

Do not start Step 4. Enforce FMA-compatible algorithm selection for the
full-model convolution forward and backward in the reusable precision
path, add a full-forward regression, and rerun Step 3 in a fresh
`OUT`/`LOG_DIR` with unchanged `atol=1e-5` / `rtol=1e-4`. Step 4
(optimizer probe plus schedule boundaries at `t=1`, `W`, `W+1`, `T`)
starts only after all four Step 3 summaries PASS.
