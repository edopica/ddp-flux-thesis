# Provenance — static-scaling Phase 3 Step 3 full three-update comparison

Date: 2026-10-17 (UTC run window 19:35–19:43).
Status: **FAIL**. The unchanged elementwise acceptance rule
`|a - b| <= atol + rtol * max(|a|, |b|)` with `atol=1e-5`, `rtol=1e-4`
did not pass. No tolerance, elementwise-rule, model-math, optimizer, or
precision-configuration change was made.

The driver generated a new PyTorch reference. That reference is
array-for-array identical (337/337 exact) to the preserved attempt-3
reference; the failures are Flux-versus-reference differences, not a
reference change.

## 1. Source identity

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis source-state hash | `49e4f70b9e7583faec438654f91cc1bb4a0c4029ab5569990afcdeba671ea8e2` |
| Flux worktree commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux worktree status | clean |
| Flux source-state hash | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty-diff hash) |
| `scripts/static_scaling/check_phase3_flux.jl` SHA-256 | `633c7f3794eed8ddf74c4202c96a2e31042e0dbff9e88465c577e9d08eb45710` |
| `scripts/static_scaling/precision.jl` SHA-256 | `d6b157f25a293cc5310a183169de596dd88f2880425af4fe90fa357c8d7f14a0` |
| Remote thesis `check_phase3_flux.jl` SHA-256 | `633c7f3794eed8ddf74c4202c96a2e31042e0dbff9e88465c577e9d08eb45710` (MATCH) |

Source-state recipe (same as Phase 2/3):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

The source state was computed after the Step 3 per-leaf diagnostics were
added to `check_phase3_flux.jl`. The remote thesis tree has no `.git`;
identity is anchored by the file hashes above.

## 2. Remote environment

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| Precompile job / node | `60904055` / `lrdn3183` |
| Comparison job / node | `60904928` / `lrdn0125` |
| GPU | NVIDIA A100-SXM-64GB (one GPU allocated) |
| Julia | 1.12.6 |
| PyTorch reference env | torch 2.9.1+cu126, CUDA 12.6, numpy 2.4.6 |
| cuDNN (PyTorch reference) | 9.10.2.21 (`torch.backends.cudnn.version()=91002`) |
| cuDNN (Julia/cuDNN.jl) | library 9.26.0; `cuDNN.version()` prints `92.60.0` |
| Flux | integration worktree `f7b4520c`, Flux 0.16.11 |
| Remote Julia manifest SHA-256 | `c8077e0c6504adedfdce5a35dd1a2ba6bde057fc45ee9bd7ba1edbd35ee23874` |
| Remote Julia project SHA-256 | `6e75702e0e98a6fe3a3c253a278be827f771295d950dee67eae38367bb8f032d` |
| Remote Flux test project SHA-256 | `13970e0a9b8595595ce63a9e05d69de0b381fff86d16f774c4e0d04d8bb38853` |
| Remote Flux root manifest SHA-256 | `4c4eaf77633c493d8463a1a57236197423a8995f92f92eff00879475fdce5837` |
| Python freeze SHA-256 | `cd39f2829c56037a3c68d05299f6d823f4ec603d7422c5d377104fc48d8c7532` |

The Julia stack is the same pinned Leonardo depot used by Step 2:
CUDA.jl 6.2.2, cuDNN.jl 6.2.2, NNlib 0.9.45, Flux 0.16.11.
`CUDA.allowscalar(false)` and the `CUDNN_FMA_MATH` mapping were active:
the checker printed `cuda_math_mode=PEDANTIC_MATH`,
`cuda_math_precision=TensorFloat32`, `cudnn_requested=CUDNN_FMA_MATH(3)`,
`cudnn_effective=CUDNN_FMA_MATH(3)`, `cudnn_version=92.60.0`.

## 3. Result

Required summaries: `PHASE3_REFERENCE PASS` and
`PHASE3_REFERENCE_CHECK PASS`; `PHASE3_FLUX_CHECK FAIL`;
`PHASE3_REMOTE_CHECK FAIL`; driver exit 1.

| Category | Result | max_abs | violations |
|---|---|---:|---:|
| phase3.eta | PASS | 0 | 0 |
| phase3.activations | FAIL | 7.64757395e-3 | 2,705,938 |
| phase3.logits | FAIL | 5.64932823e-4 | 4,981 |
| phase3.loss | PASS | 1.31607056e-4 | 0 |
| phase3.grads.u1 | FAIL | 5.24188578e-3 | 3,842,324 |
| phase3.grads.u3 | FAIL | 7.48962909e-3 | 3,906,482 |
| phase3.params.u3 | FAIL | 1.1857599e-5 | 5 |
| phase3.m.u3 | FAIL | 1.47970638e-3 | 721,437 |
| phase3.v.u3 | PASS | 3.09478492e-6 | 0 |
| phase3.beta_power.u3 | PASS | 0 | 0 |
| phase3.step | PASS | — | — |

The failing activation arrays and violation counts are exactly the
pre-FMA full-forward values recorded in
`artifacts/logs/leonardo/2026-10-17-phase3-diagnose-gpu.log`
(job 60887065): `act.stage2.block0` 616,769, `act.stage2.block1`
864,972, `act.stage3.block0` 346,553, `act.stage3.block1` 460,153,
`act.stage4.block0` 179,821, `act.stage4.block1` 233,331,
`act.pooled` 4,339; maximum `0.00764757395`. The FMA configuration
therefore did not change the executed full-model convolution numerics.

Per-leaf relative-L2 summaries (supplementary only; not part of the
acceptance rule):

| Category | leaves | max_rel_l2 | worst leaf | zero_ref/nonzero_actual | nonzero_ref/zero_actual | below_floor |
|---|---:|---:|---|---:|---:|---:|
| grads.u1 | 62 | 4.897306e-2 | grad.u1.stage1.block1.conv1.weight | 0 | 0 | 0 |
| grads.u3 | 62 | 5.608752e-2 | grad.u3.stage1.block1.conv1.weight | 0 | 0 | 0 |
| params.u3 | 62 | 1.148905e-1 | param.u3.stage1.block1.gn1.bias | 0 | 0 | 0 |
| m.u3 | 62 | 3.943915e-2 | m.u3.stage1.block1.conv1.weight | 0 | 0 | 0 |
| v.u3 | 62 | 5.166543e-2 | v.u3.stem.conv.weight | 0 | 0 | 0 |
| param_increment.u3 | 62 | 1.148905e-1 | delta.u3.stage1.block1.gn1.bias | 0 | 0 | 0 |

## 4. Reference equality

The driver wrote a new reference in a new output directory. Independent
NumPy comparison against the preserved attempt-3 reference:

- array sets equal (337 vs 337), no missing or extra arrays;
- dtype/shape identical for all 337 arrays;
- exact-equal arrays: **337/337**;
- differing arrays: **0**;
- maximum absolute difference over all arrays: **0.0**.

The new `.npz` SHA-256 is `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`,
the same as the preserved reference. The new manifest SHA-256 is
`a301a93fda32b32506a8a34d394a2c0822cdb60b2fb4e3a03b35ecba34c53a33`
(it differs from the preserved manifest only in creation/provenance
fields; its `reference_file.sha256` points to the identical archive).

The independent NumPy artifact checker was run locally against the
copied new reference and manifest: `PHASE3_REFERENCE_CHECK PASS
arrays=337 atol=1e-05 rtol=0.0001`, exit 0. Log:
`local-reference-check.log`.

## 5. Interpretation (bounded)

- This is **not** ordinary FP32 reduction-order residual behaviour. The
  full-model forward output is identical to the pre-FMA, TF32-enabled
  run (same per-array violation counts and same maximum), so the
  executed convolution kernels did not change when `CUDNN_FMA_MATH` was
  requested and reported effective.
- This is **not** an optimizer/model-math defect. Forward activations
  already fail before any optimizer update; loss, eta, beta powers, the
  step counter, and `v.u3` pass. The parameter, `m.u3`, and gradient
  differences are downstream of the forward convolution divergence.
- The evidence is consistent with the algorithm-selection caveat
  documented in Step 2: cuDNN.jl 6.2.2's
  `cudnnConvolutionAlgoPerfChoose` sets
  `skipMathTypeCheck = (dtype == Float32)`, so the Float32 autotuner can
  select a TF32-capable algorithm even when the convolution descriptor
  requests `CUDNN_FMA_MATH`. The isolated stage-2 experiment happened to
  select FMA-compatible algorithms; this full-model run selected a set
  that reproduces the TF32-scale error. This explains why Step 2's
  focused experiment passed while the full path did not.
- No conclusion is drawn about a residual FP32 reduction-order
  distribution, because the run never reached a TF32-off execution.

## 6. Exact next action

Do not start the Step 4 optimizer probe yet: the Phase 3 prerequisite
(full comparison under the accepted FP32/TF32-off policy) is not met.
Close the precision-policy enforcement gap in the reusable precision
path so the full-model convolution forward and backward select
FMA-compatible algorithms (for example, pin the forward/data-gradient/
filter-gradient algorithms as in `check_precision_math.jl`, or make the
Float32 algorithm choice honor `ps.mathType == CUDNN_FMA_MATH`). Add a
full-forward regression, then rerun Step 3 in a fresh `OUT`/`LOG_DIR`
with unchanged `atol=1e-5`, `rtol=1e-4` and require all four summaries
(`PHASE3_REFERENCE`, `PHASE3_REFERENCE_CHECK`, `PHASE3_FLUX_CHECK`,
`PHASE3_REMOTE_CHECK`) to PASS before Step 4.
