# Results — static-scaling Phase 3 final acceptance (2026-10-18)

Status: **PASS — Phase 3 accepted.**
Driver job `60956582` on `lrdn0165`, exit 0, all four required summaries
PASS. Evidence: [`remote-check.log`](remote-check.log),
[`COMMANDS.md`](COMMANDS.md), [`PROVENANCE.md`](PROVENANCE.md).

## 1. Frozen contract used (unchanged)

- FP32 in both frameworks; TF32 disabled
  (`cuda_matmul_allow_tf32=False`, `cudnn_allow_tf32=False`,
  `float32_matmul_precision='highest'`, `cudnn_deterministic=True`).
- cuBLAS `PEDANTIC_MATH`; cuDNN `CUDNN_FMA_MATH` with the FMA-aware
  algorithm chooser; `CUDA.allowscalar(false)`.
- Elementwise rule `|a - b| <= atol + rtol * max(|a|, |b|)`.
  Base `atol=1e-5`, `rtol=1e-4`.
  Only three frozen round-logic floors:
  `phase3.grads.u1 atol=1e-3`, `phase3.grads.u3 atol=2e-3`,
  `phase3.m.u3 atol=2e-4`, each with `rtol=1e-4`; all other categories use
  the base rule.
- Batch-512 schedule `W=975`, `T=9750`; three-update correctness fixture.
  OD-2/OD-3/OD-4/OD-10/OD-11 frozen.

The checker printed the rule it applied:

```text
PHASE3_ACCEPTANCE_RULE base="|a - b| <= 1.0e-5 + 0.0001 * max(|a|, |b|)" phase3.grads.u1="|a - b| <= 0.001 + 0.0001 * max(|a|, |b|)" phase3.grads.u3="|a - b| <= 0.002 + 0.0001 * max(|a|, |b|)" phase3.m.u3="|a - b| <= 0.0002 + 0.0001 * max(|a|, |b|)" other_categories=base
```

No `ATOL`, `RTOL`, or category override was passed by the driver.

## 2. Per-category acceptance table

| Check | Status | max abs diff | violations |
|---|---|---:|---:|
| `phase3.eta` | PASS | 0 | 0 |
| `phase3.activations` | PASS | 1.56164169e-5 | 0 |
| `phase3.logits` | PASS | 2.2649765e-6 | 0 |
| `phase3.loss` | PASS | 4.76837158e-7 | 0 |
| `phase3.grads.u1` | PASS | 3.02568078e-4 | 0 |
| `phase3.grads.u3` | PASS | 6.21065497e-4 | 0 |
| `phase3.params.u3` | PASS | 5.93136065e-6 | 0 |
| `phase3.m.u3` | PASS | 7.2401017e-5 | 0 |
| `phase3.v.u3` | PASS | 2.32830644e-7 | 0 |
| `phase3.beta_power.u3` | PASS | 0 | 0 |
| `phase3.step` | PASS | reference 3 = Flux 3 | 0 |

The three overridden categories are 3.30x (`grads.u1`), 3.22x
(`grads.u3`), and 2.76x (`m.u3`) below their frozen absolute floors.
No category needed a widened `rtol`.

## 3. Per-leaf maxima (supplementary)

| Category | Leaves | max rel L2 | Worst leaf | zero/nonzero disagreements |
|---|---:|---:|---|---:|
| `grads.u1` | 62 | 1.998783e-3 | `grad.u1.stage1.block0.conv1.weight` | 0 |
| `grads.u3` | 62 | 4.648105e-3 | `grad.u3.stage1.block1.conv1.weight` | 0 |
| `params.u3` | 62 | 4.420585e-3 | `param.u3.stem.gn.bias` | 0 |
| `param_increment.u3` | 62 | 1.263869e-2 | `delta.u3.stage1.block1.conv1.weight` | 0 |
| `m.u3` | 62 | 2.640012e-3 | `m.u3.stage1.block0.conv1.weight` | 0 |
| `v.u3` | 62 | 2.925289e-3 | `v.u3.stage1.block1.conv1.weight` | 0 |

Per-leaf relative L2 is diagnostic only; it does not replace the elementwise
rule. Every category has zero below-floor norms and zero zero/nonzero
gradient disagreements.

## 4. Algorithm choices

```text
PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0 enabled=true
CHECK cudnn.fma_algo_selection PASS
```

All 32 choices are `CUDNN_FMA_MATH(3)` with `fallback=false`.

| Kind | Chosen algorithms (count) |
|---|---|
| forward | algo 0 (7), algo 6 (4) |
| backward data | algo 0 (7), algo 4 (3) |
| backward filter | algo 0 (3), algo 3 (5), algo 5 (3) |

These are the same FP32 FMA families used by the accepted Step 3c run
(legacy direct, Winograd, FFT-tiling). The residual against the deterministic
PyTorch reference is ordinary FP32 backward-algorithm rounding, not TF32.

## 5. Precision policy and isolated-convolution resolution

The original failure was reproduced as a precision-policy defect, not a model
defect:

- Stock cuDNN.jl maps `PEDANTIC_MATH` to `CUDNN_DEFAULT_MATH`, which permits
  TF32. On the isolated stage-2 convolution this produced
  `conv1 3.5725832e-3` / `conv2 3.64590809e-3` against the unchanged
  PyTorch reference.
- An explicit `CUDNN_FMA_MATH` descriptor removed the discrepancy
  (`conv1 0`, `conv2 1.90734863e-6`, zero violations).
- cuDNN.jl 6.2.2's Float32 autotuner skipped the math-type check, so the
  descriptor alone did not constrain the full-model algorithm selection.
  `scripts/static_scaling/precision.jl` now installs an FMA-aware
  `math_mode` mapping and more-specific forward/backward chooser methods,
  and `configure_precision!` clears the three algorithm caches.
- Focused regression on the final snapshot: `PHASE3_PRECISION_REGRESSION
  PASS` (job `60954547`, `lrdn2211`). The FMA case passed
  (`conv_fixed_fwd/dx/dw`, `classifier_matmul`, autotuned `conv_nnlib_fwd`)
  with the chosen algorithm `algo=0`, `CUDNN_FMA_MATH(3)`,
  `fallback=false`; the stock `CUDNN_DEFAULT_MATH` control failed on
  `conv_fixed_fwd/dx/dw` at `~2.93e-4`.

The final full-model log also verifies both frameworks execute FP32 with TF32
off, `CUDA.math_mode()=PEDANTIC_MATH`, effective
`CUDNN_FMA_MATH(3)`, cleared algorithm caches, and rejected scalar indexing.

## 6. Optimizer and schedule probe

`PHASE3_OPTIMIZER_PROBE PASS` (job `60955690`, `lrdn0279`): 69 checks,
34/34 negative controls rejected, driver exit 0.

- Real training-path configuration asserted in both frameworks:
  Flux 62 leaves = 21 decay (`lambda=1e-4`) / 41 no-decay (`lambda=0`);
  PyTorch two groups 21/41 with `weight_decay=1e-4`/`0.0`;
  `beta=(0.9,0.999)`, `eps=1e-8`.
- Controlled three-update cases with nonzero `m`/`v`; Flux/PyTorch/oracle
  maximum final-update differences `5.38e-8..1.07e-7`.
- Negative controls rejected: omitted update (ratio 1,204.8), wrong decay
  scale (44.9), omitted decay (5.10), decay on a no-decay leaf (5.36),
  wrong epsilon placement (1,204.7), wrong bias correction (996.0), and
  warm-up formula at `W+1` (10.2).
- Schedule boundaries exact in both frameworks:
  `eta(1)=1.0256410256410257e-6`, `eta(W)=1e-3`,
  `eta(W+1)=9.9999996795609018e-4`, `eta(T)=0`; `t=0` and `t=T+1` are
  rejected.

## 7. Reference artifact and independent check

| Item | Value |
|---|---|
| Reference NPZ SHA-256 | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| Manifest SHA-256 | `5ba6c29f3a6167f4341e8e78b0685cb93a5b76e83575beca8db82d8585753eb7` |
| Reference arrays | 337 |
| Independent NumPy checker | `PHASE3_REFERENCE_CHECK PASS arrays=337` |
| Comparison to preserved attempt-3 | 337/337 exact-equal, max diff `0.0` |
| Remote/local SHA-256 | 15/15 MATCH |

The reference generator itself passed 32 checks; the independent artifact
checker passed 337 arrays. The reference is byte-identical to the preserved
attempt-3 archive.

## 8. Environment and precompile

- Precompile job `60954207`, `lrdn3362`, `FLUX_PRECOMPILE_OK`, exit 0.
- No later log contains `Precompiling packages...`.
- A100-SXM-64GB, driver 535.274.02, Julia 1.12.6, CUDA.jl 6.2.2,
  cuDNN.jl 6.2.2 (library reported `92.60.0`), NNlib 0.9.45,
  Optimisers 0.4.9; Python 3.11.17, torch 2.9.1+cu126 (CUDA 12.6,
  cuDNN 9.10.2.21), NumPy 2.4.6. Full record:
  [`runtime-versions.txt`](runtime-versions.txt).

## 9. Verdict

Phase 3 is accepted. The matched single-GPU Flux/PyTorch correctness baseline
passes the frozen FP32/TF32-off contract and the frozen OD-10 rule on the
final source snapshot. The isolated convolution failure is resolved and
documented as a TF32-permitting cuDNN mapping plus a Float32 autotuner
math-type skip, corrected in the reusable precision path. The optimizer and
schedule contract and the hardened Phase 2 remote prerequisite have evidence.

Benchmarks are not part of Phase 3. The next action is Phase 4 distributed
correctness at two and four ranks.
