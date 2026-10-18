# Step 3b results — FMA algorithm chooser, full three-update comparison (FAILED)

Date: 2026-10-17. Source: `2026-10-17-phase3-step3b.log`
(driver job `60910786`, node `lrdn0001`; precompile job `60909956`,
node `lrdn0034`; precision regression job `60910394`, node `lrdn0365`).
The unchanged rule `|a-b| <= 1e-5 + 1e-4*max(|a|,|b|)` still fails.
Tolerances, elementwise rule, model math, optimizer, and Flux source are
unchanged. Per-leaf relative L2 remains supplementary evidence only.

## What changed since Step 3

`precision.jl` now installs, at module load, (a) the FMA-aware
`cuDNN.math_mode(::CUDACore.MathMode)` mapping and (b) more-specific
`cudnnConvolutionAlgoPerfChoose` methods for the forward, backward-data,
and backward-filter performance results. When FMA is enabled and the
descriptor math type is `CUDNN_FMA_MATH`, the chooser requires
`ps.mathType == CUDNN_FMA_MATH` and does not apply cuDNN.jl's Float32
skip. `configure_precision!` also clears the three cuDNN algorithm perf
caches (their keys omit the math type) and verifies they are empty.

## Precision regression (job 60910394)

`PHASE3_PRECISION_REGRESSION PASS`. The new end-to-end autotuned
`Flux.CrossCor` item (stage-2 conv1 shape, 64->128, stride 2, pad 1,
batch 12) matched the Float64 CPU reference at
`rel_l2=4.248559e-7`; its recorded choice was
`algo=0 math_type=CUDNN_FMA_MATH(3) fallback=false`. The `--mode
current` negative control still fails on the three pinned convolution
items (`rel_l2≈2.93e-4`).

## Per-category checks (full model)

| Check line | Status | Detail |
|---|---|---|
| `phase3.eta` | PASS | max_abs=0 violations=0 |
| `phase3.activations` | PASS | max_abs=1.56164169e-5 violations=0 worst=act.stage4.block1 |
| `phase3.logits` | PASS | max_abs=1.66893005e-6 violations=0 |
| `phase3.loss` | PASS | max_abs=4.76837158e-7 violations=0 |
| `phase3.grads.u1` | FAIL | max_abs=3.02594155e-4 violations=18,248 worst=grad.u1.stem.conv.weight |
| `phase3.grads.u3` | FAIL | max_abs=6.67203218e-4 violations=105,443 worst=grad.u3.stem.conv.weight |
| `phase3.params.u3` | PASS | max_abs=4.77582216e-6 violations=0 |
| `phase3.m.u3` | FAIL | max_abs=6.58482313e-5 violations=1,304 worst=m.u3.stem.conv.weight |
| `phase3.v.u3` | PASS | max_abs=2.32830644e-7 violations=0 |
| `phase3.beta_power.u3` | PASS | max_abs=0 violations=0 |
| `phase3.step` | PASS | reference_step=3 flux_updates=3 |

Failing arrays: `grads.u1` affects 20 of 62 leaves; `grads.u3` affects
50 of 62; `m.u3` affects 10 of 62. All forward outputs, final
parameters, and second moments pass. The remaining errors are confined
to the backward pass.

## Per-leaf L2 summary (supplementary)

| Summary line | Data |
|---|---|
| `PHASE3_LEAF_L2_SUMMARY` | `category=grads.u1 leaves=62 max_rel_l2=1.998793e-03 worst_leaf=grad.u1.stage1.block0.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=grads.u3 leaves=62 max_rel_l2=4.536716e-03 worst_leaf=grad.u3.stage1.block1.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=params.u3 leaves=62 max_rel_l2=4.412696e-03 worst_leaf=param.u3.stem.gn.bias zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=param_increment.u3 leaves=62 max_rel_l2=1.260064e-02 worst_leaf=delta.u3.stage1.block1.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=m.u3 leaves=62 max_rel_l2=2.615754e-03 worst_leaf=m.u3.stage1.block0.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=v.u3 leaves=62 max_rel_l2=2.856201e-03 worst_leaf=v.u3.stage1.block1.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |

## Algorithm-choice evidence

`PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0 enabled=true`.
All 32 choices report `CUDNN_FMA_MATH(3)` and `fallback=false`; every
recorded candidate math type is `3`. Chosen algorithms:

| Kind | Chosen algorithms (count) |
|---|---|
| forward | algo 0 (7), algo 6 (4) |
| backward data | algo 0 (7), algo 4 (3) |
| backward filter | algo 0 (2), algo 3 (6), algo 5 (3) |

The non-TF32 residual therefore comes from FP32 cuDNN algorithm
families, not from a tensor-core candidate: algo 6 is Winograd
(forward, stride-1 shapes), algo 4 is Winograd backward-data, algo 3 is
FFT-tiling backward-filter, algo 5 is Winograd-nonfused backward-filter,
and algo 0 is the legacy direct algorithm. Full per-choice lines and
shapes are in `algorithm-choices.txt`.

## Residual-error interpretation

- TF32 has been removed: all recorded choices and all recorded candidate
  math types are `CUDNN_FMA_MATH`. The forward now passes at
  `max_abs=1.56e-5` (previously `7.65e-3`), logits at `1.67e-6`
  (previously `5.65e-4`), and final parameters at `4.78e-6`.
- The remaining failures are backward-only. The gradient maxima are
  `3.03e-4` (u1), `6.67e-4` (u3), and `6.58e-5` (AdamW m), roughly 8-25x
  smaller than the TF32-run gradient failures (`5.24e-3` / `7.49e-3`).
- The selected backward algorithms are FP32 FFT/Winograd/direct variants
  (3, 4, 5, 0). The PyTorch reference runs with `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, TF32 off, cuDNN 9.10.2.21; the Flux path uses
  cuDNN 9.26.0 and picks the fastest FMA algorithm within the stock 10%
  time window. This is an algorithm-family/numerical-rounding difference,
  not an FMA/TF32-policy difference.
- The unchanged acceptance rule does not pass on those residual FP32
  differences. This run does not establish that the model or optimizer is
  wrong, and it does not establish a tolerance that would be justified.

## Reference equality and independent check

The driver generated a new reference. Its SHA-256 is
`97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`,
identical to the preserved attempt-3 reference. Byte-level comparison:
337/337 arrays exact-equal, zero differing arrays, maximum absolute
difference `0.0`. The independent NumPy artifact checker passes locally:
`PHASE3_REFERENCE_CHECK PASS arrays=337 atol=1e-05 rtol=0.0001`, exit 0.

## Outcome

`PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK FAIL`, `PHASE3_REMOTE_CHECK FAIL`, driver exit 1.
Phase 3 acceptance does **not** pass.

## Next decision (user)

The algorithm-selection gap is closed. The residual blocker is FP32
backward algorithm choice under the frozen tolerances. Do not widen the
tolerances or pin algorithms without a decision. The two evidence-backed
options are:

1. constrain backward selection to deterministic/direct algorithms that
   match the PyTorch deterministic reference (for example exclude
   FFT/Winograd and nondeterministic algo 0), then rerun the full driver;
2. keep the current algorithm set and revise the cross-framework
   tolerance with a numerical justification recorded before the final
   acceptance run.

Do not start Step 4 until all four Phase 3 summaries PASS.
