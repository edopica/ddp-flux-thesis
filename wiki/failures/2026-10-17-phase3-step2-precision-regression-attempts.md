# Phase 3 Step 2 precision: failed attempts and corrections

Date: 2026-10-17
Status: resolved; final experiment and regression PASS.

This record keeps the bounded failures from the Step 2 precision work. None
changed the model math, the PyTorch reference, or the acceptance tolerances.

## F1 — Experiment driver rule assumed conv1

First experiment run (job 60896513, node lrdn0102) produced:
`cuda_current conv1 max_abs=0 violations=0`, `conv2 max_abs=2.42638588e-3
violations=1,196,926`; `cuda_fma` matched both at zero violations. The driver
required the current failure specifically on conv1, so it printed
`PHASE3_PRECISION_EXPERIMENT FAIL`.

Cause: cuDNN's autotuner placed the TF32 algorithm on conv2 in that process.
The failure had moved compared with the preserved run (conv1).

Correction: the PASS rule now requires the current case to retain a failure on
conv1 **or** conv2 (violations > 0 and max_abs > 1e-4), while FMA must have
zero violations and both convolution max_abs <= 1e-4. Documented in the driver
output. Final run 60896936 PASS.

## F2 — High-level synthetic convolution masked the math-mode effect

The first regression (job 60897348) used `Flux.conv` (flipped convolution) and
failed its own FMA case: conv2 relative L2 `2.9e-4`, identical in FMA and
current. A second attempt (job 60898845) used `Flux.CrossCor` and showed the
same masking.

Cause: the stock autotuner selected FP32 Winograd (algo 6) in both modes for
the 3x3 stride-1 shape. Winograd transform arithmetic has ~3e-4 relative error
against Float64; it is not TF32, and it is allowed under FMA. Descriptor
queries and probes confirmed the math types were correct
(`CUDNN_FMA_MATH(3)` versus `CUDNN_DEFAULT_MATH(0)`). A separate probe showed
the selected algorithm can change between processes even for the same shape.

Correction: the regression now pins
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` and
`CUDNN_CONVOLUTION_BWD_{DATA,FILTER}_ALGO_1`. Probes 60900208/60900215
(forward) and 60900228/60900308 (gradients) measured FMA 1e-7 to 1e-6 and
current ~2.9e-4, deterministically. The end-to-end Flux/NNlib path remains
covered by the experiment driver, not by the synthetic regression.

## F3 — Regression driver summary regex

Job 60900328 ran the pinned-algorithm regression correctly: FMA PASS, current
FAIL. The driver printed `PRECISION_REGRESSION_CHECK current FAIL` because its
grep expected item names `conv[12]_*`, while the script emits `conv_fixed_*`.

Correction: the regex now accepts `conv[^ ]*_(fwd|dx|dw)`. Re-run 60900440
printed `PHASE3_PRECISION_REGRESSION PASS`.

## Evidence

- `artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/PROVENANCE.md`
- `artifacts/logs/leonardo/2026-10-17-phase3-step2-precision-attempt1.log`
- `artifacts/logs/leonardo/2026-10-17-phase3-step2-precision-regression.log`
- devlog `wiki/devlog/03-static-scaling/2026-10-17-phase3-step2-precision-experiment.md`
