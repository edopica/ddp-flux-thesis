# Phase 3 Step 2: FMA precision experiment and focused regression PASS

Date: 2026-10-17
Status: experiment PASS, focused regression PASS. Step 3 (full three-update
comparison) not run.

## Question and bounded scope

Step 2 asked whether the isolated stage-2 GPU convolution discrepancy against
the unchanged PyTorch reference came from TF32 remaining enabled. The installed
cuDNN.jl maps `CUDA.PEDANTIC_MATH` to `CUDNN_DEFAULT_MATH`; the controlled
experiment compared that stock mapping with an explicit
`PEDANTIC_MATH -> CUDNN_FMA_MATH` mapping. Each case ran in a fresh Julia
process. The PyTorch reference, the stage-2 arrays, the model math, and the
`atol=1e-5` / `rtol=1e-4` tolerances were not changed.

No Flux source changed and no commit was made.

## Result (job 60896936, node lrdn0102)

| Case | conv1 max_abs / violations | conv2 max_abs / violations | proj |
|---|---|---|---|
| cuda current | 3.5725832e-3 / 1,225,601 | 3.64590809e-3 / 1,316,175 | 0 / 0 |
| cuda FMA | 0 / 0 | 1.90734863e-6 / 0 | 0 / 0 |
| CPU control | 7.62939453e-6 / 0 | 5.48362732e-6 / 0 | 0 / 0 |

Requested/effective math types: current requested and effective
`CUDNN_DEFAULT_MATH(0)`; FMA requested and effective `CUDNN_FMA_MATH(3)` for
conv1, conv2, and proj; cuBLAS `PEDANTIC_MATH` in both. `cuDNN.version()` prints
`92.60.0` (parser; library 9.26.0).

FMA removes the discrepancy: the GPU convolutions are at CPU level with zero
violations, while the current case retains the failure. The experiment driver
prints `PHASE3_PRECISION_EXPERIMENT PASS`.

## Algorithm-selection caveat

The failing convolution is not fixed to conv1. The first run (job 60896513)
matched conv1 exactly and failed conv2 at `2.42638588e-3`. cuDNN's stock
autotuner selects algorithms by timing, not by the requested math type. A
synthetic high-level `Flux.CrossCor` test can select FP32 Winograd in both
math modes (algo 6, relative L2 ~2.9e-4 against Float64), which masks the
TF32-vs-FMA difference. Pinning
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` and
`CUDNN_CONVOLUTION_BWD_{DATA,FILTER}_ALGO_1` separates the modes
deterministically: FMA ~1e-7 to 1e-6, `CUDNN_DEFAULT_MATH` ~2.9e-4.

The experiment uses the unchanged Flux/NNlib model path and both stage-2
convolutions, so the moving failure is visible. The regression pins algorithms
to test the math-mode contract.

## Reusable path and regression

Changed/added:

- `scripts/static_scaling/precision.jl`: `StaticScalingPrecision` with
  `configure_precision!`, `cudnn_requested_math_type`,
  `cudnn_effective_math_type`, `cudnn_version`, and `math_type_label`.
  The FMA mapping is a new, more-specific `cuDNN.math_mode` method; the stock
  method is preserved. Scope is process-wide; call before any GPU work.
- `scripts/static_scaling/diagnose_phase3_flux_stage2.jl`: `--precision
  current|fma` (default fma) and the `FLUX_STAGE2_CONFIG` line.
- `scripts/remote/leonardo/run-phase3-precision-experiment.sh`: three fresh
  processes (`cuda_current`, `cuda_fma`, `cpu_current`) and the PASS rule.
- `scripts/static_scaling/check_phase3_flux.jl` and
  `scripts/static_scaling/diagnose_phase3_flux.jl`: the cuda path now calls
  `configure_precision!(fma=true)`.
- `scripts/static_scaling/check_precision_math.jl`: GPU-only regression for
  requested/effective math types, pinned convolution forward/data-gradient/
  weight-gradient against a Float64 CPU reference, and classifier matmul
  against a Float64 CPU reference; bound `REL_L2_BOUND=1e-4`.
  `--mode current` is the negative control.
- `scripts/remote/leonardo/run-phase3-precision-regression.sh`: fma must PASS,
  current must FAIL with a convolution discrepancy.
- `scripts/static_scaling/model_flux.jl`: docstrings state that GPU training
  paths must call `StaticScalingPrecision.configure_precision!()` before GPU
  work; no model math change.

Regression (job 60900440, node lrdn2176): FMA `conv_fixed_fwd` 6.0e-7,
`dx` 2.1e-7, `dw` 1.1e-6, matmul 1.1e-7, all PASS. Current `fwd` 2.93e-4,
`dx` 2.93e-4, `dw` 2.93e-4, all FAIL; matmul 1.1e-7 PASS. Driver prints
`PHASE3_PRECISION_REGRESSION PASS`.

## Evidence

Run root:
`artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/`
(`PROVENANCE.md`, `SHA256SUMS`, five remote logs, four console logs).
Source identity: thesis HEAD `3088f61020137912af32d9c969e87ea9ae2a71c3`,
state `b72b0851fb0a94b2deb31cc06c029ae50327170b0305d819d0868af85e5f93ae`;
Flux`f7b4520c` clean. Remote and local log/source SHA-256 all MATCH.
Jobs: precompile 60896027; experiment 60896513 (rule FAIL) and 60896936
(PASS); regression 60897348/60898845 (Winograd masking), 60900328 (driver
regex), 60900440 (PASS). All on `boost_qos_dbg`, 1 A100.

Failure history:
[failure record](../../failures/2026-10-17-phase3-step2-precision-regression-attempts.md).

**Exact next action:** run Step 3, the complete three-update Flux comparison
with `configure_precision!(fma=true)` and unchanged tolerances, in a new output
directory; retain per-category maxima, violation counts, and relative L2.
