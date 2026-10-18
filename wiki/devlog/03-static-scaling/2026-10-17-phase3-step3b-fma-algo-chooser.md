# Phase 3 Step 3b: FMA algorithm chooser — precision policy fixed, backward residual remains

Date: 2026-10-17
Status: precision regression PASS; full three-update comparison FAIL on
`phase3.grads.u1`, `phase3.grads.u3`, `phase3.m.u3`. Phase 3 acceptance
not met; Step 4 not started.

## Correction scope

`scripts/static_scaling/precision.jl` now installs its cuDNN overrides
at module load, controlled by `FMA_ENABLED = Ref(false)`:

- a more-specific `cuDNN.math_mode(::CUDACore.MathMode)` method: with
  FMA enabled, PEDANTIC -> `CUDNN_FMA_MATH`, DEFAULT -> tensor-op,
  FAST -> tensor-op-allow-conversion; with FMA disabled it reproduces
  the stock mapping exactly;
- more-specific `cuDNN.cudnnConvolutionAlgoPerfChoose` methods for the
  forward, backward-data, and backward-filter perf types. With FMA
  enabled and an FMA descriptor they require
  `ps.mathType == CUDNN_FMA_MATH` and do not apply cuDNN.jl's Float32
  math-type skip. The stock 10% time/memory/status selection rule is
  retained. A fallback is allowed only for a successful non-tensor-core
  algo 0/1; otherwise the stock algo-0 backup is used and recorded as
  `fallback=true, math_unknown=true`. Choices are recorded in a
  lock-protected vector (`precision_algo_choices()`), with candidate
  lists when `STATIC_SCALING_PRECISION_LOG_CANDIDATES=1`.

`configure_precision!` sets `FMA_ENABLED` once (conflicting
reconfiguration throws), sets `CUDA.allowscalar(false)` and
`CUDA.math_mode!(CUDA.PEDANTIC_MATH)`, clears the three cuDNN algorithm
perf caches under their locks, verifies they are empty, and verifies the
requested/effective descriptor math types. Process-wide assumptions:
cuDNN.jl 6.2.2 / cuDNN 9.26.0, one configuration per process, called
before any GPU work. The perf-cache keys omit the math type, hence the
clearing requirement.

`check_phase3_flux.jl` now prints every choice as `PHASE3_CUDNN_ALGO`,
summarizes with `PHASE3_CUDNN_ALGO_SUMMARY`, and adds
`CHECK cudnn.fma_algo_selection`, which passes only if every choice is
`CUDNN_FMA_MATH` and not a fallback (a successful algo 0/1 fallback is
tolerated only when every acceptance category passes).
`check_precision_math.jl` adds an autotuned `Flux.CrossCor` NNlib-path
item (stage-2 conv1 shape, stride 2, pad 1, 64->128, batch 12) against a
Float64 CPU reference and asserts non-empty, all-FMA, no-fallback
choices in `--mode fma`. The regression driver still requires the
deterministic fixed-algorithm negative control to fail in `--mode
current`.

## Precision regression PASS

Job `60910394` on `lrdn0365` printed `PHASE3_PRECISION_REGRESSION PASS`.
The autotuned NNlib item chose `algo=0`, `CUDNN_FMA_MATH(3)`,
`fallback=false`, with `rel_l2=4.248559e-7`; the fixed-algorithm
forward/data-gradient/filter-gradient and classifier matmul items pass.
`--mode current` still fails on `conv_fixed_fwd/dx/dw` at
`rel_l2≈2.93e-4`.

## Full comparison FAIL

Job `60910786` on `lrdn0001` (precompile `60909956`, `lrdn0034`) ran in
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/`
with unchanged `atol=1e-5`, `rtol=1e-4`; thesis HEAD `3088f610`, state
`4f2da8a3…`; Flux `f7b4520c`, clean.

| Category | Result | max_abs | violations |
|---|---|---:|---:|
| eta | PASS | 0 | 0 |
| activations | PASS | 1.56164169e-5 | 0 |
| logits | PASS | 1.66893005e-6 | 0 |
| loss | PASS | 4.76837158e-7 | 0 |
| grads.u1 | FAIL | 3.02594155e-4 | 18,248 |
| grads.u3 | FAIL | 6.67203218e-4 | 105,443 |
| params.u3 | PASS | 4.77582216e-6 | 0 |
| m.u3 | FAIL | 6.58482313e-5 | 1,304 |
| v.u3 | PASS | 2.32830644e-7 | 0 |
| beta_power.u3 | PASS | 0 | 0 |
| step | PASS | — | — |

All 32 recorded choices are `CUDNN_FMA_MATH(3)` with
`fallback=false`, and every recorded candidate math type is `3`; no
tensor-op/TF32 candidate was present. Chosen algorithms: forward algo 0
(7) and 6 (4); backward data algo 0 (7) and 4 (3); backward filter
algo 0 (2), 3 (6), and 5 (3). The forward now passes; the residual is
confined to the backward pass and is consistent with FP32 FFT/Winograd
and legacy-direct algorithm rounding versus the PyTorch deterministic
reference (cuDNN 9.10.2.21). It is not a remaining TF32 path.

Per-leaf relative L2 (supplementary): `grads.u1` max `1.998793e-3`
(`stage1.block0.conv1.weight`), `grads.u3` max `4.536716e-3`
(`stage1.block1.conv1.weight`), `m.u3` max `2.615754e-3`; no
zero/nonzero disagreements or below-floor leaves.

## Reference and independent check

The new reference SHA-256 is `97e5007c…`, identical to the preserved
attempt-3 reference; 337/337 arrays exact-equal, maximum difference
`0.0`. The local independent NumPy checker passes
(`PHASE3_REFERENCE_CHECK PASS arrays=337`, exit 0). The new manifest
SHA-256 is `d9985934…`.

## Decision required

The algorithm-selection gap is closed; the residual blocker is FP32
backward algorithm choice under frozen tolerances. Do not widen
tolerances or pin algorithms without a user decision. Evidence-backed
options: (1) restrict backward selection to deterministic/direct
algorithms matching the PyTorch reference, then rerun; or (2) justify a
cross-framework tolerance revision numerically before a final run.
Step 4 remains blocked.

## Evidence

- `artifacts/logs/static-scaling/2026-10-17-phase3-step3b/`
  (`RESULTS.md`, `PROVENANCE.md`, `COMMANDS.md`, `remote-check.log`,
  `algorithm-choices.txt`, `precision-regression-summary.txt`,
  `local-reference-check.log`, `reference-comparison.txt`, SHA256SUMS)
- `artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/`
- Console logs:
  `artifacts/logs/leonardo/2026-10-17-phase3-step3b.log`,
  `…-precision-regression.log`, `…-precompile.log`
- No Flux source change and no commit.
