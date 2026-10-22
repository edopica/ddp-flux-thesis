# Static-scaling Phase 3 acceptance-review response

Date: 2026-10-18
Review: [static-scaling-phase3-acceptance-review.md](static-scaling-phase3-acceptance-review.md)
Focused regression evidence: [artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/](../artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/)
Full recheck logs: [artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/](../artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/)
Full recheck reference: [artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/](../artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/)
Deathstar policy log: [artifacts/logs/deathstar/2026-10-18-r1-r2-precision-hardening-policy.log](../artifacts/logs/deathstar/2026-10-18-r1-r2-precision-hardening-policy.log)
Devlog: [2026-10-18-r1-r2-precision-policy-fixes.md](../wiki/devlog/03-static-scaling/2026-10-18-r1-r2-precision-policy-fixes.md)
Status: **Phase 3 acceptance retained; R1/R2 hardened and verified; R3 open as OD-14; R4 corrected here.**

## Verdict

The Phase 3 acceptance recorded on 2026-10-18 is retained. The final hardened
helper run used 32 FMA choices and zero fallbacks, so neither R1 nor R2
invalidated the recorded result. The helper was hardened again after an
independent review; that review found no blocking defect, but it identified
reuse risks that are now closed:

- R1: strict FMA is fail-closed and never returns a non-FMA fallback.
- R2: supported CUDA.jl/cuDNN.jl versions, the override methods, and the
  configured process state are enforced before convolution work.
- R3: the benchmark algorithm-selection and determinism policy is recorded as
  open OD-14 with a proposed default. It is a Phase 6 decision, not a Phase 4
  blocker, and needs user approval before timed runs.
- R4: the two documentation overclaims are withdrawn in this response.

Sealed evidence under `artifacts/` was not edited. This report is the
corrigendum for the affected documentation claims.

The primary jobs are precompile `61187225`, focused precision regression
`61188854`, and full Phase 3 recheck `61191281`. The earlier intermediate
helper runs (precompile `61074887`, focused `61077382`, full recheck
`61091402`) are superseded and retained only as history.

## R1 — fail-closed strict FMA

`scripts/static_scaling/precision.jl` enforces the strict FMA policy:

- A descriptor math type other than `CUDNN_FMA_MATH` is rejected with
  `PrecisionPolicyError` before candidate selection.
- Strict mode searches only for successful `CUDNN_FMA_MATH` candidates. It
  does not accept an algo-0/1 fallback merely because the algorithm number
  suggests a non-tensor-core kernel.
- If no successful FMA candidate exists, the resolver throws
  `PrecisionPolicyError` with the kind and each candidate's
  algo/mathType/status.
- Resolver validation checks index bounds, success status, and the FMA math
  type.
- Selection is split into the pure helpers `_select_candidate` and
  `_resolve_candidate`.
- The stock `fma=false` behavior is unchanged: Float32 math-type skip, the
  same success/time/memory rule, algo-0 backup, and warnings.
- `PrecisionPolicyError` is a typed exception, so callers can distinguish a
  policy violation from other failures.
- `scripts/static_scaling/test_precision_policy.jl` now covers ten synthetic
  cases.

Focused regression on Leonardo, job `61188854` (`lrdn1845`, UTC start
`2026-10-18T12:24:49Z`, driver `12:24:54Z`, exit 0), printed
`PHASE3_PRECISION_REGRESSION PASS`:

| Case | Result | Key evidence |
|---|---|---|
| fma | PASS | `fallback=0`; `CUDNN_FMA_MATH(3)`; fixed-conv rel-L2 fwd `5.998539e-07`, dx `2.103432e-07`, dw `1.121881e-06`; classifier `1.107716e-07`; autotuned NNlib `4.248559e-07` |
| current | PASS as the expected nonzero control | exit 1; `conv_fixed_fwd` about `2.932134e-4` |
| policy | PASS | 10 case PASS lines, 0 FAIL; `CHECK_PRECISION_POLICY PASS` |

The policy cases are `strict_fma_prefers_fma`,
`strict_fma_rejects_non_fma`, `strict_fma_rejects_all_failed`,
`stock_float32_selection`, `stock_algo0_fallback`, `n_truncation`,
`versions_supported`, `versions_unsupported_gate`,
`strict_resolve_never_returns_non_fma`, and `descriptor_policy`.

The run recorded CUDA.jl `6.2.2`, cuDNN.jl `6.2.2`, and cuDNN library
`92.60.0`.

The full Phase 3 recheck on the hardened helper, job `61191281` (`lrdn2911`,
UTC start `2026-10-18T12:30:16Z`, driver `12:30:21Z`, elapsed `00:07:52`,
exit 0), recorded `PHASE3_CUDNN_ALGO_SUMMARY choices=32 fma=32 fallback=0`
and `CHECK cudnn.fma_algo_selection PASS`. All 32 choices are
`CUDNN_FMA_MATH(3)` with `fallback=false`.

## R2 — package and initialization enforcement

`configure_precision!` enforces the reuse contract:

- It requires the supported CUDA.jl/cuDNN.jl `6.2.2` pair before any state
  changes. Unsupported pairs are rejected with `PrecisionPolicyError`.
- It records the loaded cuDNN library version separately from the Julia
  package versions. The verified run records package versions `6.2.2`/`6.2.2`
  and library version `92.60.0`.
- `_check_override_methods` uses `which` to verify that the `math_mode`
  override and all three chooser methods still dispatch to this module. It
  checks both the abstract `AbstractVector{T}` signatures and the concrete
  `Vector{T}`/`Cint` call signatures.
- The chooser throws `PrecisionPolicyError` if `configure_precision!` has not
  run, so every process must initialize the same policy before convolution
  work.
- The public `cudnn_effective_math_type` requires configuration;
  `configure_precision!` uses an internal helper for its own descriptor check.
- The diagnostic `allow_unsupported_versions=true` path emits a warning and
  records `versions_supported=false`. It is intended only for clearly marked
  diagnostics, not for accepted runs.

### Reuse hardening after independent review

The independent review found no blocking defect. It identified reuse risks
that are now closed:

- The cached configure path revalidates the override methods and the process
  math state. A cached unsupported-version configuration is rejected unless
  `allow_unsupported_versions=true`.
- A state-mutating configure failure is permanent for the process; later calls
  must start a fresh process.
- Override-method checks cover both the abstract `AbstractVector{T}` and the
  concrete `Vector{T}`/`Cint` chooser signatures.
- Strictness comes from the committed configuration; a mismatch with the
  mutable `FMA_ENABLED` throws.
- Public `cudnn_effective_math_type` requires configuration; configure uses an
  internal helper.
- Resolver checks index bounds, success status, and the FMA math type.
- Policy tests expanded from 7 to 10 cases; all 10 pass on the hardened helper.

The GPU-free policy test also passed on deathstar under CUDA.jl `6.3.1` /
cuDNN.jl `6.3.1`. That pair would be rejected by the strict runtime gate, so
the result is a synthetic-logic portability check only; it does not authorize
GPU work on 6.3.1.

The canonical Phase 3 precompile is the applicable gate: job `61187225`
(`lrdn0241`, UTC `12:20:59`-`12:24:36`, `FLUX_PRECOMPILE_OK`, exit 0). The
default `scripts/remote/precompile.sh leonardo` remains the wrong target
because the remote thesis root `Manifest.toml` is excluded by `.rsyncignore`
and compute nodes are offline. See
[the failure note](../wiki/failures/2026-10-18-leonardo-precompile-default-sh.md).

## R3 — benchmark algorithm-selection and determinism policy

R3 is a benchmark-policy decision for Phase 6, not a Phase 4 blocker. It is
recorded in [open-decisions.md](../experiments/static-scaling/open-decisions.md)
as **OD-14, open/proposed**:

- Timing claims use framework-native algorithm selection with the exact
  policy and cuDNN versions recorded.
- No numerical claim is made across a policy change.
- A deterministic-mode sensitivity (both frameworks) must accompany
  cross-framework speedups.
- Do not force identical algorithm IDs.

The decision remains open and requires user approval before timed runs.

## R4 — documentation corrections

Two claims in the earlier documentation exceeded the evidence. Both are
withdrawn here:

1. The numerics report's statement that there are "no sign flips" is
   withdrawn. The final gradient comparison reports `max_rel_diff=2`, which
   requires opposite-sign nonzero elements under the implemented metric.
   Near-zero sign changes are compatible with accepted FP32 variation.
2. The zero/nonzero check compares whole-leaf L2 norms, not individual
   elements. Zero whole-leaf disagreements therefore do not establish zero
   elementwise disagreements.

The residual is consistent with FP32 backend/algorithm differences between
valid backends, including possible sensitivity near nonlinear boundaries. The
CPU/GPU controls support that explanation, but they do not isolate every
residual to backward-kernel rounding. The frozen floors remain scoped to this
fixture and cross-framework comparison; they are not Phase 4 DDP tolerances.

Sealed evidence under `artifacts/` was not edited. This response is the
corrigendum.

## Full recheck results

Source identity for job `61191281` (generation-time; later documentation edits
move the live worktree hash): thesis HEAD
`a3786d97f3263cad20b254206f09c095880f624d`, state
`da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199`; Flux
`f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`, state
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
(clean).

| Item | Result |
|---|---|
| `PHASE3_REFERENCE` | PASS, `checks=32` |
| `PHASE3_REFERENCE_CHECK` | PASS, `arrays=337` |
| `PHASE3_FLUX_CHECK` | PASS |
| `PHASE3_REMOTE_CHECK` | PASS |
| cuDNN choices | 32/32 `CUDNN_FMA_MATH`, `fallback=0` |
| `Precompiling packages...` | absent |

| Category | Max abs diff | Violations |
|---|---:|---:|
| eta | 0 | 0 |
| activations | 1.56164169e-05 | 0 |
| logits | 1.7285347e-06 | 0 |
| loss | 4.76837158e-07 | 0 |
| grads.u1 | 3.02582979e-04 | 0 |
| grads.u3 | 6.08410686e-04 | 0 |
| params.u3 | 4.77954745e-06 | 0 |
| m.u3 | 6.32628798e-05 | 0 |
| v.u3 | 2.32830644e-07 | 0 |
| beta_power.u3 | 0 | 0 |
| step | reference 3 = Flux 3 | 0 |

The new reference NPZ SHA-256 is
`97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`;
the manifest SHA-256 is
`bfd979e2d575e24004860c07c216779325923b73e9873bb063c899acab04109a`.
The reference is 337/337 exact-equal to
`artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/`
with `max_abs_diff=0`; the independent reference check passed, and the
remote/local SHA-256 comparison matched 3/3 for precision logs and 5/5 for the
Phase 3 reference. Superseded intermediate evidence remains under
`artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-fixes/` and
`artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-recheck/`.

## Exact next action

Start Phase 4 distributed correctness at two and four ranks per
`temp/static-graph-gpu-scaling-implementation-phases.md` section 6.
