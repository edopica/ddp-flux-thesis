# Results — R1/R2 precision hardening (2026-10-18)

Status: **PASS — hardened helper verified remotely.**
Primary jobs: precompile `61187225` (`lrdn0241`) and focused precision
regression `61188854` (`lrdn1845`, UTC start `2026-10-18T12:24:49Z`, driver
`12:24:54Z`, exit 0).

## 1. Scope

This root is the primary evidence for the hardened R1/R2 precision helper. The
independent review found no blocking defect; it identified reuse risks that
were closed before this run:

- Cached configure revalidates override methods and process math state; a
  cached unsupported-version configuration is rejected unless
  `allow_unsupported_versions=true`.
- A state-mutating configure failure is permanent for the process; later calls
  must start a fresh process.
- Override-method checks cover both abstract `AbstractVector{T}` and concrete
  `Vector{T}`/`Cint` chooser signatures.
- Strictness comes from the committed configuration; a mismatch with mutable
  `FMA_ENABLED` throws.
- Public `cudnn_effective_math_type` requires configuration; configure uses an
  internal helper.
- Resolver checks index bounds, success status, and FMA math type.
- Policy tests expanded from 7 to 10 cases.

The earlier runs on the pre-hardening helper (precompile `61074887`, focused
`61077382`, full recheck `61091402`) are superseded history.

## 2. Focused precision regression

Job `61188854`, node `lrdn1845`, driver exit 0,
`PHASE3_PRECISION_REGRESSION PASS`.

| Case | Result | Key evidence |
|---|---|---|
| fma | PASS | `fallback=0`; `CUDNN_FMA_MATH(3)`; fixed-conv rel-L2 fwd `5.998539e-07`, dx `2.103432e-07`, dw `1.121881e-06`; classifier `1.107716e-07`; autotuned NNlib `4.248559e-07` |
| current | PASS as expected negative control | exit 1; `conv_fixed_fwd rel_l2=2.932134e-04`, dx `2.928844e-04`, dw `2.930009e-04` FAIL; `conv_nnlib_fwd` PASS |
| policy | PASS | 10 case PASS lines, 0 FAIL; `CHECK_PRECISION_POLICY PASS` |

Policy cases:

1. `strict_fma_prefers_fma`
2. `strict_fma_rejects_non_fma`
3. `strict_fma_rejects_all_failed`
4. `stock_float32_selection`
5. `stock_algo0_fallback`
6. `n_truncation`
7. `versions_supported`
8. `versions_unsupported_gate`
9. `strict_resolve_never_returns_non_fma`
10. `descriptor_policy`

The `CUDA.jl=6.2.1` warning is from the synthetic unsupported-version case, not
the running environment.

## 3. Configuration

| Item | Value |
|---|---|
| CUDA.jl | `6.2.2` |
| cuDNN.jl | `6.2.2` |
| cuDNN library | `92.60.0` |
| cuBLAS math mode | `PEDANTIC_MATH` |
| cuDNN requested/effective | `CUDNN_FMA_MATH(3)` |
| Algorithm choices | `choices=1 fma=1 fallback=0` |

## 4. Caveats

- The deathstar ten-case policy test passes under CUDA.jl/cuDNN.jl `6.3.1`.
  The strict runtime gate would reject that pair; the result is synthetic-logic
  portability only and does not authorize GPU work on 6.3.1.
- The current-mode convolution failure is the expected nonzero negative
  control, not an acceptance failure.
- Source identity is generation-time; later documentation edits move the live
  worktree hash.

## 5. Evidence

- [`console/precision-regression-console.log`](console/precision-regression-console.log)
- [`policy-cases.txt`](policy-cases.txt)
- [`precision/logs/fma.log`](precision/logs/fma.log)
- [`precision/logs/current.log`](precision/logs/current.log)
- [`precision/logs/policy.log`](precision/logs/policy.log)
- [`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt)
- [`SHA256SUMS`](SHA256SUMS)
