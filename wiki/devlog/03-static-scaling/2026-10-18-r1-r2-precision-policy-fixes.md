# R1/R2 precision-policy fixes — 2026-10-18

Status: implemented and verified remotely; hardened after independent review;
full Phase 3 re-acceptance PASS.

## R1 — strict FMA is fail-closed

`scripts/static_scaling/precision.jl` no longer searches for successful
algo-0/1 fallbacks and no longer returns an algo-0 backup in FMA mode.
When FMA is enabled, a descriptor math type other than `CUDNN_FMA_MATH`
throws `PrecisionPolicyError` before selection. Candidate selection is
split into the pure helpers `_select_candidate` and `_resolve_candidate`;
if no successful `CUDNN_FMA_MATH` candidate exists, the resolver throws
with the kind and every candidate's algo/mathType/status. The resolver
checks index bounds, success status, and the FMA math type. Stock behavior
for `fma=false` is unchanged: Float32 math-type skip, the same
success/time/memory selection rule, algo-0 backup, and warnings.

## R2 — package and initialization enforcement

`configure_precision!` reads `Base.pkgversion(CUDA)` and
`Base.pkgversion(cuDNN)`, requires the supported CUDA.jl/cuDNN.jl 6.2.2
pair, and records the loaded cuDNN library version separately. The
diagnostic `allow_unsupported_versions=true` path warns and records
`versions_supported=false`. `_check_override_methods` uses `which` to
verify the `math_mode` override and all three chooser methods still
dispatch to this module. The chooser throws `PrecisionPolicyError` if
`configure_precision!` has not run. The public `cudnn_effective_math_type`
requires configuration; configure uses an internal helper.

The returned configuration adds `cuda_package_version`,
`cudnn_package_version`, `cudnn_library_version`, and
`versions_supported`; `cudnn_version`, `caches_before`, and
`caches_cleared` remain.

## Reuse hardening after independent review

The independent review found no blocking defect. It identified reuse risks
that are now closed:

- The cached configure path revalidates the override methods and the process
  math state. A cached unsupported-version configuration is rejected unless
  `allow_unsupported_versions=true`.
- A state-mutating configure failure is permanent for the process; later
  calls must start a fresh process.
- Override-method checks cover both the abstract `AbstractVector{T}` and the
  concrete `Vector{T}`/`Cint` chooser signatures.
- Strictness comes from the committed configuration; a mismatch with the
  mutable `FMA_ENABLED` throws.
- Public `cudnn_effective_math_type` requires configuration; configure uses an
  internal helper.
- Resolver checks index bounds, success status, and the FMA math type.
- Policy tests expanded from 7 to 10 cases.

## Tests and checkers

- `scripts/static_scaling/test_precision_policy.jl`: ten synthetic cases
  covering strict selection, non-FMA rejection (algo 0/1), all-failed
  candidates, stock Float32 selection, the stock algo-0 fallback, `n`
  truncation, supported/mismatched versions, the strict resolver invariant,
  and descriptor policy.
- `check_precision_math.jl` records package/library versions and adds a
  `CHECK_PRECISION_MATH_VERSION` gate.
- `check_phase3_flux.jl` accepts only `CUDNN_FMA_MATH` with
  `fallback=false` and adds `cudnn.package_versions_supported`.
- `run-phase3-precision-regression.sh` adds a third fresh-process `policy`
  case and requires ten passing policy cases.

## Verification performed

```bash
julia --startup-file=no -e 'for f in ARGS; Meta.parseall(read(f, String)); println("PARSE_OK ", f); end' \
  scripts/static_scaling/precision.jl scripts/static_scaling/test_precision_policy.jl \
  scripts/static_scaling/check_precision_math.jl scripts/static_scaling/check_phase3_flux.jl
bash -n scripts/remote/leonardo/run-phase3-precision-regression.sh
git diff --check
```

All commands exited 0. No local package load, GPU job, or remote command
was run for this documentation change.

## Remote verification — final hardened run (2026-10-18)

Canonical Phase 3 precompile: job `61187225`, node `lrdn0241`, UTC
`12:20:59`-`12:24:36`, `FLUX_PRECOMPILE_OK`, exit 0. The default
`scripts/remote/precompile.sh leonardo` remains the wrong target: it
precompiles the thesis root env, whose `Manifest.toml` is excluded by
`.rsyncignore`, on an offline compute node. The failed default-script log
from the intermediate run is retained at
`artifacts/logs/leonardo/2026-10-18-r1-r2-precision-fixes-precompile.log`.
Use `precompile-phase3.sh`.

Focused precision regression: job `61188854`, node `lrdn1845`, UTC start
`2026-10-18T12:24:49Z` (driver `12:24:54Z`), exit 0,
`PHASE3_PRECISION_REGRESSION PASS`.

| Case | Result | Key lines |
|---|---|---|
| fma | PASS | `CHECK_PRECISION_MATH PASS`; `cuda_package_version=6.2.2`, `cudnn_package_version=6.2.2`, `cudnn_library_version=92.60.0`; `CHECK_PRECISION_MATH_ALGO_CHOICES choices=1 fma=1 fallback=0 PASS` |
| current | PASS (expected nonzero) | exit 1; `conv_fixed_fwd rel_l2=2.932134e-04`, `conv_fixed_dx=2.928844e-04`, `conv_fixed_dw=2.930009e-04` FAIL; `conv_nnlib_fwd` PASS |
| policy | PASS | ten `CHECK_PRECISION_POLICY case=... PASS` lines, 0 FAIL; `CHECK_PRECISION_POLICY PASS` |

Policy cases: `strict_fma_prefers_fma`, `strict_fma_rejects_non_fma`,
`strict_fma_rejects_all_failed`, `stock_float32_selection`,
`stock_algo0_fallback`, `n_truncation`, `versions_supported`,
`versions_unsupported_gate`, `strict_resolve_never_returns_non_fma`, and
`descriptor_policy`.

The policy warning about `CUDA.jl=6.2.1` comes from the synthetic
unsupported-version case, not the running 6.2.2 environment.

Deathstar portability (GPU-free policy test): Julia 1.12.6 with
CUDA.jl/cuDNN.jl `6.3.1`; `CHECK_PRECISION_POLICY PASS`, exit 0, all ten
cases PASS. The strict runtime gate would reject 6.3.1 for real GPU runs;
this only exercises the synthetic policy logic.

## Full Phase 3 recheck on the hardened helper

The full Phase 3 driver ran on the hardened helper in a fresh evidence root:
job `61191281`, node `lrdn2911`, UTC start `2026-10-18T12:30:16Z` (driver
`12:30:21Z`), elapsed `00:07:52`, exit 0. All four summaries PASS:
`PHASE3_REFERENCE PASS checks=32`, `PHASE3_REFERENCE_CHECK PASS arrays=337`,
`PHASE3_FLUX_CHECK PASS`, and `PHASE3_REMOTE_CHECK PASS`. The log contains no
`Precompiling packages...` line. All 32 cuDNN choices are
`CUDNN_FMA_MATH` with `fallback=0`.

| Category | max abs diff | violations |
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
with `max_abs_diff=0`. The independent reference check passed and the
remote/local SHA-256 comparison matched 3/3 for precision logs and 5/5 for
the Phase 3 reference.

Source identity (generation-time; later documentation edits move the live
worktree hash): thesis HEAD
`a3786d97f3263cad20b254206f09c095880f624d`, state
`da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199`; Flux
`f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`, state
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
(clean).

## Superseded intermediate runs

The pre-hardening helper runs are retained as history, not primary evidence:

- Precompile `61074887` (`lrdn0421`, `FLUX_PRECOMPILE_OK`)
- Focused regression `61077382` (`lrdn0200`, 7 policy cases)
- Full recheck `61091402` (`lrdn0870`, all four summaries PASS)

Their roots are
`artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-fixes/`,
`artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-recheck/`, and
`artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-recheck/`.
They are superseded by `61187225`, `61188854`, and `61191281`.

## R3 — benchmark decision (OD-14, Phase 6)

R3 is a benchmark-policy decision, not a Phase 4 blocker. The proposed
default is recorded as open OD-14 in
[open-decisions.md](../../../experiments/static-scaling/open-decisions.md):
timing claims use framework-native algorithm selection with the exact policy
and cuDNN versions recorded; no numerical claim is made across a policy
change; a deterministic-mode sensitivity (both frameworks) must accompany
cross-framework speedups; do not force identical algorithm IDs. The decision
requires user approval before timed runs.

## R4 — documentation corrections

The numerics report's "no sign flips" claim is withdrawn: the final gradient
comparison reports `max_rel_diff=2`, which requires opposite-sign nonzero
elements under the implemented metric, and near-zero sign changes are
compatible with FP32 variation. The zero/nonzero check compares whole-leaf L2
norms, not individual elements. The residual is consistent with FP32
backend/algorithm differences, including possible sensitivity near nonlinear
boundaries; the controls support but do not isolate every residual to
backward-kernel rounding. Sealed evidence was not edited; the
[acceptance-review response](../../../reports/static-scaling-phase3-acceptance-review-response.md)
is the corrigendum.

Evidence: focused root
`artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/` (10
hashed files; 10/10 OK), full recheck logs
`artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/` (13 hashed
files; 13/13 OK), and recheck reference
`artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/`
(2 hashed files; 2/2 OK). No source-code change and no commit.

## Deferred non-blocking findings (2026-10-18)

The independent delta review found **no blocking defect**. The following
reuse/test gaps are recorded and deliberately deferred so Phase 4 can start:

1. `configure_precision!` reads `cudnn_version()` and `CUDA.math_mode()` after
   the mutation `try` block; an exception there would not set
   `_CONFIGURE_FAILED`. This is very unlikely after the descriptor checks
   succeed. Move the reads inside the `try` on the next touch.
2. The cached-revalidation and permanent-failure branches are not directly
   unit-tested; each remote case configures once in a fresh process.
3. `descriptor_policy` tests the helper, not the chooser call site; the full
   run's 32/32 FMA evidence covers the wiring.
4. The configured guard covers the autotuned chooser and the public
   `cudnn_effective_math_type`; direct fixed-algorithm calls and cuDNN.jl
   cache hits remain outside it. Accepted scripts call
   `configure_precision!` first.
5. Nits: the policy driver's zero-FAIL regex misses exception-suffixed lines;
   the stock-fallback and `n_truncation` cases do not assert the warning or the
   resolver bounds directly; the descriptor error message no longer includes
   `kind`.

None of these affects the accepted Phase 3 result or the Phase 4 correctness
scope. Do not block Phase 4 on them.

## Exact next action

Start Phase 4 distributed correctness at two and four ranks per
`temp/static-graph-gpu-scaling-implementation-phases.md` section 6.

## Focused Phase 4 readiness review (2026-10-18)

Verdict: **GO**. The review found no blocking defect in the R1/R2 fixes.
Scope: helper source, focused regression changes, and retained evidence for
jobs `61188854` and `61191281`.

Strict selection rejects unsuccessful or non-FMA candidates with
`PrecisionPolicyError`. Configuration enforces the supported package pair and
override dispatch. The chooser requires configuration, and the accepted runner
configures before model transfer and convolution.

The focused log reports 10/10 policy cases and `fallback=0`. The full log
reports all four summaries PASS and 32/32 FMA choices with `fallback=0`.
The retained array comparison reports 337/337 exact reference arrays.
Fresh hash checks passed for 10 focused evidence files and 13 full-run files.
All seven current script hashes match the recorded provenance.

Exact local commands (all exit 0):

```bash
# Working directory: artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening
sha256sum -c SHA256SUMS
# Working directory: artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening
sha256sum -c SHA256SUMS
# Working directory: repository root
sha256sum scripts/static_scaling/precision.jl scripts/static_scaling/test_precision_policy.jl scripts/static_scaling/check_precision_math.jl scripts/static_scaling/check_phase3_flux.jl scripts/remote/leonardo/run-phase3-precision-regression.sh scripts/remote/leonardo/run-phase3-check.sh scripts/remote/leonardo/precompile-phase3.sh
```

This review ran no GPU tests or remote jobs. The reference-array comparison
is retained evidence, not a fresh numerical rerun. Only this devlog and
`wiki/context/current.md` changed during the review.

Exact next action: start Phase 4 distributed correctness at two and four ranks
per `temp/static-graph-gpu-scaling-implementation-phases.md` section 6.
