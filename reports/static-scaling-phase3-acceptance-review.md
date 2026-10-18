# Phase 3 acceptance: review of implementation choices

Date: 2026-10-18
Verdict: retain Phase 3 acceptance for the recorded snapshot and fixture.
Scope: source inspection, saved GPU evidence, and fresh local artifact checks.
This review ran no new GPU or remote jobs.

## Summary

The precision diagnosis and independent-backend calibration justify acceptance.
The process corrected a performance-relevant precision difference before it relaxed the numerical thresholds.
The final run uses the frozen rule, a fresh evidence directory, and an unchanged reference archive.
The optimizer probe addresses the weak sensitivity of the three warm-up updates.

Two precision-helper defects need correction before reuse on new shapes or environments.
The accepted run used 32 FMA choices and zero fallbacks, so neither defect invalidates its recorded result.
The benchmark also needs an explicit algorithm-selection and determinism policy.

## Findings

### R1 — P2: FMA fallback is not fail-closed

Locations: `scripts/static_scaling/precision.jl:240-250,279-306`.

`_first_fallback_candidate` accepts successful algorithms numbered 0 or 1 without checking their math type.
If that search fails, the helper returns an algo-0 result without a success or math-type requirement.
The documentation calls the first branch a known non-tensor-core fallback. Algorithm number alone does not establish that property.

The retained negative control demonstrates the problem directly:
`artifacts/logs/static-scaling/2026-10-18-phase3-final/precision/logs/current.log:4-7` uses algorithm 1 for forward and both gradients.
All three fail the precision probe under `CUDNN_DEFAULT_MATH`, with relative L2 errors around `2.93e-4`.
Thus, algo 1 is not intrinsically TF32-free.

The Phase 3 checker separately requires every recorded choice to report FMA (`check_phase3_flux.jl:447-453`).
That requirement protects the recorded acceptance from non-FMA fallback results.
The reusable helper itself can still return them to future training runners.
Warnings do not enforce the precision contract.

Recommendation: under strict FMA, throw when no successful FMA candidate exists.
Alternatively, implement and independently check an explicit FMA fallback that preserves the descriptor and execution contract.
Do not classify precision from algorithm IDs.
Add a focused negative case with only successful non-FMA algo-0/1 candidates, plus an all-failed case.
No full-model numerical investigation is necessary for this correction.

### R2 — P2: package assumptions are documented but not enforced

Locations: `precision.jl:62-82,455-495`.

The module targets cuDNN.jl 6.2.2 internals and claims that changed assumptions cause a loud failure.
The initializer checks CUDA availability, enum presence, caches, and descriptor math type.
It does not check package versions or establish that the intended chooser methods still control the installed backend.
Compatible-looking internals can change semantics without breaking these checks.

Recommendation: reject unsupported CUDA/cuDNN package versions before GPU work.
Record the loaded library version separately from the Julia package versions.
Retain an explicit revalidation path for future versions.
Require each distributed process to initialize the same precision policy before convolution work.

The current snapshot is pinned and the final source hashes match. This finding concerns reuse, not the historical run.

### R3 — benchmark decision: deterministic PyTorch versus autotuned Flux

Locations: `scripts/static_scaling/model_torch.py:350-368` and `precision.jl:266-276`.

PyTorch sets `cudnn.deterministic=True` and `cudnn.benchmark=False`.
Flux retains cuDNN.jl's timing-based chooser and does not filter candidates by determinism.
The retained runtime record also shows different cuDNN library versions between frameworks.

These differences do not invalidate the numerical comparison.
They can affect throughput, memory, repeatability, and the interpretation of cross-framework timings.
TF32-off alone does not make the algorithm-selection policies identical.

Before timed runs, choose and record a benchmark policy:

- A deterministic comparison requires corresponding constraints on both frameworks.
- A native performance comparison can retain framework-specific selection, but must state its constraints and repeat numerical acceptance after policy changes.

Do not silently carry the reference-only PyTorch restrictions into a claim about unconstrained native performance.
Do not force identical algorithm IDs merely to improve numerical agreement.
Complete autotuning and diagnostic output before the timing window for every shape that the window uses.
Report the cuDNN versions as part of the implementation stack. Matching versions is not mandatory for a documented stack-level comparison.

### R4 — calibration accepted, but its scope and claims need precision

The CPU/GPU controls are a good reason to relax the three backward categories.
PyTorch CPU versus GPU also fails the base rule on those categories.
Flux CPU and GPU results remain in the same numerical range.
The probe isolates optimizer correctness from these backend differences.

The frozen absolute floors are 100x, 200x, and 20x the original floor.
These are substantial relaxations, even though they are limited to three categories.
For `atol=2e-3` and `rtol=1e-4`, the absolute term dominates until magnitude 20.
Consequently, the rule does not enforce a uniform `1e-4` relative error on typical smaller gradient components.
The retained supplementary norms are important evidence:

| Quantity | Maximum per-leaf relative L2 |
|---|---:|
| Gradient, update 1 | 0.200% |
| Gradient, update 3 | 0.465% |
| First moments | 0.264% |
| Cumulative parameter increment | 1.264% |

The increment result is comparable to the independent PyTorch CPU/GPU control, about 1.204%.
This supports bounded acceptance. It does not prove identical long-run training trajectories.

Keep these floors scoped to this fixture and cross-framework comparison.
Do not automatically reuse them for Phase 4 rank agreement or DDP versus its own framework reference.
Use the existing same-framework contract first and diagnose any failure before revising it.
An aggregate relative-norm gate can strengthen future acceptance without tighter elementwise tolerances.
It needs an explicit decision and calibration for the relevant shapes.

Two documentation claims exceed the evidence:

1. The numerics report says there are no sign flips.
   The final gradient comparisons report `max_rel_diff=2`, which requires opposite-sign nonzero elements under the implemented metric.
   Tiny near-zero sign changes are compatible with accepted numerical variation.
2. The reported zero/nonzero checks compare whole-leaf norms (`check_phase3_flux.jl:585-595`), not individual gradient elements.
   Zero whole-leaf disagreements do not establish zero elementwise disagreements.

Describe the residual as consistent with FP32 backend/algorithm differences, including possible sensitivity near nonlinear boundaries.
The controls support that explanation but do not isolate every residual to backward-kernel rounding alone.

## Choices worth retaining

- Controlled FMA intervention before tolerance calibration: addresses a real precision and performance confounder.
- Both descriptor mapping and algorithm selection corrected: the second correction closes the backend dispatch path.
- Cache clearing and fresh processes: prevent reuse of choices made under a different precision policy.
- Native FMA algorithm selection retained: avoids benchmarking an artificially pinned slow kernel merely for close agreement.
- Independent CPU/GPU controls: a stronger calibration basis than the failing Flux comparison alone.
- Small Float64 optimizer oracle and negative controls: sensitive evidence without long training or a custom production optimizer.
- Preserved original reference, fresh final run, and source hashes: the reference did not move to match the implementation.

The process-wide method extensions remain a controlled workaround rather than an upstream package fix.
The benchmark must describe this patched cuDNN selection path as part of the Flux implementation.
An upstream issue or patch is useful follow-up work, but need not block Phase 4.

## Local checks in this review

Commands from the repository root, all exit 0:

```bash
sha256sum -c artifacts/logs/static-scaling/2026-10-18-phase3-final/SHA256SUMS
sha256sum -c artifacts/logs/static-scaling/2026-10-18-phase3-final/source-sha256.txt
python3 scripts/static_scaling/check_phase3_reference.py --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --reference artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference.npz --manifest artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference_manifest.json
```

Results: 30/30 evidence hashes, 29/29 source hashes, and the independent 337-array reference check PASS.
The saved final log contains all four PASS summaries and `choices=32 fma=32 fallback=0`.
This review does not independently rerun the GPU numerical comparison.

## Recommendation

Retain Phase 3 acceptance. Correct R1 and R2 before using the helper for new batch shapes and distributed runners.
Record the benchmark selection policy before timed runs.
Continue with Phase 4 rather than further refinement of the accepted three-update numerical differences.

**Exact next action:** make strict-FMA selection fail closed and enforce supported package versions, then run focused regressions before Phase 4 GPU acceptance.
