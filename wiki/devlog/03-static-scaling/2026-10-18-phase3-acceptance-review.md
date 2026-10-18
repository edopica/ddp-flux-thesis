# Phase 3 acceptance choices review

Date: 2026-10-18
Status: recorded Phase 3 acceptance retained, two precision-helper follow-ups identified.

The user requested an evaluation of the choices used to complete Phase 3.
The review inspected the precision overrides, focused regression, calibration controls, optimizer comparator, and final evidence.
Details and exact local commands: [acceptance review](../../../reports/static-scaling-phase3-acceptance-review.md).

Fresh local checks passed 30/30 evidence hashes, 29/29 source hashes, and the independent 337-array checker.
No GPU or remote job ran during this review. No production source changed.

Findings:

- Strict-FMA fallback accepts algorithm IDs 0/1 without an FMA requirement. The negative control demonstrates that algo 1 is not intrinsically TF32-free.
- The precision helper documents package-version assumptions but does not enforce them.
- PyTorch deterministic selection and Flux timing-based selection need an explicit benchmark policy.
- The tolerance calibration is reasonable for the fixture. Whole-leaf zero checks do not establish elementwise sign agreement.

The accepted run used 32 FMA selections and no fallback. Its numerical acceptance remains supported.
The local documentation check `git diff --check` passed.

**Exact next action:** make strict-FMA selection fail closed and enforce supported package versions, then run focused regressions before Phase 4 GPU acceptance.
