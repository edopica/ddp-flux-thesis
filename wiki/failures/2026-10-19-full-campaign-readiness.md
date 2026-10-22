# Campaign launcher failures — 2026-10-19

**Correction:** the batch-256 follow-up fixes the precompile declaration and removes the faulty fallback path.
New regressions cover shell failures, early readiness checks, and missing per-rank samples.
Six regression tests pass locally. GPU preflight remains pending.
See the [correction record](../devlog/03-static-scaling/2026-10-19-batch256-launcher-corrections.md).

Local review reproduced two failures before remote execution.

- The precompile function expands an unset `name` in its `local` declaration.
- A completed batch-512 candidate with excessive memory causes an exit before the batch-256 fallback.

Shell syntax and the Python self-test pass because they do not execute these shell paths.
No remote allocation or benchmark result was affected during this review.
The launcher remains unchanged and requires corrections.

Exact commands, source references, and next actions appear in the
[readiness review](../devlog/03-static-scaling/2026-10-19-full-campaign-readiness-review.md).
