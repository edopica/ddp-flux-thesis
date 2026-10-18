# Phase 3 acceptance failures

Date: 2026-10-17
Status: numerical mismatch unresolved. Recorded during a later review of saved logs.

## Saved attempts

- `artifacts/logs/leonardo/2026-10-17-phase3-remote-attempt1.log`: the artifact checker rejected one-element scalar arrays.
  The current checker accepts scalar shapes `()` and `(1,)`.
  Attempt 3 passed this checker.
- `artifacts/logs/leonardo/2026-10-17-phase3-remote-attempt2.log`: Julia raised a world-age error for the `CUDA` binding.
  The current script imports CUDA at top level.
  Attempt 3 reached the numerical comparisons.
- `artifacts/logs/leonardo/2026-10-17-phase3-remote-attempt3.log`: job 60885091 failed six numerical categories and exited 1.
  Reference generation and artifact checks passed.

Each log records its exact remote command on line 1.
The review did not rerun these commands.

## Numerical failure

Activations first exceed tolerance at stage2.block0.
The isolated stride-2 convolution differs by `3.5725832e-3` on GPU and `7.62939453e-6` on CPU.
The CPU comparison has zero tolerance violations.
The projection convolution matches exactly on both devices.

The leading hypothesis is a cuDNN math-mode mismatch under `CUDA.PEDANTIC_MATH`.
No saved FMA intervention proves this cause yet.
Details: [status review](../../reports/static-scaling-phase3-review.md).

**Exact next action:** preserve the failing reference, then run a controlled current-math versus FMA convolution comparison.
