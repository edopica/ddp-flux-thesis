# Phase 3 numerical calibration: PyTorch CPU/GPU envelope — 2026-10-17

Status: diagnostic only. No tolerance, elementwise rule, model math, optimizer,
checker, or Flux source change. No commit. The preserved GPU reference was not
overwritten.

## What was run

A new PyTorch CPU reference was generated on Leonardo with the same generator,
fixture, bundle, three updates, and provenance as the preserved GPU reference
(job `60912525`, node `lrdn3450`). The full Flux checker then ran on CPU against
the preserved PyTorch GPU reference and against the new PyTorch CPU reference
(job `60912664`, node `lrdn3450`). A new numpy-only comparison script applied
the declared elementwise rule to the PyTorch CPU-vs-GPU pair.

Artifacts:

- `artifacts/baselines/static-scaling/2026-10-17-phase3-numerics/`
  (`phase3_reference.npz` SHA-256 `a7ff467b...`, manifest `device="cpu"`);
- `artifacts/logs/static-scaling/2026-10-17-phase3-numerics/`
  (`RESULTS.md`, `COMMANDS.md`, `PROVENANCE.md`, both checker logs, the
  reference comparison log/JSON, and the console logs);
- `scripts/static_scaling/compare_reference_arrays.py` (read-only comparison);
- `scripts/remote/leonardo/run-phase3-numerics-reference.sh`,
  `scripts/remote/leonardo/run-phase3-numerics-flux-cpu.sh`.

## Results

PyTorch CPU vs PyTorch GPU, same generator and inputs, unchanged rule:

| Category | max_abs | violations | max per-leaf rel L2 |
|---|---:|---:|---:|
| activations | 1.764297e-5 | 0 | 1.549527e-6 |
| logits | 2.235174e-6 | 0 | 9.038754e-7 |
| grads.u1 | 3.021732e-4 | 14,365 | 1.839324e-3 |
| grads.u3 | 4.099458e-4 | 128,170 | 3.636289e-3 |
| params.u3 | 2.624467e-6 | 0 | 3.398208e-3 |
| m.u3 | 5.057361e-5 | 1,137 | 2.296527e-3 |
| v.u3 | 6.893606e-8 | 0 | 2.115951e-3 |

Flux CPU vs the preserved PyTorch GPU reference fails only `grads.u1`
(`4.126653e-4`, 32,212 violations), `grads.u3` (`7.480253e-4`, 102,364), and
`m.u3` (`7.226411e-5`, 946). The forward path, logits, loss, final parameters,
second moments, beta powers, and step counter pass. Flux CPU vs PyTorch CPU
shows the same envelope (and one activation element at the `1e-5` atol floor).

The same-framework PyTorch CPU-vs-GPU control reproduces the gradient/moment
residual with the same worst leaves. The Flux GPU residual is within about
1.6x of that independent-backend envelope. The residual is ordinary FP32
algorithm/reduction variation across valid convolution kernels, not an
outlier that suggests a Flux backward defect; the CPU path does not use cuDNN,
so the FMA policy is not the cause.

## Proposed (not applied) tolerance revision

Keep the declared elementwise form and `rtol=1e-4`; raise only the absolute
floor for the three failing backward categories, all with at least a 2x margin
over the four-comparison envelope:

| Category | Proposed | Envelope max_abs | Margin |
|---|---|---:|---:|
| `grads.u1` | `atol=1e-3`, `rtol=1e-4` | 4.126653e-4 | 2.42x |
| `grads.u3` | `atol=2e-3`, `rtol=1e-4` | 7.480253e-4 | 2.67x |
| `m.u3` | `atol=2e-4`, `rtol=1e-4` | 9.655068e-5 | 2.07x |
| all others | unchanged `1e-5` / `1e-4` | — | — |

The user must approve the revision before any specification, configuration, or
checker change. What remains covered: forward/loss/parameters/second
moments/beta/step at the strict rule, large gradient/moment elements at
`rtol=1e-4`, and zero/nonzero disagreement checks. A backward defect confined
to low-magnitude elements below the new floors would not be caught by the
elementwise rule; the supplementary per-leaf relative L2 remains the guard
for that regime.

## Exact next action

User decision: approve the three category floors and update OD-10 plus the
specification, `experiment.toml`, and checker consistently, or reject the
revision and run the next bounded diagnostic (force a deterministic/direct
backward algorithm family and repeat the full comparison). Only after that
decision: rerun the full driver in a fresh evidence root and require all four
summaries to PASS. Do not start handoff Steps 4–6 before then.
