# Phase 4 GPU acceptance — 2026-10-18

**Accepted under the user-approved absolute error rule: finite `abs(a-b) < 1e-3`.**
The final artifact recheck passed all 202 categories at W=1,2,4 for Flux and PyTorch.
CPU validation is deferred by user decision.

## Evidence and rule change

The GPU evidence comes from Leonardo output
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4`.
Slurm reports final allocation `61296074`, node `lrdn0983`, `COMPLETED`, exit `0:0`.
This session reused the saved arrays and reran the artifact checker locally.
It did not rerun model workloads locally.

The previous `check/` and `local-check/` reports used CUDA trajectory overrides.
Those reports do not establish acceptance under the original strict same-framework rule.
The checker now removes those CUDA-only overrides.
The explicit `--fp32-absolute-1e3` option applies the approved strict absolute bound, with no relative allowance.
The report records this policy in `tolerances.effective_policy`.
Legacy numerical arguments do not override this option.
Hashes, array shapes, sample order, integer counters, rank replicas, and collective protocol checks retain exact comparisons.

Canonical evidence: [approved report](approved-check/phase4_report.json)
(SHA-256 `a5ca161c3e7b7b57a9d5b3d24c7ddb9236ce613554db5ab95f391976a1fa7f8c`),
[summary](approved-check/phase4_check_summary.md)
(SHA-256 `215ca36449afd3b2a2afa1e5ac80332b03b8474b7076e573ad29d9507c319eb7`),
and [checker log](approved-check.log)
(SHA-256 `ba5262254e6e01dd057ba939bbc4f6cb5040f56cc51756bd2671a4228fa9e4d5`).
The canonical recheck finished at `2026-10-18T21:44:16Z` and printed
`PHASE4_CHECK PASS categories=202`.
Earlier reports remain as historical evidence.

## Selected observed maxima

| Comparison | W=1 | W=2 | W=4 |
|---|---:|---:|---:|
| Flux DDP vs shard reference, gradient u3 | 4.99079004e-4 | 2.44891271e-4 | 3.59893776e-4 |
| PyTorch DDP vs shard reference, gradient u3 | 0 | 0 | 5.30967489e-4 |
| Flux vs PyTorch DDP, gradient u2 | 5.08360565e-4 | 5.20486385e-4 | 3.77124175e-4 |
| Flux vs PyTorch DDP, gradient u3 | 4.76382673e-4 | 2.84809619e-4 | 3.15119512e-4 |
| Flux vs PyTorch DDP, parameter u3 | 4.71249223e-6 | 7.88737088e-6 | 4.83170152e-6 |

All rank-state hashes agree exactly within each configuration.
Flux records 64 ordered collectives per update, on every rank.
The independent Float64 AdamW oracle passes, with maximum reported error `2.33121093e-7`.
The largest shard-versus-full diagnostic is `7.18047842e-4`.
These diagnostics remain explicitly non-gating in the checker.
They also fit the approved absolute bound.

The Flux split-control artifact evaluates its u3 comparison at the initial state.
It is not evidence of a matched u3 trajectory.
The separate shard-reference artifacts provide the multi-update control.
The saved CUDA trajectories do not prove the handoff's claim of strict DDP-versus-shard equality.
Acceptance uses the approved `1e-3` rule instead.

## Reproduction

Run from the repository root:

```bash
python3 -m py_compile scripts/static_scaling/check_phase4.py
python3 scripts/static_scaling/check_phase4.py --self-test
bash artifacts/logs/static-scaling/2026-10-18-phase4/recheck-approved.sh
```

The self-test passes 36 checks, including strict-boundary rejection, nonfinite rejection, and preservation of exact comparisons.
The artifact command exits 0 with `PHASE4_CHECK PASS categories=202`.

## Next action

Phase 4 is finalized for the GPU scope. GPU timing is deferred and no timing job is active.
The partial timing pilot (`61305319`, `61305371`) is not Phase 4 acceptance evidence.
Campaign `61305371` was canceled at user request after 23 of 30 records.
When timing is explicitly authorized, use global batch 12 and the alternating saved fixture batches.
Larger-batch memory selection and full-dataset timing remain separate work.
