# Phase 2 prep: make_fixtures linear fixture broadcast failure

Date: 2026-10-16
Scope: Leonardo job 60755769 (`run-phase2-prep.sh`, prep step 4/4), static scaling Phase 2
Status: FIXED and verified in retry-5 (2026-10-16); this entry records the retry-4 failure

## Failed command

Prep step 4/4 on `lrdn0018`:

```bash
bash scripts/remote/leonardo/run-phase2-prep.sh
```

with `FORCE=1`, `SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3`,
`SOURCE_STATE_SHA256=3e9ba4d8013ec67383454d86d39ec0caaae2616e43cc5ea4cc963843d6e72561`.

## Failure (verbatim)

```text
=== 4/4 make_fixtures ===
Traceback (most recent call last):
  File "/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_fixtures.py", line 655, in <module>
    sys.exit(main())
             ^^^^^^
  File "/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_fixtures.py", line 631, in main
    linear = build_linear_fixture_torch()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_fixtures.py", line 212, in build_linear_fixture_torch
    y = w @ x + b
        ~~~~~~^~~
RuntimeError: The size of tensor a (5) must match the size of tensor b (3) at non-singleton dimension 1
PHASE2_PREP FAIL (step exited non-zero, see log)
```

`run-phase2-prep: EXIT 1`; srun task 0 exited 1; job 60755769 `FAILED 1:0`
(elapsed 00:25:34).

## Root cause

In `scripts/static_scaling/make_fixtures.py` `build_linear_fixture_torch`
(lines 209-212):

```python
w = torch.randn((3, 7), ...)
b = torch.randn((3,), ...)
x = torch.randn((7, 5), ...)
y = w @ x + b
```

`w @ x` has shape `(3, 5)`; bias `b` has shape `(3,)`. Broadcasting aligns
`b` with the last dimension `5`, so torch raises
`RuntimeError: ... tensor a (5) must match ... tensor b (3)`. The function's
own shape check (line 219) expects `y` shape `(3, 5)`, so the bias must be
broadcastable to `(3, 5)`, e.g. shape `(3, 1)` / `unsqueeze(1)`. The retry-4
runbook was execution-only and stopped at the failure; the correction was
applied afterwards in retry-5.

The bug was previously unreachable: earlier attempts died in
`prepare_dataset.py` (NameError `std64`), so `make_fixtures` had never run
past its first fixture. `build_kernel_fixture_torch` (line 224) was not
reached in retry-4; it was audited before retry-5 (no bias term, uses
`F.conv2d` on a per-example loop, so it has no analogous defect) and was
exercised successfully in retry-5.

## Fix

`scripts/static_scaling/make_fixtures.py` broadcasts the bias explicitly in
both builders:

- `build_linear_fixture_torch` line 212: `y = w @ x + b[:, None]`
- `build_linear_fixture_numpy` line 304: `y = w @ x + b[:, None]`

The expected-output check and the stored fixture contract are unchanged.

## Verification (retry-5, 2026-10-16)

- Prep job 60758615 (`lrdn0018`, lprod) COMPLETED 0:0; fixture step logged
  `FIXTURES wrote ... linear_fixture.npz arrays=7 ...`,
  `FIXTURES verified files=4 arrays=35 sha256=true`,
  `FIXTURES DONE synthetic=False`.
- `linear_fixture.npz` (2,238 B) SHA-256
  `8efa5278981b2b11842d2a701b6f81789c976e033a678a0322e961ecd8f1fa2d`.
- Local `check_all.jl` on the real fixtures: `CHECK linear.forward` and the
  `linear.x_grad` / `linear.w_grad` / `linear.b_grad` checks PASS; 75 checks
  total, exit 0.
- Independent verification re-derived the linear fixture math bitwise and
  caught the adversarial perturbation of `linear_fixture.y[0,0]`
  (`CHECK linear.forward FAIL`), so the fixture is actually consumed by the
  checks. See
  [verification report](../../reports/static-scaling-phase2-verification.md)
  and [`COMMANDS.md`](../../artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md).
- Campaign record:
  [2026-10-16-phase2-artifacts-complete](../devlog/03-static-scaling/2026-10-16-phase2-artifacts-complete.md).

## Avoidance rule

Never write `w @ x + b` when `b` represents per-row biases; use
`b[:, None]` and keep the explicit shape assertion. Audit every fixture
builder locally (`selftest_local.py` plus a synthetic-fixture run) before an
expensive remote job, and never let a failure mask the remaining pipeline.

## Evidence

- Remote prep log: `artifacts/logs/static-scaling/2026-10-16-phase2/remote-prep.log`
  (copied to `$THESIS/artifacts/logs/static-scaling/2026-10-16-phase2/` on Leonardo).
- Local driver log: `artifacts/logs/leonardo/2026-10-16-phase2-prep.log`.
- SACCT and remote listings: `artifacts/logs/leonardo/2026-10-16-phase2-prep-failure-3-fixtures.log`.
- Prepared dataset hashes (steps 1-3 products): `artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt`.
- Fixture directory `$FIX` on Leonardo was empty after retry-4; no fixture npz
  existed until retry-5.

## Impact

Prep steps 1/4-3/4 PASS (`DATASET_PREP DONE`, `SCHEDULE DONE`,
`BUNDLE ... verified ... true`); step 4/4 `make_fixtures` failed before
writing any fixture. The acceptance check (STEP 3), fixture copy-back, and
`COMMANDS.md` were not run because the runbook required stopping. All of
these completed in retry-5 (see Verification).
