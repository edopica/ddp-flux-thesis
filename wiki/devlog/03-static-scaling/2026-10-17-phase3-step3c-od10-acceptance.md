# Phase 3 Step 3c: OD-10 category floors applied, full acceptance PASS

Date: 2026-10-17 (UTC run start 2026-10-18T00:08:54Z)
Status: **PASS**. All four required summaries pass and the driver exits 0.
Step 4 is unblocked; Steps 4–6 were not run.

## Approved change

The user approved the OD-10 revision after the Step 3b calibration. The
elementwise rule `|a - b| <= atol + rtol * max(|a|, |b|)` is unchanged.
The base `atol = 1e-5`, `rtol = 1e-4` still covers every category except:

| Category | atol | rtol |
|---|---:|---:|
| `phase3.grads.u1` | 1e-3 | 1e-4 |
| `phase3.grads.u3` | 2e-3 | 1e-4 |
| `phase3.m.u3` | 2e-4 | 1e-4 |

`check_phase3_flux.jl` gained `--grads-u1-atol`, `--grads-u3-atol`, and
`--m-u3-atol` with those frozen defaults, validates all thresholds as
non-negative, prints a `PHASE3_FLUX_CHECK_CONFIG` line and a
`PHASE3_ACCEPTANCE_RULE` line, and applies the overrides only to the three
named categories. `experiment.toml` records the base rule, a
machine-readable `category_overrides` table, and the calibration evidence.
`specification.md` section 11 lists the three override rows and the
numerical basis. `open-decisions.md` marks OD-10 recorded with a
2026-10-17 decision row. OD-2, OD-3, OD-4, and OD-11 remain open.

The numerical basis is the same-framework PyTorch CPU-vs-GPU control
(job `60912525`): it fails the old rule on the same three categories at
3.021732e-4 / 14,365 violations, 4.099458e-4 / 128,170, and
5.057361e-5 / 1,137, with the same worst leaves. The four-comparison
envelope maxima are 4.126653e-4, 7.480253e-4, and 9.655068e-5; each new
floor has at least 2x margin (2.42x, 2.67x, 2.07x). Per-leaf relative L2
stays supplementary. Step 3b is superseded by this recorded calibration,
not by a silent tolerance change.

## Acceptance run

The Step 3b explicit-exclusion sync was repeated (adding the newer
local-only evidence dirs), all changed files were hash-verified remotely,
and the Flux project was precompiled on compute node `lrdn2500`
(job `60941770`, `FLUX_PRECOMPILE_OK`). The full driver ran in a new root
`$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/` on
`lrdn0129` (job `60942933`) with no `ATOL`, `RTOL`, or category override:

| Category | Status | max_abs | violations |
|---|---|---:|---:|
| eta | PASS | 0 | 0 |
| activations | PASS | 1.56164169e-5 | 0 |
| logits | PASS | 1.66893005e-6 | 0 |
| loss | PASS | 4.76837158e-7 | 0 |
| grads.u1 | PASS | 3.02597880e-4 | 0 |
| grads.u3 | PASS | 5.43192029e-4 | 0 |
| params.u3 | PASS | 4.77582216e-6 | 0 |
| m.u3 | PASS | 6.515719e-5 | 0 |
| v.u3 | PASS | 2.32830644e-7 | 0 |
| beta_power.u3 | PASS | 0 | 0 |
| step | PASS | ref 3 = Flux 3 | 0 |

All 32 cuDNN choices are `CUDNN_FMA_MATH(3)` with `fallback=false`; all
225 candidate math types are FMA. The same FP32 algorithm families were
chosen as in Step 3b: forward 0/6, backward data 0/4, backward filter
0/3/5. Per-leaf relative L2 maxima (supplementary): `grads.u1`
1.998759e-3, `grads.u3` 4.288785e-3, `m.u3` 2.571253e-3; no zero/nonzero
disagreements.

The new reference `.npz` SHA-256 is `97e5007c...`, identical to the
preserved attempt-3 reference; 337/337 arrays are exact-equal and the
independent NumPy checker passes locally (`PHASE3_REFERENCE_CHECK PASS
arrays=337`, exit 0). The new manifest SHA-256 is
`221d185968ed3fb30a7d3a33fa8eb1afabad3719d259a4876765b73d1c78260d`.

## Evidence

- `artifacts/logs/static-scaling/2026-10-17-phase3-step3c/`
  (`RESULTS.md`, `PROVENANCE.md`, `COMMANDS.md`, `remote-check.log`,
  `algorithm-choices.txt`, `local-reference-check.log`,
  `reference-comparison.txt`, `SHA256SUMS`)
- `artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/`
- Console logs:
  `artifacts/logs/leonardo/2026-10-17-phase3-step3c.log`,
  `...-precompile.log`
- No Flux source change and no commit.

**Exact next action:** run Step 4 of
`temp/static-graph-gpu-scaling-phase3-handoff.md`: add the compact
cross-framework optimizer-contract probe (decay/no-decay leaves, nonzero
moments, tiny-gradient epsilon placement, negative controls) and schedule
boundary checks at `1`, `W`, `W+1`, and `T`. Do not run Steps 5–6 yet.
