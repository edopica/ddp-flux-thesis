# Phase 2 acceptance fixes: R1-R3 corrected and independently verified

Date: 2026-10-17
Scope: static-network GPU scaling (scope 3), Phase 2 acceptance follow-up
Status: PASS (follow-up to the implementation review) — R1-R3 corrected and
independently verified; positive checks rerun; `NEGTEST PASS`; remote re-check
deferred by a Leonardo SSH access blocker
Contract: [implementation plan](../../../temp/static-graph-gpu-scaling-implementation-phases.md)
section 4; [specification](../../../experiments/static-scaling/specification.md)
sections 2, 3, 4, and 8
Evidence package: [response report](../../../reports/static-scaling-phase2-review-response.md),
[negative harness](../../../scripts/static_scaling/negative_tests_local.py)

## Objective

Correct the three P2 acceptance gaps from the
[implementation review](../../../reports/static-scaling-phase2-review.md).
The review found no numerical defect in the saved fixtures; the gaps let
altered scratch inputs pass acceptance. The follow-up had to enforce the
prepared-file and fixture manifests, cover all GroupNorm arrays, rerun the
positive checks, and reproduce the reviewer's three false-positive scenarios as
rejections.

The fixes are in
[`check_dataset.py`](../../../scripts/static_scaling/check_dataset.py),
[`check_fixtures_data.jl`](../../../scripts/static_scaling/check_fixtures_data.jl),
and the new [`negative_tests_local.py`](../../../scripts/static_scaling/negative_tests_local.py)
harness. [`check_all.jl`](../../../scripts/static_scaling/check_all.jl) inherits
the Julia gate.

## Fixes

### R1 — prepared dataset and schedule manifests

`check_dataset.py` now enforces both manifests before any prep-dependent
semantic check:

- `PREP_DATASET_REQUIRED` and `PREP_SCHEDULE_REQUIRED` define the expected
  file sets at lines 116-119; `_check_prep_manifest` (lines 122-181) checks
  manifest presence, exact file set, every listed file's SHA-256, and byte size
  when the manifest records `bytes`; `check_prep_manifests` (lines 184-200)
  runs both gates.
- The six check names are `prep.dataset_manifest_present`,
  `prep.dataset_file_set`, `prep.dataset_files_sha256_bytes`,
  `prep.schedule_manifest_present`, `prep.schedule_file_set`, and
  `prep.schedule_files_sha256_bytes`.
- The gate runs at lines 695-701, before counts, statistics, schedule, bundle,
  and fixture recomputation. On failure it still checks the bundle and fixture
  files, logs `prep.semantic_checks SKIP`, and exits 1.
- `check_permutation_augmentation` regenerates and compares all 50 stored
  epochs bitwise at lines 375-385 (`schedule.regenerated_all_epochs_bitwise`),
  replacing the old epoch-49 regeneration whose result was discarded. The
  regenerated epoch 49 is included in the all-epoch comparison. The check costs
  about 29 s locally (27.5-28.4 s in the response runs).

### R2 — fixture manifest in the Julia entry point

`check_fixtures_data.jl` now enforces `fixture_manifest.json` before numerical
checks:

- Constants at lines 96-101 freeze the four canonical file names, the
  35-array count, and the 20/20 GroupNorm counts.
- `check_fixture_manifest_integrity` (lines 144-319) logs
  `fixtures.manifest_present`, `fixtures.file_set`,
  `fixtures.file_sha256_bytes`, `fixtures.array_set`,
  `fixtures.array_metadata`, and `fixtures.array_sha256`. It checks four
  files, 35 arrays, dtype, shape, and C-order SHA-256.
- The gate runs at lines 336-340. On failure, fixture semantics are SKIPped,
  the bundle checks still run, and the final line fails.
- `check_all.jl` inherits the gate because it calls the same data check.

### R3 — all GroupNorm initial values

Both selectors now cover `.gn`, `.gn1`, `.gn2`, and `.proj_gn` for weights and
biases:

- Python: lines 485-493 select 20 weights and 20 biases and require weights
  equal to 1 and biases equal to 0.
- Julia: `_is_gn_name` (lines 321-326) and lines 485-496 apply the same rule.
- Both checkers fail `bundle.gn_initial_values` on a count mismatch or a value
  mismatch, so the check no longer stops at the four non-residual layers.

## Positive acceptance reruns

| Check | Command (abbreviated) | Exit | Result |
|---|---|---|---|
| Python self-test | `TMPDIR=/tmp/opencode python3 scripts/static_scaling/selftest_local.py` | 0 | five groups PASS, `SELFTEST PASS` |
| Local Julia acceptance | `julia --startup-file=no scripts/static_scaling/check_all.jl --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle ... --bundle-manifest ...` | 0 | 81 CHECK lines (47 data + 34 operator), all PASS; previously 75 |
| Python positive | `check_dataset.py` against the harness-built synthetic PREP | 0 | 63 CHECK lines: 62 PASS plus the usual local `--dataset-root` SKIP (`data.annotation_mapping`); `PHASE2_DATASET_CHECK PASS` |
| Negative harness | `TMPDIR=/tmp/opencode python3 scripts/static_scaling/negative_tests_local.py --workdir /tmp/opencode/phase2-negtest` | 0 | six tests, `NEGTEST PASS` |

The Python positive uses the synthetic PREP that the harness builds by default;
the real `$PREP` is remote and was not re-checked (see below). The Julia
positive uses the real fixture directory and the real bundle. The response
report records 81 unique Julia check names and 63 unique Python names.

## Negative-case harness

`negative_tests_local.py` reproduces the review's three scenarios against
scratch copies. It builds a synthetic PREP by default for the Python-side
tests, uses the real fixture directory and bundle for the Julia-side tests, and
asserts both the positive runs and the rejections.

| Finding | Mutation | Rejection |
|---|---|---|
| R1 | `augmentation[49, 0, 0]` changed to another valid crop offset; schedule manifest left stale | exit 1; `prep.schedule_files_sha256_bytes FAIL`, `prep.semantic_checks SKIP` |
| R1 | same mutation with the schedule SHA-256 and byte size made self-consistent in scratch manifests | `schedule.regenerated_all_epochs_bitwise FAIL bad_epochs=['49']` and `schedule.epoch_independence FAIL` |
| R2 | `data_fixture.npz::train_labels[0]` changed 0→1; fixture manifest left stale | exit 1; `fixtures.file_sha256_bytes FAIL` |
| R3 | `stage1.block0.gn1.weight[0]` changed 1→2; per-array SHA-256 and `bundle_hash` updated | exit 1 in both checkers; `bundle.gn_initial_values FAIL` while `bundle.manifest_sha256 PASS` and `bundle.bundle_hash PASS` |

Harness result lines:

```text
NEGTEST positive_python ok rc=0 final=PHASE2_DATASET_CHECK PASS checks=63 failed=0
NEGTEST positive_julia ok rc=0 data=PASS checks=81 operator=PASS
NEGTEST mutation_prep_epoch49 ok rc=1 reject=prep.schedule_files_sha256_bytes skip=prep.semantic_checks aug49_0_0=4->5 original_sha_unchanged=true
NEGTEST mutation_fixture_label ok rc=1 reject=fixtures.file_sha256_bytes label0=0->1 original_dir_unchanged=true
NEGTEST mutation_gn_julia ok rc=1 reject=bundle.gn_initial_values gn0=1->2 manifest_sha256=PASS bundle_hash=PASS operator=PASS
NEGTEST mutation_gn_python ok rc=1 reject=bundle.gn_initial_values gn0=1->2 manifest_sha256=PASS bundle_hash=PASS
NEGTEST PASS
```

The original fixture directory and augmentation file were hash-checked before
and after the mutations and were unchanged. The harness logs live under
`/tmp/opencode/phase2-negtest/logs/`; they are ephemeral and are not durable
evidence.

## Residual limitations

- A fully self-consistent rewrite of `data_fixture.npz` plus
  `fixture_manifest.json` still passes the local Julia-only check because Julia
  has no PREP reference. The Python PREP cross-check catches it
  (`data_fixture.train_labels FAIL`), and the remote combined acceptance runs
  Python first. This bounds what the local Julia-only check can prove; it does
  not reopen R2.
- A self-consistent dataset-image rewrite is bounded by the manifest hashes,
  global statistics, and the sampled fixture images. Schedule changes remain
  detectable even when their hashes are made self-consistent because all 50
  epochs are regenerated and compared bitwise.
- Minor provenance-record asymmetry: the real `schedule_manifest.json` records
  SHA-256 but no `bytes` fields, so byte size is enforced only where recorded.
  SHA-256 remains the content-identity check.

## Remote re-check and SSH blocker

The hardened checkers were not run against the real Leonardo `$PREP`. On
2026-10-17, `ssh leonardo` failed with:

```text
epicazio@login.leonardo.cineca.it: Permission denied (publickey,gssapi-keyex,gssapi-with-mic)
```

The local SSH agent has no identities (`ssh-add -l`: `The agent has no
identities.`), and the on-disk `~/.ssh/id_ed25519` (`SHA256:TDVaBv4y...`) is
rejected by the server. This is an access/ops blocker, not a code result; no
remote command ran. The remote re-check is the first step once SSH is restored.
The response report also records the real-`$PREP` invocation of
`negative_tests_local.py` for the same remote session. Details:
[Leonardo SSH publickey failure](../../failures/2026-10-17-leonardo-ssh-publickey.md).

## Artifact integrity

No canonical artifact changed during the fixes or verification. The response
report rehashed all five files before and after the probes:

- `initial_bundle.npz`:
  `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`.
- `data_fixture.npz`, `schedule_fixture.npz`, `linear_fixture.npz`, and
  `kernel_fixture.npz` all match their `fixture_manifest.json` entries.
- Mutations ran on scratch copies under `/tmp/opencode`; the canonical files
  stayed read-only.

## Next action

**Exact next action:** restore Leonardo SSH access, then re-run
`scripts/remote/leonardo/run-phase2-check.sh` on the real `$PREP`/fixtures with
the hardened checkers (sync first), then proceed to Phase 3 per
`temp/static-graph-gpu-scaling-implementation-phases.md`.

Prior review entry:
[2026-10-17-phase2-implementation-review](2026-10-17-phase2-implementation-review.md).
Campaign record:
[2026-10-16-phase2-artifacts-complete](2026-10-16-phase2-artifacts-complete.md).

## Original-reviewer follow-up

The original reviewer inspected the corrected Python and Julia checks and the negative harness.
R1–R3 are closed for the reviewed failure scenarios. No new blocking finding arose.
The real-PREP remote re-check remains pending SSH access restoration.

Fresh evidence:

- Canonical `check_all.jl`: 81 checks PASS, exit 0.
- Harness: six cases PASS, final `NEGTEST PASS`, exit 0.
- Python positive: 63 CHECK lines, with the expected absent-raw-dataset SKIP.
- Julia positive: canonical real fixtures, 81 checks, including operators.
- Negative cases: stale epoch-49 hash, stale fixture-label hash, and self-consistent residual GroupNorm mutation all rejected.
- GroupNorm integrity checks passed while its semantic check failed in both languages.

Commands from the repository root:

```bash
JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" julia --startup-file=no scripts/static_scaling/check_all.jl --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --bundle-manifest artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/initial_bundle.manifest.json
mkdir /tmp/opencode/phase2-reviewer-followup-green && cp -al /tmp/opencode/phase2-negtest/synthetic /tmp/opencode/phase2-reviewer-followup-green/synthetic && TMPDIR=/tmp/opencode python3 scripts/static_scaling/negative_tests_local.py --workdir /tmp/opencode/phase2-reviewer-followup-green
```

The second command reused the complete existing synthetic PREP without another full image-array copy.
The harness detaches the augmentation hardlink before mutation.
Logs: `/tmp/opencode/phase2-reviewer-followup-green/logs/` (temporary evidence).

An earlier invocation exceeded the scratch quota during the fixture-directory copy.
It passed positive acceptance and the epoch-49 rejection before the copy failed.
The failure did not establish a checker defect.
Exact failed command and recovery: [scratch-quota record](../../failures/2026-10-17-phase2-review-scratch-quota.md).

**Exact next action:** restore Leonardo SSH access, sync the corrected checkers, and run `scripts/remote/leonardo/run-phase2-check.sh` on the real PREP.
