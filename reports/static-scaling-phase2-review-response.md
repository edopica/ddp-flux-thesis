# Static-scaling Phase 2 review response — independent verification

Date: 2026-10-17
Role: independent verification agent (read-only with respect to production files)
Scope: the fixes for R1, R2, and R3 in `reports/static-scaling-phase2-review.md`.
The verifier wrote only this report and scratch files under `/tmp/opencode`.

## Verdict

**R1, R2, and R3 are corrected for the reviewed failure scenarios. The evidence is adequate.**
The original positive acceptance commands pass, the new negative harness passes, and the
reviewer's three exact false-positive scenarios are now rejected.

Two residual limitations are recorded below. They do not reopen R1-R3, but they bound what
the local Julia-only check can prove:

1. R2: a *fully self-consistent* rewrite of `data_fixture.npz` plus `fixture_manifest.json`
   still passes the local Julia-only check, because Julia has no PREP reference. The Python
   checker catches it when PREP is available (`data_fixture.train_labels FAIL`). The remote
   combined acceptance runs Python first, so the combined acceptance is not blind to it.
2. R1: a self-consistent dataset-image rewrite is not independently detectable beyond the
   manifest hashes, global statistics, and the sampled fixture images. Schedule changes are
   caught even when their hashes are made self-consistent, because all 50 epochs are
   regenerated and compared bitwise.

No canonical artifact changed during verification.

## Positive acceptance reruns

All commands ran from the repository root
`/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

| # | Command | Exit | Check count | Result |
|---|---|---|---|---|
| 1 | `TMPDIR=/tmp/opencode python3 scripts/static_scaling/selftest_local.py` | 0 | 5 groups | Five `SELFTEST ... ok` lines, `SELFTEST PASS` |
| 2 | `JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" julia --startup-file=no scripts/static_scaling/check_all.jl --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --bundle-manifest artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/initial_bundle.manifest.json` | 0 | 81 CHECK lines (47 data + 34 operator) | 81 PASS, 0 FAIL, 0 SKIP; both `PHASE2_JULIA_DATA_CHECK PASS` and `PHASE2_JULIA_OPERATOR_CHECK PASS` |
| 3 | `TMPDIR=/tmp/opencode python3 scripts/static_scaling/negative_tests_local.py --workdir /tmp/opencode/verify2-negtest` | 0 | 6 tests | All six `NEGTEST ... ok`, `NEGTEST PASS` |
| 4 (supplementary) | `python3 scripts/static_scaling/check_dataset.py --prep-dir /tmp/opencode/verify2-negtest/synthetic --fixture-dir /tmp/opencode/verify2-negtest/synthetic --bundle /tmp/opencode/verify2-negtest/synthetic/initial_bundle.npz --bundle-manifest /tmp/opencode/verify2-negtest/synthetic/initial_bundle.manifest.json` | 0 | 63 CHECK lines | 62 PASS, 0 FAIL, 1 expected SKIP (`data.annotation_mapping`: no local raw dataset root); `PHASE2_DATASET_CHECK PASS` |

Positive Python detail, including the new all-epoch regeneration:

```text
CHECK prep.dataset_manifest_present PASS .../dataset_manifest.json
CHECK prep.dataset_file_set PASS required=6 listed=8 missing=none extra=none
CHECK prep.dataset_files_sha256_bytes PASS checked=8 mismatches=none
CHECK prep.schedule_manifest_present PASS .../schedule_manifest.json
CHECK prep.schedule_file_set PASS required=2 listed=2 missing=none extra=none
CHECK prep.schedule_files_sha256_bytes PASS checked=2 mismatches=none
CHECK schedule.regenerated_all_epochs_bitwise PASS bad_epochs=none 50 epochs in 27.49s
CHECK bundle.gn_initial_values PASS weights==1 (20 arrays) biases==0 (20 arrays)
CHECK fixtures.manifest_present PASS synthetic=True
CHECK fixtures.file_sha256 PASS files=4
CHECK fixtures.data_fixture.npz PASS arrays=9 extra=none missing=none bad=none
CHECK fixtures.schedule_fixture.npz PASS arrays=5 extra=none missing=none bad=none
CHECK fixtures.linear_fixture.npz PASS arrays=7 extra=none missing=none bad=none
CHECK fixtures.kernel_fixture.npz PASS arrays=14 extra=none missing=none bad=none
PHASE2_DATASET_CHECK PASS
```

Positive Julia counts by section: `bundle` 10, `data` 9, `fixtures` 6, `kernel` 14,
`kernel_a` 5, `kernel_b` 3, `labels` 4, `linear` 12, `normalization` 1, `schedule` 12,
`tail` 2, `transform` 3. Total 81, all unique.

### Harness setup note (environmental, not a checker result)

The first harness invocation attempted the synthetic PREP build and died with exit `-7`
(SIGBUS) while writing the 1.2 GB `train_images.npy` under `/tmp` (tmpfs memory pressure):

```text
NEGTEST log: .../logs/build_synthetic.log (exit -7)
NEGTEST setup error: synthetic PREP build failed (exit -7)
NEGTEST FAIL setup
```

After removing only the verifier's incomplete scratch, the rerun reused the complete
synthetic PREP already present in `/tmp/opencode/phase2-negtest`:

```text
NEGTEST build reuse: /tmp/opencode/verify2-negtest/synthetic (synthetic fixtures + bundle present)
```

That PREP is independently revalidated by the positive Python run above: all 50 schedule
epochs regenerate bitwise, both statistics are recomputed, and all 35 fixture arrays match
their manifest hashes. The failed build did not reach any checker. This is a host-memory
limitation of `/tmp`, not a defect in the fixes.

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

## R1 — prepared dataset and schedule manifest enforcement

### Fix summary

- `scripts/static_scaling/check_dataset.py:116-119` defines the required file sets:
  dataset `train_images.npy`, `train_labels.npy`, `val_images.npy`, `val_labels.npy`,
  `class_wnids.txt`, `stats.json`, with `permutations.npy`/`augmentation.npy` allowed as
  extras; schedule exactly `permutations.npy` and `augmentation.npy`.
- `check_dataset.py:122-181` (`_check_prep_manifest`) checks manifest presence, the exact
  file set, every listed file's SHA-256, and byte size when the manifest records `bytes`.
- `check_dataset.py:184-200` (`check_prep_manifests`) requires both
  `dataset_manifest.json` and `schedule_manifest.json`; a missing manifest logs
  `manifest_present FAIL` and SKIPs the dependent checks.
- `check_dataset.py:695-701` runs the manifest gate **before** every prep-dependent semantic
  section. On failure it still runs the bundle and fixture-integrity checks, then exits 1.
- `check_dataset.py:375-385` regenerates and compares **all 50** stored epochs bitwise
  (`schedule.regenerated_all_epochs_bitwise`), replacing the old epoch-49 regeneration
  whose result was discarded.
- `check_dataset.py:485-493` selects all 20 GroupNorm weights and 20 biases (see R3).

### Positive evidence

- Real `dataset_manifest.json` lists exactly the six required files; real
  `schedule_manifest.json` lists exactly the two required files.
- Synthetic `dataset_manifest.json` lists eight files (six required plus the two allowed
  extras); the synthetic schedule manifest lists exactly two. Both pass the same gate:
  `required=6 listed=8` and `required=2 listed=2`.
- Missing-manifest probe on a scratch copy: removing `schedule_manifest.json` produced
  `CHECK prep.schedule_manifest_present FAIL`, SKIPs for file set and hashes,
  `prep.semantic_checks SKIP`, and exit 1. There is no silent skip on the positive path.

### Negative reproduction (reviewer-style, independent of the harness)

Scratch copy of the synthetic PREP; `augmentation[49,0,0]` changed from `4` to `7`
(valid crop range); `schedule_manifest.json` left stale; the source file's SHA-256 was
unchanged (`9f6bbea465cf2493...` before and after).

```text
CHECK prep.dataset_files_sha256_bytes FAIL checked=8 mismatches=['augmentation.npy:sha256']
CHECK prep.schedule_files_sha256_bytes FAIL checked=2 mismatches=['augmentation.npy:sha256']
CHECK prep.semantic_checks SKIP skipped: prep manifest integrity failed
PHASE2_DATASET_CHECK FAIL prep.dataset_files_sha256_bytes,prep.schedule_files_sha256_bytes
```

Exit 1. On the synthetic PREP both manifests list `augmentation.npy`, so both SHA checks
fail. On the real PREP the dataset manifest lists only six files (no augmentation entry),
so only `prep.schedule_files_sha256_bytes` fails; that is the line required by the review.

Supplementary self-consistent probe: after also updating the augmentation SHA-256 and byte
size in both scratch manifests, the manifest checks pass and the semantic layer rejects the
mutation:

```text
CHECK prep.dataset_files_sha256_bytes PASS checked=8 mismatches=none
CHECK prep.schedule_files_sha256_bytes PASS checked=2 mismatches=none
CHECK schedule.epoch_independence FAIL epoch 0 unchanged after regenerating epoch 49 last_epoch_matches_stored=False
CHECK schedule.regenerated_all_epochs_bitwise FAIL bad_epochs=['49'] 50 epochs in 28.36s
PHASE2_DATASET_CHECK FAIL schedule.epoch_independence,schedule.regenerated_all_epochs_bitwise
```

This closes the "valid-range change after epoch 1" requirement even when the hashes are
made self-consistent.

### Code-review findings

- Enforcement ordering is correct: the gate is the first prep check after the directory
  existence check.
- SHA-256 is enforced for every listed file. Byte size is enforced where recorded:
  real and synthetic dataset manifests and the synthetic schedule manifest carry `bytes`.
  The **real `schedule_manifest.json` has no `bytes` fields** (its producer records only
  `path`, `layout`, and `sha256`), so the schedule byte-size branch is not exercised there.
  This is a minor deviation from the literal "bytes+sha256" wording, not from the review's
  requirement: SHA-256 is the stronger identity check, and a byte-size mismatch cannot
  occur without a SHA-256 mismatch in practice. Adding `bytes` to `make_schedule.py` would
  make the provenance records symmetric.
- The Python file-set test accepts a manifest that lists a required file twice (set
  equality plus entry-count equality, but no duplicate-name rejection). This is harmless
  for acceptance (the duplicate is hashed twice and no extra file is admitted), and the
  Julia file-set check rejects duplicates by count. No action is required for R1.
- A malformed manifest entry with no `path` can raise a `TypeError` inside the set sorting;
  `main()` catches it as `unhandled_exception FAIL` and exits 1. It still rejects, but the
  failure message is less precise than the other paths.
- Full-50-epoch regeneration costs about 27.5 s locally (50 epochs, permutations and
  augmentations). This is the expected cost increase and matches the author's ~29 s
  estimate.

## R2 — fixture manifest enforcement in the Julia entry point

### Fix summary

- `scripts/static_scaling/check_fixtures_data.jl:96-101` freezes the four canonical file
  names and the 35-array and 20/20 GroupNorm counts.
- `check_fixtures_data.jl:103-117` maps Julia element types to manifest dtype names for all
  standard integer and float types, `Bool`, and a lowercase fallback.
- `check_fixtures_data.jl:122-127` computes C-order SHA-256; the supported fixture dtypes
  (`Float32`, `UInt8`, `Int64`) use `replay.jl`'s `c_order_sha256`, the general path uses
  the same C-order byte rule.
- `check_fixtures_data.jl:144-319` (`check_fixture_manifest_integrity`) logs six checks:
  `fixtures.manifest_present`, `fixtures.file_set`, `fixtures.file_sha256_bytes`,
  `fixtures.array_set`, `fixtures.array_metadata`, `fixtures.array_sha256`.
- `check_fixtures_data.jl:336-340` runs the integrity gate before reading the fixtures for
  the numerical checks. When it fails, `fixtures.semantic_checks` is SKIPped; bundle checks
  still run, and the final line fails.

### Positive evidence

```text
CHECK fixtures.manifest_present PASS .../fixture_manifest.json
CHECK fixtures.file_set PASS n=4 missing=none extra=none
CHECK fixtures.file_sha256_bytes PASS n_checked=4 sha_mismatch=none bytes_mismatch=none
CHECK fixtures.array_set PASS n=35 want=35 missing=none extra=none duplicates=none unreadable=none
CHECK fixtures.array_metadata PASS n_checked=35 mismatches=none
CHECK fixtures.array_sha256 PASS n_checked=35 mismatch=none
```

The actual manifest contains 35 arrays: 4 `uint8`, 10 `int64`, and 21 `float32` arrays,
with no 0-dimensional/scalar entries. All three dtypes are covered by the Julia mapping and
all 35 C-order hashes pass. There were zero SKIP lines in the positive Julia run.

### Negative reproduction (reviewer-style, independent of the harness)

Full copy of the real fixture directory; `data_fixture.npz::train_labels[0]` changed from
`0` to `1`; `fixture_manifest.json` left stale.

```text
CHECK fixtures.file_sha256_bytes FAIL n_checked=4 sha_mismatch=data_fixture.npz:sha256 bytes_mismatch=none
CHECK fixtures.array_sha256 FAIL n_checked=35 mismatch=data_fixture.npz::train_labels
CHECK fixtures.semantic_checks SKIP fixture manifest integrity not satisfied
PHASE2_JULIA_DATA_CHECK FAIL fixtures.file_sha256_bytes,fixtures.array_sha256
PHASE2_JULIA_OPERATOR_CHECK PASS
```

Exit 1. The real fixture directory was unchanged after the probe
(`data_fixture.npz` still `6614a00ba7cac588...`).

Missing-manifest probe on a scratch copy: `CHECK fixtures.manifest_present FAIL`, five
SKIP lines, `PHASE2_JULIA_DATA_CHECK FAIL fixtures.manifest_present`, exit 1.

### Code-review findings and residual risk

- The expected array-name set is derived from the four loaded NPZ archives, not from a
  separate hardcoded 35-name list. The manifest must match that set exactly, with count 35,
  and every one of the 35 names is still required by a frozen semantic contract
  (`data.*`, `schedule.*`, `linear.*`, `kernel.*`), so a rename or substitution fails a
  contract. No exploitable gap was found in the reviewed threat model.
- Residual limitation, confirmed by an independent probe: if an attacker rewrites
  `data_fixture.npz` **and** updates the fixture manifest's file/array hashes consistently,
  the local Julia-only check passes:

  ```text
  CHECK fixtures.file_sha256_bytes PASS n_checked=4 sha_mismatch=none bytes_mismatch=none
  CHECK fixtures.array_sha256 PASS n_checked=35 mismatch=none
  CHECK labels.train_range_0_199 PASS n=12
  PHASE2_JULIA_DATA_CHECK PASS
  PHASE2_JULIA_OPERATOR_CHECK PASS
  ```

  The Python checker with PREP available catches the same artifact:

  ```text
  CHECK fixtures.file_sha256 PASS files=4
  CHECK fixtures.data_fixture.npz PASS arrays=9 extra=none missing=none bad=none
  CHECK data_fixture.train_labels FAIL
  PHASE2_DATASET_CHECK FAIL data_fixture.train_labels
  ```

  This is inherent: Julia's local entry point has no PREP to recompute labels from. The
  reviewed R2 scenario (stale manifest after an in-range label change) is correctly
  rejected. The remote combined acceptance runs the Python PREP cross-check, so it retains
  the stronger guarantee. If local Julia-only label provenance is ever required, the
  manifest needs a frozen label digest or an explicit PREP cross-check; that is a Phase 3
  design question, not a defect in this fix.

## R3 — all GroupNorm initial values

### Fix summary

- Python: `scripts/static_scaling/check_dataset.py:485-493` selects suffixes
  `.gn.weight`, `.gn1.weight`, `.gn2.weight`, `.proj_gn.weight` (and the matching biases),
  asserts exactly 20 weights and 20 biases, and requires weights equal to 1 and biases
  equal to 0.
- Julia: `scripts/static_scaling/check_fixtures_data.jl:321-326` (`_is_gn_name`) covers the
  same four suffixes; `check_fixtures_data.jl:485-496` asserts 20/20 counts and the 1/0
  values. `classifier` and other arrays cannot match these suffixes.

### Positive evidence

Both checkers now report full coverage:

```text
CHECK bundle.gn_initial_values PASS weights==1 (20 arrays) biases==0 (20 arrays)
```

(Python positive log and Julia positive log, verbatim and identical in meaning.)

### Negative reproduction (reviewer-style, independent of the harness)

Copy of the real bundle and manifest; `stage1.block0.gn1.weight[0]` changed from `1.0` to
`2.0`; that array's SHA-256 and `bundle_hash` updated so all hashes are self-consistent
(the checker recomputed the same value `35e91592de37...`).

Python (`check_dataset.py`, consistent synthetic PREP/fixture set; exit 1):

```text
CHECK bundle.gn_initial_values FAIL weights!=1 (20 arrays) biases==0 (20 arrays)
CHECK bundle.manifest_sha256 PASS checked=62 sha_mismatch=none meta_mismatch=none
CHECK bundle.bundle_hash PASS recomputed=35e91592de37e9e3270513fc1eec0f26bd5a038e49981d985b837b6b3188c3a2 recorded=35e91592de37e9e3270513fc1eec0f26bd5a038e49981d985b837b6b3188c3a2 rule="concat C-order bytes, manifest order"
PHASE2_DATASET_CHECK FAIL bundle.gn_initial_values
```

Julia (`check_all.jl`, exit 1):

```text
CHECK bundle.gn_initial_values FAIL weights!=1 (20 arrays) biases==0 (20 arrays)
CHECK bundle.manifest_sha256 PASS n_checked=62 missing=none mismatch=none
CHECK bundle.bundle_hash PASS recomputed rule=sha256(concat C-order bytes, manifest order)
PHASE2_JULIA_DATA_CHECK FAIL bundle.gn_initial_values
PHASE2_JULIA_OPERATOR_CHECK PASS
```

The original bundle remained `a790b03c1d68361a...` before and after. The Python run that
first used the synthetic PREP with the real fixture directory produced unrelated
`data_fixture.*` FAILs because PREP and fixture directory were inconsistent; the recorded
R3 Python evidence uses the matching synthetic fixture directory and fails only on
`bundle.gn_initial_values`. Operators must not mix a PREP with a fixture directory generated
from a different PREP.

## General code-review findings

- **Check counts.** Python positive: 63 CHECK lines (62 PASS, 1 expected local SKIP, 0
  FAIL), all 63 names unique. Julia data check: 47 CHECK lines. Full `check_all.jl`: 81
  CHECK lines (47 + 34 operator). These match the author's stated counts and the harness's
  assertions.
- **No check-name regressions.** Comparing the saved pre-fix Julia logs
  (`/tmp/opencode/phase2-local-checkall.log`, 75 lines; `/tmp/opencode/baseline-data-check.log`,
  41 lines) with the new runs shows no removed names. The six new Julia names are exactly
  the fixture-integrity checks: `fixtures.manifest_present`, `fixtures.file_set`,
  `fixtures.file_sha256_bytes`, `fixtures.array_set`, `fixtures.array_metadata`,
  `fixtures.array_sha256`. The operator check remains 34 lines. On the Python side, all
  previously documented semantic sections remain; the additions are the six `prep.*`
  checks and `schedule.regenerated_all_epochs_bitwise`.
- **Gating behavior.** On a prep-manifest failure the Python checker skips all
  prep-dependent semantics but still verifies the bundle and fixture files, then exits 1.
  On a fixture-manifest failure the Julia checker skips fixture semantics but still checks
  the bundle, then exits 1. Both are conservative and observable in the final FAIL list.
- **Runtime.** Full-50-epoch regeneration: 27.49 s positive, 28.36 s in the negative
  self-consistent probe. The Python positive run also rehashes the 1.2 GB train and 128 MB
  validation arrays. This is acceptable for acceptance, but it is materially slower than
  the pre-fix check.
- **Deviation from the literal R1 wording.** The real schedule manifest carries SHA-256
  but no byte-size fields, so byte size is only enforced when recorded. SHA-256 remains
  sufficient for content identity. Recorded as a minor provenance-record asymmetry.

## Reviewer's exact three false-positive scenarios

All three are now rejected:

1. **R1, later-epoch valid-range change with stale schedule manifest.** Rejected with exit
   1 and `CHECK prep.schedule_files_sha256_bytes FAIL`. If the schedule hashes are made
   self-consistent, `CHECK schedule.regenerated_all_epochs_bitwise FAIL bad_epochs=['49']`
   and `CHECK schedule.epoch_independence FAIL` still reject it.
2. **R2, in-range fixture label change with stale fixture manifest.** Rejected with exit 1
   and `CHECK fixtures.file_sha256_bytes FAIL n_checked=4
   sha_mismatch=data_fixture.npz:sha256` (plus
   `CHECK fixtures.array_sha256 FAIL n_checked=35 mismatch=data_fixture.npz::train_labels`).
3. **R3, self-consistent residual GroupNorm value change.** Rejected by **both** checkers
   with exit 1 and `CHECK bundle.gn_initial_values FAIL weights!=1 (20 arrays)
   biases==0 (20 arrays)`, while `CHECK bundle.manifest_sha256 PASS` and
   `CHECK bundle.bundle_hash PASS`.

## Artifact-integrity statement

The verifier recorded SHA-256 values before any mutation and rechecked them after all
positive and negative probes. All five canonical files are unchanged and match the frozen
records:

| File | SHA-256 (before and after) | Cross-check |
|---|---|---|
| `data_fixture.npz` | `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3` | `fixture_manifest.json` bytes+sha match |
| `schedule_fixture.npz` | `aba66a8ef7bbc82fc81db8cd629092b917cd7d85efafed82553f853c45352662` | fixture manifest match |
| `linear_fixture.npz` | `8efa5278981b2b11842d2a701b6f81789c976e033a678a0322e961ecd8f1fa2d` | fixture manifest match |
| `kernel_fixture.npz` | `3b3795f6c6ff2ec68198a645cc65f4ba97e0b6438da61e6cfd2ad564ff441f3d` | fixture manifest match |
| `initial_bundle.npz` | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` | matches Leonardo retry-5 `$PREP` record |

The retry-5 prepared manifests in
`artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/` also match the
Leonardo retry-5 SHA-256 record: `initial_bundle.manifest.json` `49965174...`,
`dataset_manifest.json` `584b9005...`, `schedule_manifest.json` `0136a0a2...`, and
`stats.json` `6bf4de44...`. (`artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt`
contains an earlier run followed by the retry-5 section; the retry-5 values are the ones
recorded here.)

No production file was edited during this verification. All mutations were applied to
copies under `/tmp/opencode`; the canonical artifacts were read-only. A supplementary
hardlink-copy probe wrote manifests into the prebuilt synthetic scratch tree; that scratch
was repaired from the actual file hashes, and the canonical artifacts were rechecked after
the repair (table above). The repaired synthetic PREP re-passes the full positive Python
check.

## Exact next action

Proceed to **Phase 3, "Implement matched single-GPU training"**, per
`temp/static-graph-gpu-scaling-implementation-phases.md:131` (section 5):

1. Implement the small-image ResNet-18 in Flux and PyTorch with the frozen 62-array
   initialization, GroupNorm placement, and head defined in the plan.
2. Import the same `initial_bundle.npz` into both models; keep the 11,271,432-parameter
   contract.
3. Implement single-process training with mean cross-entropy and Optimisers.jl for Flux;
   match AdamW equations through documented library configuration, including epsilon
   placement, bias correction, and decoupled decay.
4. Disable TF32 explicitly and enable `CUDA.allowscalar(false)` in Flux GPU execution.
5. Compare intermediate activations, logits, loss, gradients, parameters, and AdamW state
   over several updates on the frozen Phase 2 fixtures; a decreasing loss alone is not
   acceptance.

Keep the updated local acceptance suite as the Phase 3 entry gate: `selftest_local.py`,
the full `check_all.jl` command, and `negative_tests_local.py`.

**Remote re-check pending/orchestrator-owned:** the verifier did not run the updated
`check_dataset.py` or `negative_tests_local.py` against the Leonardo `$PREP` tree. Before
Phase 3 acceptance, run the harness on a compute node with the real triple:

```bash
PREP=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \
  scripts/static_scaling/negative_tests_local.py \
  --prep-dir "$PREP" \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle "$PREP/initial_bundle.npz" \
  --bundle-manifest "$PREP/initial_bundle.manifest.json" \
  --workdir "$TMPDIR/phase2-negtest"
```

Expected: `NEGTEST PASS`. This run also exercises the real 6-file dataset manifest and the
real schedule manifest (SHA-256 only, no `bytes` fields) on the canonical PREP.
