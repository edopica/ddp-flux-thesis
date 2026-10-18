# Phase 2 static-scaling artifacts — independent verification report

- Date: 2026-10-16
- Verifier: independent verification agent (adversarial pass, no involvement in producing the artifacts)
- Scope: `artifacts/baselines/static-scaling/2026-10-16-phase2/` (4 fixtures + `fixture_manifest.json` + canonical `initial_bundle.npz`), `artifacts/logs/static-scaling/2026-10-16-phase2/` (`COMMANDS.md`, `remote-prep.log`, `remote-check.log`, `prepared-manifests/`), and the Leonardo driver logs in `artifacts/logs/leonardo/2026-10-16-phase2-*.log`.
- Method: independent recomputation from the raw files (Python/torch 2.14.1+cpu/numpy 2.5.4, Julia 1.12.6 with the repo's `julia-env` stacked on the `flux-integration` worktree), plus deliberately corrupted copies to prove the Julia checks are sensitive. No modification of repo code or artifacts was made; all scratch work lives in `/tmp/opencode/`.

Provenance anchor re-derived before starting (recipe from `COMMANDS.md`), from the repo root:

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Result: `STATE=4d4fd2ac75f192c049d0c7eedfcdf4c1daf83ece770e66ff7bf181d76b651ad1`, `HEAD=3088f61020137912af32d9c969e87ea9ae2a71c3` — both match `COMMANDS.md`.

## Verdict table

| Item | Check | Verdict |
|---|---|---|
| A | Hash audit (COMMANDS.md table, fixture manifest, remote sha list) | PASS |
| B | Bundle audit (traversal, shapes, per-array sha256, bundle_hash) | PASS |
| C | Fixture audit (contract, transforms bitwise, labels, schedule, shards) | PASS (1 prompt-wording discrepancy, see D1) |
| D | Cross-version determinism (recompute bundle + schedule) | PASS |
| E | Generated-manifest consistency (provenance, decoder, stats, fixture manifest) | PASS |
| F | Adversarial negative tests (4 perturbations + positive control) | PASS |
| G | Log audit (PASS lines, job table, retries, transport caveat) | PARTIAL (1 documentation discrepancy, see D2) |

Overall: **PASS for the artifacts** — every byte-level claim about the fixture set, the bundle, and the manifests was independently reproduced. Two discrepancies were found, neither is an artifact defect: one is an incorrect literal rule in the verification instruction (D1), one is an overstatement in `COMMANDS.md` note 5 (D2).

## A — Hash audit: PASS

Command: `/tmp/opencode/torchvenv/bin/python /tmp/opencode/verify-phase2/a_hash_audit.py`

- All six file hashes in the `COMMANDS.md` table reproduce exactly:

| File | Bytes | SHA-256 |
|---|---|---|
| `data_fixture.npz` | 1821284 | `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3` |
| `schedule_fixture.npz` | 3797486 | `aba66a8ef7bbc82fc81db8cd629092b917cd7d85efafed82553f853c45352662` |
| `linear_fixture.npz` | 2238 | `8efa5278981b2b11842d2a701b6f81789c976e033a678a0322e961ecd8f1fa2d` |
| `kernel_fixture.npz` | 7114 | `3b3795f6c6ff2ec68198a645cc65f4ba97e0b6438da61e6cfd2ad564ff441f3d` |
| `initial_bundle.npz` | 45103060 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| `fixture_manifest.json` | 10643 | `51448d5905c5dfd9a47b945d96102d19ea55073fbfe9f249fd268cca03455a65` |

- The four `files[]` entries embedded in `fixture_manifest.json` match the actual npz bytes.
- The brought-home `initial_bundle.npz` hash equals the retry-5 entry in `artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt`.
- The local `prepared-manifests/` copies match the remote prepared hashes: `initial_bundle.manifest.json` `4996517497edfd66…`, `dataset_manifest.json` `584b900565ebfd02…`, `schedule_manifest.json` `0136a0a2d1d768bf…`, `stats.json` `6bf4de44cb7fd035…`.

## B — Bundle audit: PASS

Command: `/tmp/opencode/torchvenv/bin/python /tmp/opencode/verify-phase2/b_bundle_audit.py`

- Independent Functors/torch-free traversal of the architecture in `specification.md` §4 reproduces the manifest array order exactly (62 arrays, conv/GN/classifier decomposition 11159232 + 9600 + 102600 = 11271432 = manifest total).
- Key set, shapes, float32 dtype, and per-array SHA-256 over raw C-order bytes: 0 mismatches across all 62 arrays.
- Recomputed bundle hash over concatenated manifest-order bytes: `a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23` — equals the recorded value and the remote log line.
- GN init checks: all 20 weights==1, all 20 biases==0 (two GN layers per stage × 4 stages × weight+bias); classifier shapes `(200, 512)` / `(200,)`.

## C — Fixture audit: PASS (with discrepancy D1)

Command: `/tmp/opencode/torchvenv/bin/python /tmp/opencode/verify-phase2/c_fixture_audit.py`

- Contract arrays (names, shapes, dtypes) for all four fixtures: PASS.
- Normalization constants bitwise equal `stats.json` float32 arrays; transform replay (pad4 → crop64 → horizontal flip → float32/255 → normalize) reimplemented independently is bitwise equal to `train_transformed_ref` epochs 0 and 1 and to `val_transformed_ref` (using the 12/8 sampled ids).
- Labels: train ids in 0..199 with expected values; Julia +1 conversion consistent (checked by the Julia data check locally, see F positive control).
- `epoch0_permutation` is a true permutation of 0..99999; tail 160 ids dropped, unique; R2/R4 stored rank ids tile the 99840 epoch-0 sequence exactly.
- Augmentation arrays: shape `(50, 99840, 3)` uint8, top/left in 0..8, flip in 0..1; fixture rows are a subset of the schedule rows.
- Rank sharding: the stored `rank_ids_R1/R2/R4` are reproduced **bitwise** by the documented per-global-batch split (each 512-batch sharded into contiguous rank chunks, then concatenated), matching `dataset_adapter`/`specification.md` §2. See D1 for the verification-instruction rule that does *not* match.

## D — Cross-version determinism: PASS

Commands:

```bash
/tmp/opencode/torchvenv/bin/python scripts/static_scaling/make_initial_bundle.py --out-dir /tmp/opencode/verify-bundle --force
/tmp/opencode/torchvenv/bin/python scripts/static_scaling/make_schedule.py --out-dir /tmp/opencode/verify-sched --force
/tmp/opencode/torchvenv/bin/python /tmp/opencode/verify-phase2/d_determinism_cmp.py
```

- Regenerated bundle (local torch 2.14.1+cpu / numpy 2.5.4) vs remote-prepared manifest (torch 2.9.1+cu126 / numpy 2.4.6): identical `bundle_hash`, 0 per-array sha mismatches. The regenerated `initial_bundle.npz` is **byte-identical** to the shipped file (`a790b03c…`), i.e. all 62 arrays bitwise equal.
- Regenerated schedule: `permutations.npy` `7f687b61877…`, `augmentation.npy` `9f6bbea465c…` — both equal to the remote `schedule_manifest.json` entries; regenerated `permutations[0]` and `augmentation[0:2]` bitwise equal the stored fixtures.

## E — Generated-manifest consistency: PASS

Command: `/tmp/opencode/torchvenv/bin/python /tmp/opencode/verify-phase2/e_manifest_consistency.py`

- All four generated manifests carry `source_commit=3088f610…`, `source_state_sha256=4d4fd2ac…`, `synthetic=false`, and the expected `producer` script.
- `dataset_manifest.json`: train `(100000, 3, 64, 64)` / val `(10000, 3, 64, 64)`, uint8; JPEG containers `{JPEG: 100000}` / `{JPEG: 10000}`; decoder identity `PIL.Image.open(path).convert('RGB')`, Pillow 12.3.0; `.npy` byte arithmetic exact.
- `schedule_manifest.json`: invariants `samples_per_epoch=99840`, `tail_size=160`, `updates_per_epoch_b512=195`, `updates_per_epoch_b256=390`; root seed 20261016; 50 epochs.
- `stats.json`: mean/std float32 bitwise equal the retry-5 `DATASET_PREP` log lines and the `data_fixture` normalization arrays; the `decimal_*_f64` values round to the stored float32 constants.
- `fixture_manifest.json`: all 35 array entries verified by recomputing sha256 from the npz bytes (9+5+7+14), file entries verified; bundle hash in the manifest equals the retry-5 log line.

## F — Adversarial negative tests: PASS

Positive control (exact artifact set, exit 0, none skipped):

```bash
JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" \
  julia --startup-file=no scripts/static_scaling/check_all.jl \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --bundle-manifest artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/initial_bundle.manifest.json
# EXIT=0; 75 `CHECK` lines; PHASE2_JULIA_DATA_CHECK PASS + PHASE2_JULIA_OPERATOR_CHECK PASS
```

Perturbed copies were made under `/tmp/opencode/adv-{1,2,3}` (real artifacts untouched) by `/tmp/opencode/verify-phase2/f_make_negatives.py`:

| Test | Perturbation | Exit | First failing check (verbatim) |
|---|---|---|---|
| adv-1 | `linear_fixture.y[0,0] += 1e-3` | 1 | `CHECK linear.forward FAIL max_abs_diff=0.000999987125` |
| adv-2 | swap `rank_ids_R4[2,100]` ↔ `[2,101]` | 1 | `CHECK schedule.rank_sequence_R4 FAIL shape=(4, 24960) first_bad_rank=3` |
| adv-3 | flip LSB byte of `train_transformed_ref[0,0,0,0,0]` (Δ=1.86e-9) | 1 | `CHECK transform.train_epoch0 FAIL max_abs_diff=1.86264515e-09` |
| adv-4 | corrupt `bundle_hash` in a manifest copy | 1 | `CHECK bundle.bundle_hash FAIL recompute failed … tried=8 rules` |

All four targeted failures were caught (single-ULP float detection included); the positive control confirms no false positives.

## G — Log audit: PARTIAL (with discrepancy D2)

- `remote-check.log` contains exactly three PASS lines: `PHASE2_DATASET_CHECK PASS` (line 70), `PHASE2_JULIA_DATA_CHECK PASS` (line 122), `PHASE2_REMOTE_CHECK PASS` (line 129); `python_check_exit=0 julia_check_exit=0`; the driver log records `=== run-phase2-check: EXIT 0 ===`.
- `sacct` evidence matches `COMMANDS.md` exactly: job 60758615 COMPLETED 0:0, 23:33:16→23:35:11, elapsed 00:01:55 (step `.0` 00:01:50); job 60758839 COMPLETED 0:0, 23:36:39→23:38:54, elapsed 00:02:15 (step `.0` 00:02:09).
- Retry-5 scoping confirmed: `remote-prep.log` retry-5 section starts at line 84 with `SOURCE_STATE_SHA256=4d4fd2ac…`; the only `PHASE2_PREP FAIL` (line 79) is inside the retry-4 section; retry-5 ends with `FIXTURES DONE synthetic=False` and `=== run-phase2-prep: DONE ===`; step times match `COMMANDS.md` (1m18.382s / 22.600s / 2.432s / 4.544s).
- Transport-drop caveat confirmed: driver log shows `Read from remote host … Connection reset by peer` / `client_loop: send disconnect: Broken pipe` / `PREP_EXIT=255`, followed by polling that observed the job at `COMPLETED 0:0`; the in-log `DONE` marker plus `sacct` support success (the `EXIT` line is outside the remote log by design).
- `state-v4.log` records retry-5 start `2026-10-16T23:32:15+02:00 COMMIT=3088f610… STATE=4d4fd2ac…`.
- CHECK-line count: the remote log has **100** `CHECK` lines (59 `check_dataset.py` + 41 `check_fixtures_data.jl`). The **75**-line count applies to the local `check_all.jl` reproduction (data + operator checks), which the verifier reproduced locally (see F); it does **not** describe `remote-check.log`.

## Discrepancies (verbatim)

**D1 — verification instruction vs documented sharding contract (prompt wording error; artifacts correct).**
The verification instruction for item C stated the rank split rule as: *“rank r takes rows [r*len//w:(r+1)*len//w] of the 99840-epoch0 sequence”*. Applied literally (whole-sequence contiguous split), this fails for R2 (first mismatch at `[0,256]`) and R4 (first mismatch at `[0,128]`) and is not reproduced by the stored arrays. The documented contract (`experiment/…/specification.md` §2, `dataset_adapter` docstring) shards each 512-batch into contiguous rank chunks before concatenation; that rule reproduces `rank_ids_R1/R2/R4` bitwise. No artifact action required; the report records the literal-rule failure for transparency.

**D2 — `COMMANDS.md` note 5 overstates remote verification coverage.**
`COMMANDS.md` lines 223–226 (verbatim):

> local Julia `check_all.jl` passed `PHASE2_JULIA_DATA_CHECK PASS` + `PHASE2_JULIA_OPERATOR_CHECK PASS` (75 checks, exit 0) and `check_dataset.py` passed `PHASE2_DATASET_CHECK PASS`. The remote retry-5 reproduced every marker.

Fact: `remote-check.log` contains **no** operator-check section (no `PHASE2_JULIA_OPERATOR_CHECK` line anywhere in `artifacts/logs/`); `run-phase2-check.sh` runs only `check_dataset.py` and `check_fixtures_data.jl`. The remote reproduced the dataset/data markers (`PHASE2_DATASET_CHECK PASS`, `PHASE2_JULIA_DATA_CHECK PASS`, `PHASE2_REMOTE_CHECK PASS`), and the operator/fixture math was verified locally by this verification (positive control F) — but not on Leonardo. Additionally the attached “75 checks” belongs to the local `check_all.jl` run; the remote log has 100 `CHECK` lines. Suggest correcting note 5 to distinguish local vs remote coverage; no artifact action required.

## What was NOT verified

- `prepare_dataset.py` was not re-run from the raw Tiny ImageNet-200 source (requires the remote dataset tree and ~2h); dataset-level recomputation (`check_dataset.py`, incl. full mean/std recompute and histogram checks) was taken from the recorded remote run only.
- The full 100000×3×64×64 pixel content was not re-decoded and re-hashed: the decoder/transform path was verified bitwise on the 12 train + 8 val sampled ids plus the 256 distinct-byte probe; JPEG container counts were verified only via the Python checker output (59 dataset markers), not by re-counting files on Leonardo.
- No GPU/NCCL/training-side behavior is in scope here (Phase 2 covers data preparation and fixtures only).
- The remote execution environment itself (node identity, user, venv contents) was not independently inspected; trust rests on the hash chain + `sacct` records.

## Overall verdict

**PASS** — the Phase 2 artifacts (4 fixtures, `fixture_manifest.json`, canonical `initial_bundle.npz`, 4 prepared manifests) are internally consistent, byte-reproducible from the repo at `HEAD=3088f61` / `STATE=4d4fd2ac`, and cross-version deterministic; the Julia checks catch all four adversarial perturbations and pass clean on the real artifacts (75/75, exit 0). Two non-blocking discrepancies are documented above: D1 is an error in the verification instruction’s sharding rule (artifacts follow the documented per-batch contract, verified bitwise), D2 is an overstatement in `COMMANDS.md` note 5 about remote operator-check coverage that should be corrected. No changes to artifacts are required.
