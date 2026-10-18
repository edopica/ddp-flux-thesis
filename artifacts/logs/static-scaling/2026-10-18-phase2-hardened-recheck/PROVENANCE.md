# Provenance — hardened Phase 2 real-PREP re-check

Date: 2026-10-18 (UTC run start 2026-10-18T00:44:27Z).
Status: **PASS**. Read-only check of the real Leonardo `$PREP` and fixture
directory with the hardened Phase 2 checkers (R1-R3 fixes). No artifact,
checker, model, optimizer, or tolerance was modified.

This run closes the Phase 2 remote prerequisite of the Phase 3 handoff.
The 2026-10-16 remote PASS predates the hardening; this is the first
real-`$PREP` run of `check_dataset.py` with manifest gates and all-epoch
regeneration, and of `check_fixtures_data.jl` with fixture-manifest gates.

## 1. Source identity

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Workspace state | uncommitted project work as of 2026-10-18; no commit made |
| Remote thesis tree | `~/projects/ddp-flux-thesis` (latest synced sources) |

Checker hashes, local and remote (identical):

| File | SHA-256 |
|---|---|
| `scripts/static_scaling/check_dataset.py` | `ee3f10b248ff9a6075f5e828e030096bf5b6f46b5d30a608dcfd5e16b464f038` |
| `scripts/static_scaling/check_fixtures_data.jl` | `8135a2417c271b811bc39b2f32a8f800d14475764b60348351637629af07abda` |
| `scripts/static_scaling/check_all.jl` | `62f3c64d7a2bb9ca28133545ec5303140458660ca3784823be56238aa82febeb` |
| `scripts/remote/leonardo/run-phase2-check.sh` | `2fe3690912a029053012714c59452966932e4aadd6c970d53a3b307141f633d7` |

## 2. Inputs (read-only; unchanged)

Prepared dataset `$PREP`:
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared`

| File | Bytes | SHA-256 |
|---|---:|---|
| `initial_bundle.npz` | 45,103,060 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| `initial_bundle.manifest.json` | 16,437 | `4996517497edfd66cf259bb7d51d2ca8f0d44076bc958438af33df5920b0c571` |
| `dataset_manifest.json` | 4,288 | `584b900565ebfd02d6d323db393425b6263170f55ebf6a928a49fb5a956b875a` |
| `schedule_manifest.json` | 1,773 | `0136a0a2d1d768bf2c2a8544bd10f7893b533911bad37d77b09a1ae277121295` |
| `stats.json` | 1,238 | `6bf4de44cb7fd035812917e2c34a8b25cdbc503be9b947885bd33e0b2fc324bc` |
| `train_images.npy` | 1,228,800,128 | `ba43b39f8384d5b6230f676f3f649640a302a952b848a193c034828e6bd35510` |
| `train_labels.npy` | 800,128 | `fd297be84d7737ab295f758c1e5829a00e5015a983c7f2de51e2baafa6dc6e79` |
| `val_images.npy` | 122,880,128 | `5e25fedd066f1e0bf9b3ed767438f6155414f1594ea2557b98c6278b290a2b46` |
| `val_labels.npy` | 80,128 | `631d5760a11d79c22b5b4de5767b738c81bcc7b6cf1346b0bea1f89b2357b487` |
| `class_wnids.txt` | 2,000 | `a36eb4b603007f1985cdfbfc5987166a2710b3c6f0f6d3e29636fc129bba527b` |
| `permutations.npy` | 40,000,128 | `7f687b61877483a9154bd6abf871533751e1e06aba993a218474f234b8c85bfd` |
| `augmentation.npy` | 15,000,128 | `9f6bbea465cf24938ef326c92b7516209123520cad3cb50a65928b4f9c4d528b` |

All files are dated 2026-10-16 23:33-23:35 (+0200); no file changed during
this run.

Fixtures `$FIX`:
`~/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2`

| File | Bytes | SHA-256 |
|---|---:|---|
| `data_fixture.npz` | 1,821,284 | `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3` |
| `schedule_fixture.npz` | 3,797,486 | `aba66a8ef7bbc82fc81db8cd629092b917cd7d85efafed82553f853c45352662` |
| `linear_fixture.npz` | 2,238 | `8efa5278981b2b11842d2a701b6f81789c976e033a678a0322e961ecd8f1fa2d` |
| `kernel_fixture.npz` | 7,114 | `3b3795f6c6ff2ec68198a645cc65f4ba97e0b6438da61e6cfd2ad564ff441f3d` |
| `fixture_manifest.json` | 10,643 | `51448d5905c5dfd9a47b945d96102d19ea55073fbfe9f249fd268cca03455a65` |
| `initial_bundle.npz` (copy) | 45,103,060 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |

`fixture_manifest.json` lists exactly the four fixture `.npz` files and 35
arrays; all hashes match.

## 3. Remote environment and run identity

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| Allocation | 1 node, 1 task, 8 CPUs, `--gres=none` (CPU-only), 30 min cap |
| Slurm job / node | `60951247` / `lrdn0171` |
| Job state / exit | `COMPLETED` / `0:0` |
| Elapsed | `00:00:56` (allocation), `00:00:50` (task) |
| Start / end | `2026-10-18T02:44:21` / `2026-10-18T02:45:17` (+0200) |
| Run UTC start | `2026-10-18T00:44:27Z` |
| Python / numpy / pillow / torch | 3.11.17 / 2.4.6 / 12.3.0 / 2.9.1+cu126 |
| Julia | 1.12.6 |
| Python venv | `/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling` |
| Julia depot | `/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot` |

Remote evidence root (outside the rsync-managed thesis tree):
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase2-hardened-recheck/logs/`

## 4. Results

```text
CHECK prep.dataset_file_set PASS required=6 listed=6 missing=none extra=none
CHECK prep.dataset_files_sha256_bytes PASS checked=6 mismatches=none
CHECK prep.schedule_files_sha256_bytes PASS checked=2 mismatches=none
CHECK schedule.regenerated_all_epochs_bitwise PASS bad_epochs=none 50 epochs in 21.21s
CHECK bundle.gn_initial_values PASS weights==1 (20 arrays) biases==0 (20 arrays)
CHECK fixtures.file_sha256_bytes PASS n_checked=4 sha_mismatch=none bytes_mismatch=none
CHECK fixtures.array_sha256 PASS n_checked=35 mismatch=none
PHASE2_DATASET_CHECK PASS
PHASE2_JULIA_DATA_CHECK PASS
python_check_exit=0 julia_check_exit=0
PHASE2_REMOTE_CHECK PASS
```

- Python `check_dataset.py`: 66 CHECK lines, all PASS; exit 0.
- Julia `check_fixtures_data.jl`: 47 CHECK lines, all PASS; exit 0.
- Total: 113 `CHECK ... PASS`, 0 FAIL, 0 SKIP.
- Driver exit 0.

## 5. Evidence hashes

| File | SHA-256 |
|---|---|
| `remote-check.log` | `42b33dc3d6753c08f1e55064b1fc864668f6f67782a0da713a43c3e70de62bf5` |
| `local-console.log` | `3c9867cdae5ae784bc54fad8ffde227ab78176353201214f705111c498e259b7` |

`local-console.log` is the operator-side console capture (includes the
`salloc` allocation lines); `remote-check.log` is the compute-node script
log. Both contain the same CHECK output and PASS lines.

## 6. Scope

This is a single-node CPU check of prepared data, fixtures, and optimizer-free
artifacts. It does not run training, collectives, GPU kernels, or the Flux
operator check. Rank-0/output rules do not apply.
