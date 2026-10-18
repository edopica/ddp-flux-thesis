# Phase 2 complete: static-scaling dataset, schedule, bundle, and fixtures verified

Date: 2026-10-16
Scope: static-network GPU scaling (scope 3), Phase 2 data pipeline on Leonardo
Status: PASS — prepared dataset and fixture set generated, remote acceptance
PASS, local operator check PASS, independent verification PASS
Contract: [specification.md](../../../experiments/static-scaling/specification.md)
sections 2, 3, 4, and 8; [implementation plan](../../../temp/static-graph-gpu-scaling-implementation-phases.md)
section 4
Evidence package: [`COMMANDS.md`](../../../artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md)
(authoritative command, job, and hash record)

## Objective

Implement Phase 2 of the static-network GPU scaling plan: one canonical data
path that both frameworks reproduce. The phase had to produce the decoded
dataset, the 50-epoch sample schedule, the saved augmentation decisions, the
normalization statistics, the canonical initial parameter bundle, and the
shared `.npz`-plus-manifest correctness fixtures, including the review's
rectangular-linear and asymmetric-kernel cases. Both frameworks must consume
identical inputs under the conversion rules of specification section 8.

## Deliverables

All deliverables are generation-time items under the source state recorded
below; the Python scripts write every manifest.

| Path | Purpose |
|---|---|
| `scripts/static_scaling/dataset_adapter.py` | Shared schedule/sharding/transform contract (mirrors `replay.jl`): global batches before rank sharding, per-512-batch contiguous rank chunks, saved transform replay |
| `scripts/static_scaling/prepare_dataset.py` | Decode Tiny ImageNet-200 once in canonical order; write `train/val_images.npy`, `train/val_labels.npy`, `class_wnids.txt`, `stats.json`, `dataset_manifest.json` |
| `scripts/static_scaling/make_schedule.py` | 50 epochs of `permutations.npy` and `augmentation.npy` (`(crop_top, crop_left, flip)` per epoch and sample); `schedule_manifest.json` |
| `scripts/static_scaling/make_initial_bundle.py` | Canonical 62-array float32 initial bundle, per-array SHA-256, and `bundle_hash`; `initial_bundle.manifest.json` |
| `scripts/static_scaling/make_fixtures.py` | Four correctness fixtures and `fixture_manifest.json` |
| `scripts/static_scaling/check_dataset.py` | Python acceptance: dataset, stats, schedule, sharding, bundle, and fixture checks (remote check step 1/2) |
| `scripts/static_scaling/selftest_local.py` | NumPy-only local self-test of the frozen adapter contract (no torch, PIL, or prepared data needed) |
| `scripts/static_scaling/replay.jl` | Julia NPZ reading, C-order array hashing, schedule/shard/transform rebuild |
| `scripts/static_scaling/check_fixtures_data.jl` | Julia data check against the fixtures (remote check step 2/2) |
| `scripts/static_scaling/check_fixtures_operators.jl` | Flux operator check on the fixtures: rectangular `Dense(7 => 3)` forward and Zygote gradients, asymmetric kernel cases A/B with an orientation canary and reversed-kernel/`CrossCor` match |
| `scripts/static_scaling/check_all.jl` | Local data + operator acceptance entry point (75 checks) |
| `scripts/static_scaling/julia-env/` | NPZ-only Julia project (`Project.toml`, `Manifest.toml`); Flux is stacked through `JULIA_LOAD_PATH` only for the operator check |
| `scripts/remote/leonardo/run-phase2-prep.sh` | Remote prep driver, four steps, fail-fast |
| `scripts/remote/leonardo/run-phase2-check.sh` | Remote acceptance driver, NPZ-only |

## Provenance

- Thesis repository HEAD:
  `3088f61020137912af32d9c969e87ea9ae2a71c3` ("Record implementation-phases
  devlog entry", 2026-10-16).
- Generation-time source state:
  `4d4fd2ac75f192c049d0c7eedfcdf4c1daf83ece770e66ff7bf181d76b651ad1`.
  Recipe (repository root):

  ```bash
  STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
    | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
  ```

  Every generated manifest records `source_commit` and
  `source_state_sha256`. Later comment-only edits (for example
  `scripts/remote/hosts/leonardo.conf`) move the current tree hash, so the
  recorded state is generation-time provenance, not a live tree hash.
- `bash scripts/remote/sync_code.sh leonardo` synced the thesis repo and
  `flux-integration` at this HEAD and state before the successful run
  (`Sync complete!` in `2026-10-16-phase2-sync.log`).

## Attempt history

Five attempts were needed. Each failure was separated and is recorded in its
own failure note.

| Attempt | Job | Outcome | Root cause and fix |
|---|---|---|---|
| 1 | 60729163 (`lrdn2229`, dbg) | FAILED 13:0 in ~2 s | `srun ... env VAR=...` resolved the `env` token to the non-executable uv activation script on `PATH`; `execve` failed with `Permission denied` (exit 13). Fixed by `bash -c 'VAR=... cmd'`; the uv layout was not modified |
| 2 | 60729410 (`lrdn3102`, dbg) | TIMEOUT/CANCELLED | Killed at the 30-minute `boost_qos_dbg` cap mid-decode at 80000/100000. A `boost_qos_bprod` retry was rejected at submit (`QOSMinCpuNotSatisfied`, MinTRES node=65). Fixed with `QOS=boost_qos_lprod TIME=02:00:00` |
| 3 | 60731702 (`lrdn3432`, lprod) | prep FAILED but exit 0 (masked) | `prepare_dataset.py` raised `NameError: name 'std64' is not defined` after ~36m54s of decode; the driver still printed DONE. Fixed `std64` computation and the driver fail-fast contract |
| 4 | 60755769 (`lrdn0018`, lprod) | FAILED 1:0, 00:25:34 | Steps 1-3 PASS; step 4/4 `make_fixtures` raised `RuntimeError: The size of tensor a (5) must match the size of tensor b (3)` at `make_fixtures.py:212` (`y = w @ x + b`, bias `(3,)` against `(3, 5)`). Fixed with `b[:, None]` in the torch and numpy builders |
| 5 | 60758615 + 60758839 | COMPLETED 0:0 | Prep 00:01:55 (steps 1m18.382s / 22.600s / 2.432s / 4.544s); check 00:02:15 (steps 11.432s / 45.926s). PASS |

Failure records:
[env-token and QoS](../../failures/2026-10-16-phase2-prep-leonardo-env-and-qos.md),
[`std64` NameError and masked exit](../../failures/2026-10-16-phase2-prep-std64-nameerror-masked-exit.md),
[linear fixture broadcast](../../failures/2026-10-16-phase2-prep-make-fixtures-linear-broadcast.md).

## Retry-5 evidence

Prep job 60758615 (`lrdn0018`, `boost_qos_lprod`, COMPLETED 0:0, 00:01:55)
and check job 60758839 (`lrdn0011`, default `boost_qos_dbg`, COMPLETED 0:0,
00:02:15). Verbatim markers from
[`remote-prep.log`](../../../artifacts/logs/static-scaling/2026-10-16-phase2/remote-prep.log)
and
[`remote-check.log`](../../../artifacts/logs/static-scaling/2026-10-16-phase2/remote-check.log):

```text
DATASET_PREP train_mean=[0.48023695 0.44806704 0.39750364]
DATASET_PREP train_std=[0.27643642 0.2688633  0.28158993]
DATASET_PREP containers_train=[('JPEG', 100000)] containers_val=[('JPEG', 10000)]
DATASET_PREP DONE
SCHEDULE DONE
BUNDLE elements=11271432 bundle_hash=a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23
BUNDLE verified name_set=true per_array_sha256=true bundle_hash=true torch=2.9.1+cu126 numpy=2.4.6
FIXTURES verified files=4 arrays=35 sha256=true
FIXTURES DONE synthetic=False
=== run-phase2-prep: DONE ===
PHASE2_DATASET_CHECK PASS
PHASE2_JULIA_DATA_CHECK PASS
python_check_exit=0 julia_check_exit=0
PHASE2_REMOTE_CHECK PASS
```

The remote check job is NPZ-only (`check_dataset.py` and
`check_fixtures_data.jl`); it does not run the Flux operator check. The SSH
stdio stream dropped mid-decode (`Connection reset by peer`, exit 255); the
Slurm job kept running and completed, so this is transport-only. The exit
status is evidenced by `sacct` (COMPLETED 0:0) and the in-log
`=== run-phase2-prep: DONE ===` marker.

Local acceptance used the copied-back real fixtures:

```bash
JULIA_LOAD_PATH="$PWD/scripts/static_scaling/julia-env:/home/kurapica/Projects/ddp_flux/flux-integration:@stdlib" \
  julia --startup-file=no scripts/static_scaling/check_all.jl \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --bundle-manifest artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/initial_bundle.manifest.json
```

Result: `PHASE2_JULIA_DATA_CHECK PASS` + `PHASE2_JULIA_OPERATOR_CHECK PASS`,
75 `CHECK` lines, exit 0.

## Artifacts and hashes

Local fixtures,
[`artifacts/baselines/static-scaling/2026-10-16-phase2/`](../../../artifacts/baselines/static-scaling/2026-10-16-phase2/):

| File | Bytes | SHA-256 |
|---|---|---|
| `data_fixture.npz` | 1,821,284 | `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3` |
| `schedule_fixture.npz` | 3,797,486 | `aba66a8ef7bbc82fc81db8cd629092b917cd7d85efafed82553f853c45352662` |
| `linear_fixture.npz` | 2,238 | `8efa5278981b2b11842d2a701b6f81789c976e033a678a0322e961ecd8f1fa2d` |
| `kernel_fixture.npz` | 7,114 | `3b3795f6c6ff2ec68198a645cc65f4ba97e0b6438da61e6cfd2ad564ff441f3d` |
| `initial_bundle.npz` | 45,103,060 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| `fixture_manifest.json` | 10,643 | `51448d5905c5dfd9a47b945d96102d19ea55073fbfe9f249fd268cca03455a65` |

Prepared dataset at
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/`
(train/val `.npy`, `class_wnids.txt`, `stats.json`, `dataset_manifest.json`,
`permutations.npy`, `augmentation.npy`, `schedule_manifest.json`,
`initial_bundle.npz` plus manifest; every file has a recorded SHA-256; see
[`prepared-sha256.txt`](../../../artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt)).
Key values from the retry-5 block: `permutations.npy` `7f687b61…`,
`augmentation.npy` `9f6bbea4…`, `stats.json` `6bf4de44…`,
`dataset_manifest.json` `584b9005…`, `schedule_manifest.json` `0136a0a2…`,
`initial_bundle.manifest.json` `49965174…`.

The canonical bundle is 62 arrays and 11,271,432 float32 elements; the
`bundle_hash` is the SHA-256 over the concatenated per-array raw C-order bytes
in manifest order. It was reproduced identically by local torch 2.14.1+cpu /
numpy 2.5.4 and remote torch 2.9.1+cu126 / numpy 2.4.6, and the regenerated
`initial_bundle.npz` is byte-identical (cross-version determinism). The
regenerated schedule also matches the stored `permutations` and `augmentation`
hashes.

## Verification

- Remote acceptance: jobs 60758615 and 60758839, all three PASS markers above.
- Local acceptance on the real copied fixtures: `check_all.jl` 75 checks,
  exit 0.
- Independent verification:
  [static-scaling-phase2-verification.md](../../../reports/static-scaling-phase2-verification.md),
  overall **PASS**. It re-derived the hash chain, the bundle traversal and
  per-array hashes, the fixture contract and transforms bitwise, the
  cross-version determinism, and the manifest consistency. Four adversarial
  perturbations (linear output, rank-split swap, transform LSB flip,
  `bundle_hash` corruption) were all caught; the clean positive control passed
  with no false positives. Two disclosed discrepancies are non-blocking:
  D1 is an error in the verification prompt's rank-rule wording; the stored
  `rank_ids_R1/R2/R4` match the documented per-512-batch sharding bitwise.
  D2 was a stale note in `COMMANDS.md` note 5, now corrected: the remote check
  job is NPZ-only and does not run the Flux operator check, which ran locally.

## Findings and corrections

- **Container format.** The decoder records container types dynamically and
  this copy reports JPEG for both splits:
  `containers_train=[('JPEG', 100000)]`,
  `containers_val=[('JPEG', 10000)]`, in `dataset_manifest.json` and the
  retry-5 log. Treat that manifest entry as authoritative; no extension-based
  assumption (for example, that a `.JPEG` file is PNG-encoded) is needed.
- **Decode time is filesystem-bound.** `prepare_dataset` took 23m52s
  (retry-4) and 1m18s (retry-5) on the same node (`lrdn0018`) with identical
  inputs; user time was ~20-28s. The variation is `$WORK`/Lustre latency, not
  CPU. Long 1-node prep jobs need the `lprod` QoS.
- **Never-tested step reached late.** Steps 1-3 had passed several times
  before `make_fixtures` first ran past its first fixture; the shape bug then
  surfaced only after a ~25-minute job. The local `selftest_local.py` path
  covers the adapter contract but not the fixture builders.

## Lessons

1. `srun` resolves the `env` exec token through `PATH`; the uv activation
   script must never shadow it. Use `bash -c 'VAR=... cmd'` and keep the uv
   layout untouched.
2. A batch driver must propagate the first failing step. `set -e` plus an ERR
   trap turned a masked exit 0 into `PHASE2_PREP FAIL` and exit 1.
3. Check `MinTRES` before selecting a QoS: `boost_qos_bprod` cannot run `-N1`
   jobs; `boost_qos_lprod` is the long single-node QoS.
4. Pin generation-time provenance (`source_commit` plus dirty-tree state) and
   hash every artifact; this made cross-version determinism and independent
   verification possible.
5. A dropped SSH stream is transport-only evidence until `sacct` says
   otherwise: poll the job before rerunning.
6. Audit each fixture builder before the expensive remote run; a two-line
   broadcasting defect otherwise costs a full decode cycle.

## Next actions

1. Phase 3 (training-side consumption / single-GPU baseline, plan section 5):
   sync `flux-integration` `f7b4520c` to Leonardo, resolve on the login node,
   precompile on a compute node under `boost_qos_lprod`, then run the remote
   operator check against the frozen fixtures.
2. Keep the pinned `COMMIT` / `STATE` from this entry as the Phase 2 artifact
   provenance.
