# Full-dataset loader validation — job 61450014

**Status: PASS (loader validation only).** This is not a benchmark result and
does not establish throughput, scaling, numerical equivalence, training quality,
or the OD-8 memory-margin decision. The measured updates are validation
metadata only.

## Job

| Item | Value |
|---|---|
| Job | `61450014` (`static-full-loader-validate`) |
| Node | `lrdn0667.leonardo.local` |
| Allocation | `boost_qos_dbg`, 1 node, 1 task, 32 CPUs, 4×A100-SXM-64GB, `00:30:00` |
| Slurm state | `COMPLETED`, exit `0:0`, elapsed `00:14:09` |
| Start/end | `2026-10-19T17:43:59` / `2026-10-19T17:58:08` (Slurm local) |
| Driver start/end (UTC) | `2026-10-19T15:44:15Z` / `2026-10-19T15:58:08Z` |
| Precompile | `FLUX_PRECOMPILE_OK` for current and historical Flux |
| Driver marker | `FULL_VALIDATION PASS` |
| Slurm log | `artifacts/logs/leonardo/2026-10-19-static-full-loader-validation-61450014.log` |

Superseded: job `61447030` completed the same checks but recorded
`host_maxrss_bytes` with a 1024× unit error (Julia 1.12 `Sys.maxrss()` already
returns bytes). Job `61446924` failed before any workload because of a
`set -u` variable-expansion bug in the driver. See
`wiki/failures/2026-10-19-full-loader-validation-driver.md`.

## Dataset identity (read-only prepared dataset)

`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared`

| File | Bytes | SHA-256 |
|---|---:|---|
| `train_images.npy` | 1,228,800,128 | `ba43b39f8384d5b6230f676f3f649640a302a952b848a193c034828e6bd35510` |
| `train_labels.npy` | 800,128 | `fd297be84d7737ab295f758c1e5829a00e5015a983c7f2de51e2baafa6dc6e79` |
| `permutations.npy` | 40,000,128 | `7f687b61877483a9154bd6abf871533751e1e06aba993a218474f234b8c85bfd` |
| `augmentation.npy` | 15,000,128 | `9f6bbea465cf24938ef326c92b7516209123520cad3cb50a65928b4f9c4d528b` |
| `stats.json` | 1,238 | `6bf4de44cb7fd035812917e2c34a8b25cdbc503be9b947885bd33e0b2fc324bc` |
| `initial_bundle.npz` | 45,103,060 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| `initial_bundle.manifest.json` | 16,437 | `4996517497edfd66cf259bb7d51d2ca8f0d44076bc958438af33df5920b0c571` |
| `dataset_manifest.json` | 4,288 | `584b900565ebfd02d6d323db393425b6263170f55ebf6a928a49fb5a956b875a` |
| `schedule_manifest.json` | 1,773 | `0136a0a2d1d768bf2c2a8544bd10f7893b533911bad37d77b09a1ae277121295` |
| `class_wnids.txt` | 2,000 | `a36eb4b603007f1985cdfbfc5987166a2710b3c6f0f6d3e29636fc129bba527b` |
| `val_images.npy` | 122,880,128 | `5e25fedd066f1e0bf9b3ed767438f6155414f1594ea2557b98c6278b290a2b46` |
| `val_labels.npy` | 80,128 | `631d5760a11d79c22b5b4de5767b738c81bcc7b6cf1346b0bea1f89b2357b487` |

Frozen order hashes (epoch 0, batch 512):

- sample order: `f773fcec7d5b5bd9f7b0f98c3fdf0364be5233ffb15cfa05f97ad06c3d865335`
- update-1 global order: `23c183e8e1b9b777a70013926c30590851cda4ae2c7fbb18ec8a0aa21515a61a`
- transform spec: `e82e0703f40b15e2cf7f95c404a21e0ca8ae3efd403fa54da8c30a19cd5825fa`
- transformed update-1 sample (NCHW float32): `ab5730ec5bd7f2e144355de91ba889e7220937ac9ef812f4b6dfa6b4ef0f4cf0`
- bundle `bundle_hash`: `a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23`
- bundle file SHA-256: `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`

## Source identity

| Item | Value |
|---|---|
| Thesis base commit | `a3786d97f3263cad20b254206f09c095880f624d` |
| Thesis synced state SHA-256 | `336f2bb618e9a8c7a7b6797b8a990b84e4df609cf363111cb1cb523cdbd8dda3` |
| Flux current commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux current git-state SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (clean) |
| Flux current source-tree SHA-256 | `83da2929ab7ebd66dbceb83452c8e55eae5e0444c8e96866cc2a25b513c3eb0d` |
| Flux historical commit | `8add0e42b7d9388ff365335152126e5b1e9de657` |
| Flux historical git-state SHA-256 | `4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c` |
| Flux historical source-tree SHA-256 | `ad114ed385f7cc78e8409fa8e1f09d38133555f7fd58d5d9aca6a846bcf5e757` |
| Julia / PyTorch | 1.12.6 / 2.9.1+cu126 (Python 3.11.17, NumPy 2.4.6) |

The historical worktree retains its pre-existing uncommitted
`test/Project.toml` change that adds CUDA/cuDNN/NCCL; it was synced with
`ALLOW_DIRTY_SOURCE=1`. No Flux source was modified for this validation.

## Loader contract checks

The Python reference was generated with `--verify-large-hashes`; the Julia
current check independently re-hashed the large files. All checks are exact
except the declared floating tolerance, and the observed transformed-tensor
difference is exactly zero.

| Check | Result |
|---|---|
| Python reference | `FULL_DATASET_REFERENCE PASS` |
| Python independent check | `FULL_DATASET_PYTHON_CHECK PASS checks=169` |
| Julia current loader | `FULL_DATASET_JULIA_CHECK PASS label=flux-current checks=111` |
| Python combined, current | `FULL_DATASET_PYTHON_CHECK PASS checks=177` |
| Julia historical loader | `FULL_DATASET_JULIA_CHECK PASS label=flux-historical checks=108` |
| Python combined, historical | `FULL_DATASET_PYTHON_CHECK PASS checks=177` |
| Epoch ids | `99,840`, `int64`, exact match to Python `dataset_adapter.rank_sequence` reference |
| Tail | `160` ids dropped; missing set equals the final 160 permutation ids |
| Updates | `195` at batch 512; `390` at batch 256 |
| Rank sequences W=1/2/4 | exact shape/value match; shard concatenation in rank order reproduces the global batch; union is exact; no duplicates |
| Transformed update-1 tensor W=1/2/4 | shape `(512,3,64,64)`, `Float32`, sample order exact, `max_abs_diff=0.0` |
| Labels | exact match to Python `replay_transform`/adapter output |
| One-hot | `(200,512)`, one `1` per column, argmax row = label+1 |
| Bundle | 62 arrays, 11,271,432 parameters, `bundle_hash` and file SHA-256 match the frozen values |

Exact key lines from `logs/julia-check-current.log`:

```text
CHECK rank_sequence.R4.concatenation PASS per-update rank order
CHECK contract.updates.b512 PASS updates=195 expected=195
CHECK contract.updates.b256 PASS updates=390 expected=390
CHECK contract.missing_tail PASS missing=160 tail=160
CHECK update1.R1.transformed.max_abs_diff PASS max_abs_diff=0.0
CHECK update1.R2.transformed.max_abs_diff PASS max_abs_diff=0.0
CHECK update1.R4.transformed.max_abs_diff PASS max_abs_diff=0.0
CHECK onehot.argmax PASS
CHECK bundle.bundle_hash PASS julia=a7632ca9... ref=a7632ca9...
CHECK dataset_file.train_images.npy.sha256 PASS actual=ba43b39f... ref=ba43b39f...
```

## Fixture timing path regression

The existing fixture timing path was run unchanged at W=1 with 3 measured and
2 warmup updates. Both records retain the existing schema and fields:

```text
TIMING_PASS flux 1 0.058378593
TIMING_PASS pytorch 1 0.08964534476399422
FIXTURE_TIMING_JSON PASS flux-current-w1.json schema static-timing-pilot-v1
FIXTURE_TIMING_JSON PASS pytorch-w1.json schema static-timing-pilot-v1
FIXTURE_TIMING_PATH PASS
```

## 1-update GPU dry runs (validation only)

Each run used epoch 0, global batch 512, W=1, 20 discarded warmup updates, and
one measured update (`K=1`). The dataset mmap was opened before timing. The
timed region includes batch assembly, the frozen transform, one-hot creation,
host-to-device transfer, forward, backward, and the optimizer step.

| Framework | seconds (validation only) | final loss | loss finite | memory |
|---|---:|---:|---|---|
| Flux current `f7b4520c` | 0.50566162 | 5.492467880249023 | yes | device used `58,796,081,152` / `68,099,571,712` bytes (`86.34%`); host max RSS `3,088,822,272` bytes |
| Flux historical `8add0e42` | 0.512345482 | 5.492467880249023 | yes | device used `56,615,043,072` / `68,099,571,712` bytes (`83.14%`); host max RSS `3,071,229,952` bytes |
| PyTorch 2.9.1+cu126 | 0.5795233901590109 | 5.492467880249023 | yes | max allocated `15,394,088,960` (`22.61%` of total), max reserved `24,540,872,704` (`36.04%`); host max RSS `1,280,143,360` bytes |

`device used = total - free` from `CUDA.memory_info()`. It includes the CUDA
context and allocator caches, not only live tensors. PyTorch reports allocator
counters separately. **The proposed OD-8 80% margin is not established here.**
Flux device-used is 83–86% of the A100 total at batch 512/W=1 in these
validation runs, so the Phase 5 larger-batch memory pilot must measure and
decide OD-7/OD-8 before any campaign. Batch 512 did not OOM; no silent fallback
to 256 was performed.

The identical final loss across the three configurations is a useful
cross-check of the shared batch/transform/schedule, not a training-quality or
equivalence claim.

## Exact commands

Submission (one command; actually executed):

```bash
ssh leonardo 'sbatch --parsable -A iscrc_unmasked -p boost_usr_prod \
  --qos=boost_qos_dbg -N1 -n1 --cpus-per-task=32 --gres=gpu:4 --time=00:30:00 \
  --job-name=static-full-loader-validate \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-loader-validation-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-validation.sh"'
```

Inside the allocation the driver ran, in order: `precompile-phase3.sh` for each
worktree; `full_dataset_reference.py`; `check_full_dataset.py`;
`check_full_dataset.jl` under current and historical Flux; the fixture timing
smoke; and the three 1-update full-dataset runs. Every exact `RUN` line is in
`logs/` and in `COMMANDS.md`; the complete driver transcript is
`artifacts/logs/leonardo/2026-10-19-static-full-loader-validation-61450014.log`.

## Campaign command (documented, NOT executed)

The full campaign driver is a template and refuses to run without
`CONFIRM_CAMPAIGN=1`. It executes the validated full-dataset path for Flux
current, Flux historical, and PyTorch at W=1,2,4 with epoch 0, batch 512,
K=195, and 20 warmup updates:

```bash
ssh leonardo 'CONFIRM_CAMPAIGN=1 sbatch --parsable -A iscrc_unmasked \
  -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=32 \
  --gres=gpu:4 --time=04:00:00 --job-name=static-full-campaign \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"'
```

This command was **not** executed. The campaign driver is not acceptance
evidence and needs its own preflight, repeat policy, and provenance check.

## Remaining risks and limitations

- No full campaign ran; no throughput or scaling claim is made.
- No W=2 or W=4 GPU dry run ran. The loader contract is validated at W=1,2,4
  on CPU against the Python reference, but distributed GPU execution is not
  re-validated here.
- Flux batch 512/W=1 device-used is 83–86% of the A100 total in these runs.
  This may violate the proposed 80% OD-8 margin. Measure before OD-7/OD-8.
- `device used` is not peak live tensor memory; Flux uses `CUDA.memory_info()`
  because this CUDA.jl version does not expose a max-allocated counter.
- The historical Flux worktree has a pre-existing uncommitted
  `test/Project.toml` environment change (CUDA/cuDNN/NCCL). It is recorded, not
  provenance-clean.
- The fixture path was smoke-tested at W=1 only; its code path was not edited.
- CPU loader validation uses the real dataset and the real manifests; it is
  not a GPU numerical comparison.
- The campaign template is unexecuted and its repeat/provenance design is not
  accepted.
