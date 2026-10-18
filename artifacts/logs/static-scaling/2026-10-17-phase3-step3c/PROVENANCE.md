# Provenance — static-scaling Phase 3 Step 3c (OD-10 revision, acceptance)

Date: 2026-10-17 (UTC run start 2026-10-18T00:08:54Z).
Status: **PASS**. Three approved OD-10 category floors only; base rule
`1e-5` / `1e-4` otherwise. No Flux source change, no commit, no model-math
or optimizer change, no reference change.

## 1. Source identity

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis source-state hash | `e1970ef82a09e7864ec13e453afab727486b658954c46c842e28cbcd39f9e7e5` |
| Flux worktree commit | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| Flux worktree state | clean; source-state hash below |
| Flux source-state hash | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty-diff hash) |

Source-state recipe (unchanged):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Changed/added source files in this revision:

| File | SHA-256 |
|---|---|
| `scripts/static_scaling/check_phase3_flux.jl` | `16a9a4c51f78e8d2f659a06c4fc575ee211bcdc4bc6cd69c5f979967c7a71c44` |
| `scripts/static_scaling/precision.jl` | `60e003ced5cb50fd546edddc2976bc403ee3d855977274f0bb09d7f34946916d` |
| `experiments/static-scaling/experiment.toml` | `ff08e3fd8054999b0104ed9a9f18d7a18104ba50908e3209bb532900c3ae2720` |
| `experiments/static-scaling/specification.md` | `14be9f10143850a6c2b2be4a7bc9c1a6fd45aa6d2ae1e793327e076e2a6fdcd7` |
| `experiments/static-scaling/open-decisions.md` | `91b9e5708da22e7a0aaf8fbd12bf5f546c632bd7a091bb383dc70d7f803b18d4` |

`precision.jl` is the post-Step-3b fallback-only `@warn` fix
(`kind` -> `kind = kind`). It was synced before this run; the branch did
not execute. The algorithm-selection logic is unchanged from the Step 3b
run-time file `7d47c687...`.

Remote copies of all five files matched these hashes after sync (checked
with `sha256sum` over SSH).

## 2. Remote environment and run identity

| Item | Value |
|---|---|
| Cluster / partition / QoS | Leonardo, `boost_usr_prod`, `boost_qos_dbg` |
| Precompile job / node | `60941770` / `lrdn2500` (`FLUX_PRECOMPILE_OK`) |
| Full driver job / node | `60942933` / `lrdn0129` (exit 0) |
| GPU | NVIDIA A100-SXM-64GB (one GPU allocated), driver 535.274.02 |
| Julia | 1.12.6 |
| CUDA.jl / cuDNN.jl | 6.2.2 / 6.2.2 |
| cuDNN library | 9.26.0 (`cuDNN.version()` prints `92.60.0`) |
| PyTorch reference env | torch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2.21 |
| Driver tolerances | base `atol=1e-5`, `rtol=1e-4`; frozen overrides in the checker defaults |

Remote run root:

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/
  reference/phase3_reference.npz
  reference/phase3_reference_manifest.json
  logs/remote-check.log
```

Evidence root name `2026-10-17-phase3-step3c` is new; no previous run
directory was reused or overwritten.

## 3. Sync

`scripts/remote/sync_code.sh` was not used. The Step 3b explicit-exclusion
rsync was repeated, with the newer local-only evidence directories also
excluded so that large local reference copies were not uploaded:

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3b/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-numerics/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-numerics/' \
  . leonardo:projects/ddp-flux-thesis/
```

The transfer was 151,342 bytes. The first compute-node run after sync was
the precompile; the driver log contains no `Precompiling packages...` line.

## 4. Reference equality and independent check

| Item | SHA-256 |
|---|---|
| New `phase3_reference.npz` (remote == local) | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| New `phase3_reference_manifest.json` | `221d185968ed3fb30a7d3a33fa8eb1afabad3719d259a4876765b73d1c78260d` |
| `remote-check.log` | `0e3caeddca3f807908eda4367109dbf51a9685d424091fec070d1d0473e3bc73` |

The new `.npz` SHA-256 equals the preserved attempt-3 reference, so the
local preserved file was copied byte-for-byte into this evidence directory
instead of downloading 286 MB again. Local array comparison against
`artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz`:
337/337 arrays exact-equal, 0 differing, maximum absolute difference
`0.0`. The independent NumPy checker passes locally on the copied
reference/manifest: `PHASE3_REFERENCE_CHECK PASS arrays=337`, exit 0.

## 5. Algorithm evidence

All 32 choices are `CUDNN_FMA_MATH(3)` with `fallback=false`; all 225
recorded candidate math types are `3`; no tensor-op/TF32 candidate was
present. Chosen algorithms:

- forward: algo 0 (7), algo 6 (4)
- backward data: algo 0 (7), algo 4 (3)
- backward filter: algo 0 (2), algo 3 (6), algo 5 (3)

Per-choice lines, candidate lists, and counts are in
`algorithm-choices.txt`.

## 6. Outcome

`PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK PASS`, `PHASE3_REMOTE_CHECK PASS`; driver exit 0.
Phase 3 acceptance passes under the recorded OD-10 rule.
