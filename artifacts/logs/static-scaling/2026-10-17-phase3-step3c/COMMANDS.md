# Commands — static-scaling Phase 3 Step 3c (OD-10 revision, acceptance)

All commands were run from the thesis repository root unless noted.
Date: 2026-10-17 (UTC run start 2026-10-18T00:08:54Z).

## 1. Local source checks

```bash
julia --startup-file=no -e \
  'for f in ["scripts/static_scaling/check_phase3_flux.jl"]
     Meta.parseall(read(f, String)); println("PARSE_OK ", f)
   end'

python3 - <<'PY'
import tomllib
with open('experiments/static-scaling/experiment.toml','rb') as f:
    d=tomllib.load(f)
t=d['tolerances']
assert t['cross_framework']['atol'] == 1e-5
assert t['cross_framework']['rtol'] == 1e-4
assert t['category_overrides']['phase3.grads.u1']['atol'] == 1e-3
assert t['category_overrides']['phase3.grads.u3']['atol'] == 2e-3
assert t['category_overrides']['phase3.m.u3']['atol'] == 2e-4
print('TOML_OK')
PY
```

Both checks passed (`PARSE_OK`, `TOML_OK`).

## 2. Source identity

```bash
COMMIT=$(git rev-parse HEAD)
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
FLUX_COMMIT=$(git -C ../flux-integration rev-parse HEAD)
FLUX_STATE=$( { git -C ../flux-integration diff HEAD; \
  git -C ../flux-integration ls-files --others --exclude-standard \
  | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Result:

```text
COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3
STATE=e1970ef82a09e7864ec13e453afab727486b658954c46c842e28cbcd39f9e7e5
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Changed source hashes:

```text
16a9a4c51f78e8d2f659a06c4fc575ee211bcdc4bc6cd69c5f979967c7a71c44  scripts/static_scaling/check_phase3_flux.jl
60e003ced5cb50fd546edddc2976bc403ee3d855977274f0bb09d7f34946916d  scripts/static_scaling/precision.jl
ff08e3fd8054999b0104ed9a9f18d7a18104ba50908e3209bb532900c3ae2720  experiments/static-scaling/experiment.toml
14be9f10143850a6c2b2be4a7bc9c1a6fd45aa6d2ae1e793327e076e2a6fdcd7  experiments/static-scaling/specification.md
91b9e5708da22e7a0aaf8fbd12bf5f546c632bd7a091bb383dc70d7f803b18d4  experiments/static-scaling/open-decisions.md
```

## 3. Sync to Leonardo (explicit exclusions, not `sync_code.sh`)

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

Remote hashes of the five changed files matched the local values above.

## 4. Compute-node precompile

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh"
```

Job `60941770`, node `lrdn2500`; ended with `FLUX_PRECOMPILE_OK`, exit 0.
Console log: `artifacts/logs/leonardo/2026-10-17-phase3-step3c-precompile.log`
(copied here as `2026-10-17-phase3-step3c-precompile.log`).

## 5. Full driver (new evidence root)

No `ATOL`, `RTOL`, or category override was passed. The checker used its
frozen defaults.

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/reference \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/logs \
    SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
    SOURCE_STATE_SHA256=e1970ef82a09e7864ec13e453afab727486b658954c46c842e28cbcd39f9e7e5 \
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
    STATIC_SCALING_PRECISION_LOG_CANDIDATES=1 \
    bash scripts/remote/leonardo/run-phase3-check.sh'"
```

Job `60942933`, node `lrdn0129`, one A100, 8 CPUs, `boost_qos_dbg`.
Result: `PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK PASS`, `PHASE3_REMOTE_CHECK PASS`; driver exit 0.
Console log: `artifacts/logs/leonardo/2026-10-17-phase3-step3c.log`
(copied here as `2026-10-17-phase3-step3c.log`).

## 6. Copy evidence home

```bash
mkdir -p artifacts/baselines/static-scaling/2026-10-17-phase3-step3c \
         artifacts/logs/static-scaling/2026-10-17-phase3-step3c
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/reference/phase3_reference_manifest.json' \
  artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3c/logs/remote-check.log' \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3c/
cp artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz \
  artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/
cp artifacts/logs/leonardo/2026-10-17-phase3-step3c.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3c/2026-10-17-phase3-step3c.log
cp artifacts/logs/leonardo/2026-10-17-phase3-step3c-precompile.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3c/
```

The remote `.npz` SHA-256
(`97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`)
equals the preserved local file, so the local file was copied instead of
downloading 286 MB. Remote manifest and log hashes are recorded in
`PROVENANCE.md`.

## 7. Independent artifact checker and reference comparison

```bash
python3 scripts/static_scaling/check_phase3_reference.py \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --reference artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/phase3_reference.npz \
  --manifest artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/phase3_reference_manifest.json
```

Result: `PHASE3_REFERENCE_CHECK PASS arrays=337 atol=1e-05 rtol=0.0001`,
exit 0. Log: `local-reference-check.log`.

```bash
python3 - <<'PY'
import numpy as np
new = np.load('artifacts/baselines/static-scaling/2026-10-17-phase3-step3c/phase3_reference.npz')
old = np.load('artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz')
# key-set, dtype/shape, and np.array_equal comparison over all arrays
# exact count, differing list, and maximum absolute difference are printed
PY
```

Result: 337/337 exact-equal, 0 differing, maximum absolute difference
`0.0`, `REFERENCE_COMPARISON PASS`. Output:
`reference-comparison.txt`.

## 8. Outcome

This run is the Step 3c acceptance evidence. All four required summaries
PASS and the driver exits 0. The three OD-10 category floors are applied;
no other tolerance changed. Step 3b is superseded by the approved
calibration, not by a silent tolerance change. Steps 4–6 were not run.
