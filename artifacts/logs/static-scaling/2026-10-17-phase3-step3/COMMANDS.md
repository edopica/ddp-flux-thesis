# Commands — static-scaling Phase 3 Step 3 (FAILED)

All commands were run from the thesis repository root unless noted.
Date: 2026-10-17.

## 1. Local source identity

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
STATE=49e4f70b9e7583faec438654f91cc1bb4a0c4029ab5569990afcdeba671ea8e2
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

`git -C ../flux-integration status --porcelain=v1` returned empty
(clean worktree); `git -C ../flux-integration diff HEAD | sha256sum`
returned the same empty-diff hash.

## 2. Sync to Leonardo

Deliberately not `scripts/remote/sync_code.sh`. The two preserved
evidence directories are excluded and were not uploaded or modified.

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  . leonardo:projects/ddp-flux-thesis/
```

Remote source hashes after sync:

```text
633c7f3794eed8ddf74c4202c96a2e31042e0dbff9e88465c577e9d08eb45710  .../scripts/static_scaling/check_phase3_flux.jl
098cc7ab2072f9c2f7bebe577db26f9a757020de8eb29c80e2f3a1cdbe4ff63e  .../scripts/static_scaling/replay.jl
```

Both match the local files.

## 3. Compute-node precompile

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh"
```

Job `60904055`, node `lrdn3183`; ended with `FLUX_PRECOMPILE_OK`, exit 0.
Local console log: `artifacts/logs/leonardo/2026-10-17-phase3-step3-precompile.log`
(copied here as `console-precompile.log`).

## 4. Full driver (new evidence root)

`OUT` and `LOG_DIR` identify this run. The four provenance variables are
set inside the inner `bash -c`, exactly as in the historical attempt-3
command; the `env` token was not used.

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c 'OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3/reference LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3/logs SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 SOURCE_STATE_SHA256=49e4f70b9e7583faec438654f91cc1bb4a0c4029ab5569990afcdeba671ea8e2 FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 bash scripts/remote/leonardo/run-phase3-check.sh'"
```

Job `60904928`, node `lrdn0125`, `boost_qos_dbg`, one A100, 8 CPUs.
Console: `artifacts/logs/leonardo/2026-10-17-phase3-step3.log`
(copied here as `console-driver.log`).

The driver did not pass `ATOL` or `RTOL`; defaults `1e-5` / `1e-4` were
used (printed in `PHASE3_FLUX_CHECK_CONFIG`).

Result: `PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK FAIL`, `PHASE3_REMOTE_CHECK FAIL`, exit 1.

## 5. Copy evidence home

```bash
mkdir -p artifacts/baselines/static-scaling/2026-10-17-phase3-step3 \
         artifacts/logs/static-scaling/2026-10-17-phase3-step3
rsync -avz leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3/reference/ \
  artifacts/baselines/static-scaling/2026-10-17-phase3-step3/
rsync -avz leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3/logs/ \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3/
cp artifacts/logs/leonardo/2026-10-17-phase3-step3.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3/console-driver.log
cp artifacts/logs/leonardo/2026-10-17-phase3-step3-precompile.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3/console-precompile.log
```

Remote and local SHA-256 MATCH for the copied files:

| File | SHA-256 |
|---|---|
| `phase3_reference.npz` | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| `phase3_reference_manifest.json` | `a301a93fda32b32506a8a34d394a2c0822cdb60b2fb4e3a03b35ecba34c53a33` |
| `remote-check.log` | `8b69ccb788409d9154d55f4aa8bbfad9cb774d3ff785886c8595930b3b5ded2e` |

## 6. New-versus-preserved reference comparison

```bash
python3 - <<'PY'
import numpy as np
new = np.load('artifacts/baselines/static-scaling/2026-10-17-phase3-step3/phase3_reference.npz')
old = np.load('artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz')
# key set, dtype/shape, exact equality, and max_abs for differing arrays
PY
```

Result: 337/337 arrays exact-equal; 0 differing; maximum absolute
difference 0.0. Full output: `reference-comparison.txt`.

## 7. Local independent artifact checker (numpy only)

```bash
python3 scripts/static_scaling/check_phase3_reference.py \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --reference artifacts/baselines/static-scaling/2026-10-17-phase3-step3/phase3_reference.npz \
  --manifest artifacts/baselines/static-scaling/2026-10-17-phase3-step3/phase3_reference_manifest.json
```

Result: `PHASE3_REFERENCE_CHECK PASS arrays=337 atol=1e-05 rtol=0.0001`,
exit 0. Log: `local-reference-check.log`.

## 8. Remote environment metadata

```bash
ssh leonardo 'sha256sum \
  ~/projects/flux-integration/test/Project.toml \
  ~/projects/flux-integration/Manifest.toml \
  ~/projects/ddp-flux-thesis/scripts/static_scaling/julia-env/Manifest.toml \
  ~/projects/ddp-flux-thesis/scripts/static_scaling/julia-env/Project.toml \
  $WORK/epicazio/venvs/pytorch-scaling/requirements-freeze.txt'
```

`~/projects/flux-integration/test/Manifest.toml` does not exist on the
remote copy; the test project is resolved from `test/Project.toml` and
the pinned depot. Recorded hashes are listed in `PROVENANCE.md`.

## 9. Outcome

The run is retained as a **failed** Step 3 attempt. Do not treat it as
acceptance evidence. The next action is the precision-policy enforcement
fix described in `PROVENANCE.md` section 6, followed by a fresh Step 3
run.
