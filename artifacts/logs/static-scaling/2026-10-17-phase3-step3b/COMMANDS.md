# Commands — static-scaling Phase 3 Step 3b (FMA algorithm chooser)

All commands were run from the thesis repository root unless noted.
Date: 2026-10-17.

## 1. Local checks

```bash
julia --startup-file=no -e \
  'for f in ["scripts/static_scaling/precision.jl",
             "scripts/static_scaling/check_phase3_flux.jl",
             "scripts/static_scaling/check_precision_math.jl"]
     Meta.parseall(read(f, String)); println("PARSE_OK ", f)
   end'
bash -n scripts/remote/leonardo/run-phase3-precision-regression.sh
```

The precision module was also loaded on a CPU-only machine with
`using CUDA, cuDNN; include("scripts/static_scaling/precision.jl")` to
confirm that the module-load method definitions parse and install
without invoking CUDA runtime functions. `configure_precision!` was not
called there (no GPU).

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
STATE=4f2da8a364b277839b4dc77f1f642866d900e9b5e025de17791a0bf28cfc3c1e
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Run-time `scripts/static_scaling/precision.jl` SHA-256:
`7d47c687bf003eb9cb2357e6950d95a44263afa88707e35fa26b213bf8bb9bf8`.
After the run, the fallback-only `@warn` metadata was corrected
(`kind` -> `kind = kind`); the current local file SHA-256 is
`60e003ced5cb50fd546edddc2976bc403ee3d855977274f0bb09d7f34946916d`.
That branch did not execute; the remote tree was not re-synced.

## 3. Sync to Leonardo

`scripts/remote/sync_code.sh` was not used. The two preserved-evidence
directories were excluded as in Step 3. Two further excludes cover the
local-only copies of the failed Step 3 evidence, which are not run
inputs; the first full-sync attempt without them was stopped after
15 minutes while uploading the 286 MB local reference at about
0.1–0.3 MB/s.

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-step3/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-step3/' \
  . leonardo:projects/ddp-flux-thesis/
```

The changed scripts were also transferred directly while the full sync
was being diagnosed:

```bash
rsync -avz scripts/ leonardo:projects/ddp-flux-thesis/scripts/
```

## 4. Compute-node precompile

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh"
```

Job `60909956`, node `lrdn0034`; ended with `FLUX_PRECOMPILE_OK`, exit 0.
Console log: `artifacts/logs/leonardo/2026-10-17-phase3-step3b-precompile.log`
(copied here as `2026-10-17-phase3-step3b-precompile.log`).

## 5. Precision regression (fresh processes)

`STATIC_SCALING_PRECISION_LOG_CANDIDATES=1` records the candidate lists
(no behavioral effect).

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/precision \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/precision/logs \
    STATIC_SCALING_PRECISION_LOG_CANDIDATES=1 \
    bash scripts/remote/leonardo/run-phase3-precision-regression.sh'"
```

Job `60910394`, node `lrdn0365`. Result:
`PHASE3_PRECISION_REGRESSION PASS` (fma exit 0; current exit 1 with the
expected `conv_fixed_fwd` failure). Console log:
`artifacts/logs/leonardo/2026-10-17-phase3-step3b-precision-regression.log`.

## 6. Full driver (new evidence root)

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/reference \
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/logs \
    SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
    SOURCE_STATE_SHA256=4f2da8a364b277839b4dc77f1f642866d900e9b5e025de17791a0bf28cfc3c1e \
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e \
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 \
    STATIC_SCALING_PRECISION_LOG_CANDIDATES=1 \
    bash scripts/remote/leonardo/run-phase3-check.sh'"
```

Job `60910786`, node `lrdn0001`, one A100, 8 CPUs, `boost_qos_dbg`.
The driver did not pass `ATOL` or `RTOL`; defaults `1e-5` / `1e-4`
were used (`PHASE3_FLUX_CHECK_CONFIG` in the log).

Result: `PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`,
`PHASE3_FLUX_CHECK FAIL`, `PHASE3_REMOTE_CHECK FAIL`, exit 1.
Console log: `artifacts/logs/leonardo/2026-10-17-phase3-step3b.log`
(copied here as `2026-10-17-phase3-step3b.log`).

## 7. Copy evidence home

```bash
mkdir -p artifacts/baselines/static-scaling/2026-10-17-phase3-step3b \
         artifacts/logs/static-scaling/2026-10-17-phase3-step3b/precision
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/reference/phase3_reference_manifest.json' \
  artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/logs/remote-check.log' \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3b/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/precision/logs/fma.log' \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3b/precision/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step3b/precision/logs/current.log' \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3b/precision/
cp artifacts/baselines/static-scaling/2026-10-17-phase3-step3/phase3_reference.npz \
  artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/
cp artifacts/logs/leonardo/2026-10-17-phase3-step3b*.log \
  artifacts/logs/static-scaling/2026-10-17-phase3-step3b/
```

Remote new reference SHA-256 is
`97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783`,
identical to the preserved local copy used above. Manifest SHA-256 is
`d9985934156c91404a7d02563de63e07b631fcf028859202b3ee6104882effca`;
remote-check log SHA-256 is
`b4cc21f8ba39242ad1f43b50200d641328c6e45b354a994ede035a61ae15bc14`.

## 8. Independent artifact checker and reference comparison

```bash
python3 scripts/static_scaling/check_phase3_reference.py \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --reference artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/phase3_reference.npz \
  --manifest artifacts/baselines/static-scaling/2026-10-17-phase3-step3b/phase3_reference_manifest.json
```

Result: `PHASE3_REFERENCE_CHECK PASS arrays=337 atol=1e-05 rtol=0.0001`,
exit 0. Log: `local-reference-check.log`.

The reference comparison used NumPy to compare key sets, dtype/shape, and
exact equality against
`artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/phase3_reference.npz`.
Result: 337/337 exact-equal, 0 differing, maximum absolute difference
`0.0`. Output: `reference-comparison.txt`.

## 9. Outcome

This run is retained as a **failed** Step 3b acceptance attempt. It is
not acceptance evidence. The precision-policy fix works (all choices and
candidates are FMA; forward passes); the remaining blocker is FP32
backward algorithm selection under the frozen tolerances. The next
decision belongs to the user.
