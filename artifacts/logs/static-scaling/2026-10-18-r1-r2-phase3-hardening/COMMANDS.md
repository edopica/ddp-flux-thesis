# Commands — Phase 3 hardening recheck (2026-10-18)

All commands ran from the thesis root or inside the Leonardo Slurm allocation.
No source code was changed during this campaign. No commit was made.

## 1. Source snapshot

Generation-time identity:

```text
THESIS_HEAD=a3786d97f3263cad20b254206f09c095880f624d
THESIS_STATE_SHA256=da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Later documentation edits move the live worktree hash; the values above are
the run-time provenance. The remote run log also records
`THESIS_GIT=<no-git>` and `FLUX_GIT=<no-git>` because the remote trees are
rsync deployments, not Git checkouts.

## 2. Sync

The campaign workflow synced the thesis tree to Leonardo without `--delete`:

```bash
rsync -az --exclude-from='.rsyncignore' \
  . leonardo:projects/ddp-flux-thesis/
```

The run-time console records the exact executed job command.

## 3. Precompile

Job `61187225`, node `lrdn0241`, UTC `12:20:59`-`12:24:36`, exit 0,
`FLUX_PRECOMPILE_OK`.

```bash
JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot \
PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin:$PATH \
SLURM_MPI_TYPE=pmi2 JULIA_PKG_PRECOMPILE_AUTO=0 \
salloc --account=iscrc_unmasked --partition=boost_usr_prod \
  --qos=boost_qos_dbg -N1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --time=00:30:00 \
  srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh
```

## 4. Full Phase 3 driver

Job `61191281`, node `lrdn2911`, UTC start `2026-10-18T12:30:16Z`, driver
`12:30:21Z`, elapsed `00:07:52`, exit 0.

```bash
JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot \
PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin:$PATH \
SLURM_MPI_TYPE=pmi2 JULIA_PKG_PRECOMPILE_AUTO=0 \
salloc --account=iscrc_unmasked --partition=boost_usr_prod \
  --qos=boost_qos_dbg -N1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --time=00:30:00 \
  srun --mpi=pmi2 -n1 -c8 --chdir=$HOME/projects/ddp-flux-thesis bash -c '
    OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-phase3-hardening/reference
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-phase3-hardening/logs
    SOURCE_COMMIT=a3786d97f3263cad20b254206f09c095880f624d
    SOURCE_STATE_SHA256=da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    STATIC_SCALING_PRECISION_LOG_CANDIDATES=1
    bash scripts/remote/leonardo/run-phase3-check.sh'
```

No `ATOL`, `RTOL`, or category override was passed; the frozen checker
defaults supplied the rule.

## 5. Copy back

```bash
rsync -avz \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-phase3-hardening/reference/phase3_reference.npz' \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-phase3-hardening/reference/phase3_reference_manifest.json' \
  artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/
rsync -avz \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-phase3-hardening/logs/remote-check.log' \
  artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/
cp artifacts/logs/leonardo/2026-10-18-r1-r2-phase3-hardening.log \
  artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/console/phase3-driver-console.log
cp artifacts/logs/leonardo/2026-10-18-r1-r2-precision-hardening-precompile-phase3.log \
  artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/console/precompile-phase3.log
```

## 6. Local independent checks

```bash
python3 scripts/static_scaling/check_phase3_reference.py \
  --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \
  --bundle artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz \
  --reference artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/phase3_reference.npz \
  --manifest artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/phase3_reference_manifest.json
```

The exact array comparison is in
[`local-array-comparison.txt`](local-array-comparison.txt): 337/337 exact,
`max_abs_diff=0`. The independent checker output is
[`local-reference-check.log`](local-reference-check.log).

```bash
sha256sum -c SHA256SUMS
```

The remote/local comparison is in
[`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt): precision 3/3 MATCH and
Phase 3 5/5 MATCH.
