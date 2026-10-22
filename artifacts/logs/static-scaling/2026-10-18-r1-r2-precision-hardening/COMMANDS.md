# Commands — R1/R2 precision hardening (2026-10-18)

All remote commands ran from the Leonardo login node or inside the Slurm
allocation. No source code was changed during this campaign. No commit was
made.

## 1. Source identity

Generation-time identity passed to the driver:

```text
SOURCE_COMMIT=a3786d97f3263cad20b254206f09c095880f624d
SOURCE_STATE_SHA256=da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199
FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Later documentation edits move the live worktree hash. The values above are
the run-time provenance.

## 2. Sync

The campaign workflow synced the thesis tree to Leonardo without `--delete`,
then copied the changed helper scripts into the existing remote trees:

```bash
rsync -az --exclude-from='.rsyncignore' \
  . leonardo:projects/ddp-flux-thesis/
```

The runtime console records the exact executed job command; the source
identity above pins the synced thesis state.

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

## 4. Focused precision regression

Job `61188854`, node `lrdn1845`, UTC start `2026-10-18T12:24:49Z`, driver
`12:24:54Z`, exit 0, `PHASE3_PRECISION_REGRESSION PASS`.

```bash
JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot \
PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin:$PATH \
SLURM_MPI_TYPE=pmi2 JULIA_PKG_PRECOMPILE_AUTO=0 \
salloc --account=iscrc_unmasked --partition=boost_usr_prod \
  --qos=boost_qos_dbg -N1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --time=00:30:00 \
  srun --mpi=pmi2 -n1 -c8 --chdir=$HOME/projects/ddp-flux-thesis bash -c '
    OUT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-precision-hardening/precision
    LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-precision-hardening/precision/logs
    SOURCE_COMMIT=a3786d97f3263cad20b254206f09c095880f624d
    SOURCE_STATE_SHA256=da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199
    FLUX_COMMIT=f7b4520c4bd307c67bbc5ac92a1c68493eccff6e
    FLUX_STATE_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    bash scripts/remote/leonardo/run-phase3-precision-regression.sh'
```

The driver runs three fresh processes: `fma`, `current`, and `policy`.

## 5. Copy back and checks

```bash
rsync -avz \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-r1-r2-precision-hardening/precision/' \
  artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/precision/
cp artifacts/logs/leonardo/2026-10-18-r1-r2-precision-hardening-precision-regression.log \
  artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/console/precision-regression-console.log
cp artifacts/logs/leonardo/2026-10-18-r1-r2-precision-hardening-precompile-phase3.log \
  artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/console/precompile-phase3.log
sha256sum -c SHA256SUMS
```

The remote/local precision log comparison is `3/3 MATCH`; see
[`SHA256-COMPARISON.txt`](SHA256-COMPARISON.txt).
