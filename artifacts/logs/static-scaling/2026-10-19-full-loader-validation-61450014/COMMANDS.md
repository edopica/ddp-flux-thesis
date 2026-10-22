# Commands — full-dataset loader validation 61450014

Status: PASS (validation only). The command transcript below is copied from
the Slurm log. Paths are remote Leonardo paths unless marked local.

## Local sync (executed before submission)

```bash
scripts/remote/sync_code.sh leonardo
scripts/remote/sync_flux_integration.sh leonardo
ALLOW_DIRTY_SOURCE=1 scripts/remote/sync_flux_baseline.sh leonardo
```

## Submission (executed)

```bash
ssh leonardo 'sbatch --parsable -A iscrc_unmasked -p boost_usr_prod \
  --qos=boost_qos_dbg -N1 -n1 --cpus-per-task=32 --gres=gpu:4 --time=00:30:00 \
  --job-name=static-full-loader-validate \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-loader-validation-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-validation.sh"'
```

## Driver commands (executed on lrdn0667)

Precompile commands are the driver `STAGE precompile-*` blocks invoking
`scripts/remote/leonardo/precompile-phase3.sh` with `FLUX_DIR` set to the
current and historical worktrees. In the transcript, `julia_current` and
`julia_historical` are driver functions:

```bash
# SCRIPTS=$THESIS/scripts/static_scaling
# FLUX_CURRENT=$HOME/projects/flux-integration
# FLUX_HISTORICAL=$HOME/projects/flux-static-baseline
# julia_current:
env JULIA_DEPOT_PATH=$WORK/epicazio/julia-depot \
    JULIA_LOAD_PATH="$SCRIPTS/julia-env:$FLUX_CURRENT/test:@stdlib" \
    JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    $WORK/epicazio/julia-1.12.6/bin/julia --startup-file=no <args>
# julia_historical: identical with FLUX_HISTORICAL in JULIA_LOAD_PATH
```

The data and run commands are:

```bash
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/full_dataset_reference.py --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference --epoch 0 --batch-size 512 --worlds 1,2,4 --verify-large-hashes
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_full_dataset.py --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --reference-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference/python-check.json --epoch 0 --batch-size 512 --worlds 1,2,4 --strict-contract true
julia_current /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_full_dataset.jl --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --reference-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/check-julia-current.json --batch-size 512 --epoch 0 --worlds 1,2,4 --label flux-current --verify-large-hashes true --strict-contract true
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_full_dataset.py --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --reference-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference/python-check-current-combined.json --epoch 0 --batch-size 512 --worlds 1,2,4 --strict-contract true --julia-json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/check-julia-current.json
julia_historical /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_full_dataset.jl --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --reference-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/check-julia-historical.json --batch-size 512 --epoch 0 --worlds 1,2,4 --label flux-historical --verify-large-hashes false --strict-contract true
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_full_dataset.py --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --reference-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/reference/python-check-historical-combined.json --epoch 0 --batch-size 512 --worlds 1,2,4 --strict-contract true --julia-json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/check-julia-historical.json
julia_current /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/timing_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --world-size 1 --updates 3 --warmup 2 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/fixture-smoke/flux-current-w1.json --source-commit a3786d97f3263cad20b254206f09c095880f624d --source-state-sha256 336f2bb618e9a8c7a7b6797b8a990b84e4df609cf363111cb1cb523cdbd8dda3 --flux-commit f7b4520c4bd307c67bbc5ac92a1c68493eccff6e --flux-state-sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/timing_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --world-size 1 --updates 3 --warmup 2 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/fixture-smoke/pytorch-w1.json --source-commit a3786d97f3263cad20b254206f09c095880f624d --source-state-sha256 336f2bb618e9a8c7a7b6797b8a990b84e4df609cf363111cb1cb523cdbd8dda3
julia_current /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/timing_flux.jl --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --world-size 1 --updates 1 --warmup 20 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/dryrun/flux-current-w1.json --source-commit a3786d97f3263cad20b254206f09c095880f624d --source-state-sha256 336f2bb618e9a8c7a7b6797b8a990b84e4df609cf363111cb1cb523cdbd8dda3 --flux-commit f7b4520c4bd307c67bbc5ac92a1c68493eccff6e --flux-state-sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
julia_historical /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/timing_flux.jl --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --world-size 1 --updates 1 --warmup 20 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/dryrun/flux-historical-w1.json --source-commit a3786d97f3263cad20b254206f09c095880f624d --source-state-sha256 336f2bb618e9a8c7a7b6797b8a990b84e4df609cf363111cb1cb523cdbd8dda3 --flux-commit 8add0e42b7d9388ff365335152126e5b1e9de657 --flux-state-sha256 4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c
/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/timing_torch.py --dataset-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared --world-size 1 --updates 1 --warmup 20 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/dryrun/pytorch-w1.json --source-commit a3786d97f3263cad20b254206f09c095880f624d --source-state-sha256 336f2bb618e9a8c7a7b6797b8a990b84e4df609cf363111cb1cb523cdbd8dda3 --flux-commit  --flux-state-sha256 
```

## Campaign command (documented, NOT executed)

```bash
ssh leonardo 'CONFIRM_CAMPAIGN=1 sbatch --parsable -A iscrc_unmasked \
  -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=32 \
  --gres=gpu:4 --time=04:00:00 --job-name=static-full-campaign \
  --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-%j.log \
  "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"'
```

This command was not executed. See `RESULTS.md` for the acceptance scope and risks.
