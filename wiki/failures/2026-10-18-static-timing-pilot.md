# Static timing pilot failures — 2026-10-18

## Smoke job 61304553

The first smoke job stopped before precompile or training, with exit 128.
The driver requested Git metadata on Leonardo, but the sync workflow excludes `.git`.
The log reports `fatal: not a git repository` on `lrdn1772`.

Correction: preserve the actual Flux `src/`, `ext/`, project, and test manifest in each output directory.
The driver also preserves all static-scaling scripts and their SHA-256 hashes.
These snapshots identify uncommitted source without a remote Git repository.

Log: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-timing-smoke-61304553.log`.

## Smoke job 61304730

The second smoke stopped before precompile because `flux-integration/test/Manifest.toml` does not exist on Leonardo.
The resolved Flux manifest is at the worktree root. The driver now preserves that manifest instead.
The static-scaling environment has its own preserved manifest.
The diagnostic SSH command also requested a nonexistent historical `logs/a-precompile.log` and exited 1.
Neither failure ran a model workload.

## Smoke job 61304948

Compute-node precompile passed with `FLUX_PRECOMPILE_OK` on `lrdn0566`.
Slurm then warned that the W=1 step requested 32 CPUs per task against an eight-CPU allocation limit.
The operator canceled this job before its training step started.
The replacement allocation uses one task with 32 CPUs and four GPUs, as the established Phase 4 driver requires.
Its internal steps divide those 32 CPUs across W=1,2,4.
The replacement also preserves warmed PyTorch DDP buckets during the post-warmup mathematical reset.

## Campaign job 61305371 — canceled by user decision

The smoke `61305319` passed all six configurations.
The dependent campaign `61305371` started on `lrdn3063`.
The user then directed that Phase 4 be finalized and that timing jobs not run yet.
The operator canceled `61305371` after 23 of 30 records.
The partial records are not analyzed, are not Phase 4 acceptance evidence, and do not support throughput claims.
No timing job is active.
Timing remains deferred until the user explicitly authorizes it.
