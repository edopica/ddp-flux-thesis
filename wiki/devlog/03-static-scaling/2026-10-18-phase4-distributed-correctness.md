# Phase 4 GPU acceptance and timing pilot — 2026-10-18

## Acceptance

The user approved finite absolute numerical error `<1e-3` for FP32 comparisons.
Structural and rank consistency checks remain exact. CPU validation is deferred.
The artifact recheck passed all 202 categories for Flux and PyTorch at W=1,2,4.
The final GPU allocation `61296074` completed on `lrdn0983`, exit `0:0`.
No model workload ran locally in this session.

The previous checker used CUDA-only trajectory overrides.
This session removed those overrides and added the explicit `--fp32-absolute-1e3` option.
The option rejects errors equal to `1e-3`, nonfinite values, and relative-tolerance escapes.
The self-test passes 36 checks. Exact comparisons retain their original behavior.
The canonical recheck finished at `2026-10-18T21:44:16Z` and printed
`PHASE4_CHECK PASS categories=202`. Canonical report SHA-256:
`a5ca161c3e7b7b57a9d5b3d24c7ddb9236ce613554db5ab95f391976a1fa7f8c`.

Results and limitations: [Phase 4 results](../../../artifacts/logs/static-scaling/2026-10-18-phase4/RESULTS.md).
The old strict same-framework criterion is not the acceptance claim.
The Flux split-control u3 artifact still uses the initial state.
The separate shard-reference artifacts supply the multi-update control.

## Timing implementation

- `scripts/static_scaling/timing_flux.jl`: existing Flux model, Optimisers state, and NCCL backend.
- `scripts/static_scaling/timing_torch.py`: existing PyTorch model, AdamW, and native DDP.
- `scripts/remote/leonardo/run-static-timing.sh`: compute-node precompile, bounded matrix, source snapshots, and logs.
- `scripts/static_scaling/check_timing.py`: artifact-only workload, completeness, duration, and throughput checks.

The pilot uses global batch 12 and alternates the two accepted Phase 2 fixture batches.
It uses W=1,2,4, with ordinary single-GPU training at W=1.
Each campaign configuration uses 20 warmup updates and 300 measured updates.
Five fresh-process repeats give 30 timing records.
The driver alternates framework and world-size order between repeat blocks.
All six configurations run sequentially on the same allocated four-A100 node.

Both runners restore model parameters, optimizer state, schedule position, and batch position after warmup.
PyTorch retains its warmed DDP wrapper and buckets during reset.
CUDA synchronization brackets the measured loop. A rank barrier precedes the start.
The maximum synchronized rank duration determines images/s and update time.
Data staging, initialization, hashes, diagnostics, and output writes remain outside the measured loop.
Flux retains the distributed optimizer guards and strict FMA helper.
The pilot does not optimize the helper or claim to resolve OD-14 overhead.

Final rank hashes must agree exactly before rank 0 writes the result.
Each launch has a 1,200-second watchdog. Slurm also bounds the complete job.
The outputs include raw durations, workload counts, fixture hashes, node, job, and precision policy.
The driver preserves the scripts, Flux source, manifests, and Python dependency freeze.

This is a training-only fixture pilot. The full-dataset campaign, memory pilot, and learning curves remain pending.

Timing is deferred. The smoke (`61305319`) passed all six configurations.
Campaign `61305371` was canceled at user request after 23 of 30 records.
The partial records are not analyzed and are not Phase 4 acceptance evidence.
No timing job is active.

## Commands and evidence

Local checks:

```bash
python3 -m py_compile scripts/static_scaling/timing_torch.py scripts/static_scaling/check_phase4.py
python3 scripts/static_scaling/check_phase4.py --self-test
bash -n scripts/remote/leonardo/run-static-timing.sh
bash artifacts/logs/static-scaling/2026-10-18-phase4/recheck-approved.sh
```

Deployment:

```bash
scripts/remote/sync_code.sh leonardo
scripts/remote/sync_flux_integration.sh leonardo
```

The batch driver runs `bash scripts/remote/leonardo/precompile-phase3.sh` on its allocated compute node before training.
The first two smoke attempts stopped during provenance capture, before precompile or training.
Job `61304553` assumed a remote Git repository. Job `61304730` assumed a test manifest path.
Both assumptions are corrected. See the [failure record](../../failures/2026-10-18-static-timing-pilot.md).

The third smoke (`61304948`) passed precompile, then stopped at a Slurm CPU-allocation warning.
The operator canceled it before training. The corrected smoke is `61305319`.

Corrected smoke submission command:

```bash
ssh -o BatchMode=yes leonardo 'sbatch --parsable -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg -N1 -n1 --cpus-per-task=32 --gres=gpu:4 --time=00:30:00 --job-name=static-timing-smoke --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-timing-smoke-%j.log "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-timing.sh" smoke'
```

Campaign submission returned job `61305371`:

```bash
ssh -o BatchMode=yes leonardo 'sbatch --parsable -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=32 --gres=gpu:4 --time=02:00:00 --dependency=afterok:61305319 --kill-on-invalid-dep=yes --job-name=static-timing-campaign --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-timing-campaign-%j.log "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-timing.sh" campaign'
```

Slurm releases this campaign only after the smoke exits successfully.
The two-hour wall limit and per-launch watchdog bound the workload.

Remote output root:
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/`.
Each attempt has its own `smoke-JOBID/` or `campaign-JOBID/` directory.

## Next action

Phase 4 is finalized. The canonical artifact recheck passes 202 categories.
GPU timing is deferred; no timing job is active.
The partial pilot is preserved under
`artifacts/logs/static-scaling/2026-10-18-timing-pilot/` and is not acceptance evidence.
Start timing only on explicit request.
