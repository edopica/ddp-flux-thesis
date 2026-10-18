# Phase 2 prep on Leonardo: `env`-token exec failure and QoS selection

Date: 2026-10-16
Scope: Leonardo retries 1-2 (jobs 60729163 and 60729410; `boost_qos_bprod`
submit rejection), static scaling Phase 2
Status: FIXED and verified in retry-5 (2026-10-16); no artifacts were lost

## Symptom

The first two prep attempts died before a long run could finish. Job 60729163
failed in about 2 seconds with an `execve` error. Job 60729410 was killed at
the 30-minute `boost_qos_dbg` wall cap mid-decode. A follow-up attempt on
`boost_qos_bprod` was rejected at submit time.

## Failure 1 (job 60729163): the `env` token under `srun`

### Failed command

```bash
salloc --account=iscrc_unmasked --partition=boost_usr_prod \
  --qos=boost_qos_dbg -N1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=00:30:00 \
  srun --mpi=pmi2 -n1 -c8 --chdir=$HOME/projects/ddp-flux-thesis \
  env SOURCE_COMMIT=3088f610... SOURCE_STATE_SHA256=caa7f00e... \
  bash scripts/remote/leonardo/run-phase2-prep.sh
```

### Failure (verbatim)

```text
salloc: Granted job allocation 60729163
salloc: Nodes lrdn2229 are ready for job
[2026-10-16T17:00:32.630] error: execve(): /leonardo_work/IscrC_UNMASKED/epicazio/tools/uv/env: Permission denied
srun: error: lrdn2229: task 0: Exited with exit code 13
salloc: Relinquishing job allocation 60729163
PIPELINE_EXIT=13
```

`sacct`: `60729163 FAILED 13:0`, `60729163.0 env FAILED 13:0`.

### Root cause

The uv tools activation script
`/leonardo_work/IscrC_UNMASKED/epicazio/tools/uv/env` is mode 644
(non-executable) and its directory is prepended to `PATH`. The file is
sourced by `.bash_profile`, `.bashrc`, and `.profile`. `srun` resolves the
`env` exec token through `PATH` and calls `execve` on that file, which fails
with `Permission denied` (exit 13). A login `bash -l` skips non-executable
files and resolves `env` to `/usr/bin/env`; the failure appears only under
`srun`. Diagnosis transcript:
`artifacts/logs/leonardo/2026-10-16-phase2-prep-failure-diagnosis.log`.

### Fix

Pass the environment metadata inside a shell instead of using the `env` exec
token:

```bash
srun ... bash -c 'SOURCE_COMMIT=... SOURCE_STATE_SHA256=... bash scripts/remote/leonardo/run-phase2-prep.sh'
```

The uv layout and its permissions were not modified.

## Failure 2 (job 60729410 and the bprod rejection): QoS selection

### Failure (verbatim)

```text
salloc: Job 60729410 has exceeded its time limit and its allocation has been revoked.
srun: Job step aborted: Waiting up to 182 seconds for job step to finish.
[2026-10-16T17:37:26.017] error: *** STEP 60729410.0 ON lrdn3102 CANCELLED AT 2026-10-16T17:37:26 DUE TO TIME LIMIT ***
RETRY_EXIT=143
```

`remote-prep.log` tail: decode reached `train_decoded=80000` of 100000 when
the kill arrived (started 17:04:10, cancelled 17:37:26). The next submit on
`boost_qos_bprod` failed immediately:

```text
salloc: error: QOSMinCpuNotSatisfied
salloc: error: Job submit/allocate failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
```

Verified QoS policy (`sacctmgr show qos` output recorded in
`artifacts/logs/leonardo/2026-10-16-phase2-prep.log`):

```text
boost_qos_bprod MinTRES: cpu=2080,gres/gpu=260,mem=31280G,node=65  (MaxWall 1-00:00:00)
boost_qos_lprod MinTRES: (none)  MaxWall 4-00:00:00  MaxTRES node=8
boost_qos_dbg   MinTRES: (none)  MaxWall 00:30:00   MaxTRES node=8
```

### Root cause

Two independent selection errors. `boost_qos_dbg` caps wall time at 30
minutes; the decode step needs about 41 minutes at the observed pace
(80,000/100,000 in ~33 min). `boost_qos_bprod` declares a `MinTRES` of
`node=65`, so any `-N1` request violates the policy and is rejected. Only
`boost_qos_lprod` has no `MinTRES` and a wall time long enough for a
single-node job.

### Fix

Use `QOS=boost_qos_lprod TIME=02:00:00` for long single-node jobs. The
resource overrides are passed as environment prefixes on
`scripts/remote/run.sh` (that wrapper forwards them to `salloc`):

```bash
QOS=boost_qos_lprod TIME=02:00:00 NTASKS=1 CPUS_PER_TASK=8 GRES=gpu:1 \
  bash scripts/remote/run.sh leonardo "srun --mpi=pmi2 -n1 -c8 ..."
```

Retries 3-5 (jobs 60731702, 60755769, 60758615) all ran under `lprod` and
reached a scheduler outcome, not a wall-time kill. The stale `bprod` advice in
`scripts/remote/hosts/leonardo.conf` was corrected.

## Impact

No usable artifacts were produced by the failed attempts. The cancelled job
left a partial `$PREP` directory (`class_wnids.txt` plus a truncated
`train_images.npy`, 1,228,800,128 B); the prep script's non-empty-`$PREP`
guard then required deliberate `FORCE=1` on the retries.

## Avoidance rules

1. Never pass the `env` exec token to `srun`; use `bash -c 'VAR=... cmd'`.
2. Do not fix PATH shadowing by changing the uv layout or its permissions.
3. Check `MinTRES` before choosing a QoS; never assume a longer QoS class
   accepts a smaller job.
4. Treat `boost_qos_dbg` as a 30-minute box only; long single-node work uses
   `boost_qos_lprod`.
5. After a cancelled job, inspect `$PREP` for partial products before a rerun
   and pass `FORCE=1` deliberately.

## Evidence

- `artifacts/logs/leonardo/2026-10-16-phase2-prep.log` (both commands, QoS
  policy dump; lines 1-117).
- `artifacts/logs/leonardo/2026-10-16-phase2-prep-failure-diagnosis.log`
  (env-shadowing diagnosis, `sacct`, PATH and rc sourcing, uv script contents).
- `artifacts/logs/leonardo/2026-10-16-phase2-prep-failure-2-timeout.log`
  (timeout transcript and partial `$PREP`).
- [`COMMANDS.md`](../../artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md)
  notes 1-3.
- Campaign record:
  [2026-10-16-phase2-artifacts-complete](../devlog/03-static-scaling/2026-10-16-phase2-artifacts-complete.md).
- Operational guidance:
  [Leonardo cluster context](../context/leonardo.md).
