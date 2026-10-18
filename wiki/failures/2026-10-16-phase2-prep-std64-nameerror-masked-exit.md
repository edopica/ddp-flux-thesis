# Phase 2 prep: `std64` NameError after full decode, masked by the prep driver

Date: 2026-10-16
Scope: Leonardo job 60731702 (`run-phase2-prep.sh`, prep step 1/4), static
scaling Phase 2, retry-3
Status: FIXED and verified — fail-fast demonstrated in retry-4, full pipeline
PASS in retry-5

## Symptom

After a full decode of the training split, `prepare_dataset.py` raised
`NameError: name 'std64' is not defined`. The remote job still looked
successful: the driver printed `=== run-phase2-prep: DONE ===` and exited 0
(masked failure).

## Failed command

Retry-3 on `lrdn3432` (lprod):

```bash
QOS=boost_qos_lprod TIME=02:00:00 NTASKS=1 CPUS_PER_TASK=8 GRES=gpu:1 \
  bash scripts/remote/run.sh leonardo "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
  bash -c 'SOURCE_COMMIT=3088f610... SOURCE_STATE_SHA256=f8574e36... FORCE=1 bash scripts/remote/leonardo/run-phase2-prep.sh'"
```

## Failure (verbatim)

```text
NameError: name 'std64' is not defined
```

The traceback appeared after the 100,000th `DATASET_PREP train_decoded=` line
(of 100000). The reported decode duration was ~36m54s. The retry-3 remote log
was overwritten when retry-4 started, so the retained
`artifacts/logs/leonardo/2026-10-16-phase2-prep.log` retry-3 section stops at
`DATASET_PREP train_decoded=20000`. The error text and the masked-exit
sequence are recorded verbatim in
[`COMMANDS.md`](../../artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md)
note 3.

## Root cause

Two defects.

1. **Statistics computed late.** In `prepare_dataset.py` `build_train`, the
   float32 normalization constants were built from `std64` on the path that
   runs only after all 100,000 train images decode. The
   `std64 = sqrt(var64)` step was not executed before those casts. Earlier
   attempts never reached this path: retry-2 died at the 30-minute cap
   mid-decode, so retry-3 was the first run to complete the full decode and
   expose the defect.
2. **Masked exit.** `run()` in `run-phase2-prep.sh` executed the four steps
   without a fail-fast contract. The pipeline status was the `tee` status, so
   a crashed step did not stop the run and did not affect the exit code. The
   driver printed DONE and exited 0.

## Fix

- `scripts/static_scaling/prepare_dataset.py`: compute
  `std64 = np.sqrt(np.maximum(var64, 0.0))` before the float32 casts
  (line 157), so `std32` is defined on all paths.
- `scripts/remote/leonardo/run-phase2-prep.sh`: inside `run()`,
  `set -e` plus an ERR trap (lines 113-114):

  ```bash
  # Any failing step must abort the run with a non-zero status: without this,
  # a crashed step was masked and the script printed DONE/exit 0 (2026-10-16).
  set -e
  trap 'echo "PHASE2_PREP FAIL (step exited non-zero, see log)"' ERR
  ```

  The outer wrapper preserves the pipeline status:
  `status=${PIPESTATUS[0]}`, `=== run-phase2-prep: EXIT $status ===`,
  `exit "$status"`.

## Verification

- Fail-fast contract, retry-4 (job 60755769): steps 1-3 PASS, then step 4/4
  failed and the driver printed `PHASE2_PREP FAIL (step exited non-zero, see
  log)` and `=== run-phase2-prep: EXIT 1 (log: ...) ===`; `sacct` shows
  `FAILED 1:0`. A masked exit is no longer possible.
- `std64` fix, retry-5 (job 60758615): `DATASET_PREP train_mean=[0.48023695
  0.44806704 0.39750364]`, `DATASET_PREP train_std=[0.27643642 0.2688633
  0.28158993]`, `DATASET_PREP DONE`, full prep COMPLETED 0:0; all acceptance
  markers PASS.
- Independent verification matched `stats.json` bitwise to the retry-5 log
  values and the fixture normalization arrays, and the recorded float64
  decimals round to the stored float32 constants (a full-pixel re-decode was
  out of scope). See
  [verification report](../../reports/static-scaling-phase2-verification.md).
- Campaign record:
  [2026-10-16-phase2-artifacts-complete](../devlog/03-static-scaling/2026-10-16-phase2-artifacts-complete.md).

## Impact

Retry-3 spent ~37 minutes on a decode whose statistics were lost. Because the
driver exited 0, the failure was recorded rather than surfaced by the job
status. The retry-4 page
([linear fixture broadcast](2026-10-16-phase2-prep-make-fixtures-linear-broadcast.md))
documents the failure the fixed driver then caught.

## Avoidance rules

1. Every batch driver must abort on the first failing step and propagate its
   exit code (`set -e` + ERR trap + `PIPESTATUS`); never let `tee` status
   become the pipeline status.
2. Compute every derived value before the first consumer, and keep the
   statistics block free of forward references.
3. Keep per-attempt remote logs: retry-3's log was overwritten, so its
   traceback survives only in `COMMANDS.md`.
4. Do not trust `DONE` without checking each step marker and the exit code.
