# Phase 2 prep retry-4: dataset/schedule/bundle PASS, fixtures blocked by a linear-shape bug

Date: 2026-10-16
Scope: static-network GPU scaling, Phase 2 prepared dataset on Leonardo, retry-4 (lprod)
Status: RESOLVED in retry-5 — this entry records the retry-4 failure state

**RESOLVED in retry-5 (2026-10-16):** the linear-bias fix (`b[:, None]` in the
torch and numpy builders) and the audited kernel builder produced the full
fixture set; prep job 60758615 and check job 60758839 both COMPLETED 0:0 with
`PHASE2_DATASET_CHECK`/`PHASE2_JULIA_DATA_CHECK`/`PHASE2_REMOTE_CHECK` PASS.
Full campaign record:
[2026-10-16-phase2-artifacts-complete](2026-10-16-phase2-artifacts-complete.md);
evidence
[`COMMANDS.md`](../../../artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md).

## Provenance and sync

- Local repo HEAD: `3088f61020137912af32d9c969e87ea9ae2a71c3`.
- Dirty-tree state hash (recipe: `git diff HEAD` + sha256 of sorted untracked
  files excluding `artifacts/`, then sha256 of the concatenation):
  `3e9ba4d8013ec67383454d86d39ec0caaae2616e43cc5ea4cc963843d6e72561`.
- `bash scripts/remote/sync_code.sh leonardo` completed with `Sync complete!`
  (log `artifacts/logs/leonardo/2026-10-16-phase2-sync.log`).
- Remote verification: `prepare_dataset.py` line 157 has
  `std64 = np.sqrt(np.maximum(var64, 0.0))`; `run-phase2-prep.sh` line 114 has
  the `trap 'echo "PHASE2_PREP FAIL ..."' ERR`; the protected
  `scripts/static_scaling/julia-env/Manifest.toml` is intact.

## Prep job

Exact command (no `env` token; `bash -c 'VAR=... cmd'` quoting):

```bash
QOS=boost_qos_lprod TIME=02:00:00 NTASKS=1 CPUS_PER_TASK=8 GRES=gpu:1 \
  bash scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
  bash -c 'SOURCE_COMMIT=<commit> SOURCE_STATE_SHA256=<state> FORCE=1 \
  bash scripts/remote/leonardo/run-phase2-prep.sh'"
```

- Job 60755769 (salloc; step 60755769.0 bash), node `lrdn0018`,
  `boost_qos_lprod`, start 22:59:11, end 23:24:45, elapsed `00:25:34`,
  state FAILED `1:0`.
- 1/4 `prepare_dataset`: PASS, `real 23m52.511s` (dominant; `$WORK` latency).
  `DATASET_PREP train_mean=[0.48023695 0.44806704 0.39750364]`,
  `DATASET_PREP train_std=[0.27643642 0.2688633  0.28158993]`,
  `DATASET_PREP containers_train=[('JPEG', 100000)] containers_val=[('JPEG', 10000)]`,
  `DATASET_PREP DONE`.
- 2/4 `make_schedule`: PASS (`SCHEDULE ... epochs=50`, `SCHEDULE DONE`), 22.4 s.
- 3/4 `make_initial_bundle`: PASS (`BUNDLE ... arrays=62`,
  `bundle_hash=a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23`,
  `BUNDLE verified ... true`, `BUNDLE DONE`), 11.4 s.
- 4/4 `make_fixtures`: FAIL, 2.4 s.
  `RuntimeError: The size of tensor a (5) must match the size of tensor b (3)
  at non-singleton dimension 1` at `make_fixtures.py:212` (`y = w @ x + b`)
  inside `build_linear_fixture_torch`; bias is `(3,)` but must broadcast to
  `(3, 5)`. The fixed prep driver caught the failure:
  `PHASE2_PREP FAIL (step exited non-zero, see log)`,
  `run-phase2-prep: EXIT 1`. Details:
  [failure record](../../failures/2026-10-16-phase2-prep-make-fixtures-linear-broadcast.md).

## What was and was not done

- Done: sync, provenance hashes, prep job, remote prep log copy-back,
  prepared-dataset sha256 (`artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt`;
  `initial_bundle.npz` = `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`).
- NOT done (blocked by the failure): acceptance check job, fixture copy-back,
  `COMMANDS.md`. `$FIX` on Leonardo is empty; no fixture npz exists. No code
  was modified.

## Operation notes (carried forward)

1. Never use the `env` exec token under `srun` on Leonardo: the uv tools
   activation script (`$WORK/epicazio/tools/uv/env`, mode 644, on `PATH`,
   sourced by `.bashrc`) shadows `/usr/bin/env`. Use `bash -c 'VAR=... cmd'`.
2. Long 1-node jobs need `QOS=boost_qos_lprod TIME=02:00:00`: `boost_qos_dbg`
   caps at 30 min; `boost_qos_bprod` rejects `-N1` (`QOSMinCpuNotSatisfied`,
   MinTRES node=65).
3. Job history: 60729163 (`env` token failure), 60729410 (dbg 30-min cap),
   60731702 (prepare_dataset `std64` NameError + masked exit 0), 60755769
   (this run: steps 1-3 PASS, make_fixtures linear broadcast bug).

## Exact next action

Fix `build_linear_fixture_torch` in `scripts/static_scaling/make_fixtures.py`
(bias broadcastable to `(3, 5)`, e.g. `(3, 1)`), audit
`build_kernel_fixture_torch` for the same pattern (never exercised), then
sync and re-run the prep. Prepared steps 1-3 products on Leonardo are already
complete and verified, so a non-`FORCE` rerun (if the script supports reuse)
would only need step 4; otherwise `FORCE=1` redoes all four.
