# Phase 2 static-scaling data pipeline on Leonardo — retry-5 (PASS)

Date: 2026-10-16
Status: **PASS** — prep job 60758615 and check job 60758839 both exit 0.
Scope: prepare the canonical static-scaling dataset/fixtures on Leonardo and bring evidence home.

## Provenance

- Repository branch: `main` (thesis repo), HEAD `3088f61020137912af32d9c969e87ea9ae2a71c3`
  ("Record implementation-phases devlog entry", 2026-10-16T14:21:25+02:00).
- `COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3`
- `STATE=4d4fd2ac75f192c049d0c7eedfcdf4c1daf83ece770e66ff7bf181d76b651ad1`
  Recipe (from the thesis repo root):

  ```bash
  STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
    | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
  ```

  The state hash covers the modified `make_fixtures.py` (linear-bias fix) and the
  untracked Phase 2 scripts; it intentionally excludes `artifacts/`. Files under
  `~/projects/` on the remote resolved to this exact state (sync verified).

## Exact commands

### 1. Sync and remote verification

```bash
bash scripts/remote/sync_code.sh leonardo        # ended with "Sync complete!"
ssh leonardo 'grep -n "b\[:, None\]" ~/projects/ddp-flux-thesis/scripts/static_scaling/make_fixtures.py'
# 212:    y = w @ x + b[:, None]        (torch path)
# 304:    y = (w @ x + b[:, None]).astype(np.float32)   (numpy path)
```

### 2. Prep job (retry-5, lprod)

```bash
QOS=boost_qos_lprod TIME=02:00:00 NTASKS=1 CPUS_PER_TASK=8 GRES=gpu:1 \
  bash scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
  bash -c 'SOURCE_COMMIT=3088f61020137912af32d9c969e87ea9ae2a71c3 \
  SOURCE_STATE_SHA256=4d4fd2ac75f192c049d0c7eedfcdf4c1daf83ece770e66ff7bf181d76b651ad1 \
  FORCE=1 bash scripts/remote/leonardo/run-phase2-prep.sh'"
```

No `env` exec token; the inner `bash -c 'VAR=... cmd'` form is mandatory (see notes).

### 3. Check job

```bash
NTASKS=1 CPUS_PER_TASK=8 GRES=gpu:1 \
  bash scripts/remote/run.sh leonardo \
  'srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-phase2-check.sh'
```

### 4. Prepared-dataset sha256 (login node, read-only)

```bash
ssh leonardo 'cd /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared && \
  sha256sum train_images.npy train_labels.npy val_images.npy val_labels.npy \
  class_wnids.txt permutations.npy augmentation.npy stats.json \
  initial_bundle.npz initial_bundle.manifest.json dataset_manifest.json \
  schedule_manifest.json && ls -lA'
```

### 5. Evidence copy-back

```bash
rsync -avz leonardo:projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/ \
  artifacts/baselines/static-scaling/2026-10-16-phase2/
rsync -avz leonardo:projects/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-16-phase2/ \
  artifacts/logs/static-scaling/2026-10-16-phase2/
rsync -avz leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/stats.json \
  leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/dataset_manifest.json \
  leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/schedule_manifest.json \
  leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/initial_bundle.manifest.json \
  artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/
# addition for local bundle hashing (original STEP 5): initial_bundle.npz from $PREP
rsync -avz leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/initial_bundle.npz \
  artifacts/baselines/static-scaling/2026-10-16-phase2/
```

## Jobs

| Job | What | Node | QoS | Start | End | Elapsed | State |
|---|---|---|---|---|---|---|---|
| 60758615 (+ .0 bash) | prep retry-5 | lrdn0018 | boost_qos_lprod | 2026-10-16 23:33:16 (step 23:33:21) | 23:35:11 | 00:01:55 (step 00:01:50) | COMPLETED 0:0 |
| 60758839 (+ .0 bash) | acceptance check | lrdn0011 | boost_qos_dbg (default) | 2026-10-16 23:36:39 (step 23:36:45) | 23:38:54 | 00:02:15 (step 00:02:09) | COMPLETED 0:0 |

Step times from the remote logs:

- prep: 1/4 `prepare_dataset` 1m18.382s; 2/4 `make_schedule` 22.600s;
  3/4 `make_initial_bundle` 2.432s; 4/4 `make_fixtures` 4.544s.
- check: 1/2 `check_dataset.py` 11.432s; 2/2 `check_fixtures_data.jl` 45.926s
  (3 NPZ-stack dependencies precompiled once on first use inside the job).

Note: the prep SSH stdio stream dropped mid-run (`Connection reset by peer`,
exit 255); the Slurm job kept running and completed. The script's
`=== run-phase2-prep: EXIT $status ===` line is printed outside the remote log,
so the exit status is evidenced by `sacct` (COMPLETED 0:0) and the in-log
`=== run-phase2-prep: DONE ===` marker.

## Verbatim success lines

Prep (remote-prep.log, retry-5 section):

```text
DATASET_PREP train_mean=[0.48023695 0.44806704 0.39750364]
DATASET_PREP train_std=[0.27643642 0.2688633  0.28158993]
DATASET_PREP val_decoded=10000
DATASET_PREP containers_train=[('JPEG', 100000)] containers_val=[('JPEG', 10000)]
DATASET_PREP wrote /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/dataset_manifest.json files=6 pixel_count_per_channel=409600000
DATASET_PREP DONE
SCHEDULE wrote /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/schedule_manifest.json epochs=50
SCHEDULE DONE
BUNDLE wrote /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/initial_bundle.npz arrays=62
BUNDLE elements=11271432 bundle_hash=a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23
BUNDLE verified name_set=true per_array_sha256=true bundle_hash=true torch=2.9.1+cu126 numpy=2.4.6
BUNDLE DONE
FIXTURES wrote .../data_fixture.npz arrays=9
FIXTURES wrote .../schedule_fixture.npz arrays=5
FIXTURES wrote .../linear_fixture.npz arrays=7
FIXTURES wrote .../kernel_fixture.npz arrays=14
FIXTURES wrote .../fixture_manifest.json arrays_total=35
FIXTURES verified files=4 arrays=35 sha256=true
FIXTURES DONE synthetic=False
=== run-phase2-prep: DONE ===
```

Check (remote-check.log):

```text
PHASE2_DATASET_CHECK PASS
PHASE2_JULIA_DATA_CHECK PASS
=== summary ===
python_check_exit=0 julia_check_exit=0
PHASE2_REMOTE_CHECK PASS
=== run-phase2-check: EXIT 0 (log: .../remote-check.log) ===
```

(`PHASE2_PREP FAIL` does not occur in the retry-5 section of remote-prep.log;
the one occurrence at line 79 belongs to the retry-4 failure.)

## Fixture files (local `artifacts/baselines/static-scaling/2026-10-16-phase2/`)

The produced fixture set is four npz plus `fixture_manifest.json`
(`FIXTURES verified files=4 arrays=35`; Julia/Python checks confirm). The
canonical `initial_bundle.npz` was copied here from `$PREP` for local hashing.
All hashes match `fixture_manifest.json` and the remote prepared dataset.

| File | Bytes | SHA-256 |
|---|---|---|
| `data_fixture.npz` | 1,821,284 | `6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3` |
| `schedule_fixture.npz` | 3,797,486 | `aba66a8ef7bbc82fc81db8cd629092b917cd7d85efafed82553f853c45352662` |
| `linear_fixture.npz` | 2,238 | `8efa5278981b2b11842d2a701b6f81789c976e033a678a0322e961ecd8f1fa2d` |
| `kernel_fixture.npz` | 7,114 | `3b3795f6c6ff2ec68198a645cc65f4ba97e0b6438da61e6cfd2ad564ff441f3d` |
| `initial_bundle.npz` | 45,103,060 | `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317` |
| `fixture_manifest.json` | 10,643 | `51448d5905c5dfd9a47b945d96102d19ea55073fbfe9f249fd268cca03455a65` |

## Prepared dataset hashes

File: `artifacts/logs/leonardo/2026-10-16-phase2-prepared-sha256.txt`
(retry-5 section appended). Key values:

- `initial_bundle.npz` = `a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`
- `initial_bundle.manifest.json` = `4996517497edfd66cf259bb7d51d2ca8f0d44076bc958438af33df5920b0c571`
- `dataset_manifest.json` = `584b900565ebfd02d6d323db393425b6263170f55ebf6a928a49fb5a956b875a`
- `schedule_manifest.json` = `0136a0a2d1d768bf2c2a8544bd10f7893b533911bad37d77b09a1ae277121295`
- `stats.json` = `6bf4de44cb7fd035812917e2c34a8b25cdbc503be9b947885bd33e0b2fc324bc`
- data arrays (`train_images.npy`, `train_labels.npy`, `val_images.npy`,
  `val_labels.npy`, `class_wnids.txt`, `permutations.npy`, `augmentation.npy`)
  are byte-identical to retry-4; manifests differ only in the embedded
  `source_state_sha256`.

Evidence copies: `artifacts/logs/static-scaling/2026-10-16-phase2/`
(`remote-prep.log`, `remote-check.log`, `prepared-manifests/`).

## Remote paths

- Thesis repo: `/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis`
- Prepared dataset `$PREP`: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared`
- Fixtures `$FIX`: `$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2`
- Remote logs: `$THESIS/artifacts/logs/static-scaling/2026-10-16-phase2`
- Dataset root: `/leonardo_work/IscrC_UNMASKED/epicazio/datasets/tiny-imagenet-200`
- Python venv: `/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling` (torch 2.9.1+cu126, numpy 2.4.6, pillow 12.3.0)
- Julia: `/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6`, depot `/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot`

## Operation notes

1. **Never use the `env` exec token under `srun` on Leonardo.** The uv tools
   activation script (`$WORK/epicazio/tools/uv/env`, mode 644, on `PATH`,
   sourced by `.bashrc`) shadows `/usr/bin/env` and the job dies in ~2 s.
   Use `bash -c 'VAR=... cmd'`; set wrapper resources as environment prefixes
   on `scripts/remote/run.sh`.
2. **QoS for long 1-node jobs:** `QOS=boost_qos_lprod TIME=02:00:00`.
   `boost_qos_dbg` caps at 30 min (killed job 60729410 mid-decode);
   `boost_qos_bprod` rejects `-N1` (`QOSMinCpuNotSatisfied`, MinTRES node=65).
3. **Failure history:**
   - 60729163 — died in 2 s: exec'd `env` hit the non-executable uv activation
     script (see note 1).
   - 60729410 — killed at the 30-min `boost_qos_dbg` cap mid-decode.
   - 60731702 — completed but prep FAILED with `NameError: name 'std64' is not
     defined` in `prepare_dataset.py`; the prep script still printed DONE and
     exited 0 (masked failure).
   - 60755769 — steps 1-3 PASS; step 4/4 `make_fixtures` failed with
     `RuntimeError: The size of tensor a (5) must match the size of tensor b (3)`
     at `make_fixtures.py:212` (`y = w @ x + b`; bias `(3,)` vs `(3, 5)`).
   - Three code bugs fixed before this retry: (a) `env` token usage replaced by
     `bash -c 'VAR=... cmd'`; (b) `std64` computed from `var64` before the
     float32 cast; (c) linear-fixture bias broadcast fixed to `b[:, None]` in
     the torch and numpy paths, and the prep driver now aborts with
     `PHASE2_PREP FAIL` + non-zero exit on any failed step.
   - Operational incident this retry: the SSH stdio stream dropped
     (`Connection reset by peer`, exit 255) mid-decode; the Slurm job survived
     to COMPLETED 0:0, so this is transport-only, not a job failure.
4. **Decode time varies with `$WORK` latency:** `prepare_dataset` took 23m52s
   in retry-4 and 1m18s in retry-5 on the same node (`lrdn0018`), dominated by
   Lustre/`$WORK` filesystem latency, not CPU.
5. **Local preflight (orchestrator, torch 2.14.1+cpu on a full-size fake
   dataset):** all four fixture builders completed with sha256 round-trip
   verification; `make_initial_bundle` reproduced the SAME
   `bundle_hash=a7632ca90c9c51cd022b3c82e0dc81391608f317c1afbd17889deeb2d1dcce23`
   as Leonardo's torch 2.9.1+cu126 run (cross-version determinism); local Julia
   `check_all.jl` passed `PHASE2_JULIA_DATA_CHECK PASS` +
   `PHASE2_JULIA_OPERATOR_CHECK PASS` (75 checks, exit 0) and
    `check_dataset.py` passed `PHASE2_DATASET_CHECK PASS`. The remote retry-5
    reproduced the prep markers, `PHASE2_DATASET_CHECK PASS` and
    `PHASE2_JULIA_DATA_CHECK PASS`; the remote check job is NPZ-only and does
    NOT run the Flux operator check. The operator check was run separately,
    locally, against the fixtures copied back from retry-5 (75 checks, exit 0),
    and independently re-run by the verification agent (see
    `reports/static-scaling-phase2-verification.md`).
6. The acceptance check used the default `boost_qos_dbg` (fits in 30 min) and
   precompiled 3 NPZ-stack deps inside the job (first use of the tiny Julia
   env); no Julia precompile concerns for the Flux env apply here.

## Next actions

1. Operator/orchestrator: fold this campaign into the wiki docs (devlog,
   current context) — not done by this runbook by instruction.
2. Proceed to the next Phase 2 step (training-side consumption of the fixtures
   / single-GPU baseline) with the pinned COMMIT/STATE above.
