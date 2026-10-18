# Commands — hardened Phase 2 real-PREP re-check

All commands ran from the thesis repository root unless noted.
Date: 2026-10-18 (UTC run start 2026-10-18T00:44:27Z).

## 1. Read-only inspection of the real `$PREP` and fixtures

```bash
ssh leonardo 'hostname; ls -la --time-style=full-iso \
  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared/; \
  ls -la --time-style=full-iso \
  ~/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/; \
  cd /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared && \
  sha256sum initial_bundle.npz initial_bundle.manifest.json dataset_manifest.json \
    schedule_manifest.json stats.json train_images.npy train_labels.npy \
    val_images.npy val_labels.npy class_wnids.txt permutations.npy augmentation.npy; \
  sha256sum ~/projects/ddp-flux-thesis/scripts/static_scaling/check_dataset.py \
    ~/projects/ddp-flux-thesis/scripts/static_scaling/check_fixtures_data.jl \
    ~/projects/ddp-flux-thesis/scripts/static_scaling/check_all.jl \
    ~/projects/ddp-flux-thesis/scripts/remote/leonardo/run-phase2-check.sh'
```

Result: all required files present; fixture manifest lists four files and 35
arrays; all file hashes match the frozen records (see `PROVENANCE.md`). The
four remote checker hashes equal the local hashes, so the hardened sources
are in place. No file was modified.

## 2. Compute-node hardened check (new LOG_DIR outside the remote tree)

```bash
GRES=none NTASKS=1 CPUS_PER_TASK=8 TIME=00:30:00 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase2-hardened-recheck/logs \
    bash scripts/remote/leonardo/run-phase2-check.sh'" \
  2>&1 | tee artifacts/logs/leonardo/2026-10-18-phase2-hardened-recheck.log
```

Job `60951247`, node `lrdn0171`, `boost_qos_dbg`, 1 node, 1 task, 8 CPUs,
`--gres=none`, `COMPLETED 0:0`, elapsed `00:00:56`. No `Precompiling
packages...` line appeared.

Results:

```text
PHASE2_DATASET_CHECK PASS
PHASE2_JULIA_DATA_CHECK PASS
python_check_exit=0 julia_check_exit=0
PHASE2_REMOTE_CHECK PASS
=== run-phase2-check: EXIT 0 ===
```

Python `check_dataset.py`: 66 CHECK lines, all PASS (including
`prep.dataset_files_sha256_bytes`, `prep.schedule_files_sha256_bytes`,
`schedule.regenerated_all_epochs_bitwise`, and the 20/20 GroupNorm check).
Julia `check_fixtures_data.jl`: 47 CHECK lines, all PASS (including the six
`fixtures.*` manifest gates). 113 PASS, 0 FAIL, 0 SKIP.

## 3. Job record

```bash
ssh leonardo 'sacct -j 60951247 --format=JobID,State,ExitCode,Elapsed,Start,End,NodeList -P'
```

```text
60951247|COMPLETED|0:0|00:00:56|2026-10-18T02:44:21|2026-10-18T02:45:17|lrdn0171
60951247.extern|COMPLETED|0:0|00:00:56|2026-10-18T02:44:21|2026-10-18T02:45:17|lrdn0171
60951247.0|COMPLETED|0:0|00:00:50|2026-10-18T02:44:27|2026-10-18T02:45:17|lrdn0171
```

## 4. Copy evidence home

```bash
mkdir -p artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck
rsync -avz \
  leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase2-hardened-recheck/logs/' \
  artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/
cp artifacts/logs/leonardo/2026-10-18-phase2-hardened-recheck.log \
  artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/local-console.log
```

## 5. Verification

```bash
cd artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck
sha256sum remote-check.log local-console.log
grep -E 'PHASE2_(DATASET_CHECK|JULIA_DATA_CHECK|REMOTE_CHECK)|python_check_exit' remote-check.log
sha256sum -c SHA256SUMS
```

Hash values and check lines: see `PROVENANCE.md` and `SHA256SUMS`.

## 6. Outcome

The hardened Phase 2 real-`$PREP` re-check PASSes; both checkers exit 0 and
the driver exits 0. This closes the Phase 2 remote prerequisite for the
Phase 3 handoff. `$PREP` and the fixture artifacts were not modified; no
commit was made.
