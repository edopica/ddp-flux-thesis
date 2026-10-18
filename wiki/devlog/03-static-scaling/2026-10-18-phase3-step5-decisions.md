# Phase 3 Step 5: hardened Phase 2 re-check PASS and decisions frozen

Date: 2026-10-18
Status: **PASS** for Step 5. The hardened Phase 2 remote re-check passes on
the real `$PREP`; OD-2, OD-3, OD-4, and OD-11 are recorded (OD-10 was recorded
2026-10-17). At the time of this entry, Step 6 was still pending.

**Update (2026-10-18):** Step 6 subsequently passed; Phase 3 is accepted.
See [the final-acceptance devlog](2026-10-18-phase3-final-acceptance.md).

## Phase 2 hardened re-check (real `$PREP`)

The 2026-10-16 remote PASS predates the R1-R3 hardening, so it was repeated
with the hardened checkers. Read-only inspection first confirmed that all
prepared files and fixtures match their frozen hashes and that the remote
checker hashes equal the local ones.

Compute-node run (new `LOG_DIR` outside the rsync-managed tree):

```bash
GRES=none NTASKS=1 CPUS_PER_TASK=8 TIME=00:30:00 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis bash -c \
   'LOG_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase2-hardened-recheck/logs \
    bash scripts/remote/leonardo/run-phase2-check.sh'"
```

| Item | Value |
|---|---|
| Job / node | `60951247` / `lrdn0171` |
| State / exit / elapsed | `COMPLETED` / `0:0` / `00:00:56` |
| UTC start | `2026-10-18T00:44:27Z` |
| Python / numpy / pillow / torch | 3.11.17 / 2.4.6 / 12.3.0 / 2.9.1+cu126 |
| Julia | 1.12.6 |

Result: 113 `CHECK ... PASS`, 0 FAIL, 0 SKIP.

```text
PHASE2_DATASET_CHECK PASS          (66 CHECK lines, exit 0)
PHASE2_JULIA_DATA_CHECK PASS       (47 CHECK lines, exit 0)
python_check_exit=0 julia_check_exit=0
PHASE2_REMOTE_CHECK PASS           (driver exit 0)
```

The hardened gates exercised the real data: `prep.dataset_file_set`
`required=6 listed=6`, `prep.schedule_files_sha256_bytes` `checked=2`,
`schedule.regenerated_all_epochs_bitwise` `bad_epochs=none` (50 epochs in
21.21 s), `bundle.gn_initial_values` 20/20, and all six `fixtures.*` manifest
gates with 35/35 array hashes. This closes the Phase 2 remote prerequisite.

Inputs were not modified. Key hashes: bundle
`a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317`,
`dataset_manifest.json` `584b9005...`, `schedule_manifest.json`
`0136a0a2...`, `data_fixture.npz` `6614a00b...`. Evidence:
[results and provenance](../../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/PROVENANCE.md),
[commands](../../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/COMMANDS.md),
[`SHA256SUMS`](../../../artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/SHA256SUMS).

## Decisions frozen (2026-10-18, authority: user)

| ID | Recorded value |
|---|---|
| OD-2 | AdamW `lr = 1e-3`, `wd = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8` |
| OD-3 | Decay only on the 20 convolution weights and classifier weight (21 leaves, `lambda = 1e-4`); no decay on the 40 GroupNorm `gamma`/`beta` arrays and classifier bias (41 leaves, `lambda = 0`) |
| OD-4 | 5 warm-up epochs, 50-epoch budget indexed by global update; batch 512 `W = 975`, `T = 9,750`; batch-256 alternate `W = 1,950`, `T = 19,500` retained for Phase 5 |
| OD-10 | Already recorded 2026-10-17: base `atol = 1e-5`, `rtol = 1e-4`; `grads.u1 atol = 1e-3`, `grads.u3 atol = 2e-3`, `m.u3 atol = 2e-4` |
| OD-11 | Explicit `Zygote.withgradient` + `Optimisers.update!` on `DistributedOptimizerState`; communication outside AD; 64 collectives per update (eager path adds the loss vote, 65) |

Files updated: `experiments/static-scaling/open-decisions.md` (status, table,
notes, four decision-record rows plus the consistency note),
`experiments/static-scaling/experiment.toml` (`[optimizer] status`,
`[schedule] status`, `flux_entry_point_status` set to recorded; schedule epoch
keys renamed from `*_proposed`), and
`experiments/static-scaling/specification.md` (sections 6.1, 6.2, 6.3, 10.4,
12 and the header). OD-7, OD-8, OD-9, OD-12, and OD-13 remain open with
unchanged proposed defaults. No values changed relative to the configuration
used by Steps 3c and 4.

Verification: Python `tomllib` and Julia `TOML` both parse `experiment.toml`;
the values read back as `lr=0.001`, `wd=0.0001`, `betas=[0.9,0.999]`,
`eps=1e-8`, `W/T=975/9750` (b512) and `1950/19500` (b256). The decision-record
table contains the OD-2/3/4/10/11 rows and no stale "proposed" wording for
them.

## Phase 3 status

Steps 1-4 are complete. Step 3c acceptance PASSed under the recorded OD-10
rule (job `60942933`), and Step 4's optimizer-contract probe PASSed (job
`60948431`). Phase 3 final acceptance is **not** claimed yet: Step 6 must
rerun the complete driver in a fresh root with the frozen precision policy
and tolerances, copy the evidence home, and run the independent NumPy
checker.

**Exact next action:** execute Step 6 of
`temp/static-graph-gpu-scaling-phase3-handoff.md`: sync the final source
snapshot, precompile on a compute node, run
`scripts/remote/leonardo/run-phase3-check.sh` in a fresh `OUT`/`LOG_DIR`,
require all four PASS summaries and driver exit 0, copy the artifacts home,
and run `check_phase3_reference.py` against the copied reference.
