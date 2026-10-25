# Bounded W=1 cache diagnostic — 2026-10-25

## Authorization and scope

The user authorized delegated harness preparation followed by a Leonardo run.
Three agents implement the opt-in runner instrumentation, independent artifact
checker with negative controls, and compute-node launcher. Root reviews and
runs the integrated harness. No upstream Flux source change is authorized or
needed for this diagnostic.

The planned matrix is current Flux and PyTorch at W=1, batch 256, epoch 0,
20 discarded warmup updates, reset from the shared initial bundle, and 20
measured updates, with two sequential fresh-process repeats each. Retain the
4-CPU allocation policy. Use the full-dataset loader/training path from the
accepted campaign, not the earlier fixture-only memory probe.

The diagnostic is separate from the completed 390-update capacity campaign.
Cache release is outside the timer and only after the measured updates.
Capture initialization, warmup, reset, post-training, post-GC, and post-cache-
release snapshots with synchronized process and allocator counters. Hold the
model, optimizer and dataset consistently during the final lifecycle samples.
Record unavailable counters explicitly and verify loaded API semantics. A
point sample is not a transient peak or a pure live-tensor comparison.

## Remote workflow

Loaded `/home/kurapica/.config/opencode/skills/remote/SKILL.md` and read
`wiki/context/leonardo.md` before remote commands. The server page explicitly
selects `precompile-phase3.sh` for the active Flux test environment rather than
the generic root-env precompile target. Only the thesis harness needs deployment;
keep remote Flux worktrees and prior campaign evidence intact. The launcher
must precompile current Flux on a compute node before measurements.

Read-only availability command (successful after required network escalation):

```bash
ssh -F /home/kurapica/.ssh/config -o BatchMode=yes -o ConnectTimeout=15 leonardo 'hostname; date -u; squeue -u epicazio -o "%.18i %.22j %.8T %.10M %.24R"'
```

Observed `login02.leonardo.local`, `2026-10-25 09:22:21 UTC`, no active jobs.
Prepared dataset, Julia, PyTorch venv and current Flux source exist; the home
and work filesystems have sufficient free space. This is availability evidence,
not a measurement. Preparation failures are recorded in
`wiki/failures/2026-10-25-cache-diagnostic-preparation.md`.

## Current status

Completed: job `62241616`, `COMPLETED 0:0`, remote/local artifact checks PASS,
118/118 original evidence hashes match. See the completion record below.

**Exact next action:** integrate the verified capacity and cache findings into
the presentation; keep the OD-14 deterministic sensitivity separate.

## Local readiness

All three agents completed implementation and independent review with no open
blocking findings. Root repeated these checks successfully:

```bash
python3 -m unittest discover -s scripts/static_scaling -p 'test_cache_diagnostic.py' # 15/15
python3 -m unittest discover -s scripts/static_scaling -p 'test_full_campaign.py' # 8/8
python3 scripts/static_scaling/check_full_campaign.py --self-test # 37 checks
python3 -m py_compile scripts/static_scaling/cache_diagnostic.py scripts/static_scaling/timing_torch.py scripts/static_scaling/check_cache_diagnostic.py
bash -n scripts/remote/leonardo/run-static-cache-diagnostic.sh
git diff --check
```

Julia syntax validation recursively rejected `Expr(:error)`/`:incomplete` for
`cache_diagnostic.jl` and `full_dataset_timing.jl`; both passed. Helper mock
checks and three launcher resource rejection controls passed. The Git parent
is `976db1ffbc63472d306412b620d856a0cfd9cbc9`; these harness changes were
uncommitted at measurement time and are identified by deployment/source hashes. One GPU is
requested (previous capacity campaign exposed four); this is W=1 device 0
with the same batch/CPU/training policy. No throughput claim is made.

## Deployment and submission

Thesis-only deployment succeeded with all existing exclusions; neither Flux
worktree changed:

```bash
rsync -a --exclude-from=.rsyncignore -e 'ssh -F /home/kurapica/.ssh/config -o BatchMode=yes -o ConnectTimeout=15' ./ leonardo:projects/ddp-flux-thesis/
```

Remote SHA-256 matches local for launcher, checker, both diagnostic helpers,
and both runner integration files. Submission returned job **62241616**:

```bash
ssh -F /home/kurapica/.ssh/config -o BatchMode=yes -o ConnectTimeout=15 leonardo 'cd "$HOME/projects/ddp-flux-thesis" && EXPECTED_THESIS_COMMIT=976db1ffbc63472d306412b620d856a0cfd9cbc9 sbatch --parsable --export=ALL -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=4 --gres=gpu:1 --time=02:00:00 --job-name=static-cache-diagnostic --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-cache-%j.log scripts/remote/leonardo/run-static-cache-diagnostic.sh'
```

Remote output is
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-cache-62241616`.
Do not sync further while this job uses the deployed source.

## Completion and independent local verification

Job `62241616` completed `COMPLETED 0:0`, elapsed `00:14:00`, `lrdn3215`,
4 allocated CPUs and one A100-SXM-64GB. Scheduler start/end are
`2026-10-25T10:27:41` / `10:41:41` in its local representation; the recorded
Slurm Unix start is `2026-10-25T09:27:41+00:00` UTC. All four runs passed,
followed by `CACHE_DIAGNOSTIC_CHECK PASS`, `CACHE_DIAGNOSTIC PASS`,
`CACHE_DIAGNOSTIC_DRIVER_EXIT=0`. No OOM, timeout or run failure occurred.

Fetch commands:

```bash
rsync -a -e 'ssh -F /home/kurapica/.ssh/config -o BatchMode=yes -o ConnectTimeout=15' leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-cache-62241616/ artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616/
rsync -a -e 'ssh -F /home/kurapica/.ssh/config -o BatchMode=yes -o ConnectTimeout=15' leonardo:/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-cache-62241616.log artifacts/logs/leonardo/2026-10-25-cache-diagnostic-62241616.log
python3 scripts/static_scaling/check_cache_diagnostic.py --root artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616
ssh -F /home/kurapica/.ssh/config -o BatchMode=yes -o ConnectTimeout=15 leonardo 'sacct -j 62241616 --parsable2 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,AllocTRES; sha256sum /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-cache-62241616.log'
```

The local checker passes independently. Python SHA-256 validation over each
entry in the fetched portable `SHA256SUMS` verified 118/118 original files.
The remote/local Slurm log SHA matches
`3a196a0cd2055092463a497faa554c78e46f485efc3afe0c3f6d00d4aa124da5`.
All 26 Flux src and 10 ext files match the archived capacity campaign snapshot.
Git-free expected commit labels remain declared metadata; actual source
snapshots provide the independent content comparison.

Flux after GC: `49,792` / `49,760 MiB` process; after reclaim: `1,270` /
`1,334 MiB`. Released: `48,522` / `48,426 MiB` (97.45% / 97.32%). Pool used
falls from `21,533.365 MiB` raw post-training to `128.991 MiB` after GC,
while reservation stays high until reclaim. PyTorch after GC is `10,894 MiB`
and after release `1,786 MiB` in both repeats, releasing `9,108 MiB` (83.61%).
Every state invariant passes. Raw trace density passes (Flux 18/39 training
samples; PyTorch 19/18; maximum trace gaps below 0.29 s).

An independent subagent reviewed all records, API semantics and archived
source consistency: PASS, no blocking issue. Both frameworks cache memory;
Flux's large end-of-window footprint is predominantly reclaimable pool
reservation plus delayed collection of allocations. W=1 shows DDP is
unnecessary for this effect; it does not bound multi-rank overhead, prove
absence of leaks or measure exact transient activation peaks. PyTorch retains
gradients while Flux does not, and post-release residuals include runtime
and pool overhead; do not infer intrinsic tensor efficiency from these values.
State hashing between the raw snapshot and explicit GC may trigger ordinary
Julia GC. Keep the original operational capacity peaks unchanged and do not
insert per-step reclaim into the normal training path.

Detailed evidence and point-counter tables:
`artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616/RESULTS.md`.
Context, dashboard/progress table and devlog index updated. Harness and docs were
uncommitted at the end of the measurement session; no Flux source modification.

**Exact next action:** integrate the capacity and cache findings into the
presentation, with separate labels for sampled operational peak and lifecycle
point counters; the OD-14 deterministic sensitivity remains separately required.

## Documentation and publication follow-up

The user requested documentation updates, commit and push, and asked why
cache release is not the default. Added
`reports/static-scaling-memory-interpretation.md`, linked it from the dashboard,
quickstart, current context and diagnostic results, and qualified leak and
outlier-cause claims in live summaries. Official CUDA.jl memory documentation
and PyTorch 2.9 CUDA documentation explain allocator reuse and automatic
pressure reclamation; repeated cache release would discard reuse, with cost
not measured by this post-training diagnostic. No training code default or
upstream Flux source changed in this follow-up.

Commit scope follows `2026-10-22-artifact-commit-scope.md`: include both completed
campaigns' text evidence, raw sample traces, source snapshots, hashes, harness,
checker/tests and documentation. Existing ignore rules keep the large `.npz`
arrays and Python caches local; do not modify original hashed source snapshots.

**Exact next action:** integrate the verified memory interpretation into the
supervisor presentation; retain OD-14 deterministic sensitivity separately.

Pre-publication verification passed: 15 diagnostic tests, 8 campaign tests,
`CACHE_DIAGNOSTIC_CHECK PASS`, `FULL_CAMPAIGN_ARTIFACT_CHECK PASS`, original
118-entry diagnostic SHA manifest, and `git diff --check`. `git fetch origin`
succeeded and `git rev-list --left-right --count HEAD...origin/main` was `0 0`
before the publication commit. The completed full-campaign report also now
qualifies leak claims, timing-outlier causes and long-run numerical differences;
original source snapshots and their hashed evidence remain unchanged.

Validation commands:

```bash
python3 -m unittest discover -s scripts/static_scaling -p 'test_cache_diagnostic.py'
python3 -m unittest discover -s scripts/static_scaling -p 'test_full_campaign.py'
python3 scripts/static_scaling/check_cache_diagnostic.py --root artifacts/logs/static-scaling/2026-10-25-cache-diagnostic-62241616
python3 scripts/static_scaling/check_full_campaign.py campaign --root artifacts/logs/static-scaling/2026-10-25-campaign-62195500
git diff --check
git fetch origin
git rev-list --left-right --count HEAD...origin/main
```

Publication targets the existing `main` branch and `origin/main`, without
changing the original measurement identity or the baseline training defaults.

Staged publication scope is 448 files / approximately 7.3 MB before this final
failure note, with no `.npz`, `.pyc` or individual file above 50 MiB. The full
staged whitespace check flags existing whitespace in captured logs/upstream
source snapshots; those bytes must remain unchanged. Authored paths pass
`git diff --cached --check -- README.md reports scripts wiki`. This is recorded
in `wiki/failures/2026-10-25-memory-publication-whitespace.md`.
