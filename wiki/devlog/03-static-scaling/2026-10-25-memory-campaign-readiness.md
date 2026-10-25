# Memory campaign launch readiness — 2026-10-25

**Status: conditional GO for the measurement run; NO-GO for a final
campaign-length memory claim.** This entry reviews readiness only. No remote
command ran, no GPU was allocated, and no memory or throughput result was
produced. The `remote` skill was not available in this session, so the remote
tree and queue state are not verified here.

## What is ready

- Preflight `62167742` passed all nine batch-256 framework/world cases at
  4 CPUs and selected batch 256 for the 390-update campaign. The tightest
  cases are Flux W=1: `49,792 MiB` current and `49,728 MiB` historical against
  the `51,955.8 MiB` 80% limit (2.1–2.2 GiB margin).
- The current campaign scripts are byte-identical to the preflight source
  snapshot for the launcher, checker, both full-dataset runners, and
  `memory_flux.jl`.
- Local gates re-pass: `FULL_CAMPAIGN_SELFTEST PASS checks=37`, 8/8
  `test_full_campaign.py` tests, `git diff --check` exit 0, and
  `check_full_campaign.py ready` on the copied preflight artifact prints
  `FULL_CAMPAIGN_READY PASS batch=256 updates=390`.
- The campaign path reuses the preflight decision and samples every timing
  repeat with the 0.25 s `nvidia-smi` per-process sampler. The checker reports
  per-rank training/startup/whole-process peaks, repeat spread, and the
  simultaneous-sum aggregate.

## Conditions before launch

1. **Do not sync the thesis tree.** The preflight decision binds
   `THESIS_STATE_SHA256=1ddd2d20…` to the remote tree. At review time, before
   these doc edits, the local tree was `7f1c9518…` at HEAD `976db1f` (the
   preflight record used expected commit `660e363e…` in a Git-free copy). A
   sync changes the state hash, and campaign mode rejects the decision at
   `ready` (`preflight decision source identity differs`). The launcher fails
   closed, so this cannot silently produce bad evidence; it requires a fresh
   preflight.
2. **Set `REPEATS=3` explicitly.** The launcher default is 5 and the preflight
   `campaign-config.json` recorded 5. The checker does not bind repeats to the
   decision. Use the planned three fresh-process repeats.
3. **Keep `--cpus-per-task=4`.** The checker requires `CAMPAIGN_CPUS=4`;
   requesting 32 CPUs fails readiness.
4. **Use the stored decision as `PREFLIGHT_DECISION`.** The remote path is
   `$WORK/epicazio/static-scaling-evidence/static-full-campaign-62167742/preflight/decision.json`.
   The preflight output root is unchanged and must stay that way.
5. **Request enough walltime.** `boost_qos_lprod` allows 4 days. The three-repeat
   campaign plus diagnostics is expected to fit in roughly 2–3 hours; 8 hours
   leaves margin for the documented first-repeat W=1 outlier.

## What is not established

- **Campaign-length headroom.** The preflight is 20 warmup + 20 measured
  updates; the campaign is 20 + 390. The preflight itself says the campaign can
  peak higher. The campaign checker reports `margin_bytes` but does **not**
  re-apply the OD-8 80% gate at 390 updates. Therefore
  `FULL_CAMPAIGN_ARTIFACT_CHECK PASS` is not a campaign memory PASS. Read the
  `FULL_MEMORY` row's `margin_MiB`; a negative value means the 80% criterion
  does not hold at campaign length even though the artifact passes.
- **A clean live-tensor comparison.** At Flux W=1, 45,504 MiB of the 49,792 MiB
  peak is present before the measured window (91%); the 20-update increment is
  4,288 MiB. The harness calls `GC.gc()` but not `CUDA.reclaim()`, so the peak
  includes CUDA.jl cached blocks. The 4.6× W=1 process-footprint difference is
  real, but it is not a live-tensor measurement. The existing
  `memory_flux.jl`/`memory_torch.py` Phase 5 probe records live and reserved
  counters and calls `CUDA.reclaim()`; it has not been run on this workload.
- **Repeat spread at 390 updates.** The accepted 32-CPU campaign `61493236`
  already completed 390 updates at batch 256, so the workload fits, but it did
  not sample campaign-length memory. The new run is the first 390-update
  sampled-memory measurement.

## Recommendation

If the goal is the measurement, launch the three-repeat campaign now under the
conditions above. If avoiding a possibly invalid three-repeat run matters more
than using the whole overnight window, first run a one-repeat 390-update
full-matrix probe (`REPEATS=1`), inspect the Flux W=1 `FULL_MEMORY` margin, then
run the three-repeat campaign. Either way, run the reclaim diagnostic before
writing a live-tensor conclusion, and report the campaign peak as a sampled
process footprint with its qualifications.

## Exact next action

The user decides: (a) three-repeat measurement now, or (b) one-repeat
390-update probe first. Do not sync before either. The campaign submission is
`run-static-full-campaign.sh campaign` with `CONFIRM_CAMPAIGN=1`, `REPEATS=3`
(or 1), `PREFLIGHT_DECISION=<preflight decision>`, 1 task, 4 CPUs, 4 GPUs.
No command was run in this review.

## Launch follow-up

The three-repeat campaign was launched as job `62195500` at
`2026-10-24T23:43:05Z`; the launcher’s source identity matched the preflight
state `1ddd2d20…` and the stored decision. Both Flux worktrees precompiled
(`FLUX_PRECOMPILE_OK`), and the required markers printed at
`2026-10-24T23:48:23Z`:

```text
FULL_CAMPAIGN_READY PASS batch=256 updates=390
FULL_CAMPAIGN_preflight PASS reused_decision=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-62167742/preflight/decision.json batch=256 k=390
FULL_CAMPAIGN_CAMPAIGN START utc=2026-10-24T23:48:23Z batch=256 updates=390 repeats=3
CASE framework=flux-current world=1 updates=390 warmup=20 batch=256 snapshot=none
```

At `2026-10-24T23:49:48Z` Slurm reported `RUNNING`, `00:06:30`, node
`lrdn2909`. No sync was run. Evidence:
[launch record](../../../artifacts/logs/leonardo/2026-10-25-static-full-memory-62195500.md).
