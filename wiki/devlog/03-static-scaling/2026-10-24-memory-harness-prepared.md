# Memory benchmark harness prepared — 2026-10-24

Status: **local harness implementation and local gates only.** No remote
command ran, no GPU was allocated, and no memory, throughput, or scaling result
was measured. This entry implements the plan in
[the rerun plan](2026-10-24-memory-rerun-plan.md).

Follow-up: the nine-case preflight ran later the same day as job `62167742`;
see [the preflight devlog](2026-10-24-memory-preflight-62167742.md). The
statements below describe the harness at the time of this entry.

## Scope

The existing full-dataset campaign is reused unchanged in workload:
`flux-current`, `flux-historical`, and `pytorch` at W=1, 2, and 4, global batch
256, 20 discarded warmup updates, 390 measured updates. The changes add
instrumentation and acceptance checks only. Training math, the precision
policy, the optimizer, the dataset, the schedule, and the diagnostics are
untouched.

## Changes

1. `scripts/remote/leonardo/run-static-full-campaign.sh`
   - Campaign repeats now pass a per-repeat sample path
     (`OUT/samples/<framework>-w<W>-r<repeat>.csv`), so the existing
     `nvidia-smi` per-process sampler runs during the measured repeats instead
     of only during preflight.
   - Each repeat keeps its raw timestamped trace and its
     `<label>.ranks/rank-<r>.json` PID/GPU registration.
   - The 0.25 s target interval and the existing sampler/rank-registration
     implementation are unchanged.
   - Traces live under `OUT`, so the existing `SHA256SUMS` step covers them.

2. `scripts/static_scaling/full_dataset_timing.jl`
   - Records `rank_training_windows` as per-rank wall-clock epoch
     (`time()`) start/end pairs around the already-synchronized measured
     region.
   - The window opens immediately before the measured `steps` call (after the
     warmup, model/optimizer reset, barrier, and `CUDA.synchronize()`) and
     closes immediately after the post-update `CUDA.synchronize()`. Final
     hashing, beta-power checks, snapshots, and teardown are outside it.
   - Window starts and ends are gathered with `MPI.Allgather`, after the
     existing duration gather, so collective order stays identical on every
     rank.

3. `scripts/static_scaling/timing_torch.py`
   - Adds `training_start_epoch`/`training_end_epoch` (`time.time()`) to each
     rank record around the same synchronized boundaries.
   - Final hashing, snapshot writing, and process-group teardown remain
     outside the window.
   - No per-step synchronization and no allocator-cache clearing was added.

4. `scripts/static_scaling/check_full_campaign.py`
   - Validates that every record has complete, finite, increasing per-rank
     windows, and that each window duration agrees with the measured rank
     duration within `max(2%, 1 s)` (the epoch window uses the realtime clock,
     the duration uses the monotonic clock).
   - Preflight: every rank must have at least one sample inside its training
     window in addition to the previous repeated-PID/UUID whole-process
     coverage, and simultaneous samples must cover the ranks. The short
     preflight is not held to the campaign sample-density floor because its
     20-update window can be only a few seconds. OD-8 remains the same
     acceptance rule: the whole-process sampled peak must be at or below 80% of
     the device total. Training-window and aggregate values are reported
     alongside.
   - Campaign: every timing repeat must have a parseable trace with complete
     rank registration. The checker rejects missing traces, missing rank
     registrations, duplicate PIDs/GPUs, traces that start or stop outside a
     5 s coverage slack around the rank window, sampling gaps over 5 s, and
     overlapping-window samples that do not cover every rank. Each rank must
     also have at least `max(2, ceil(window_seconds))` training-window
     samples, i.e. one sample per second at the 0.25 s target; this rejects a
     trace with only two samples in a ten-second window. The simultaneous
     aggregate is held to the same density floor.
   - Sample timestamps must be strictly increasing. Duplicate or out-of-order
     `TIME` blocks are rejected rather than collapsed, so repeated blocks
     cannot inflate the sample count; density is counted over distinct
     instants.
   - The simultaneous aggregate is validated as its own timestamp series, not
     only by sample count. Its first and last samples must lie within the 5 s
     slack of the common training window and it must contain no internal gap
     over 5 s. A dense burst followed by an uncovered tail is rejected.
   - Reports per-rank training peak and training median, startup peak,
     whole-process peak, sample count, actual sampling gap median/max, and the
     worst rank. Across repeats it reports median/min/max/spread.
   - The aggregate footprint is the maximum sum of **simultaneous** per-rank
     samples; it is never the sum of independent rank maxima. This is covered
     by a dedicated self-test case where independent maxima sum to 2000 MiB but
     the simultaneous maximum is 1500 MiB.
   - Writes the summary into `REPORT.json` under `memory` and prints a
     nine-row `FULL_MEMORY` table plus per-rank `MEMORY_RANK` lines.
   - Records the qualifications that sampled peaks are not continuous
     high-water marks, process usage includes allocator caches/runtime
     overhead, startup is separate from training, and framework allocator
     counters remain supplementary with their own semantics.

5. Tests
   - `scripts/static_scaling/test_full_campaign.py`: sample timestamps moved
     inside the synthetic training windows.
   - `check_full_campaign.py --self-test` now has 37 checks: the previous 21,
     thirteen new rejection mutations (missing trace, missing rank
     registration, no training-window sample, sampling gap, truncated trace,
     sparse trace, missing window, invalid window order, window-duration
     mismatch, duplicate sample PID, duplicate sample timestamp,
     aggregate-start coverage, aggregate middle gap), a report-shape check,
     the simultaneous-aggregate case, and a 20 s early-burst regression whose
     simultaneous samples stop after the first 5 s.

## Semantics and compatibility

- A sampled peak is not a guaranteed continuous high-water mark.
- GPU process usage includes framework allocator caches and runtime/
  communication overhead; it is not a pure tensor footprint.
- Startup and whole-process peaks are kept separate from the training window.
- The preflight gate is deliberately unchanged so the OD-8 decision remains
  comparable to the earlier preflight.
- The checker now requires traces for every campaign repeat. The accepted
  2026-10-20 campaign artifact predates this instrumentation, so it cannot be
  re-verified with this checker revision; its frozen evidence is unchanged.
- A preflight decision produced before this change lacks training-window
  coverage and is rejected for campaign reuse. The memory rerun needs a fresh
  preflight, as the plan already requires.

## Independent review response

A read-only review of the first revision found no blockers and verified that
OD-8 is unchanged, rank order is correct, the aggregate is a simultaneous sum,
and both runners bracket the synchronized measured region. Its acceptance
gaps were addressed:

- An under-sampled campaign trace (two samples in ten seconds) could pass.
  Campaign repeats now require `max(2, ceil(window_seconds))` training
  samples per rank, and the simultaneous aggregate has the same density floor.
- The aggregate could rest on a single coincident stamp; the density floor now
  applies to it as well.
- A two-sample preflight minimum could make a valid sub-two-second preflight
  window fail. Preflight now requires one training-window sample per rank after
  the whole-process repeated-PID/UUID coverage.
- Trace coverage now requires the first and last rank sample to be within 5 s
  of the rank window, and the maximum sampling gap is 5 s (previously 10 s).
- A sampler-start failure leaked `FULL_PREFLIGHT_RANK_DIR` into later cases;
  the failure path now unsets it.

## Second review fixes (2026-10-24)

A second read-only review reproduced three acceptance/provenance gaps in the
first revision. All three are fixed; no training math changed.

1. **Aggregate coverage could stop early.** Individual ranks could remain
   covered while the simultaneous stream had a dense burst at the start and no
   samples afterwards, and the count-only density check still passed. The
   checker now builds the simultaneous timestamp series itself and requires
   its first and last samples to be within `SAMPLE_COVERAGE_SLACK_SECONDS`
   (5 s) of the common training window and every internal gap to be at most
   `MAX_SAMPLE_GAP_SECONDS` (5 s). The global gap report is the maximum of the
   raw trace gap, the per-rank gap, and the simultaneous gap. A 20 s
   regression with simultaneous samples only in the first 5 s is now rejected
   after individual rank coverage passes.
2. **Duplicate timestamps could inflate density.** `_read_gpu_samples` now
   requires each `TIME` value to be strictly greater than the previous one;
   duplicate and out-of-order blocks are rejected before density is counted.
   The self-test includes a trace with three distinct instants repeated four
   times, which previously reached the ten-sample floor in a ten-second
   window.
3. **Thesis pin was stale.** The campaign launcher, the checker, and the
   full-validation driver now default to local HEAD
   `660e363e6de5b6e4b398e542d79b6ce1adb381da` instead of
   `a3786d97…`. The source snapshot still records `THESIS_STATE_SHA256`, so
   the uncommitted harness edits are identified even though the commit is the
   base revision. The Git-free remote copy is the supported path in this
   state; if the harness is committed, both pins must move to the new HEAD
   before launch. A plain checkout of `660e363e…` cannot contain this
   uncommitted pin fix.

The two new coverage rejections also prove the distinction from the existing
staggered trace test: individual rank coverage and a simultaneous sum are
separate requirements. The aggregate remains the maximum sum of simultaneous
samples, never the sum of independent rank maxima.

## Residual risks

- The campaign density floor is one sample per second (four target intervals)
  with a two-sample minimum. A valid trace slower than that is rejected; the
  check is fail-closed and would require a rerun.
- `MAX_SAMPLE_GAP_SECONDS = 5` and `SAMPLE_COVERAGE_SLACK_SECONDS = 5` are
  fixed. A driver/NFS stall longer than five seconds fails the artifact check.
- Epoch windows use the realtime clock while throughput uses the monotonic
  clock. An NTP step during a run would fail the window/duration consistency
  check rather than silently mis-clip samples.
- The sampler writes `TIME` before running `nvidia-smi`, so each value is
  stamped one query latency early. This affects at most the boundary samples.
- Strictly increasing timestamps reject a duplicate or backward clock step
  rather than silently collapsing it. A sampler or `date` anomaly therefore
  fails closed and needs a rerun.
- Aggregate span and internal-gap checks are campaign-only. The short
  preflight still requires just one training-window sample per rank after
  whole-process coverage, so its aggregate is reported but not held to the
  campaign span.
- The thesis pin is the base commit `660e363e…`; the harness edits are
  uncommitted. The Git-free remote copy and `THESIS_STATE_SHA256` identify the
  edited tree. If the tree is committed, update both pins to the new HEAD.
- Preflight training-window coverage is one sample per rank, not the campaign
  density floor, because a 20-update window can be only a few seconds. OD-8
  still gates on whole-process coverage and the 80% peak.

## Local gates (exact commands and results)

```bash
cd scripts/static_scaling
python3 check_full_campaign.py --self-test
# FULL_CAMPAIGN_SELFTEST PASS checks=37

python3 -m unittest test_full_campaign -v
# Ran 8 tests — OK

python3 -m py_compile check_full_campaign.py timing_torch.py test_full_campaign.py
# exit 0

bash -n scripts/remote/leonardo/run-static-full-campaign.sh
bash -n scripts/remote/leonardo/run-static-full-validation.sh
# exit 0

julia --startup-file=no -e 'for f in ARGS; Meta.parseall(read(f, String)); println("JULIA_PARSE_OK ", f); end' \
  scripts/static_scaling/full_dataset_timing.jl scripts/static_scaling/timing_flux.jl
# JULIA_PARSE_OK full_dataset_timing.jl
# JULIA_PARSE_OK timing_flux.jl

git diff --check
# exit 0
```

No command failed. `shellcheck` was not run (not installed locally).

## Not done

- No remote execution, sync, precompile, preflight, campaign, or GPU-memory
  measurement. The `remote` skill is not available in this session, and the
  rule is to load it and read `wiki/context/leonardo.md` before any Leonardo
  command.
- No memory table or memory-over-time plot exists yet; both need run traces.
- No throughput measurement was repeated.

## Next action

Load the required remote skill and read `wiki/context/leonardo.md`. Confirm
`EXPECTED_THESIS_COMMIT` equals the deployed thesis HEAD; if the harness is
committed, update both pins to the new HEAD and re-run the local gates. Sync
the thesis tree (and both Flux worktrees), precompile on a compute node, run
the nine-case batch-256 preflight with sampling enabled, and require
`FULL_CAMPAIGN_READY PASS` with training-window coverage for every rank. Then
run the campaign with `REPEATS=3` (three fresh-process repeats, alternating
configuration order) and inspect `REPORT.json` `memory` rows, the
`FULL_MEMORY` table, the per-rank `MEMORY_RANK` lines, and the raw
`OUT/samples/*.csv` traces before reporting any memory result.
