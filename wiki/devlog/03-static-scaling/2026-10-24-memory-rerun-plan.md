# Memory benchmark rerun plan — 2026-10-24

User requested an essential action plan, prioritizing prompt measurements.
This session inspected local source and records only. No benchmark code,
Flux source, or remote state changed. The working tree was initially clean.

## Plan

1. Reuse Leonardo's existing full-dataset campaign: historical/static Flux,
   current/guarded Flux, and PyTorch at 1, 2, and 4 GPUs; global batch 256
   (local batches 256/128/64), same model/data/precision/source identities.
2. Minimal instrumentation: pass a sample output path to campaign `run_case`
   calls, reusing the existing timestamped PID/GPU-UUID sampler and rank
   registration. Keep its 0.25-second target interval and record actual gaps.
   Add per-rank wall-clock training start/end timestamps around the existing
   synchronized boundaries. Separate whole-process/startup peaks from the
   post-warmup training window and exclude final hashing/snapshots from that
   window. No per-step synchronization or forced allocator-cache clearing.
3. Extend the checker to reject missing rank coverage or failed sampling,
   summarize each rank's sampled peak and training median, and report the
   worst rank plus median/range of run peaks across repeats. Preserve raw
   traces, exact commands, source identities and checksums. GPU process usage
   includes allocator caches and runtime/communication overhead; call its
   maximum a sampled peak, not an exact high-water mark. Framework allocator
   counters are supplementary and must retain their own semantics.
4. Run focused sampler/checker checks, sync, precompile on the compute node,
   and run the existing nine-case short preflight. Verify source identities
   and actual sample coverage. Continue directly to 20 discarded warmup +
   390 training updates, three fresh-process repeats per configuration,
   alternating configuration order. Preserve existing correctness checks and
   the 80%-of-device-capacity preflight rule. Extend duration only if traces
   still grow materially at the end. No new correctness campaign is needed
   unless the workload/source changes or an existing check fails.
5. Deliver a nine-row memory table and memory-over-time plots. Include
   per-rank results and repeat spread. Any aggregate device footprint must
   sum simultaneous samples, not independent rank maxima. Memory runs do not
   replace the accepted uninstrumented throughput measurements.

Primary scope is GPU memory, consistent with the existing benchmark and its
memory preflight. Host RSS or phase-level allocation attribution can follow
if requested; neither blocks the first GPU measurements.

## Evidence inspected

- `run-static-full-campaign.sh`: sampler defaults to 0.25 s; preflight uses
  it, but the full-repeat loop currently passes an empty sample path.
- `full_dataset_timing.jl` and `timing_torch.py`: existing synchronized
  warmup/reset/training boundaries can anchor the measurement window.
- `check_full_campaign.py`: already checks repeated PID/UUID sample coverage.
- Latest context/devlogs, baseline/integration ADRs, compatibility audit,
  and campaign report/review: accepted campaign completed 45 timing records;
  previous memory evidence is limited to sampled preflight peaks.

Read-only inspection used `git status --short`, `rg --files`, `rg -n`,
`cat`, `wc -l`, and `sed -n` against the above sources. No command failed.
Documentation validation: `git diff --check`.

## Exact next action

Enable campaign sampling in `scripts/remote/leonardo/run-static-full-campaign.sh`,
add training-window timestamps to both full-dataset runners, and extend the
checker for coverage and memory summaries. Before remote work, load the
required remote skill and read `wiki/context/leonardo.md`.
