# Phase K reference plots and live-chart stretch

Date: 2026-10-05

## Request and changes

The user requested learning-rate visualization, a single-process comparison, and a later live speed comparison on Leonardo.

- The Phase K prompt now requires task-specific validation loss and accuracy overlays for DDP and the independent native reference.
- A separate plot shows the configured Adam learning rate before each update.
- The plan requires independent metric records, task exposure counts, metric-difference panels, and checkpoint/resume markers.
- Each reference consumes its distributed run's saved schedule, including the task mixture and globally absent heads.
- The fixed-rate baseline requires no new scheduler.
- The prompt header now reflects existing K1 results and the pending Leonardo K1 cell.
- The implementation plan, current context, and README progress row reflect these requirements.

## Optional post–Phase K stretch

The live chart compares a standalone native process on one A100 with four NCCL ranks on four A100s.
Both runs use the same global work and four-rank routing schedule.
Separate timing runs exclude the in-loop correctness reference and detailed diagnostic snapshots.
The timing contract includes warm-up, state restoration, CUDA synchronization, slowest-rank duration, repeated runs, and separate overhead reporting.
An external viewer consumes rank-0 metric records and supports explicitly labeled replay.
A live run can use the saved standalone baseline for comparison without simultaneous resource contention.
The report will state the measured speedup or slowdown. The current small model can be communication-bound.

## Verification

This session changed documentation only. It produced no new training or performance evidence.
Documentation check command:

```bash
git diff --check -- README.md temp/conditional_graph_wrapper_implementation_plan.md wiki/context/current.md
git diff --no-index --check -- /dev/null temp/conditional_graph_phase_k_prompt.md
git diff --no-index --check -- /dev/null wiki/devlog/02-conditional-graph/2026-10-05-phase-k-visualization-plan.md
```

All three commands passed with exit 0.
The last two commands cover files outside the tracked diff.

## Exact next action

Inspect the K1 metric schema and native reference.
Then extend K2 with separate reference metrics and learning-rate records under section 8 of the Phase K prompt.
