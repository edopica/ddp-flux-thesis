# Phase K review and GPU recap notebook plan

Date: 2026-10-06

The user requested a Phase K implementation review before a GPU recap notebook.
The review covered the demo, schedule, dataset, native reference, checkpoint, metrics, driver, and plotting code.
It also covered the optional timing path.

The review found no blocking defect in the inspected correctness path.
Stored K2 evidence passed local checks for two NCCL ranks and four NCCL ranks.
Fresh-process resume records contain 236/236 and 472/472 matches respectively.
The current Flux wrapper hash matches both stored campaigns.
Local figure regeneration produced 17 PNGs plus metadata with exit 0.

The timing review found three defects:

- The timed baseline snapshots absent heads to CPU, unlike the DDP path.
- The headline uses total duration, and the training field retains record-write overhead.
- The speedup figure uses a median of paired ratios instead of a ratio of medians.

The last two statistics coincide for the stored campaign.
The baseline bias remains unmeasured and requires a corrected timing campaign.
The required correctness matrix remains complete.

Report, source locations, commands, and notebook outline:
[`reports/conditional-graph-phase-k-review.md`](../../../reports/conditional-graph-phase-k-review.md).

Documentation changed in this session.
No training, remote execution, Flux changes, commits, or command failures occurred.

**Exact next action:** create `notebook/conditional_graph_gpu_recap.ipynb` from the existing two-rank and four-rank correctness artifacts.

## Follow-up: demonstration preconditions

The user reported the corrected Leonardo campaign and requested the next decision.
The stored `summary-correctness.txt` and `summary-timing.txt` both report PASS.
The latter records training-only speedup 1.888594 and total speedup 1.397335.
This follow-up read the updated documentation and summaries. It did not repeat the timing verification.

Audit 002 section 6 needs three clarifications for the demonstration:

1. Finite gradients are a separate requirement from finite losses.
   Any future finite-value guard needs an all-rank decision before a stop or skipped update.
2. `Flux.train!` does not resolve the asymmetric-loss failure in DB-1.
   The existing explicit fixed-step driver remains the notebook execution path.
3. Equal participation does not require `DistributedDataContainer` specifically.
   The saved Phase K schedule already supplies equal batch sizes and update counts.

Proposed notebook scope retains the existing correctness artifacts and corrected timing results.
Numerical-failure recovery and untested GPU paths remain explicit limitations.
The user did not yet accept these preconditions.

**Exact next action:** obtain the user's decision on the demonstration preconditions, then create the GPU recap notebook.
