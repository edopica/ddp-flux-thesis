# Phase K isolated implementation prompt

Date: 2026-10-04
Scope: prompt preparation and implementation discussion.

## Request and result

The user requested a standalone Phase K prompt from the conditional-wrapper implementation plan.
Created `temp/conditional_graph_phase_k_prompt.md` and linked it from Phase K in the plan.

The prompt defines:

- One device-generic shared-trunk classifier with two equal-shaped heads.
- Saved sample schedules and a six-step routing cycle.
- An independent native Adam trajectory for the exact routed global objective.
- Per-step parameter, moment, scalar-state, and replica comparisons.
- Exact state preservation during global absence after nonzero Adam history.
- Data-only checkpoints with fresh-process CPU and GPU continuation.
- Toy and paired MNIST/Fashion-MNIST runs at two/four CPU ranks and the two accepted NCCL configurations.
- Rank-0 presentation output and reproducible evidence requirements.

Numerical defaults remain recommendations for discussion.
The prompt keeps the global batch size fixed across rank counts.
It explains why the one-versus-rest routing schedule changes the objective across those rank counts.
Each run therefore has its own schedule-matched reference.

## Dataset decision after discussion

The user questioned whether ordinary MNIST classification gives a meaningful reason for conditional heads.
The user selected toy diagnostics followed by task-conditioned MNIST/Fashion-MNIST training.
Updated the implementation plan, isolated prompt, README, and current context.

The presentation model shares a trunk and uses separate ten-class digit and clothing heads.
Known dataset identity selects the matching head without the target class or a learned router.
The schedule selects a task and samples from that task's dataset for every local batch.
Mixed-task and single-task steps preserve the existing absence and reactivation checks.

The prompt now uses a fixed presentation budget of 120 updates instead of three nominal epochs.
Separate task pools cycle deterministically, with recorded sample reuse and exposure counts.
This definition avoids an ambiguous epoch count when the task schedule gives the datasets different exposure.
Validation uses the matching dataset head and reports task-specific loss and accuracy.
The report includes both prediction panels and makes no required accuracy-gain claim over separate models.

## Source review

Read the implementation plan, current context, latest Leonardo devlog, relevant ADRs, and compatibility audit.
Inspected the historical training example, Phase H migration recipe, Phase J device regression, and thesis environment dependencies.
The historical example calls the removed helper and saves model weights without Adam history.
The prompt specifies normal wrapper updates and complete native-tree checkpoints.

Phase J establishes native GPU gradient communication at two and four ranks.
GPU checkpoint continuation remains a new Phase K requirement.

## Evidence and scope

Command: `git status --short` from the thesis repository, exit 0.
The workspace contains existing tracked and untracked work.
Changes in this session are documentation only.
No training, remote job, or Phase K correctness test ran.
Phase K remains planned.

Documentation whitespace checks passed, exit 0:

```bash
git diff --check -- "temp/conditional_graph_wrapper_implementation_plan.md" "wiki/context/current.md"
git diff --no-index --check /dev/null "temp/conditional_graph_phase_k_prompt.md"
git diff --no-index --check /dev/null "wiki/devlog/02-conditional-graph/2026-10-04-conditional-wrapper-phase-k-prompt.md"
```

The dataset-choice update also uses this whitespace check:

```bash
git diff --check -- "temp/conditional_graph_wrapper_implementation_plan.md" "README.md" "wiki/context/current.md"
```

## Exact next action

Implement the saved toy schedule and independent native reference in `temp/conditional_graph_phase_k_prompt.md`.
