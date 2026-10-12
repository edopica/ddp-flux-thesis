# Phase K two-notebook plan

Date: 2026-10-06

## Request and outcome

The user requested two IJulia notebooks with output analysis: a wrapper tutorial and a separate one-versus-four-rank benchmark.
The tutorial uses four ranks by default and offers two ranks as a selectable configuration.
The user excluded Pluto and requested a recommendation on Python versus Julia plots.

The plan is `reports/conditional-graph-phase-k-notebook-plan.md`.
It recommends the existing Python command-line plotters, called from Julia, with inline image display.
The plan specifies graph inputs, required explanations, missing views, timing statistics, provenance, and notebook acceptance checks.
It distinguishes the different correctness objectives from the equal-work benchmark objective.
The plan retains the corrected timing results: training-only speedup 1.888594 and total speedup 1.397335.

## Inspection and checks

Source inspection covered the three Phase K plotters, the saved schedule implementation, and the training API calls.
Evidence inspection covered the implementation review, corrected timing devlog, current context, and audit 002 section 6.
`git status --short` showed substantial existing user work. The session added planning documents and targeted dashboard updates.

Documentation check command:

```bash
git diff --check -- README.md wiki/context/current.md
git diff --no-index --check /dev/null reports/conditional-graph-phase-k-notebook-plan.md
git diff --no-index --check /dev/null wiki/devlog/02-conditional-graph/2026-10-06-phase-k-two-notebook-plan.md
```

All three documentation checks passed (exit 0).
Notebook implementation remains pending. No new training evidence comes from this planning session.

**Exact next action:** implement the artifact bundle map and shared IJulia display helper, then extend the existing Python figures.
