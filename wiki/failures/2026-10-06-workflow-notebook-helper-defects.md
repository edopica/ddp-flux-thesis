# Workflow notebook helper defects

Date: 2026-10-06
Scope: new notebook tooling only; no campaign data, Flux source, or tests.

The distributed workflow notebook implementation hit two traps that are easy
to repeat. Both were caught before completion.

## 1. Greedy regex capture dropped concatenated console records

The campaign driver's parallel output sometimes writes two records on one
line: `...resume_match=trueDEMO_STEP mode=resume ...`. The helper parser used
`resume_match=(\S+)`, so the capture consumed the following `DEMO_STEP` marker
and the next record was skipped.

Visible effect: `continuation_summary` counted 364 resume records instead of
472 at four ranks; the check `n_total == ranks x steps` failed. The loss was
108 records, and the failure was silent in the parsing itself, so only the
count cross-check exposed it.

Correction: `resume_match=(true|false|n/a)`. Any future console parser for
these logs must use explicit alternatives for the trailing token, or scan
record starts independently.

## 2. Reproduction command passed the prepared directory, not the artifact

`run_training_child` and `local_reproduction_command` passed
`.../prepared` (a directory) to `--prepared`. The driver expects the
`mnist-fashion_prepared.jls` file. The artifact-mode notebook does not exercise
this path, so only actually running the opt-in child command exposed it:

```
prepared artifact .../prepared is not present
```

Correction: both functions now use
`<prepared>/mnist-fashion_prepared.jls`; the corrected path ran to a 6/6 PASS
MPI campaign. This is the reason the child path is verified by execution, not
by inspection.

## Minor iterations

- `REPO_ROOT` had to be exported from the helper module.
- `@sprintf` required `using Printf` in the notebook bootstrap cell.
- `local` is not usable as a generator variable name in the collectives check.
- `exposure_summary` first counted schedule rows rather than sample identities;
  the prepared schedule row carries a local batch of `pool_indices`.
- The child-run timestamp used `HHmmss`, which Julia expands as month plus
  milliseconds; `HHMMSS` is hour-minute-second.

## 3. Markdown.MD tables failed in KaTeX frontends

The user opened the delivered notebook in VS Code and every table output
reported:

```
ParseError: KaTeX parse error: No such environment: tabular at position 7:
\begin{̲t̲a̲b̲u̲l̲a̲r̲}̲ {r | r} field …
```

`md_table` returned `Markdown.parse(...)` (`Markdown.MD`). IJulia published
that value with four representations, including `text/latex`, where Julia's
Markdown writer emits `\begin{tabular}`. KaTeX, unlike MathJax, has no
`tabular` environment, so any KaTeX-based frontend rendered the LaTeX
representation and failed. The HTML was present in the output all along; the
frontend simply preferred LaTeX.

Correction: `md_table` returns an `HtmlTable` wrapper whose only show methods
are `text/html` and `text/plain`. No frontend can select a LaTeX form.
`Markdown.html` builds the table HTML, and the pipe table stays as the plain
fallback. The check is `showable(MIME"text/latex"(), table) == false`; the
re-executed notebooks publish exactly `text/html` and `text/plain` for all 14
table outputs each.

Lesson: for notebook display, never return a bare `Markdown.MD` when a LaTeX
representation exists. Frontends choose it, and KaTeX support is narrower than
MathJax. Prefer explicit `text/html` or `text/markdown` output types.
