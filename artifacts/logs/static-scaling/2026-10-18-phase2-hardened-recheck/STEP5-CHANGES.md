# Step 5 changes — decisions frozen and evidence package

Date: 2026-10-18. No commit was made.

## Files changed in Step 5

| File | Change |
|---|---|
| `experiments/static-scaling/open-decisions.md` | Header date/status; table cells for OD-2/OD-3/OD-4/OD-11 marked recorded 2026-10-18; notes for OD-2/OD-3, OD-4, and OD-11 updated; four decision-record rows plus the evidence/consistency paragraph |
| `experiments/static-scaling/experiment.toml` | `[optimizer] status`, `[schedule] status`, and `training_contract.flux_entry_point_status` set to recorded 2026-10-18; `warmup_epochs_proposed` → `warmup_epochs`, `learning_curve_epochs_proposed` → `learning_curve_epochs`; no numeric value changed |
| `experiments/static-scaling/specification.md` | Header date/status; sections 6.1, 6.2, 6.3, 10.4, and 12 updated; tolerance section unchanged and consistent with OD-10 |
| `wiki/devlog/03-static-scaling/2026-10-18-phase3-step5-decisions.md` | New Step 5 devlog |
| `wiki/devlog/README.md` | Scope 3 index entry for the new devlog |
| `wiki/context/current.md` | New top section for Step 5; exact next action = Step 6 |
| `wiki/context/leonardo.md` | 2026-10-18 hardened Phase 2 re-check status |
| `README.md` | Top summary line, Phase 2 status note, SS-005 row, SS-002/SS-001 status updates |
| `artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/` | New evidence package: `remote-check.log`, `local-console.log`, `PROVENANCE.md`, `COMMANDS.md`, `SHA256SUMS`, this file, `step5-worktree.diff` |

## Exact decision rows added

```text
| 2026-10-18 | OD-2 | AdamW with `lr = 1e-3`, `wd = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`; decoupled decay through the same learning-rate factor; bias correction on; both frameworks use the identical rule | user | specification.md, experiment.toml |
| 2026-10-18 | OD-3 | Decay only on the 20 convolution weights and the classifier weight (21 leaves, `lambda = 1e-4`); no decay on the 40 GroupNorm `gamma`/`beta` arrays and the classifier bias (41 leaves, `lambda = 0`); PyTorch uses `ndim >= 2`, Flux uses a Functors-compatible rule tree | user | specification.md, experiment.toml |
| 2026-10-18 | OD-4 | Linear warm-up over 5 epochs and cosine decay to zero over a 50-epoch budget, indexed by global update `t`; batch 512 uses `W = 975`, `T = 9,750`; the batch-256 alternate `W = 1,950`, `T = 19,500` remains available for Phase 5 | user | specification.md, experiment.toml |
| 2026-10-18 | OD-11 | Explicit `Zygote.withgradient` + `Optimisers.update!` on `DistributedOptimizerState`; communication outside AD; 64 collectives per update on the explicit path (1 presence + 62 gradient averages + 1 gradient-validity vote); the eager path adds the ADR-0008 loss vote (65) and is not selected | user | specification.md, experiment.toml |
```

OD-10 remains the 2026-10-17 row; together the table has the five OD-2/3/4/10/11
rows. OD-7, OD-8, OD-9, OD-12, and OD-13 remain open with unchanged proposed
defaults.

## Diff record

`step5-worktree.diff` is `git diff HEAD` for the three decision files. The
working tree already contained uncommitted OD-1/OD-5/OD-6/OD-10 work before
this session, so the file also shows those earlier edits (for example the
`[tolerances]` block). The Step 5-specific edits are the OD-2/OD-3/OD-4/OD-11
status, table, note, and decision-record changes, the three recorded status
fields, and the two schedule key renames.

## Verification

- Python `tomllib`: `TOML_PARSE PASS`; `optimizer.status`, `schedule.status`,
  and `flux_entry_point_status` read back as recorded 2026-10-18; values
  `lr=0.001`, `wd=0.0001`, `betas=[0.9,0.999]`, `eps=1e-8`,
  `W/T=975/9750` (b512), `W/T=1950/19500` (b256).
- Julia `TOML.parsefile`: `JULIA_TOML_PARSE PASS` with the same status strings.
- No stale `proposed` wording remains for OD-2/OD-3/OD-4/OD-10/OD-11; the
  decision table shows all five as recorded.
- The Phase 3 acceptance checker and tolerances were not edited in Step 5;
  the tolerance diffs in `step5-worktree.diff` predate this step.
