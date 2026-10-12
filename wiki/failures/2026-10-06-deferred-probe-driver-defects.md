# 2026-10-06 — Deferred-behavior probe/driver defects (test tooling only)

The new deferred-behavior probe/driver pair
(`scripts/checks/conditional_graph_deferred_{probe,check}.jl`) failed on its
first executions. All failures were defects in the newly written test tooling,
not in Flux or in the probe's subject. Each was diagnosed, fixed, and re-run;
the final run is 8/8 modes, 0 failures, exit 0.

Recorded per the repository rule to log failed commands.

## Defects and fixes

1. **Marker slicing across ranks.** The driver sliced the console log at the
   last `PROBE_READY`, assuming both ranks' markers would follow it. Rank
   output is interleaved, so one rank's early mode markers were dropped.
   Fixed by using the full log (`body(run) = run.output`) after making the
   warmup use a non-logging backend, so no stale collective markers remain.
2. **Literal regex prefixes.** Two assertions used `occursin("... [0-9.eE+-]+",
   ...)` with a String, treating regex syntax as literal; and two numeric
   parses anchored a field immediately after the mode name when another field
   preceded it. Fixed with `Regex(...)` and `.*` prefixes. Symptoms:
   `cross delta=nothing` and `delta_vs_shifted=nothing` while the raw log
   contained the values.
3. **Log truncation on killed runs.** `pipeline(cmd, stdout=path, stderr=path)`
   opens the same path twice; when the watchdog killed a hung run, the child
   stderr crash traces overwrote the stdout markers from offset 0, so the hang
   logs lost their evidence. Fixed by writing both streams to one shared
   `mktemp()` handle and copying the buffer to the destination log afterwards.
   One probe iteration also had an instrumentation bug (`String * Float64`);
   fixed by interpolation.
4. **Shadow semantics in `config-eta`.** The probe's independent reference
   applied each rank's local gradient, but the wrapper applies the global-mean
   gradient with each rank's local rule. With a rank-local learning rate the
   replicas diverge, so the local gradient is not the mean; the first probe
   reported a false self-mismatch. Fixed by replicating the wrapper's MPI
   averaging in the shadow. After the fix `self_maxdiff=0.0` on both ranks and
   the cross-rank delta remains the finding.

## Impact

No product defect, no Flux source or test change, no commit. The final run and
raw logs are under `artifacts/logs/conditional-wrapper/deferred/`; the report
is `wiki/audits/02-conditional-graph/002-conditional-wrapper-deferred-behaviors.md`.
