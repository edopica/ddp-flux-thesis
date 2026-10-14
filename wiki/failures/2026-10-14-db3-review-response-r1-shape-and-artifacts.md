# 2026-10-14 — DB-3 review response: R1 first-run shape and artifact failures (fixed)

Two failures stopped the first `deathstar` run of
`scripts/checks/phase_k_collective_protocol_check.jl` during the DB-3 review
response. Neither is a production defect and neither invalidates the local R1
correction; both were fixed and the rerun exits 0. The console below is
transcribed from the run output because the local log
`r1-protocol-check.log` was overwritten by the successful rerun. Recorded per
the repository rule to log failed commands.

## First-run console (transcribed)

```text
PHASE_K_PROTOCOL tutorial-2rank PASS
PHASE_K_PROTOCOL tutorial-4rank FAIL: selected bundle tutorial-4rank is incomplete; missing required artifact(s):
  artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/trajectory.jls
  artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/validation-train.tsv
PHASE_K_PROTOCOL db3-toy-mpi2 FAIL: train step 1 rank 1: trailing collective is Vector{Float32} shape (1,), expected the trailing host Float32 length-1 avg gradient-validity vote
PHASE_K_PROTOCOL guarded-pinned-legacy-5/7 PASS
PHASE_K_PROTOCOL guarded-without-protocol PASS
PHASE_K_PROTOCOL tutorial-2rank-pinned-guarded-6/8 PASS
PHASE_K_PROTOCOL_SUMMARY FAIL
  failed: tutorial-4rank
  failed: db3-toy-mpi2
```

## 1. Incomplete remote 4-rank tutorial artifacts

Symptom: `tutorial-4rank` failed before any protocol check because the selected
bundle failed its completeness test. The remote deathstar copy of
`artifacts/logs/conditional-wrapper/demo/k2-2026-10-05-leonardo-nccl4/nccl4/train/`
was missing `trajectory.jls` and `validation-train.tsv`; the local bundle is
complete.

Root cause: the thesis sync deliberately excludes `artifacts/`
(`--exclude='artifacts/'`) to protect remote-only artifacts, so files produced
elsewhere do not reach deathstar automatically. The two files had never been
uploaded to this checkout.

Fix: upload the two missing files without `--delete`, so existing remote-only
artifacts stay intact.

Evidence: local sizes `trajectory.jls` 148521848 bytes and
`validation-train.tsv` 108911 bytes; the rerun passes `tutorial-4rank`.

## 2. One-dimensional shape capture `(1,)`

Symptom: `db3-toy-mpi2` failed the guarded trailing-vote check even though the
guard behaved correctly. The error reads "trailing collective is
`Vector{Float32}` shape `(1,)`", which is the expected shape; the comparison
still failed.

Root cause: the console shape capture is the raw text inside the parentheses,
so a one-dimensional `(1,)` parses as `"1,"`. The synthetic local check had used
`"1"` and missed the trailing comma, so the defect only surfaced against the
real guarded artifacts.

Fix: new `shape_dims(text)` helper parses the dimensions (ignoring empty
segments), and the vote check uses `shape_dims(vote.shape) == [1]` instead of a
literal string compare.

Evidence: final
`notebook/phase_k_workflow_helpers.jl` md5 `b570d0777109a41eee5bb1be3b4d53d5`;
guarded check passes in the rerun.

## Green rerun

```
scripts/remote/run.sh deathstar "julia --startup-file=no scripts/checks/phase_k_collective_protocol_check.jl"
```

Exit 0; console:

```text
PHASE_K_PROTOCOL tutorial-2rank PASS
PHASE_K_PROTOCOL tutorial-4rank PASS
PHASE_K_PROTOCOL db3-toy-mpi2 PASS
PHASE_K_PROTOCOL guarded-pinned-legacy-5/7 PASS
PHASE_K_PROTOCOL guarded-without-protocol PASS
PHASE_K_PROTOCOL tutorial-2rank-pinned-guarded-6/8 PASS
PHASE_K_PROTOCOL_SUMMARY PASS
```

Log:
[`r1-protocol-check.log`](../../artifacts/logs/conditional-wrapper/numerical-guards/db3-review-response/r1-protocol-check.log)
(the overwriting rerun that replaced the failed console).

## Impact

No production code, test, or artifact was rewritten; the fixes are one artifact
upload and one parsing helper. Full response record:
[devlog](../devlog/02-conditional-graph/2026-10-14-db3-review-response.md);
commands and source identity:
[COMMANDS.md](../../artifacts/logs/conditional-wrapper/numerical-guards/COMMANDS.md#db-3-review-response-campaign-2026-10-14-deathstar).
