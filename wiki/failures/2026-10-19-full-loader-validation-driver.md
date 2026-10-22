# Full-dataset loader validation driver failures — 2026-10-19

Scope: jobs `61446924`, `61447030`, and `61450014` on Leonardo. None of these
failures changed a Flux source file, the dataset, or the accepted artifacts.
The accepted validation evidence is job `61450014`.

## F1 — `set -u` unbound variable in the driver

Job `61446924` (`static-full-loader-validate`, `lrdn1279`) failed after 17
seconds before any precompile, data read, or GPU work:

```text
/var/spool/slurmd/job61446924/slurm_script: line 118: name: unbound variable
```

Cause: `precompile_variant()` declared

```bash
local name=$1 dir=$2 log=$LOG_DIR/precompile-$name.log
```

With `set -u`, the `$name` expansion in the same `local` statement was
evaluated before the assignment. Fix: split the declarations:

```bash
local name=$1
local dir=$2
local log="$LOG_DIR/precompile-$name.log"
```

The fix was syntax-checked with `bash -n` and exercised by the later jobs.

## F2 — host RSS recorded 1024× too large

Job `61447030` (`lrdn3410`) completed all loader, fixture, and dry-run checks
with `FULL_VALIDATION PASS`, but the Flux dry-run JSON recorded
`host_maxrss_bytes = 3,158,617,096,192` (about 2.9 TiB), which is impossible on
the 514 GB node.

Cause: the code did

```julia
host_maxrss_bytes = Int64(Sys.maxrss()) * 1024
```

In Julia 1.12, `Sys.maxrss()` already returns bytes on this platform; the
historic kilobyte assumption is wrong. Local verification:

```text
Sys.maxrss()=1567686656
VmHWM=272880 kB   # /proc/self/status
```

Fix: use `Int64(Sys.maxrss())` directly. Job `61450014` re-ran the entire
validation with the fix and recorded `host_maxrss_bytes` of `3,088,822,272`
(Flux current) and `3,071,229,952` (Flux historical), which are plausible.

The `61447030` evidence directory is retained for history but is **superseded**
and must not be cited for memory values. Its device-memory values were not
affected by this unit bug.

## F3 — login-node import probe timed out

During remote inspection, a login-node command

```bash
ssh leonardo '.../venvs/pytorch-scaling/bin/python -c "import numpy, torch; ..."'
```

was terminated by the local 120 s timeout. No remote process remained
(`ps` check clean). This was an inspection mistake, not a workload failure.
All subsequent environment, precompile, Python, and Julia commands ran on the
compute node inside the batch allocation.

## Accepted run

Job `61450014` (`lrdn0667`) printed `FULL_VALIDATION PASS` and
`COMPLETED 0:0` in `00:14:09`. Evidence:
[RESULTS](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md),
[COMMANDS](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/COMMANDS.md).
