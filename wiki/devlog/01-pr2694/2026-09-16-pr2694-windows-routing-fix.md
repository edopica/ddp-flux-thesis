# 2026-09-16 - PR #2694 Windows routing-test fix

## Failure

The Windows CI job failed in `test/distributed_routing.jl`. The assertion
`occursin(runner_path, string(expr))` compared a raw Windows path with the
rendered Julia expression, where backslashes in the string literal are escaped.
The routing helper itself stored the correct path.

## Fix

Commit `69703025` replaced rendered-text matching with direct inspection of the
path argument stored in the `Expr`. Commit `fee2c310` removed a redundant
Windows-style fixture added in the first fix and retained the minimal assertion:

```julia
@test expr.args[2] == runner_path
```

The existing assertions still verify that the value is a call expression whose
callee is `include`.

## Verification

From `/home/kurapica/Projects/ddp_flux/flux-pr2694-salvage`:

```bash
julia --startup-file=no test/distributed_routing.jl
git diff --check
```

Result: routing tests 17/17 PASS, exit 0; whitespace check passed.

## Push

Command:

```bash
git push origin HEAD:ddp/upstream-pr
```

The push fast-forwarded the PR branch from `c64f695f` to
`fee2c310a32958ca2f44c5ce5c1ddbef3db87c75`. PR #2694 is open and live CI is
running, including the Windows job.

## Next action

Monitor PR #2694 CI at `fee2c310`, especially the Windows job, and inspect any
failure before merge.
