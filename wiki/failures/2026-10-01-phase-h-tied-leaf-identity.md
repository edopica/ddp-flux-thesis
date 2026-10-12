# Phase H tied-leaf synchronization: RED evidence

Date: 2026-10-01

## Historical defect

The historical `synchronize!!(::Leaf)` uses `@set!` to reconstruct each visited leaf.
Two occurrences of one leaf therefore return distinct leaves.
Native Optimisers accumulates gradients by leaf identity and caches parameter updates by `(leaf, parameter)`.
The reconstructed tree loses this contract, even though synchronized state values initially match.

Agent 1 corrected the shared cause with direct assignment to mutable `Leaf.state`.
Agent 2 added identity and continuation regressions without another production change.

## Failed commands

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
ls temp/phase_h && timeout --signal=TERM --kill-after=10s 210s julia --startup-file=no --project=../flux-integration/test temp/phase_h/02-run-tied.jl historical temp/phase_h/02-tied-RED-mpi2.log
ls temp/phase_h && timeout --signal=TERM --kill-after=10s 210s julia --startup-file=no --project=../flux-integration/test temp/phase_h/02-run-tied.jl historical temp/phase_h/02-tied-RED2-mpi2.log
```

The first command returned 1 after 18.587 seconds, without a timeout.
Its raw-tree fixture supplied integer arrays to an in-place average reduction.
The fractional mean caused `InexactError: Int64(1.5)`.
Each rank reported two errors, one for each raw update path.
Rank 0 reported 107 passes and 44 failures. Rank 1 reported 145 passes and six failures.
This run establishes no complete continuation result.
Explicit `Float64` gradient arrays correct the fixture.

The second command returned 1 after 17.780 seconds, without a timeout or test error.
The whole file reported 125 passes and 86 failures on rank 0, and 163 passes and 48 failures on rank 1.
Each raw-tree update path reported 15 passes and 24 failures per rank.
Those failures cover original-leaf identity, shared-leaf identity, parameters, and Adam moments after updates.
The wrapper retains its original tree under the historical method, but rank 0 loses scalar replacements.
Its two tied wrapper cases each reported 27 passes and 14 failures. Rank 1 passed both cases (41 each).
Agent 1's existing scalar regression also reproduced ten failures on rank 0.

## Isolation and correction

The child reads the exact historical method from Flux commit `55171e65` and evaluates it only in each disposable process.
The runner compares production bytes before and after each run.
Every log records `PRODUCTION_BYTES_UNCHANGED true`.
No source restoration is necessary because the runner never edits the source.

The unchanged final tests passed on Agent 1's corrected source:

```bash
ls temp/phase_h && timeout --signal=TERM --kill-after=10s 210s julia --startup-file=no --project=../flux-integration/test temp/phase_h/02-run-tied.jl current temp/phase_h/02-tied-GREEN-mpi2.log
```

Result: exit 0, 14.765 seconds, no timeout, 211/211 assertions per rank.
The new tied regression contributes 160/160 assertions per rank.
The report `temp/phase_h/02-tied-leaf-identity.md` records scope and handoff details.
