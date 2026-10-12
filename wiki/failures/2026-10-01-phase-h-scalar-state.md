# Phase H scalar state synchronization failure

## Cause

The leaf synchronization method reconstructed a mutable `Optimisers.Leaf` through `@set!`.
The wrapper discarded the returned tree. Mutable moment arrays changed, but Adam scalar decay products remained rank-local.
The next updates therefore used incorrect bias correction on the non-root rank.

## RED evidence before the source correction

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
ls temp/phase_h && timeout --signal=TERM --kill-after=10s 210s julia --startup-file=no --project=../flux-integration/test temp/phase_h/01-run-scalar.jl temp/phase_h/01-scalar-RED-mpi2.log
```

Exit 1, 15.865 seconds, no timeout. The two-rank synchronized file passed 92 assertions and failed 10 assertions.
Rank 0 failed five assertions per update API. Rank 1 (the root) passed all 51 assertions.
The new regression passed 22/32 on rank 0 and 32/32 on rank 1.
The failures cover the initial decay tuple and the parameters and decay tuples after two updates.
Both moment arrays and the independent native synchronization reference passed.
The log is `temp/phase_h/01-scalar-RED-mpi2.log`.

## Correction rationale

Assign the synchronized state directly to the existing mutable leaf.
This retains scalar replacements, wrapper identity, and leaf identity without a new traversal or collective.
The pre-edit rationale and final report are in `temp/phase_h/01-scalar-state.md`.

## GREEN evidence

The same focused command with `01-scalar-GREEN-mpi2.log` passed 51/51 assertions per rank.
Exit 0, 27.478 seconds, no timeout. This includes 32 new regression assertions per rank.
The production correction changed only the assignment in the leaf synchronization method and its explanatory comment.

## Inspection error

A read-tool request used `synchronize_distributedtest.jl` and returned `File not found`.
The correct existing filename is `synchronized_distributedtest.jl`. No shell command or test ran for that request.
