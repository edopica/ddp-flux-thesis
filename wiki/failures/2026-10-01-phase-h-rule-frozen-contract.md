# Phase H: full optimizer restore is a false synchronization promise

Date: 2026-10-01

## Failed command

Working directory: `/home/kurapica/Projects/ddp_flux/ddp-flux-thesis`.

```bash
ls temp/phase_h && timeout --signal=TERM --kill-after=10s 210s julia --startup-file=no --project=../flux-integration/test temp/phase_h/03-run-contract.jl hypothesis temp/phase_h/03-full-restore-RED-mpi2.log
```

Exit 1, 12.403 seconds, no timeout.
Rank 0 passed four assertions and failed eight. Root rank 1 passed all twelve assertions.
No test errors occurred.

## Cause and decision

The diagnostic deliberately asserts a complete optimizer restore after `synchronize!!`.
Both raw and wrapped paths retain rank-local rule configuration and frozen flags.
Rank 0 retains Adam learning rate `0.01`, while the root uses `0.02`.
In a separate case, rank 0 stays active while the root stays frozen.
The next updates differ despite identical gradients.

This failure rejects the full-restore hypothesis. It does not establish a metadata transport defect under the chosen contract.
The contract requires matching rule configuration and frozen flags on every rank before distributed updates.
Source documentation must state that precondition and distinguish numerical synchronization from checkpoint restore.
The prior scalar-state and tied-leaf corrections remain intact.

Evidence: `temp/phase_h/03-full-restore-RED-mpi2.log`.
Decision and final results: `temp/phase_h/03-rule-frozen-contract.md`.

## GREEN evidence

```bash
ls temp/phase_h && timeout --signal=TERM --kill-after=10s 210s julia --startup-file=no --project=../flux-integration/test temp/phase_h/03-run-contract.jl contract temp/phase_h/03-contract-GREEN-mpi2.log
```

Exit 0, 29.207 seconds, no timeout. Both ranks passed 499 assertions each.
The new contract cases contribute 288 assertions per rank.
They establish local metadata preservation, divergence outside the precondition, and matching continuation after explicit configuration alignment.
The remaining assertions retain scalar-state and tied-leaf coverage.
The GREEN tests assert the chosen contract, not the rejected full-restore hypothesis.
