# Failure: Bocconi Cluster Unreachable (2026-07-30)

## Operation

C8 negative demonstrations — revert regression guards in `public_api.jl`, run data sharding tests on Bocconi, capture failures, restore.

## Symptom

```
ssh: Could not resolve hostname slogin.hpc.unibocconi.it: Name or service not known
```

## Root cause

DNS resolution failing for `slogin.hpc.unibocconi.it`. Likely the VPN (Cisco AnyConnect) to the Bocconi university network is disconnected or the DNS tunnel is not propagating.

## Commands attempted

```bash
ssh bocconi "echo 'Bocconi reachable' && hostname"
ssh -v bocconi echo ok
```

## Impact

Both demos require Bocconi MPI execution. Cannot proceed until connectivity is restored.

## Mitigation

All commands are documented in `wiki/reports/c8-negative-demonstrations.md` for re-execution when VPN is reconnected.
