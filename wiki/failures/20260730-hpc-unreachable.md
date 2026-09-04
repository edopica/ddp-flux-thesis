# Failure: HPC Unreachable (2026-07-30)

## Operation

C8 negative demonstrations — revert regression guards in `public_api.jl`, run data sharding tests on HPC, capture failures, restore.

## Symptom

```
ssh: Could not resolve hostname slogin.hpc.unibocconi.it: Name or service not known
```

## Root cause

DNS resolution failing for `slogin.hpc.unibocconi.it`. Likely the VPN (Cisco AnyConnect) to the Bocconi university network is disconnected or the DNS tunnel is not propagating.

## Commands attempted

```bash
ssh hpc "echo 'HPC reachable' && hostname"
ssh -v hpc echo ok
```

## Impact

Both demos require HPC MPI execution. Cannot proceed until connectivity is restored.

## Mitigation

All commands are documented in `wiki/reports/c8-negative-demonstrations.md` for re-execution when VPN is reconnected.
