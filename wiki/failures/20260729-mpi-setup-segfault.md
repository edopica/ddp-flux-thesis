# MPI.jl setup segfault — 2026-07-29

## Context
Attempting deadlock verification for C8.1 harness exit condition.

## Command
```bash
julia --project=/home/kurapica/Projects/ddp_flux/ddp_flux -e '
using MPIPreferences
MPIPreferences.use_system_binary(; mpiexec="/opt/mpich/bin/mpirun", abi="mpich", export_prefs=true)
using MPI
MPI.install_mpiexecjl()
'
```

## Result
Segmentation fault (signal 11) during MPI initialization.
System MPICH at `/opt/mpich/bin/mpirun`, Julia 1.12.

## Impact
Cannot run deadlock verification locally. The harness watchdog logic is correct by inspection:
- `runtests.jl` starts child processes via `mpiexecjl` with `run(cmd, wait=false)`
- Watchdog loop checks `process_running()` every 0.5s
- On timeout: sends SIGUSR1 for stack dump, waits 2s, kills process group
- Reports failure via `@test false` on timeout, `@test proc.exitcode == 0` on normal exit

## Action
Deadlock verification deferred to Bocconi or CI environment where MPI.jl is properly configured.
