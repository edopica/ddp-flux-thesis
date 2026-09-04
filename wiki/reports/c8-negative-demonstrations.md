# C8 - Negative Demonstrations for Regression Guards

**Date:** 2026-07-30
**Status:** HPC unreachable — commands documented, not executed

## Summary

Both demos rely on running MPI tests on the HPC cluster (`slogin.hpc.unibocconi.it`). The HPC hostname cannot be resolved — DNS resolution fails, likely because the VPN to the Bocconi university network is not connected.

All commands are documented below for re-execution when connectivity is restored.

---

## Guard 1: Cyclic padding fix

**File:** `~/projects/ddp_flux/src/distributed/public_api.jl:276`

**Guard:** `[mod1(i, total_size) for i in 1:(total_padded - total_size)]` wraps padding indices to avoid out-of-bounds access when `total_padded > total_size`.

**Buggy version:** `1:(total_padded - total_size)` (linear range without wrapping)

### Steps

1. **Revert the fix:**
```bash
ssh hpc "sed -i 's|append!(indices, \[mod1(i, total_size) for i in 1:(total_padded - total_size)\])|append!(indices, 1:(total_padded - total_size))|' ~/projects/ddp_flux/src/distributed/public_api.jl"
```

2. **Run the N=1 data sharding test (expects failure — out-of-bounds error):**
```bash
ssh hpc "cd ~/projects/ddp-flux-thesis && SLURM_MPI_TYPE=pmi2 salloc --ntasks=4 --gres=gpu:0 --mem=16G --cpus-per-task=2 --account=3320522 --partition=stud --qos=stud bash -c 'FLUX_TEST_DISTRIBUTED_TIMEOUT=120 ~/.julia/bin/mpiexecjl --project=~/projects/ddp_flux/test -n 4 julia --project=~/projects/ddp_flux/test ~/projects/ddp_flux/test/ext_distributed/data_distributedtest.jl mpi 2>&1'"
```

**Expected failure:** `BoundsError` or equivalent when accessing out-of-bounds data indices (e.g., accessing `data[5]` when `total_size=3` and padding with `4,5,...`).

3. **Restore the fix:**
```bash
ssh hpc "sed -i 's|append!(indices, 1:(total_padded - total_size))|append!(indices, \[mod1(i, total_size) for i in 1:(total_padded - total_size)\])|' ~/projects/ddp_flux/src/distributed/public_api.jl"
```

4. **Confirm passing after restore:**
```bash
ssh hpc "cd ~/projects/ddp-flux-thesis && SLURM_MPI_TYPE=pmi2 salloc --ntasks=4 --gres=gpu:0 --mem=16G --cpus-per-task=2 --account=3320522 --partition=stud --qos=stud bash -c 'FLUX_TEST_DISTRIBUTED_TIMEOUT=120 ~/.julia/bin/mpiexecjl --project=~/projects/ddp_flux/test -n 4 julia --project=~/projects/ddp_flux/test ~/projects/ddp_flux/test/ext_distributed/data_distributedtest.jl mpi 2>&1'"
```

---

## Guard 2: Empty dataset check

**File:** `~/projects/ddp_flux/src/distributed/public_api.jl:266`

**Guard:** `if total_size == 0 throw(ArgumentError("Cannot distribute an empty dataset"))`

**Buggy version:** Skip the check entirely (no exception thrown for empty datasets).

### Steps

1. **Revert the fix:**
```bash
ssh hpc "sed -i 's|if total_size == 0|if false # REVERTED|' ~/projects/ddp_flux/src/distributed/public_api.jl"
```

2. **Run the N=0 data test (expects failure — test expects `ArgumentError` but doesn't get it):**
```bash
ssh hpc "cd ~/projects/ddp-flux-thesis && SLURM_MPI_TYPE=pmi2 salloc --ntasks=2 --gres=gpu:0 --mem=16G --cpus-per-task=2 --account=3320522 --partition=stud --qos=stud bash -c 'FLUX_TEST_DISTRIBUTED_TIMEOUT=120 ~/.julia/bin/mpiexecjl --project=~/projects/ddp_flux/test -n 2 julia --project=~/projects/ddp_flux/test ~/projects/ddp_flux/test/ext_distributed/data_distributedtest.jl mpi 2>&1'"
```

**Expected failure:** The test's `@test_throws ArgumentError` assertion fails because no exception was raised. Likely a `BoundsError` or division-by-zero error in subsequent code instead of the clean `ArgumentError`.

3. **Restore the fix:**
```bash
ssh hpc "sed -i 's|if false # REVERTED|if total_size == 0|' ~/projects/ddp_flux/src/distributed/public_api.jl"
```

4. **Confirm passing after restore:**
```bash
ssh hpc "cd ~/projects/ddp-flux-thesis && SLURM_MPI_TYPE=pmi2 salloc --ntasks=2 --gres=gpu:0 --mem=16G --cpus-per-task=2 --account=3320522 --partition=stud --qos=stud bash -c 'FLUX_TEST_DISTRIBUTED_TIMEOUT=120 ~/.julia/bin/mpiexecjl --project=~/projects/ddp_flux/test -n 2 julia --project=~/projects/ddp_flux/test ~/projects/ddp_flux/test/ext_distributed/data_distributedtest.jl mpi 2>&1'"
```

---

## Precompile (if needed before Demos)

```bash
bash /home/kurapica/Projects/ddp_flux/ddp-flux-thesis/scripts/remote/precompile.sh hpc
```

---

## Resolution

- **HPC connectivity issue:** `slogin.hpc.unibocconi.it` — `Name or service not known`
- **Root cause:** DNS resolution fails; VPN to Bocconi university network likely disconnected or DNS not propagated.
- **Next action:** Reconnect VPN and re-run the commands above sequentially.
