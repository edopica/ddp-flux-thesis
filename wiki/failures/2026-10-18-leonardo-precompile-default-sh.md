# Leonardo default precompile script cannot resolve the offline root env

Date: 2026-10-18
Scope: static-scaling R1/R2 remote verification; Leonardo compute-node precompile
Status: **WORKAROUND** — the canonical Phase 3 precompile passed; no source
change.

## Symptom

`scripts/remote/precompile.sh leonardo` failed on compute node `lrdn2693`
(job `61073207`) with exit 2 and no `FLUX_PRECOMPILE_OK`:

```text
┌ Warning: could not download https://pkg.julialang.org/registries
│   exception = RequestError: Failed to connect to pkg.julialang.org port 443 ...
     Cloning [533571aa-0936-420e-b4be-9c66f5f626ca] cuFFT from https://github.com/JuliaGPU/CUDA.jl.git
ERROR: failed to clone from https://github.com/JuliaGPU/CUDA.jl.git, error: GitError(Code:ERROR, Class:OS, failed to connect to github.com: Network is unreachable)
...
srun: error: lrdn2693: task 0: Exited with exit code 1
make: *** [Makefile:20: precompile] Error 1
```

Full log: `artifacts/logs/leonardo/2026-10-18-r1-r2-precision-fixes-precompile.log`.

## Root cause

`precompile.sh` runs `make precompile-all`, whose `precompile` target calls
`Pkg.precompile()` with `--project=.` (the thesis root). `.rsyncignore`
excludes `Manifest.toml`, so the remote thesis tree has no root manifest; Pkg
resolves the environment from scratch and needs to clone `cuFFT`. Leonardo
compute nodes have no outbound network. The marker `FLUX_PRECOMPILE_OK` is
printed only by `scripts/remote/leonardo/precompile-phase3.sh`, not by this
target.

This is a pre-existing workflow mismatch, not an effect of the changed
scripts.

## Workaround

Run the canonical Phase 3 precompile under `srun`, as in the accepted campaign:

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh"
```

Job `61074887` (`lrdn0421`, UTC `2026-10-18T08:05:58Z`) printed
`FLUX_PRECOMPILE_OK` and exited 0. The subsequent precision regression
(job `61077382`) found no `Precompiling packages...` line and passed. The
final hardened precompile is job `61187225` (`lrdn0241`, UTC
`12:20:59`-`12:24:36`, `FLUX_PRECOMPILE_OK`, exit 0); `61074887` is the
earlier intermediate run.

## Impact

None on the R1/R2 verdict. Both logs are retained; do not delete the failed
one. Use `precompile-phase3.sh` for the flux-integration runtime environment.

## Links

- [R1/R2 verification devlog](../devlog/03-static-scaling/2026-10-18-r1-r2-precision-policy-fixes.md)
- [Leonardo context](../context/leonardo.md)
