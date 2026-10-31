# P0 implementation failures — 2026-10-31

Inspection attempted nonexistent `scripts/local/run.sh`,
`../flux-multinode/src/distributed/DistributedUtils.jl` and
`../flux-multinode/src/distributed.jl`; rg returned exit 2. Correct module is
`src/distributed/public_api.jl`; inspected it instead.

A development acceptance attempt was started while environment precompilation
was still active. It was terminated by SIGTERM to its process tree (root PID
52641), and is excluded from acceptance. Its output is retained as
`artifacts/logs/multinode/p0/acceptance-2.log`. Run only after warm loading.

The first bounded driver attempt failed with `ArgumentError: Non-default
environment behavior is only permitted for the first interpolant` before any
worker launch. MPI.mpiexec() carries environment settings; placing it after
`timeout` in Julia command interpolation is invalid. The driver now interpolates
its argv and explicitly preserves launcher.env with setenv. See
`stage1-attempt1.log`.

The subsequent attempt repeatedly rebuilt dependency cache variants inside MPI
workers despite successful parent precompile/warm imports. Terminated its process
tree (root PID 55621), preserving `stage1-cache-attempt.log`. Final driver uses
`--compiled-modules=existing` after explicit precompilation, prohibiting new
compiled-cache writes during tests. This does not disable JIT execution or change
numerical tolerances. The precise cause of cache incompatibility is unresolved.

Commit review: `git diff --cached --check` flagged four trailing-whitespace
lines emitted by Julia's warning renderer in setup.log. Raw runtime evidence is
preserved; source/documentation whitespace checks pass when logs are excluded.
