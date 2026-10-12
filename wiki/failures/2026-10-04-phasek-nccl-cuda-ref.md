# Phase K: GPU-only buffer check failed on the first NCCL attempt

Date: 2026-10-04

## Failure: CUDA referenced in a device-generic module

Command (deathstar, first attempt):

```bash
julia --startup-file=no --project=/home/picazio/projects/flux-integration/test \
  scripts/checks/conditional_graph_demo_check.jl \
  --root artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-deathstar-nccl2 \
  --backend nccl --ranks 2 --label toy-deathstar-nccl2
```

`prepare` passed (exit 0); `nccl2-train` exited 1 with
`ERROR: LoadError: UndefVarError: CUDA not defined in Main.CGCommon`
at `scripts/examples/conditional_graph/common.jl:224` in
`check_collective_trace` (`check_collective_trace` names
`CUDA.CuArray` in the gradient-buffer type assertion).

Cause: `CGCommon` deliberately does not import CUDA so CPU runs never load
the CUDA stack. The host/device check named `CUDA.CuArray` only on the
`on_gpu` branch, which the local 2- and 4-rank MPI campaigns never evaluate.
No wrapper collective had failed; the demo aborted in its own post-step
verification.

Correction: the check is dependency-free. A device buffer is any array that
is not a host `Array` (`common.jl:224`). CPU behavior is unchanged.

Evidence:

- Failing console, synced back before the rerun:
  `artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-deathstar-nccl2-attempt1/nccl2-train/console.log`
  (exit 1, `UndefVarError`).
- Corrected 2-rank NCCL campaign PASS, 6/6 driver cells:
  `artifacts/logs/conditional-wrapper/demo/toy-2026-10-04-deathstar-nccl2/`.
