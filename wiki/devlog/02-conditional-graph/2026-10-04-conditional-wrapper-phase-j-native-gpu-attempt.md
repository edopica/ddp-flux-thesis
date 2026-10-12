# Conditional-graph wrapper Phase J: native CUDA/NCCL implementation and GPU attempt

Date: 2026-10-04
Host: local (implementation/CPU validation) + `deathstar` (GPU attempt)
Flux worktree: `/home/kurapica/Projects/ddp_flux/flux-integration` (branch
`ddp/integration-m1-spike`, HEAD `55171e65`, dirty; nothing committed)
Handoff for the next agent: `temp/phaseJ_handoff.md`

## Implemented

- New `test/ext_distributed/cuda_conditional_distributedtest.jl` (device-generic:
  `CuArray` under NCCL, `Array` under MPI; `CUDA.allowscalar(false)` before model
  construction under NCCL). Testsets: NCCL rank-to-device mapping; 6-step
  rank-dependent conditional forward (both update APIs, globally absent steps
  with nonzero Adam history, caller-gradient preservation, independent analytic
  Adam reference, ordered collective trace, presence type/placement);
  tied device parameter; nested traversal with an excluded device array;
  transpose parent-space pairing; whole-model freeze/thaw.
- `test/Project.toml`: added CUDA, NCCL, cuDNN to the test deps.
- `test/ext_distributed/runtests.jl`: conditional `import CUDA` for the NCCL
  pass (the sizer called `CUDA.devices()` without importing it).

## Local CPU/MPI evidence

- New file, 2 ranks: 212 assertions per rank (171 + 11 + 12 + 8 + 10), exit 0.
- Complete dedicated suite, 2 ranks: `Distributed | 7 7 2m28.8s`, exit 0.
- Log: `artifacts/logs/conditional-wrapper/phaseJ-local-suite-mpi2-2026-10-04.log`.

## deathstar GPU attempt

- GPUs healthy: 3× TITAN RTX, driver 555.42.02; direct `CUDA.functional()` true
  (runtime 12.9.0 artifact); MPI+CUDA probe passes on 2 ranks.
- The Phase F remote root manifest was stale (StridedViews conflict). It was
  backed up to `Manifest.toml.phaseF-bak` and regenerated fresh; the workspace
  test env then resolved Flux 0.16.11 / Optimisers 0.4.9 / Functors 0.5.3 /
  MPI 0.20.27 / CUDA 6.2.2 / NCCL 0.1.3 and precompiled 188 deps in 137 s.
- The first/native run failed in ~9 s:
  `CUDA devices are not functional and force_cuda is set` from
  `FluxMPIExt.__initialize` (`ext/FluxMPIExt/FluxMPIExt.jl:34`).
- **Root cause**: `FluxMPIExt` checks `MLDataDevices.functional(CUDADevice)`,
  which in MLDataDevices 1.17.10 is defined only in `ext/cuDNNExt.jl:37`.
  Without cuDNN loaded the generic fallback `functional(x) = false` applies,
  so NCCL init fails even though CUDA works. `test/runtests.jl` adds cuDNN only
  for `FLUX_TEST_CUDA=true`; the dedicated runner did not.
- Fix applied locally (cuDNN added to `test/Project.toml`) but not yet synced,
  resolved, or run. The GPU gate has **not** passed yet; Phase J is incomplete.

## Evidence

- `phaseJ-precompile-deathstar-2026-10-04.log`, `phaseJ-versions-deathstar-2026-10-04.log`
- `phaseJ-gpu-focused-nccl2-deathstar-2026-10-04.log` (failure + diagnosis context)
- `phaseJ-local-suite-mpi2-2026-10-04.log`
- Full operational detail: `temp/phaseJ_handoff.md`
