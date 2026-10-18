# Leonardo Cluster (`leonardo`)

The CINECA GPU cluster. The SSH alias `leonardo` (`HostName login.leonardo.cineca.it`, user `epicazio`) is defined in `~/.ssh/config`.

**Status 2026-10-04:** Operative. SSH, Slurm, modules/PMI, single-node 4-GPU use, and the Julia workflow are verified. Julia 1.12.6 lives on `$WORK` (depot `$WORK/epicazio/julia-depot`), `scripts/remote/hosts/leonardo.conf` exists, and the **4-rank NCCL gate passed**: Phase J focused regression 217/217 assertions per rank and the full dedicated suite 14/14 under both MPI and NCCL, all exit 0. Evidence: `artifacts/logs/conditional-wrapper/phaseJ-gpu-focused-nccl4-leonardo-2026-10-04.log`, `phaseJ-remote-suite-nccl4-leonardo-2026-10-04.log`, `artifacts/logs/leonardo/`, devlog `2026-10-04-leonardo-bringup-4rank-nccl.md`.

**Status 2026-10-05:** Phase K complete. The K1 toy and K2 MNIST/Fashion-MNIST CUDA/NCCL 4-rank cells PASS (driver exit 0, `PHASE_K_SUMMARY ... backend=nccl ranks=4 PASS`) with roots `artifacts/logs/conditional-wrapper/demo/k1-2026-10-05-leonardo-nccl4` and `k2-2026-10-05-leonardo-nccl4` on nodes `lrdn3394` and `lrdn0436`. Evidence: campaign record `artifacts/logs/conditional-wrapper/demo/CAMPAIGN-2026-10-05-leonardo-k2.md`, local driver logs `artifacts/logs/leonardo/2026-10-05-k1-leonardo-nccl4.log` / `...-k2-...log`, devlog `2026-10-05-leonardo-k1-k2-4rank-nccl.md`.

**Status 2026-10-05 (post-Phase K timing stretch):** the optional live 1×A100 versus 4×A100 NCCL timing stretch is complete. Correctness preflight PASS (`single` trajectory `maxabs=2.980232e-08`, `ddp` `maxabs=0.0`); baseline 5/5 and DDP 5/5 timed repeats. Jobs: correctness `59434349` on `lrdn0790`, baseline `59434611` on `lrdn1990` (1 process, device 0), DDP `59435161` on `lrdn1847` (4 ranks, devices `[1,2,3,0]`), GPU check `59434335` on `lrdn3078`. Measured diagnostic speedup `0.465324 / 0.314365 = 1.480203` for the tiny K2 schedule. Report: `reports/conditional-graph-phase-k-timing-stretch.md`; campaign record `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/CAMPAIGN.md`; devlog `2026-10-05-phase-k-timing-stretch.md`.

**Status 2026-10-06 (timing corrections):** the timing-correction campaign is complete. Correctness preflight PASS (`single` trajectory `maxabs=2.980232e-08`, `ddp` `maxabs=0.000000e+00`) and baseline 5/5 plus DDP 5/5 with the corrected driver. Jobs: correctness `59514262` on `lrdn1934`, timing `59514959` on `lrdn0009` (4× A100-SXM-64GB, driver 535.274.02, DDP devices `[1,2,3,0]`). Corrected ratio-of-medians training-only speedup `0.263440 / 0.139490 = 1.888594`; total speedup `0.438678 / 0.313939 = 1.397335`. The 2026-10-05 instrumented `1.480203` is superseded. Evidence: run root `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/` (`summary-correctness.txt`, `summary-timing.txt`, both PASS), report `reports/conditional-graph-phase-k-timing-stretch.md`, campaign record `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-06-leonardo/CAMPAIGN.md`, local logs `artifacts/logs/leonardo/2026-10-06-timing-correctness-r2.log` / `...-timing-r2.log`, devlog `wiki/devlog/02-conditional-graph/2026-10-06-timing-corrections.md`.

**Status 2026-10-15 (Python/PyTorch benchmark environment):** the static-network scaling benchmark environment is ready on `$WORK`; no experiment has run. A uv-managed CPython 3.11.17 venv with torch 2.9.1+cu126 and torchvision 0.24.1+cu126 lives at `$WORK/epicazio/venvs/pytorch-scaling` (uv 0.13.1 in `$WORK/epicazio/tools/uv`; uv cache `$WORK/epicazio/uv-cache`; uv-managed Python `$WORK/epicazio/uv-python`), with the 28-line resolved freeze in `$VENV/requirements-freeze.txt` and records in `$VENV/ENV-RECORD.md`. Compute-node environment verification PASS (`ENV_VERIFY=PASS`, exit 0, one A100-SXM-64GB, driver 535.274.02; the smoke is a single kernel, not a benchmark). Tiny ImageNet-200 is downloaded and verified at `$WORK/epicazio/datasets/tiny-imagenet-200/`: archive 248,100,043 bytes, SHA-256 `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`, published-MD5 cross-check MATCH, `unzip -t` OK; train 100,000 JPEG (200 classes, 500 each), val 10,000 JPEG with 10,000 annotation lines, test 10,000 JPEG, `wnids.txt` 200; metadata `$WORK/epicazio/datasets/tiny-imagenet-200-metadata.md`. Reusable scripts: `scripts/remote/leonardo/setup-pytorch-env.sh`, `verify-pytorch-env.py`, `requirements-pytorch-scaling.txt`, `download-tiny-imagenet.sh`. Evidence: `artifacts/logs/leonardo/2026-10-15-pytorch-env-setup.log`, `artifacts/logs/leonardo/2026-10-15-pytorch-env-freeze.txt`, `artifacts/logs/leonardo/2026-10-15-tiny-imagenet-setup.log`; devlog `wiki/devlog/03-static-scaling/2026-10-15-leonardo-benchmark-env-setup.md`. No model or training code exists yet.

**Status 2026-10-16 (static scaling Phase 2):** Phase 2 complete and independently verified. The prepared dataset (canonical decoded images, 50-epoch schedule, normalization statistics, 62-array initial bundle) lives at `$WORK/epicazio/static-scaling-prepared/`. Prep job `60758615` (`lrdn0018`, `boost_qos_lprod`, COMPLETED 0:0) and acceptance job `60758839` (`lrdn0011`, `boost_qos_dbg`, COMPLETED 0:0) both PASS (`PHASE2_DATASET_CHECK`, `PHASE2_JULIA_DATA_CHECK`, `PHASE2_REMOTE_CHECK`); the local operator check adds 75 checks, exit 0. Evidence: `artifacts/logs/static-scaling/2026-10-16-phase2/`, `artifacts/logs/leonardo/2026-10-16-phase2-*.log`; campaign devlog `wiki/devlog/03-static-scaling/2026-10-16-phase2-artifacts-complete.md`; verification `reports/static-scaling-phase2-verification.md`.

**Status 2026-10-17 (acceptance hardening):** the Phase 2 acceptance checks were hardened (local `check_all.jl` 81 checks; `negative_tests_local.py` `NEGTEST PASS`); the remote re-check on the real `$PREP` is pending the SSH publickey access issue ([failure](../failures/2026-10-17-leonardo-ssh-publickey.md)).

**Status 2026-10-17 (Phase 3 Step 2 precision):** the controlled stage-2 experiment PASSED on `lrdn0102` (job 60896936). Stock `CUDNN_DEFAULT_MATH` retained the conv failure (`conv1 3.57e-3`, `conv2 3.65e-3`), while explicit `CUDNN_FMA_MATH` was at CPU level (`0`, `1.9e-6`, zero violations). The pinned-algorithm regression PASSED on `lrdn2176` (job 60900440). Evidence: `artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/`; devlog `2026-10-17-phase3-step2-precision-experiment.md`.

**Status 2026-10-17 (Phase 3 Step 3b):** precompile job `60909956` (`lrdn0034`), precision regression job `60910394` (`lrdn0365`, `PHASE3_PRECISION_REGRESSION PASS`), full driver job `60910786` (`lrdn0001`, exit 1). The module-load FMA chooser made all 32 full-model choices `CUDNN_FMA_MATH(3)` with no fallback; the forward now passes, but `grads.u1`, `grads.u3`, and `m.u3` remain above the unchanged tolerances (FP32 backward FFT/Winograd/direct algorithm rounding). Evidence: `artifacts/logs/static-scaling/2026-10-17-phase3-step3b/`; devlog `2026-10-17-phase3-step3b-fma-algo-chooser.md`.

**Status 2026-10-18 (Phase 2 hardened re-check):** the hardened Phase 2 checkers ran against the real `$PREP`/fixtures on job `60951247` (`lrdn0171`, `boost_qos_dbg`, 1 task, 8 CPUs, `--gres=none`, COMPLETED 0:0). `PHASE2_DATASET_CHECK PASS`, `PHASE2_JULIA_DATA_CHECK PASS`, `PHASE2_REMOTE_CHECK PASS`, both checker exits 0, driver exit 0; 113 CHECK lines PASS, 0 FAIL. The evidence root is `$WORK/epicazio/static-scaling-evidence/2026-10-18-phase2-hardened-recheck/`; local copy `artifacts/logs/static-scaling/2026-10-18-phase2-hardened-recheck/`. This closes the Phase 2 remote prerequisite for Phase 3 Step 6.

**Status 2026-10-18 (Phase 3 final acceptance):** Phase 3 is accepted. Precompile job `60954207` (`lrdn3362`, `FLUX_PRECOMPILE_OK`), precision regression `60954547` (`lrdn2211`, `PHASE3_PRECISION_REGRESSION PASS`), optimizer probe `60955690` (`lrdn0279`, `PHASE3_OPTIMIZER_PROBE PASS`, 34/34 controls), and the final driver `60956582` (`lrdn0165`, UTC `2026-10-18T01:25:21Z`, exit 0) with `PHASE3_REFERENCE PASS`, `PHASE3_REFERENCE_CHECK PASS`, `PHASE3_FLUX_CHECK PASS`, `PHASE3_REMOTE_CHECK PASS`. All 32 cuDNN choices are `CUDNN_FMA_MATH(3)`/no fallback; the reference is 337/337 exact-equal to the preserved attempt-3 archive; remote/local SHA-256 matches 15/15. Evidence: `$WORK/epicazio/static-scaling-evidence/2026-10-18-phase3-final/`; local copy `artifacts/logs/static-scaling/2026-10-18-phase3-final/`. The SSH publickey rejection returned immediately after the evidence copy; no remote artifact was needed after that.

## Environment

- Login nodes `login01`…`login05`: Xeon Platinum 8358 (Ice Lake), 2 sockets × 32 cores × 2 threads, RHEL 8.8.
- Compute nodes (`boost_usr_prod`): one Xeon Platinum 8358 (32 cores, 1 thread/core, 2 NUMA nodes), 514 GB RAM, SLURM 25.05.
- GPUs: 4× NVIDIA A100-SXM-64GB per node (CC 8.0, driver 535.274.02, 65536 MiB). GRES `gpu:a100:4`; 13024 GPUs in the partition.
- Fabric: 4× InfiniBand HCA (`mlx5_0`…`mlx5_3`). NVLink links report 25 GB/s × 12. `nvidia-smi topo -m` does **not** print `NV#` entries and the fabric manager is inactive, but CUDA P2P access works on all 12 ordered pairs (44–90 GB/s measured, asymmetric). Do not assume a performance figure; measure under NCCL.
- Account/partition/QoS (usable by `epicazio`): account `iscrc_unmasked`, partition `boost_usr_prod`.
  - `boost_qos_dbg`: 30 min max, 8 nodes / 32 GPUs, cpu=256, mem≈3.95 TB — use for quick tests and precompile.
  - `boost_qos_bprod`: MaxWall 1 day, 256 nodes, **MinTRES `node=65`** — `-N1` requests are rejected (`salloc: error: QOSMinCpuNotSatisfied`, verified 2026-10-16); do not use for single-node work.
  - `boost_qos_lprod`: 4 days max, MaxTRES `node=8`, **no MinTRES** — the QoS for long single-node jobs (prep runs use `QOS=boost_qos_lprod TIME=02:00:00`).

  Verified QoS policy (2026-10-16, `sacctmgr show qos` output in `artifacts/logs/leonardo/2026-10-16-phase2-prep.log`): `boost_qos_bprod` MinTRES `cpu=2080,gres/gpu=260,mem=31280G,node=65`; `boost_qos_lprod` and `boost_qos_dbg` no MinTRES.
- Partitions `dcgp_usr_prod` (CPU), `lrd_all_viz`, `lrd_all_serial` exist but are not used by this project.

## Modules, MPI, NCCL, PMI

- `cuda/12.2` (default), `12.3`, `12.6`; `cudnn/8.9.7.29-12`; `nccl/2.22.3-1--gcc--12.2.0-cuda-12.2`; `openmpi/4.1.6--gcc--12.2.0-cuda-12.2` (default) and `5.0.9`; `hpcx-mpi/2.19` / `2.25.1`; `intel-oneapi-mpi/2021.12.1`.
- `srun --mpi=list` returns `none`, `pmix` (v3, v5), `pmi2`, `cray_shasta`.
- Julia's default `MPICH_jll` expects PMI2: use `srun --mpi=pmi2`, i.e. `SLURM_MPI_TYPE=pmi2`, exactly as on Bocconi. Never let an OpenMPI/PMIx launcher initialize the MPICH build (the `PMIX_RANK`/`OMPI_COMM_WORLD_RANK` guard applies here).
- No `julia` module is provided. Julia 1.12.6 is installed under `$WORK/epicazio/julia-1.12.6` with depot `$WORK/epicazio/julia-depot` (installed 2026-10-04; see the bring-up workflow below).

## CPU targets and precompilation

`/proc/cpuinfo` flags are byte-identical on login and compute nodes (md5 `cdd007a665abdd169b9384296f33ffec`), so the Bocconi login/compute cache incompatibility does not exist here. Still precompile through a compute node (`scripts/remote/precompile.sh leonardo`; `scripts/remote/hosts/leonardo.conf` exists since 2026-10-04): that is where the tests run, and login-node compiles are heavy for a shared node. If a run prints `Precompiling packages...`, a sync happened without a precompile: stop, precompile, rerun. `boost_qos_dbg` caps at 30 min; if the first precompile (CUDA stack) needs longer, use `boost_qos_lprod` (no MinTRES; `boost_qos_bprod` rejects single-node jobs with `QOSMinCpuNotSatisfied`).

## GPU allocation (verified 2026-10-04)

Give each rank its own GPU with `--gpus-per-task=1`; Slurm then remaps `CUDA_VISIBLE_DEVICES` per rank. `--gres=gpu:4` alone exposes all four GPUs to every rank. Verified on one node, 4 ranks, 4 distinct A100 UUIDs, CUDA kernel on each:

```bash
srun -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
  -N1 -n4 --cpus-per-task=4 --gpus-per-task=1 --time=00:10:00 \
  bash -lc 'nvidia-smi -L'
```

Like Bocconi, `salloc` control commands stay on the login node; only the inner `srun` reaches compute nodes:

```bash
salloc -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_dbg \
  -N1 -n4 --gres=gpu:4 --cpus-per-task=4 --time=00:05:00 \
  bash -c "srun --mpi=pmi2 ..."
```

## `srun` and the `env` token (verified 2026-10-16)

Never pass the `env` exec token to `srun` on Leonardo. `PATH` starts with
`$WORK/epicazio/tools/uv`, whose activation script
`$WORK/epicazio/tools/uv/env` is mode 644 (non-executable) and is sourced by
`.bash_profile`, `.bashrc`, and `.profile`. `srun` resolves `env` through
`PATH` and `execve`s that file; job 60729163 died in about 2 seconds:

```text
[2026-10-16T17:00:32.630] error: execve(): /leonardo_work/IscrC_UNMASKED/epicazio/tools/uv/env: Permission denied
srun: error: lrdn2229: task 0: Exited with exit code 13
```

A login `bash -l` skips non-executable files and resolves `env` to
`/usr/bin/env`, so the failure appears only under `srun`. Workaround: pass the
variables inside a shell:

```bash
srun ... bash -c 'VAR=... cmd'
```

Set wrapper resources (`QOS=`, `TIME=`, `NTASKS=`, `CPUS_PER_TASK=`, `GRES=`)
as environment prefixes on `scripts/remote/run.sh`; do not pass them via the
`env` token. Do not `chmod` or move the uv activation script: it is part of
the benchmark environment. Full record:
[env-token and QoS failure](../failures/2026-10-16-phase2-prep-leonardo-env-and-qos.md).

## Journal and storage

- `$HOME` (`/leonardo/home/userexternal/epicazio`) is a 50 GB filesystem.
- `$WORK` = `/leonardo_work/IscrC_UNMASKED` (1 TB, 5% used) and `$SCRATCH` = `/leonardo_scratch/large/userexternal/epicazio` are available. The Julia depot and downloaded artifacts should live on `$WORK` or `$SCRATCH`, not in the 50 GB home.
- The scaling-benchmark Python environment and dataset live under `$WORK/epicazio/`: the venv in `venvs/pytorch-scaling`, the uv tool and cache in `tools/uv` and `uv-cache`, the uv-managed CPython in `uv-python`, and Tiny ImageNet-200 plus its metadata under `datasets/`.
- Login nodes have outbound HTTPS (github.com, julialang-s3, pkg.julialang.org reachable; no proxy).

## Bring-up workflow (validated 2026-10-04)

Done once, in this order. All commands run from the thesis repo root except
where noted.

1. Install Julia 1.12.6 on `$WORK` (login node; `$HOME` is only 50 GB):
   ```bash
   ssh leonardo 'mkdir -p "$WORK/epicazio" && cd "$WORK/epicazio" && \
     curl -fsSLO https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz && \
     tar xzf julia-1.12.6-linux-x86_64.tar.gz && rm -f julia-1.12.6-linux-x86_64.tar.gz && \
     mkdir -p julia-depot && "$WORK/epicazio/julia-1.12.6/bin/julia" --version'
   ```
2. Sync `flux-integration` (manual rsync; protect `test/Manifest.toml` and the
   root `Manifest.toml`) and the thesis repo (`--exclude-from='.rsyncignore'`).
3. Resolve on the login node (downloads only, no precompile):
   ```bash
   ssh leonardo 'cd ~/projects/flux-integration && \
     JULIA_DEPOT_PATH=$WORK/epicazio/julia-depot JULIA_PKG_PRECOMPILE_AUTO=0 \
     $WORK/epicazio/julia-1.12.6/bin/julia --startup-file=no --project=test \
     -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"'
   ```
4. **Pre-download the CUDA artifacts on the login node** (compute nodes have no
   internet; the driverless login does not select them):
   ```bash
   JULIA_DEPOT_PATH=... julia --startup-file=no scripts/remote/leonardo/fetch-cuda-artifacts.jl
   ```
5. Precompile on a GPU-allocated node (driver visible, otherwise CUDA.jl bakes
   the "no driver" state):
   ```bash
   GRES=gpu:4 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
     "srun --mpi=pmi2 -n 1 -c 8 bash ~/leonardo-setup/precompile-test.sh"
   ```
6. Run through `scripts/remote/run.sh leonardo "<cmd>"`. The wrapper is
   `salloc ...` and stays on the login node; only an inner `srun` reaches
   compute nodes. Use `srun --chdir=<abs path>` instead of `cd ... && srun`:
   a `&&` splits the command so the part after it escapes the salloc wrapper
   and loses the Julia PATH/depot (see failures log 2026-10-04).

Verified 4-rank commands (one line each, both exit 0):

```bash
# focused device regression (217/217 per rank)
scripts/remote/run.sh leonardo "env FLUX_TEST_DISTRIBUTED_BACKEND=nccl \
  srun --mpi=pmi2 -n 4 -c 4 --chdir=$HOME/projects/flux-integration \
  julia --startup-file=no --project=test test/ext_distributed/cuda_conditional_distributedtest.jl"

# full dedicated suite, MPI + NCCL (14/14)
scripts/remote/run.sh leonardo "env JULIA_MPI_TEST_NPROCS=4 \
  FLUX_TEST_DISTRIBUTED_NCCL=true FLUX_TEST_DISTRIBUTED_TIMEOUT=1200 \
  srun --mpi=pmi2 -n 1 -c 8 --chdir=$HOME/projects/flux-integration \
  julia --startup-file=no --project=test test/ext_distributed/runtests.jl"
```

GPU layout: allocate `--gres=gpu:4` (all four A100 visible to every rank).
`FluxMPIExt` maps rank `r` to device `mod1(r + 2, ndevices)`, so 4 ranks use the
four distinct GPUs (observed `[1, 2, 3, 0]`). Do NOT use `--gpus-per-task=1`
for the focused regression: it asserts `CUDA.ndevices() >= ranks`.

## Original readiness checklist (all done 2026-10-04)

- [x] SSH access, Slurm account/partition/QoS, modules, PMI protocol documented
- [x] Single-node 4-GPU allocation and CUDA smoke verified
- [x] Install Julia 1.12.x (no module); `JULIA_DEPOT_PATH` on `$WORK`
- [x] `scripts/remote/hosts/leonardo.conf` (salloc wrapper, `SLURM_MPI_TYPE=pmi2`, `JULIA_PKG_PRECOMPILE_AUTO=0`, `boost_qos_dbg`)
- [x] First sync + resolve + `MPI.install_mpiexecjl` equivalent (runs use `srun --mpi=pmi2` directly)
- [x] Compute-node precompile, MPI smoke, 4-rank NCCL smoke

`scripts/remote/run.sh` rejects `leonardo` if `scripts/remote/hosts/leonardo.conf` is missing; the conf exists now.
