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

**Status 2026-10-18 (Phase 3 review fixes, hardened):** R1/R2 are fixed, hardened after independent review, and verified. Canonical precompile job `61187225` (`lrdn0241`, UTC `12:20:59`-`12:24:36`, `FLUX_PRECOMPILE_OK`, exit 0), focused precision regression `61188854` (`lrdn1845`, UTC start `2026-10-18T12:24:49Z`, driver `12:24:54Z`, `PHASE3_PRECISION_REGRESSION PASS`, `fallback=0`, policy 10/10), and full recheck `61191281` (`lrdn2911`, UTC start `2026-10-18T12:30:16Z`, driver `12:30:21Z`, elapsed `00:07:52`, exit 0) with all four summaries PASS, 337/337 exact reference, 32/32 `CUDNN_FMA_MATH`/`fallback=0`, and zero violations. Source identity (generation-time; later doc edits move the live worktree hash): thesis `a3786d97f3263cad20b254206f09c095880f624d`, state `da20dddc8616f2b53b7ab7095ee0e3bc1794bece3916b1806cf83ee725fdf199`; Flux `f7b4520c...`, state `e3b0c442...` (clean). Use `scripts/remote/leonardo/precompile-phase3.sh`; the default `scripts/remote/precompile.sh leonardo` is the wrong target (the root `Manifest.toml` is excluded by `.rsyncignore` and compute nodes are offline). Access currently works; the earlier transient SSH/DNS drop did not affect the final hardened evidence and is not evidence of a key change. Evidence: `artifacts/logs/static-scaling/2026-10-18-r1-r2-precision-hardening/`, `artifacts/logs/static-scaling/2026-10-18-r1-r2-phase3-hardening/`, `artifacts/baselines/static-scaling/2026-10-18-r1-r2-phase3-hardening/`. The intermediate runs `61074887`/`61077382`/`61091402` are superseded history.

**Status 2026-10-19 (fixture timing campaign):** job `61314790` (`static-timing-campaign`, `lrdn3064.leonardo.local`, `boost_qos_lprod`, 1 node, 1 task, 32 CPUs, 4× A100-SXM-64GB, `02:00:00`) completed `0:0` with elapsed `00:42:07` (UTC `2026-10-18T23:06:02Z`–`23:48:09Z`). The compute-node precompile printed `FLUX_PRECOMPILE_OK`; the driver printed `STATIC_TIMING_campaign PASS` and `TIMING_DRIVER_EXIT=0`. All 30/30 records are present (5 repeats × 6 configurations), with no failed runs, and the checker prints `TIMING_ARTIFACT_CHECK PASS`. Flux is flat across W=1/2/4 (median `5.483/5.715/5.664 s`; efficiency `1.0000/0.4797/0.2420`); PyTorch scales `1.5602x` at W=2 and `1.8060x` at W=4 (median `8.204/5.258/4.542 s`). This is a bounded training-only fixture pilot, not the full-dataset campaign, end-to-end timing, the Phase 5 memory pilot, or learning curves. Evidence: local `artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/`, Slurm log `artifacts/logs/leonardo/2026-10-19-static-timing-campaign-61314790.log`, remote `$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/campaign-61314790/`; [results](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/RESULTS.md), [devlog](../devlog/03-static-scaling/2026-10-19-fixture-timing-campaign.md).

**Status 2026-10-19 (timing review):** campaign `61314790` retains a qualified timing PASS. The hardened checker now validates schema, precision, job/node identity, fixture/bundle hashes, complete state hashes, all 36 source-snapshot hashes, driver markers, and `provenance.json` policy/qualifications; `python3 scripts/static_scaling/check_timing.py --self-test` prints `TIMING_ARTIFACT_SELFTEST PASS checks=13`. `run-static-timing.sh` now writes `runtime-env.txt` (UTC date, hostname, Julia depot/load path, Julia version, `Pkg.status()`, `Base.load_path()`, installed Python distribution metadata, Python/torch/numpy paths) for future runs; campaign `61314790` predates it. No new remote job ran at that point because the SSH publickey was rejected again; job `61366015` subsequently ran (see below). [Review response](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790/REVIEW-RESPONSE.md).

**Status 2026-10-19 (300-update numerics):** job `61366015` (`static-timing-numerics`, `lrdn2256.leonardo.local`, `boost_qos_lprod`, 1 node, 1 task, 32 CPUs, 4× A100-SXM-64GB, `01:00:00`) started `2026-10-19T08:53:56Z` and completed `COMPLETED` exit `0:0` in `00:13:26` (end approx. `2026-10-19T09:07:22Z`). Precompile `FLUX_PRECOMPILE_OK`; all six configurations `TIMING_PASS`; driver `STATIC_TIMING_numerics PASS` and `TIMING_DRIVER_EXIT=0`. The run produced 6/6 JSON records and 6/6 `state.npz` snapshots for one repeat of 300 measured updates and 20 warmup at global batch 12, strict FP32 FMA/TF32 off, W=1,2,4. `runtime-env.txt` records Julia 1.12.6, Flux 0.16.11, CUDA.jl/cuDNN.jl 6.2.2, Python 3.11.17, torch 2.9.1+cu126, and numpy 2.4.6 with executable/module paths; no `RUNTIME_ENV_CAPTURE_FAILED` line. The comparison is descriptive: `param` `max_abs` up to `0.0333` and `rel_l2` up to `0.00224`, `m` `rel_l2` up to `0.0485`, `v` `rel_l2` up to `0.000333`, `beta_power` exact (62/62); no `param`/`m`/`v` array is exactly equal. `NUMERICAL_EQUIVALENCE_CLAIM none`; no correctness, equivalence, or error bound is claimed. The Phase 6 numerical-acceptance policy (OD-14) remains open. Evidence: local `artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/`, Slurm log `artifacts/logs/leonardo/2026-10-19-static-timing-numerics-61366015.log`, remote `$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot/numerics-61366015/`; [results](../../artifacts/logs/static-scaling/2026-10-18-timing-pilot/numerics-61366015/RESULTS.md), [devlog](../devlog/03-static-scaling/2026-10-19-timing-review-response.md), [failed attempts](../failures/2026-10-19-timing-numerics-attempts.md).

**Status 2026-10-19 (full-dataset loader validation):** job `61450014` (`static-full-loader-validate`, `lrdn0667.leonardo.local`, `boost_qos_dbg`, 1 node, 1 task, 32 CPUs, 4× A100-SXM-64GB, `00:30:00`) completed `0:0` in `00:14:09` and printed `FULL_VALIDATION PASS`. Compute-node precompile printed `FLUX_PRECOMPILE_OK` for both `~/projects/flux-integration` and `~/projects/flux-static-baseline`. The driver built the Python reference, passed `FULL_DATASET_PYTHON_CHECK` (169 then 177 checks), `FULL_DATASET_JULIA_CHECK` for current (111 checks) and historical (108 checks), `FIXTURE_TIMING_PATH PASS`, and three 1-update/batch-512/W=1 dry runs with finite loss `5.492467880249023`. Loader transformed tensors matched the Python reference with `max_abs_diff=0.0` at W=1,2,4. Flux dry-run device-used was 86.34% (current) and 83.14% (historical) of the A100 total, so the proposed OD-8 80% margin is not established; batch 512 did not OOM and no 256 fallback ran. This is validation only; the campaign was not executed. Evidence: local `artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/`, Slurm log `artifacts/logs/leonardo/2026-10-19-static-full-loader-validation-61450014.log`, remote `$WORK/epicazio/static-scaling-evidence/2026-10-19-full-loader-validation-61450014/`; [results](../../artifacts/logs/static-scaling/2026-10-19-full-loader-validation-61450014/RESULTS.md), [devlog](../devlog/03-static-scaling/2026-10-19-full-dataset-loader-validation.md). Superseded metadata-only job `61447030` and pre-workload failure `61446924` are recorded in [the failure note](../failures/2026-10-19-full-loader-validation-driver.md).

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
