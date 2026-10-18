# Provenance — static-scaling Phase 3 Step 2 precision experiment and regression

Date: 2026-10-17 (UTC run window 18:12–18:52).
Status: **experiment PASS**, **focused regression PASS**.
Scope: isolated stage-2 convolution precision only. No PyTorch reference was
regenerated; no Phase 3 driver was rerun; no tolerance or model math changed.
No Flux source change and no commit.

## 1. Result

The unchanged PyTorch stage-2 diagnostic (`stage2_diag.npz`) was compared with
a fresh Flux stage-2 run under three independent processes. The observed
convolutions are `conv1` (stage2.block0, stride 2, pad 1), `conv2`
(stage2.block0, stride 1, pad 1), and `proj` (1x1, stride 2, pad 0).
Comparison rule is the frozen `atol=1e-5`, `rtol=1e-4` (elementwise).

### 1.1 Final experiment run (job 60896936, node lrdn0102)

| Case | Array | max_abs | max_rel | violations |
|---|---|---|---:|---:|---:|
| cuda_current | conv1 | 3.5725832e-3 | 1.99545726 | 1,225,601 |
| cuda_current | conv2 | 3.64590809e-3 | 1.99393233 | 1,316,175 |
| cuda_current | proj | 0 | 0 | 0 |
| cuda_fma | conv1 | 0 | 0 | 0 |
| cuda_fma | conv2 | 1.90734863e-6 | 0.233333333 | 0 |
| cuda_fma | proj | 0 | 0 | 0 |
| cpu_current (control) | conv1 | 7.62939453e-6 | 0.130006154 | 0 |
| cpu_current (control) | conv2 | 5.48362732e-6 | 0.869565217 | 0 |
| cpu_current (control) | proj | 0 | 0 | 0 |

Requested and effective cuDNN math types:

| Case | cuBLAS | cuDNN requested | effective conv1 | effective conv2 | effective proj |
|---|---|---|---|---:|---:|---:|
| cuda_current | `PEDANTIC_MATH` | `CUDNN_DEFAULT_MATH(0)` | `CUDNN_DEFAULT_MATH(0)` | `CUDNN_DEFAULT_MATH(0)` | `CUDNN_DEFAULT_MATH(0)` |
| cuda_fma | `PEDANTIC_MATH` | `CUDNN_FMA_MATH(3)` | `CUDNN_FMA_MATH(3)` | `CUDNN_FMA_MATH(3)` | `CUDNN_FMA_MATH(3)` |
| cpu_current | n/a | n/a | n/a | n/a | n/a |

`cuDNN.version()` printed `92.60.0` (the cuDNN.jl 6.2.2 parser divides the raw
`92600`); the library is cuDNN 9.26.0. `CUDNN_FMA_MATH` exists (value 3).

**Interpretation.** With the stock mapping (`PEDANTIC_MATH -> CUDNN_DEFAULT_MATH`)
the GPU convolution differs from the unchanged PyTorch reference by
~3.6e-3. With the more-specific mapping `PEDANTIC_MATH -> CUDNN_FMA_MATH` the
same convolutions are at CPU level (`conv1=0`, `conv2=1.9e-6`) with zero
violations. The projection conv matches exactly in both cases. This resolves
the Step 2 precision question for the reproduced failure.

### 1.2 Algorithm-selection observation

The failing convolution is not fixed to `conv1`. In the first experiment run
(job 60896513, same node) `conv1` matched exactly (`max_abs=0`) while `conv2`
failed (`2.42638588e-3`, 1,196,926 violations); the FMA case matched both.
cuDNN's stock autotuner chooses algorithms by timing, not by the requested math
type. A synthetic high-level `Flux.CrossCor` probe can therefore select FP32
Winograd in **both** modes (algo 6, relative L2 ~2.9e-4 versus Float64), which
masks the TF32-vs-FMA difference. Pinning
`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` and
`CUDNN_CONVOLUTION_BWD_{DATA,FILTER}_ALGO_1` makes the effect deterministic:
FMA ~1e-7 to 1e-6, `CUDNN_DEFAULT_MATH` ~2.9e-4. The focused regression uses
those pinned kernels; the experiment uses the unchanged Flux/NNlib model path.

## 2. Focused regression (job 60900440, node lrdn2176)

`check_precision_math.jl` with the stage-2 conv2 shape
`(32,32,128,12)`, one 3x3 stride-1 convolution, fixed cuDNN algorithms,
Float64 CPU reference, and a classifier-sized 200x512 by 512x12 matmul.
Declared relative-L2 bound: `1e-4`.

| Case | Item | rel_l2 | max_abs | result |
|---|---|---|---:|---:|---|
| fma | conv_fixed_fwd | 5.998539e-7 | 8.698975e-6 | PASS |
| fma | conv_fixed_dx | 2.103432e-7 | 1.475017e-6 | PASS |
| fma | conv_fixed_dw | 1.121881e-6 | 1.011678e-3 | PASS |
| fma | classifier_matmul | 1.107716e-7 | 4.209593e-7 | PASS |
| current | conv_fixed_fwd | 2.932134e-4 | 1.600785e-3 | FAIL (expected) |
| current | conv_fixed_dx | 2.928844e-4 | 1.434166e-3 | FAIL (expected) |
| current | conv_fixed_dw | 2.930009e-4 | 1.480980e-1 | FAIL (expected) |
| current | classifier_matmul | 1.107716e-7 | 4.209593e-7 | PASS |

`CHECK_PRECISION_MATH PASS` for fma (exit 0); `CHECK_PRECISION_MATH FAIL` for
current (exit 1). `PHASE3_PRECISION_REGRESSION PASS` requires both.

## 3. Remote and local locations

Remote run root (mutable evidence):

```text
/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-step2-precision/
  logs/cuda_current.log
  logs/cuda_fma.log
  logs/cpu_current.log
  logs/fma.log
  logs/current.log
```

Local copies (this repository):

```text
artifacts/logs/static-scaling/2026-10-17-phase3-step2-precision/
  cuda_current.log, cuda_fma.log, cpu_current.log
  fma.log, current.log
  2026-10-17-phase3-step2-precision.log             (console, final run)
  2026-10-17-phase3-step2-precision-attempt1.log    (console, first run)
  2026-10-17-phase3-step2-precision-regression.log  (console)
  2026-10-17-phase3-step2-precompile.log            (console)
```

Unchanged inputs, from the preserved attempt-3 evidence root
`/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `reference/phase3_reference.npz` | 299,701,312 | `97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783` |
| `diag/stage2_diag.npz` | 56,625,286 | `a5dd592d7427ccdfba6a68dd3e8c9985a4466e8f7eca067095e5fc96d11a9b7e` |

The local copies of both are under the preserved attempt-3 directories and were
not modified. The PyTorch generator and checker were not rerun.

## 4. SHA-256 of copied logs (remote == local)

| Log | SHA-256 |
|---|---|
| `cuda_current.log` | `c0d47c40bd9fef928e9b7656412cc7d3068a551e37c34121bba10af25ed2cdbf` |
| `cuda_fma.log` | `1175a4777868b9e588b8d6e2fc74e1cf6a85a2c8719f41c2bb6c7445a8b0edbe` |
| `cpu_current.log` | `81f9f4709a6fad50c78ac39b745be2d71ae3d8c37719f7798c3b20aea431e053` |
| `fma.log` | `330a58dcedb2198b8ed65e020c4b1734cfb95812335d965184f47dffea031059` |
| `current.log` | `547f8e72e65ede160fd9f5721479a44847aa4edb37d18c4c045d2cef0a62a7ff` |

All five were hashed remotely with `sha256sum`, copied with `rsync`, and
re-hashed locally: all MATCH.

## 5. Source identity

| Item | Value |
|---|---|
| Thesis `git rev-parse HEAD` | `3088f61020137912af32d9c969e87ea9ae2a71c3` |
| Thesis source-state hash (recipe below) | `b72b0851fb0a94b2deb31cc06c029ae50327170b0305d819d0868af85e5f93ae` |
| `git diff HEAD \| sha256sum` | `a96456534d1d467c649bc813291f8d3070a924f035743db546321b05ea0df8bc` |
| untracked non-artifact list hash | `3b4aeb97aea2618221f768d6955ad7721e4f3a3dd7f590fad4433c86d0d1835a` |
| Flux worktree commit / status | `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e`, clean |
| Remote thesis tree | no `.git` (rsynced); identity anchored by file hashes |

Source-state recipe (same as Phase 2/3):

```bash
STATE=$( { git diff HEAD; git ls-files --others --exclude-standard \
  | grep -v '^artifacts/' | sort | xargs -r sha256sum; } | sha256sum | cut -d' ' -f1 )
```

Remote and local SHA-256 MATCH for every changed/added source file:

| File | SHA-256 |
|---|---|
| `scripts/static_scaling/precision.jl` | `d6b157f25a293cc5310a183169de596dd88f2880425af4fe90fa357c8d7f14a0` |
| `scripts/static_scaling/diagnose_phase3_flux_stage2.jl` | `3316869b7ab18d69b1c7127df85b9c674743a3be25e691052babaa62326536dc` |
| `scripts/static_scaling/check_phase3_flux.jl` | `c13ce678ed64cdda4a44fffb1fe63ad8770d556a7940d098186dc865dbe833b1` |
| `scripts/static_scaling/diagnose_phase3_flux.jl` | `383f1fa02fd11e308292eda3d9e41ac916236b17cd3406e9c370bc64b2e63d16` |
| `scripts/static_scaling/check_precision_math.jl` | `8cd254987df661d7b92c9bcd73ba50b95dfd26b9ccd6182cf2ffe2da77c90117` |
| `scripts/static_scaling/model_flux.jl` | `9bc2d752bb7c5543374ed4ed8e107048bec95ec17d5e293a2bade23f0a7055a4` |
| `scripts/remote/leonardo/run-phase3-precision-experiment.sh` | `2e18443c3275f7299f54265e4a2953390d2cd671fe2e97e52a8b4ea4c8ac8004` |
| `scripts/remote/leonardo/run-phase3-precision-regression.sh` | `deae7627a24356c3a7433f236f1db3be8724f48da2a7d3550fb1a5595717db57` |

## 6. Environment

| Item | Value |
|---|---|
| Julia | 1.12.6 (`/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia`) |
| Depot | `/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot` |
| CUDA.jl | 6.2.2 |
| cuDNN.jl | 6.2.2 (`$WORK/epicazio/julia-depot/packages/cuDNN/0SFve`) |
| CUDNN_jll / library | 9.26.0 (`cuDNN.version()` prints `92.60.0`; parser artifact) |
| NNlib | 0.9.45 |
| Flux (integration worktree) | 0.16.11, commit `f7b4520c4bd307c67bbc5ac92a1c68493eccff6e` |
| GPU | NVIDIA A100-SXM-64GB, CC 8.0, 65536 MiB |
| Driver | 535.274.02 |
| PyTorch reference environment | unchanged from attempt-3 preservation (torch 2.9.1+cu126, cuDNN 9.10.2.21, TF32 disabled) |

## 7. Exact commands

### 7.1 Sync (no `sync_code.sh`; preserved reference not uploaded)

```bash
rsync -avz --delete --exclude-from='.rsyncignore' \
  --exclude='artifacts/baselines/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  --exclude='artifacts/logs/static-scaling/2026-10-17-phase3-attempt3-preserved/' \
  . leonardo:projects/ddp-flux-thesis/
```

The remote Flux worktree was not synced or modified.

### 7.2 Precompile (compute node; job 60896027, node lrdn0102)

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/precompile-phase3.sh"
```

Output ended with `FLUX_PRECOMPILE_OK`.

### 7.3 Precision experiment (compute node)

```bash
# First run, job 60896513, node lrdn0102 (current conv1 exact / conv2 failed;
# FMA passed; driver rule keyed on conv1 only, so it printed FAIL).
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-phase3-precision-experiment.sh"

# Final run, job 60896936, node lrdn0102 (PASS; logs copied above).
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-phase3-precision-experiment.sh"
```

The driver ran three fresh Julia processes: `cuda_current`, `cuda_fma`, and
`cpu_current`, each through `run-julia-phase3.sh`. Inputs were
`FIX=$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2`,
`REF=$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/reference/phase3_reference.npz`,
`DIAG=$WORK/epicazio/static-scaling-evidence/2026-10-17-phase3-attempt3/diag/stage2_diag.npz`.

### 7.4 Focused regression (compute node; final job 60900440, node lrdn2176)

```bash
GRES=gpu:1 NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh leonardo \
  "srun --mpi=pmi2 -n1 -c8 bash scripts/remote/leonardo/run-phase3-precision-regression.sh"
```

Failure history, all in fresh allocations: job 60897348 (high-level
`Flux.conv` test; FMA item failed because cuDNN selected FP32 Winograd),
job 60898845 (`Flux.CrossCor` high-level test; same Winograd masking),
job 60900328 (pinned-algorithm test: FMA and current behaved as intended, but
the driver's summary regex did not match `conv_fixed_*`), then job 60900440
(PASS after the regex fix). Pinned-algorithm probes: 60900208/60900215
(forward), 60900228/60900308 (data/weight gradients).

## 8. Exact next action

Proceed to Step 3 of `temp/static-graph-gpu-scaling-phase3-handoff.md`: run the
complete three-update Flux comparison with `configure_precision!(fma=true)`
wired into `check_phase3_flux.jl`, unchanged `atol=1e-5` / `rtol=1e-4`, in a
new output directory, and retain per-category maxima and violation counts.
