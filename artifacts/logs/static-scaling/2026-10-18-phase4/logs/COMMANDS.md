# Commands - static-scaling Phase 4 remote check

Run: Leonardo compute node, driver `scripts/remote/leonardo/run-phase4-check.sh`.
No result in this file is inferred: every listed command was executed by the driver.

| Item | Value |
|---|---|
| Host | `lrdn0983.leonardo.local` |
| SLURM job | `61296074` |
| SLURM node list | `lrdn0983` |
| Start (UTC) | `2026-10-18T19:21:44Z` |
| End (UTC) | `2026-10-18T19:25:57Z` |
| Driver SHA-256 | `e16612b5a1a6d299221bdcac929770bc1ada95365597eb5799705868be53e7fd` |
| Device / backend | `cuda` / `nccl` |
| Flux / PyTorch backend | `nccl` / `nccl` |
| World sizes | `1 2 4` |
| CPUs | `32` (reference stages `8`) |
| OUT | `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4` |
| LOG_DIR | `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs` |
| Status | **PASS** |

## Stage status

| Stage | Status | Note |
|---|---|---|
| 0 | PASS | preflight |
| a | PASS | source, versions, hashes |
| b | PASS | Flux single-GPU reference (reused) |
| c | PASS | PyTorch single-GPU reference and identity (reused) |
| d | PASS | split-control both frameworks |
| e | PASS | Flux DDP worlds: 1 2 4 (reused) |
| f | PASS | PyTorch DDP worlds: 1 2 4 |
| g | PASS | checker PHASE4_CHECK PASS |

## Environment

```text
THESIS=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis
FLUX_DIR=/leonardo/home/userexternal/epicazio/projects/flux-integration
VENV_PY=/leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python
JULIA_BIN=/leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia
JULIA_DEPOT_PATH=/leonardo_work/IscrC_UNMASKED/epicazio/julia-depot
JULIA_LOAD_PATH=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib
FIX=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2
BUNDLE=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz
PY_REF=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference.npz
PY_MANIFEST=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference_manifest.json
FLUX_REF=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz
TORCH_REF=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz
DDP_ROOT=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp
CHECK_DIR=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check
SOURCE_COMMIT=<unset>
SOURCE_STATE_SHA256=<unset>
FLUX_COMMIT=<unset>
FLUX_STATE_SHA256=<unset>
```

## Exact command transcript

```text
STAGE a source-and-tool-metadata
START_UTC 2026-10-18T18:07:01Z
END_UTC 2026-10-18T18:08:28Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/a-source-metadata.log
COMMAND bash -c $'\nset -euxo pipefail\ndate -u +%Y-%m-%dT%H:%M:%SZ\nhostname\necho "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"\necho "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"\necho "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"\necho "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"\necho "SLURM_PROCID=${SLURM_PROCID:-<unset>}"\necho "THESIS=$THESIS"\necho "FLUX_DIR=$FLUX_DIR"\necho "FIX=$FIX"\necho "BUNDLE=$BUNDLE"\necho "PY_REF=$PY_REF"\necho "PY_MANIFEST=$PY_MANIFEST"\necho "OUT=$OUT"\necho "LOG_DIR=$LOG_DIR"\necho "DEVICE=$DEVICE BACKEND=$BACKEND FLUX_BACKEND=$FLUX_BACKEND TORCH_BACKEND=$TORCH_BACKEND"\necho "WORLD_SIZES=$WORLD_SIZES TOTAL_CPUS=$TOTAL_CPUS REF_CPUS=$REF_CPUS"\necho "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"\necho "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"\nif command -v git >/dev/null 2>&1; then\n  echo "THESIS_GIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo "<no-git>")"\n  echo "FLUX_GIT=$(git -C "$FLUX_DIR" rev-parse HEAD 2>/dev/null || echo "<no-git>")"\nelse\n  echo "THESIS_GIT=<no-git-command>"\n  echo "FLUX_GIT=<no-git-command>"\nfi\necho "PROCESS_COMMAND=$(tr "\\0" " " < /proc/$$/cmdline 2>/dev/null || echo "<unavailable>")"\necho "PARENT_COMMAND=$(ps -o args= -p "$PPID" 2>/dev/null || echo "<unavailable>")"\necho "--- srun ---"\nsrun --version\nif [ "$DEVICE" = cuda ]; then\n  echo "--- nvidia-smi ---"\n  nvidia-smi -L\n  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader\nfi\necho "--- python/torch ---"\n"$VENV_PY" - <<"PY"\nimport sys\nprint("python", sys.version.split()[0])\nimport numpy\nprint("numpy", numpy.__version__)\nimport torch\nprint("torch", torch.__version__, "cuda", torch.version.cuda)\nprint("cuda_available", torch.cuda.is_available())\nprint("cudnn", torch.backends.cudnn.version())\ntry:\n    print("nccl", torch.cuda.nccl.version())\nexcept Exception as exc:\n    print("nccl unavailable:", exc)\nPY\necho "--- julia ---"\n"$JULIA_BIN" --version\necho "--- hash manifest ---"\n{\n  echo "# static_scaling sources"\n  find "$THESIS/scripts/static_scaling" -maxdepth 1 -type f \\( -name "*.jl" -o -name "*.py" \\) -print0 | sort -z | xargs -0 sha256sum\n  echo "# remote drivers"\n  for f in "$THESIS/scripts/remote/leonardo/run-phase4-check.sh" "$THESIS/scripts/remote/leonardo/precompile-phase3.sh"; do\n    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi\n  done\n  echo "# inputs"\n  sha256sum "$FIX/data_fixture.npz" "$FIX/fixture_manifest.json" "$FIX/schedule_fixture.npz" "$BUNDLE"\n  for f in "$PY_REF" "$PY_MANIFEST"; do\n    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi\n  done\n  echo "# environments"\n  for f in "$JULIA_ENV/Project.toml" "$JULIA_ENV/Manifest.toml" "$FLUX_DIR/test/Project.toml" "$FLUX_DIR/test/Manifest.toml"; do\n    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi\n  done\n} | tee "$LOG_DIR/source-sha256.txt"\n' 

STAGE b flux-reference
START_UTC 2026-10-18T18:08:28Z
END_UTC 2026-10-18T18:13:45Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/b-flux-reference.log
COMMAND srun --overlap --mpi=pmi2 -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode reference --device cuda --backend nccl --world-size 1 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE b verify-flux-reference
START_UTC 2026-10-18T18:13:45Z
END_UTC 2026-10-18T18:13:45Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/b-verify-flux-reference.log
COMMAND bash -c $'\nset -euo pipefail\ntest -f "$1"\ntest -f "$2"\nsha256sum "$1" "$2"\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.json 

STAGE c pytorch-reference
START_UTC 2026-10-18T18:13:45Z
END_UTC 2026-10-18T18:14:30Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/c-pytorch-reference.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_phase3_reference.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda --device cuda --force 

STAGE c verify-pytorch-reference
START_UTC 2026-10-18T18:14:30Z
END_UTC 2026-10-18T18:14:31Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/c-verify-pytorch-reference.log
COMMAND bash -c $'\nset -euo pipefail\ntest -f "$1"\ntest -f "$2"\nsha256sum "$1" "$2"\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference_manifest.json 

STAGE c pytorch-reference-identity
START_UTC 2026-10-18T18:14:31Z
END_UTC 2026-10-18T18:14:31Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/c-reference-identity.log
COMMAND bash -c $'\nset -euo pipefail\nactual=$(sha256sum "$1" | cut -d" " -f1)\nexpected=$(sha256sum "$2" | cut -d" " -f1)\necho "preserved=$2"\necho "preserved_sha256=$expected"\necho "generated=$1"\necho "generated_sha256=$actual"\ntest "$actual" = "$expected"\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference.npz 

STAGE d flux-split-control
START_UTC 2026-10-18T18:14:31Z
END_UTC 2026-10-18T18:17:04Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-flux-split-control.log
COMMAND srun --overlap --mpi=pmi2 -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode split-control --device cuda --backend nccl --world-size 1 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-split-control
START_UTC 2026-10-18T18:17:04Z
END_UTC 2026-10-18T18:17:11Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-split-control.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode split-control --device cuda --backend nccl --world-size 1 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d flux-shard-reference-w1
START_UTC 2026-10-18T18:17:11Z
END_UTC 2026-10-18T18:19:25Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-flux-shard-reference-w1.log
COMMAND srun --overlap --mpi=pmi2 -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 1 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-shard-reference-w1
START_UTC 2026-10-18T18:19:25Z
END_UTC 2026-10-18T18:19:32Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-shard-reference-w1.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 1 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d flux-shard-reference-w2
START_UTC 2026-10-18T18:19:32Z
END_UTC 2026-10-18T18:21:45Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-flux-shard-reference-w2.log
COMMAND srun --overlap --mpi=pmi2 -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 2 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-shard-reference-w2
START_UTC 2026-10-18T18:21:45Z
END_UTC 2026-10-18T18:21:52Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-shard-reference-w2.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 2 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d flux-shard-reference-w4
START_UTC 2026-10-18T18:21:52Z
END_UTC 2026-10-18T18:24:09Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-flux-shard-reference-w4.log
COMMAND srun --overlap --mpi=pmi2 -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 4 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-shard-reference-w4
START_UTC 2026-10-18T18:24:09Z
END_UTC 2026-10-18T18:24:16Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-shard-reference-w4.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 4 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d verify-split-outputs
START_UTC 2026-10-18T18:24:16Z
END_UTC 2026-10-18T18:24:18Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-verify-split-outputs.log
COMMAND bash -c $'\nset -euo pipefail\nfor f in "$@"; do\n  test -f "$f" || { echo "MISSING $f"; exit 1; }\ndone\nsha256sum "$@"\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/flux/cuda/split_control.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/flux/cuda/split_control.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/pytorch/cuda/split_control.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/pytorch/cuda/split_control.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.json 

STAGE e flux-ddp-w1
START_UTC 2026-10-18T18:24:18Z
END_UTC 2026-10-18T18:26:50Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/e-flux-ddp-w1.log
COMMAND srun --overlap --mpi=pmi2 -n1 -c32 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 1 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE e verify-flux-ddp-w1
START_UTC 2026-10-18T18:26:50Z
END_UTC 2026-10-18T18:26:50Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/e-verify-flux-ddp-w1.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w1 

STAGE e flux-ddp-w2
START_UTC 2026-10-18T18:26:50Z
END_UTC 2026-10-18T18:29:19Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/e-flux-ddp-w2.log
COMMAND srun --overlap --mpi=pmi2 -n2 -c16 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 2 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE e verify-flux-ddp-w2
START_UTC 2026-10-18T18:29:19Z
END_UTC 2026-10-18T18:29:19Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/e-verify-flux-ddp-w2.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w2 

STAGE e flux-ddp-w4
START_UTC 2026-10-18T18:29:19Z
END_UTC 2026-10-18T18:31:51Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/e-flux-ddp-w4.log
COMMAND srun --overlap --mpi=pmi2 -n4 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c unset\ JULIA_PROJECT\;\ export\ JULIA_DEPOT_PATH=\"\$1\"\ JULIA_LOAD_PATH=\"\$2\"\ JULIA_PKG_PRECOMPILE_AUTO=0\ JULIA_NUM_THREADS=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ /leonardo_work/IscrC_UNMASKED/epicazio/julia-depot /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env:/leonardo/home/userexternal/epicazio/projects/flux-integration/test:@stdlib 1 /leonardo_work/IscrC_UNMASKED/epicazio/julia-1.12.6/bin/julia --startup-file=no /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 4 --out /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE e verify-flux-ddp-w4
START_UTC 2026-10-18T18:31:51Z
END_UTC 2026-10-18T18:31:51Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/e-verify-flux-ddp-w4.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4 

STAGE f pytorch-ddp-w1
START_UTC 2026-10-18T18:31:51Z
END_UTC 2026-10-18T18:32:05Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-pytorch-ddp-w1.log
COMMAND srun --overlap -n1 -c32 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\ TORCH_NCCL_DEBUG=\"\$2\"\ MASTER_PORT=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ 1 INFO 29601 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 1 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE f verify-pytorch-ddp-w1
START_UTC 2026-10-18T18:32:05Z
END_UTC 2026-10-18T18:32:05Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-verify-pytorch-ddp-w1.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w1 

STAGE f pytorch-ddp-w2
START_UTC 2026-10-18T18:32:05Z
END_UTC 2026-10-18T18:32:14Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-pytorch-ddp-w2.log
COMMAND srun --overlap -n1 -c16 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\ TORCH_NCCL_DEBUG=\"\$2\"\ MASTER_PORT=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ 1 INFO 29602 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 2 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE f verify-pytorch-ddp-w2
START_UTC 2026-10-18T18:32:14Z
END_UTC 2026-10-18T18:32:14Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-verify-pytorch-ddp-w2.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2 

STAGE f pytorch-ddp-w4
START_UTC 2026-10-18T18:32:14Z
END_UTC 2026-10-18T18:32:23Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-pytorch-ddp-w4.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\ TORCH_NCCL_DEBUG=\"\$2\"\ MASTER_PORT=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ 1 INFO 29604 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 4 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE f verify-pytorch-ddp-w4
START_UTC 2026-10-18T18:32:23Z
END_UTC 2026-10-18T18:32:23Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-verify-pytorch-ddp-w4.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4 

STAGE g phase4-checker
START_UTC 2026-10-18T18:32:23Z
END_UTC 2026-10-18T18:33:27Z
EXIT 1
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/g-phase4-checker.log
COMMAND /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_phase4.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --pytorch-reference /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz --flux-reference /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz --ddp-root /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp 

STAGE a source-and-tool-metadata
START_UTC 2026-10-18T19:21:44Z
END_UTC 2026-10-18T19:23:04Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/a-source-metadata.log
COMMAND bash -c $'\nset -euxo pipefail\ndate -u +%Y-%m-%dT%H:%M:%SZ\nhostname\necho "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"\necho "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"\necho "SLURM_NTASKS=${SLURM_NTASKS:-<unset>}"\necho "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"\necho "SLURM_PROCID=${SLURM_PROCID:-<unset>}"\necho "THESIS=$THESIS"\necho "FLUX_DIR=$FLUX_DIR"\necho "FIX=$FIX"\necho "BUNDLE=$BUNDLE"\necho "PY_REF=$PY_REF"\necho "PY_MANIFEST=$PY_MANIFEST"\necho "OUT=$OUT"\necho "LOG_DIR=$LOG_DIR"\necho "DEVICE=$DEVICE BACKEND=$BACKEND FLUX_BACKEND=$FLUX_BACKEND TORCH_BACKEND=$TORCH_BACKEND"\necho "WORLD_SIZES=$WORLD_SIZES TOTAL_CPUS=$TOTAL_CPUS REF_CPUS=$REF_CPUS"\necho "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"\necho "FLUX_COMMIT=${FLUX_COMMIT:-<unset>} FLUX_STATE_SHA256=${FLUX_STATE_SHA256:-<unset>}"\nif command -v git >/dev/null 2>&1; then\n  echo "THESIS_GIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo "<no-git>")"\n  echo "FLUX_GIT=$(git -C "$FLUX_DIR" rev-parse HEAD 2>/dev/null || echo "<no-git>")"\nelse\n  echo "THESIS_GIT=<no-git-command>"\n  echo "FLUX_GIT=<no-git-command>"\nfi\necho "PROCESS_COMMAND=$(tr "\\0" " " < /proc/$$/cmdline 2>/dev/null || echo "<unavailable>")"\necho "PARENT_COMMAND=$(ps -o args= -p "$PPID" 2>/dev/null || echo "<unavailable>")"\necho "--- srun ---"\nsrun --version\nif [ "$DEVICE" = cuda ]; then\n  echo "--- nvidia-smi ---"\n  nvidia-smi -L\n  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader\nfi\necho "--- python/torch ---"\n"$VENV_PY" - <<"PY"\nimport sys\nprint("python", sys.version.split()[0])\nimport numpy\nprint("numpy", numpy.__version__)\nimport torch\nprint("torch", torch.__version__, "cuda", torch.version.cuda)\nprint("cuda_available", torch.cuda.is_available())\nprint("cudnn", torch.backends.cudnn.version())\ntry:\n    print("nccl", torch.cuda.nccl.version())\nexcept Exception as exc:\n    print("nccl unavailable:", exc)\nPY\necho "--- julia ---"\n"$JULIA_BIN" --version\necho "--- hash manifest ---"\n{\n  echo "# static_scaling sources"\n  find "$THESIS/scripts/static_scaling" -maxdepth 1 -type f \\( -name "*.jl" -o -name "*.py" \\) -print0 | sort -z | xargs -0 sha256sum\n  echo "# remote drivers"\n  for f in "$THESIS/scripts/remote/leonardo/run-phase4-check.sh" "$THESIS/scripts/remote/leonardo/precompile-phase3.sh"; do\n    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi\n  done\n  echo "# inputs"\n  sha256sum "$FIX/data_fixture.npz" "$FIX/fixture_manifest.json" "$FIX/schedule_fixture.npz" "$BUNDLE"\n  for f in "$PY_REF" "$PY_MANIFEST"; do\n    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi\n  done\n  echo "# environments"\n  for f in "$JULIA_ENV/Project.toml" "$JULIA_ENV/Manifest.toml" "$FLUX_DIR/test/Project.toml" "$FLUX_DIR/test/Manifest.toml"; do\n    if [ -f "$f" ]; then sha256sum "$f"; else echo "MISSING $f"; fi\n  done\n} | tee "$LOG_DIR/source-sha256.txt"\n' 

STAGE d pytorch-split-control
START_UTC 2026-10-18T19:23:04Z
END_UTC 2026-10-18T19:23:58Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-split-control.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode split-control --device cuda --backend nccl --world-size 1 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-shard-reference-w1
START_UTC 2026-10-18T19:23:58Z
END_UTC 2026-10-18T19:24:05Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-shard-reference-w1.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 1 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-shard-reference-w2
START_UTC 2026-10-18T19:24:05Z
END_UTC 2026-10-18T19:24:11Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-shard-reference-w2.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 2 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d pytorch-shard-reference-w4
START_UTC 2026-10-18T19:24:11Z
END_UTC 2026-10-18T19:24:17Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-pytorch-shard-reference-w4.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\;\ shift\;\ exec\ \"\$@\" _ 1 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode shard-reference --device cuda --backend nccl --world-size 4 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE d verify-split-outputs
START_UTC 2026-10-18T19:24:17Z
END_UTC 2026-10-18T19:24:21Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/d-verify-split-outputs.log
COMMAND bash -c $'\nset -euo pipefail\nfor f in "$@"; do\n  test -f "$f" || { echo "MISSING $f"; exit 1; }\ndone\nsha256sum "$@"\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/flux/cuda/split_control.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/flux/cuda/split_control.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/pytorch/cuda/split_control.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/pytorch/cuda/split_control.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.json /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.npz /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.json 

STAGE f pytorch-ddp-w1
START_UTC 2026-10-18T19:24:21Z
END_UTC 2026-10-18T19:24:32Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-pytorch-ddp-w1.log
COMMAND srun --overlap -n1 -c32 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\ TORCH_NCCL_DEBUG=\"\$2\"\ MASTER_PORT=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ 1 INFO 29601 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 1 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE f verify-pytorch-ddp-w1
START_UTC 2026-10-18T19:24:32Z
END_UTC 2026-10-18T19:24:32Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-verify-pytorch-ddp-w1.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w1 

STAGE f pytorch-ddp-w2
START_UTC 2026-10-18T19:24:32Z
END_UTC 2026-10-18T19:24:41Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-pytorch-ddp-w2.log
COMMAND srun --overlap -n1 -c16 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\ TORCH_NCCL_DEBUG=\"\$2\"\ MASTER_PORT=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ 1 INFO 29602 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 2 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE f verify-pytorch-ddp-w2
START_UTC 2026-10-18T19:24:41Z
END_UTC 2026-10-18T19:24:41Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-verify-pytorch-ddp-w2.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2 

STAGE f pytorch-ddp-w4
START_UTC 2026-10-18T19:24:41Z
END_UTC 2026-10-18T19:24:51Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-pytorch-ddp-w4.log
COMMAND srun --overlap -n1 -c8 --chdir=/leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis bash -c export\ OMP_NUM_THREADS=\"\$1\"\ MKL_NUM_THREADS=\"\$1\"\ TORCH_NCCL_DEBUG=\"\$2\"\ MASTER_PORT=\"\$3\"\;\ shift\ 3\;\ exec\ \"\$@\" _ 1 INFO 29604 /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --mode ddp --device cuda --backend nccl --world-size 4 --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4 

STAGE f verify-pytorch-ddp-w4
START_UTC 2026-10-18T19:24:51Z
END_UTC 2026-10-18T19:24:51Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/f-verify-pytorch-ddp-w4.log
COMMAND bash -c $'\nset -euo pipefail\ntest -d "$1" || { echo "MISSING directory $1"; exit 1; }\nfind "$1" -maxdepth 1 -type f -print | sort\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4 

STAGE g phase4-checker
START_UTC 2026-10-18T19:24:51Z
END_UTC 2026-10-18T19:25:57Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/g-phase4-checker.log
COMMAND /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_phase4.py --fixture-dir /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2 --bundle /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz --pytorch-reference /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz --flux-reference /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz --ddp-root /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp 

STAGE g checker-verdict
START_UTC 2026-10-18T19:25:57Z
END_UTC 2026-10-18T19:25:57Z
EXIT 0
LOG /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/g-checker-verdict.log
COMMAND bash -c $'\nset -euo pipefail\ngrep -Eq "^PHASE4_CHECK PASS( |$)" "$1" || { echo "MISSING PHASE4_CHECK PASS in $1"; exit 1; }\ntest -f "$2" || { echo "MISSING report $2"; exit 1; }\ntest -f "$3" || { echo "MISSING report $3"; exit 1; }\nsha256sum "$2" "$3"\n' _ /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/g-phase4-checker.log /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_report.txt /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_report.json 

```

## Evidence produced

- Stage logs: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/*.log`
- Source and input hashes: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/source-sha256.txt`
- Artifact hashes: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/artifact-sha256.txt`
- Flux reference: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz`
- PyTorch reference: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz`
- DDP root: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp`
- Checker reports: `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_report.txt`, `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_report.json`

## Copy evidence home (not executed by the driver)

```bash
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/' artifacts/logs/static-scaling/2026-10-18-phase4/logs/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/' artifacts/logs/static-scaling/2026-10-18-phase4/check/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/' artifacts/logs/static-scaling/2026-10-18-phase4/ddp/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/' artifacts/logs/static-scaling/2026-10-18-phase4/split/
rsync -avz leonardo:'/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/' artifacts/baselines/static-scaling/2026-10-18-phase4/reference/
```
