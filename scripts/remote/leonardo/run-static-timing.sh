#!/usr/bin/env bash
# Submit with sbatch. The batch shell runs on the allocated compute node.
set -euo pipefail
MODE=${1:?smoke, campaign, or numerics}
[[ "$MODE" == smoke || "$MODE" == campaign || "$MODE" == numerics ]]
WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=$HOME/projects/ddp-flux-thesis
FLUX_DIR=$HOME/projects/flux-integration
ROOT=$WORK/epicazio/static-scaling-evidence/2026-10-18-timing-pilot
OUT=$ROOT/$MODE-$SLURM_JOB_ID
mkdir -p "$OUT/source"
exec > >(tee "$OUT/driver.log") 2>&1
trap 'rc=$?; echo "TIMING_DRIVER_EXIT=$rc"' EXIT
hostname
date -u
nvidia-smi
cp -a "$THESIS/scripts/static_scaling/." "$OUT/source/"
cp -a "$FLUX_DIR/src" "$FLUX_DIR/ext" "$OUT/source/"
cp "$FLUX_DIR/Project.toml" "$OUT/flux-project.toml"
cp "$FLUX_DIR/Manifest.toml" "$OUT/flux-manifest.toml"
sha256sum "$OUT/source/"*.py "$OUT/source/"*.jl > "$OUT/source-sha256.txt"
cp "$THESIS/scripts/static_scaling/julia-env/Manifest.toml" "$OUT/julia-manifest.toml"
cp "$WORK/epicazio/venvs/pytorch-scaling/requirements-freeze.txt" "$OUT/requirements-freeze.txt"
export JULIA_DEPOT_PATH=$WORK/epicazio/julia-depot
export JULIA_LOAD_PATH=$THESIS/scripts/static_scaling/julia-env:$FLUX_DIR/test:@stdlib
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset JULIA_PROJECT
PY=$WORK/epicazio/venvs/pytorch-scaling/bin/python
JL=$WORK/epicazio/julia-1.12.6/bin/julia
bash "$THESIS/scripts/remote/leonardo/precompile-phase3.sh"
export JULIA_PKG_PRECOMPILE_AUTO=0
{
  echo "--- runtime identity ---"
  date -u
  hostname
  echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
  echo "JULIA_LOAD_PATH=$JULIA_LOAD_PATH"
  "$JL" --version
  "$JL" --startup-file=no --project="$FLUX_DIR/test" -e 'using Pkg; Pkg.status()'
  "$JL" --startup-file=no -e 'foreach(println, Base.load_path())'
  "$PY" - <<'PYEOF'
import importlib.metadata as metadata
import sys
import numpy
import torch
print("python", sys.version.split()[0])
print("python_executable", sys.executable)
print("torch", torch.__version__, torch.__file__)
print("numpy", numpy.__version__, numpy.__file__)
for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
    print("package", dist.metadata.get("Name"), dist.version)
PYEOF
} > "$OUT/runtime-env.txt" 2>&1 || echo "RUNTIME_ENV_CAPTURE_FAILED" >> "$OUT/runtime-env.txt"
FIX=$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2
if [[ "$MODE" == smoke ]]; then
  REPEATS=1 UPDATES=3 WARMUP=2
elif [[ "$MODE" == numerics ]]; then
  REPEATS=1 UPDATES=300 WARMUP=20
else
  REPEATS=5 UPDATES=300 WARMUP=20
fi
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  if (( repeat % 2 )); then
    WORLDS="1 2 4"; FRAMEWORKS="flux pytorch"
  else
    WORLDS="4 2 1"; FRAMEWORKS="pytorch flux"
  fi
  for world in $WORLDS; do
    for framework in $FRAMEWORKS; do
      file=$OUT/$framework-w$world-r$repeat.json
      args=(--fixture-dir "$FIX" --world-size "$world" --updates "$UPDATES" --warmup "$WARMUP" --out "$file")
      if [[ "$MODE" == numerics ]]; then
        args+=(--snapshot-dir "$OUT/snapshots/$framework-w$world")
      fi
      set -x
      if [[ "$framework" == flux ]]; then
        timeout 1200 srun --mpi=pmi2 --kill-on-bad-exit=1 -n "$world" -c "$((32/world))" \
          "$JL" --startup-file=no "$OUT/source/timing_flux.jl" "${args[@]}"
      elif [[ "$world" == 1 ]]; then
        timeout 1200 srun --kill-on-bad-exit=1 -n1 -c32 "$PY" "$OUT/source/timing_torch.py" "${args[@]}"
      else
        timeout 1200 srun --kill-on-bad-exit=1 -n1 -c32 "$PY" -m torch.distributed.run \
          --standalone --nnodes=1 --nproc_per_node="$world" "$OUT/source/timing_torch.py" "${args[@]}"
      fi
      set +x
      test -s "$file"
    done
  done
done
echo "STATIC_TIMING_$MODE PASS"
