#!/usr/bin/env bash
# Phase 5 memory pilot and selected-batch correctness on one Leonardo node.
#
# Submit with sbatch, for example:
#
#   ssh -o BatchMode=yes leonardo 'sbatch --parsable -A iscrc_unmasked \
#     -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=32 \
#     --gres=gpu:4 --time=01:30:00 --job-name=static-memory-pilot \
#     --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-memory-pilot-%j.log \
#     "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-memory.sh" pilot'
#
# Modes:
#   pilot                 run batch 512, fall back to 256, then correctness
#   matrix <512|256>      run one memory matrix and its checker
#   correctness <512|256> run the selected-batch numerical checks only
set -euo pipefail
MODE=${1:?pilot, matrix, or correctness}
[[ "$MODE" == pilot || "$MODE" == matrix || "$MODE" == correctness ]]
WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=$HOME/projects/ddp-flux-thesis
FLUX_DIR=$HOME/projects/flux-integration
ROOT=$WORK/epicazio/static-scaling-evidence/2026-10-19-phase5
OUT=$ROOT/$MODE-$SLURM_JOB_ID
mkdir -p "$OUT/source"
exec > >(tee "$OUT/driver.log") 2>&1
trap 'rc=$?; echo "PHASE5_DRIVER_EXIT=$rc"' EXIT
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
bash "$THESIS/scripts/remote/leonardo/precompile-phase3.sh"
export JULIA_PKG_PRECOMPILE_AUTO=0
PY=$WORK/epicazio/venvs/pytorch-scaling/bin/python
JL=$WORK/epicazio/julia-1.12.6/bin/julia
FIX=$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2
PREP=$WORK/epicazio/static-scaling-prepared
WARMUP=2
UPDATES=3
SOURCE_STATE_SHA256=$(cd "$OUT/source" && find . -type f -print0 | sort -z | \
  xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
SOURCE_COMMIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || echo unknown)

make_fixture() {
  local batch=$1
  local dir=$OUT/fixture-b$batch
  "$PY" "$OUT/source/make_phase5_fixture.py" \
    --prep-dir "$PREP" \
    --schedule-source "$FIX" \
    --bundle-source "$FIX" \
    --batch-size "$batch" --epoch 0 --out-dir "$dir" --force \
    --source-commit "$SOURCE_COMMIT" \
    --source-state-sha256 "$SOURCE_STATE_SHA256"
  test -s "$dir/data_fixture.npz"
  test -s "$dir/schedule_fixture.npz"
  test -s "$dir/initial_bundle.npz"
  test -s "$dir/fixture_manifest.json"
}

run_matrix() {
  local batch=$1
  local root=$OUT/memory-b$batch
  local fixture=$OUT/fixture-b$batch
  local failed=0
  mkdir -p "$root"
  for world in 1 2 4; do
    for framework in flux pytorch; do
      local dir=$root/$framework-w$world
      mkdir -p "$dir"
      local rc=0
      set +e
      if [[ "$framework" == flux ]]; then
        timeout 1200 srun --mpi=pmi2 --kill-on-bad-exit=1 -n "$world" -c "$((32/world))" \
          "$JL" --startup-file=no "$OUT/source/memory_flux.jl" \
          --fixture-dir "$fixture" --world-size "$world" --global-batch "$batch" \
          --warmup "$WARMUP" --updates "$UPDATES" --out-dir "$dir"
        rc=$?
      elif [[ "$world" == 1 ]]; then
        timeout 1200 srun --kill-on-bad-exit=1 -n1 -c32 \
          "$PY" "$OUT/source/memory_torch.py" \
          --fixture-dir "$fixture" --world-size 1 --global-batch "$batch" \
          --warmup "$WARMUP" --updates "$UPDATES" --out-dir "$dir"
        rc=$?
      else
        timeout 1200 srun --kill-on-bad-exit=1 -n1 -c32 \
          "$PY" -m torch.distributed.run --standalone --nnodes=1 \
          --nproc_per_node="$world" "$OUT/source/memory_torch.py" \
          --fixture-dir "$fixture" --world-size "$world" --global-batch "$batch" \
          --warmup "$WARMUP" --updates "$UPDATES" --out-dir "$dir"
        rc=$?
      fi
      set -e
      if [[ $rc -ne 0 ]]; then
        echo "MEMORY_CONFIG_FAIL framework=$framework world=$world batch=$batch rc=$rc"
        failed=1
      elif [[ ! -s "$dir/summary.json" ]]; then
        echo "MEMORY_CONFIG_FAIL framework=$framework world=$world batch=$batch missing_summary"
        failed=1
      fi
    done
  done
  return $failed
}

check_memory_batch() {
  local batch=$1
  "$PY" "$OUT/source/check_memory.py" \
    --batch-root "$batch=$OUT/memory-b$batch" \
    --out "$OUT/decision-b$batch.json"
}

run_correctness() {
  local batch=$1
  local root=$OUT/correctness-b$batch
  local fixture=$OUT/fixture-b$batch
  mkdir -p "$root"
  for framework in flux pytorch; do
    for world in 1 2 4; do
      if [[ "$framework" == flux ]]; then
        timeout 1200 srun --mpi=pmi2 --kill-on-bad-exit=1 -n "$world" -c "$((32/world))" \
          "$JL" --startup-file=no "$OUT/source/phase4_flux.jl" \
          --fixture-dir "$fixture" --mode ddp --device cuda --backend nccl \
          --world-size "$world" --global-batch "$batch" --out "$root" --force
      elif [[ "$world" == 1 ]]; then
        timeout 1200 srun --kill-on-bad-exit=1 -n1 -c32 \
          "$PY" "$OUT/source/phase4_torch.py" \
          --fixture-dir "$fixture" --mode ddp --device cuda --backend nccl \
          --world-size 1 --global-batch "$batch" --out-dir "$root" --force
      else
        timeout 1200 srun --kill-on-bad-exit=1 -n1 -c32 \
          "$PY" -m torch.distributed.run --standalone --nnodes=1 \
          --nproc_per_node="$world" "$OUT/source/phase4_torch.py" \
          --fixture-dir "$fixture" --mode ddp --device cuda --backend nccl \
          --world-size "$world" --global-batch "$batch" --out-dir "$root" --force
      fi
    done
  done
  "$PY" "$OUT/source/check_phase5.py" \
    --fixture-dir "$fixture" --ddp-root "$root/ddp" \
    --world-sizes 1,2,4 --out "$root/check.json"
}

select_batch() {
  if [[ "$MODE" == pilot ]]; then
    make_fixture 512
    if run_matrix 512 && check_memory_batch 512; then
      SELECTED=512
      echo "PHASE5_SELECTED_BATCH=512"
    else
      echo "PHASE5_BATCH_512_NOT_ACCEPTED"
      make_fixture 256
      if run_matrix 256 && check_memory_batch 256; then
        SELECTED=256
        echo "PHASE5_SELECTED_BATCH=256"
      else
        echo "PHASE5_NO_BATCH_PASSED"
        return 1
      fi
    fi
  else
    SELECTED=${2:?matrix or correctness requires a batch size}
    [[ "$SELECTED" == 512 || "$SELECTED" == 256 ]]
    make_fixture "$SELECTED"
    if [[ "$MODE" == matrix ]]; then
      run_matrix "$SELECTED"
      check_memory_batch "$SELECTED"
    fi
  fi
  printf '%s\n' "$SELECTED" > "$OUT/selected-batch.txt"
}

select_batch "$@"
if [[ "$MODE" == pilot || "$MODE" == correctness ]]; then
  run_correctness "$SELECTED"
  {
    echo "# Phase 5 memory pilot summary"
    echo
    echo "- job: \`$SLURM_JOB_ID\` on \`$(hostname)\`"
    echo "- UTC: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo "- selected global batch: **$SELECTED**"
    echo "- memory decision: \`$OUT/decision-b$SELECTED.json\`"
    echo "- selected-batch numerical check: \`$OUT/correctness-b$SELECTED/check.json\`"
    echo "- source-state SHA-256: \`$SOURCE_STATE_SHA256\`"
  } > "$OUT/SUMMARY.md"
  echo "PHASE5_MEMORY_PILOT PASS batch=$SELECTED"
else
  echo "PHASE5_MATRIX PASS batch=$SELECTED"
fi
