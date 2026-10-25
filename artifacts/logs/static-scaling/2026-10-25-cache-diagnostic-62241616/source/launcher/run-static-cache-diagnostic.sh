#!/usr/bin/env bash
# Bounded cache diagnostic, independent of the OD-8 capacity campaign.
# Submit from the deployed thesis root, after source review:
# EXPECTED_THESIS_COMMIT=<local HEAD> sbatch --export=ALL -A iscrc_unmasked \
#   -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 --cpus-per-task=4 \
#   --gres=gpu:1 --time=02:00:00 --job-name=static-cache-diagnostic \
#   --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-cache-%j.log \
#   scripts/remote/leonardo/run-static-cache-diagnostic.sh
# Each measurement is a fresh process. Cache release happens after training.
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FLUX_CURRENT=${FLUX_CURRENT:-$HOME/projects/flux-integration}
PREP=${PREP:-$WORK/epicazio/static-scaling-prepared}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
PYTHON_BIN=${PYTHON_BIN:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
SCRIPTS=$THESIS/scripts/static_scaling
CHECKER=$SCRIPTS/check_cache_diagnostic.py
PRECOMPILE=$THESIS/scripts/remote/leonardo/precompile-phase3.sh
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/static-cache-${SLURM_JOB_ID:-manual}}
EXPECTED_CURRENT_COMMIT=${EXPECTED_CURRENT_COMMIT:-f7b4520c4bd307c67bbc5ac92a1c68493eccff6e}
EXPECTED_CURRENT_SOURCE_TREE=${EXPECTED_CURRENT_SOURCE_TREE:-83da2929ab7ebd66dbceb83452c8e55eae5e0444c8e96866cc2a25b513c3eb0d}

fail() { echo "CACHE_DIAGNOSTIC FAIL $*" >&2; exit 1; }
host=$(hostname)
case "$host" in login*) fail "compute node required: $host" ;; esac
[ -n "${SLURM_JOB_ID:-}" ] || fail 'Slurm allocation required'
[ "${SLURM_NTASKS:-0}" = 1 ] || fail 'exactly one task required'
[ "${SLURM_CPUS_PER_TASK:-0}" = 4 ] || fail 'exactly four CPUs per task required'
[ "${SLURM_PROCID:-0}" = 0 ] || fail 'driver must be rank zero'
for path in "$JULIA_BIN" "$PYTHON_BIN" "$PRECOMPILE" "$CHECKER" \
            "$FLUX_CURRENT/src" "$FLUX_CURRENT/ext" \
            "$PREP/train_images.npy" "$PREP/initial_bundle.npz"; do
  [ -e "$path" ] || fail "missing path: $path"
done
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then fail "output exists: $OUT"; fi
[ ! -e "$OUT" ] || [ -d "$OUT" ] || fail "output is not a directory: $OUT"
mkdir -p "$OUT/logs" "$OUT/source/launcher" "$OUT/records" "$OUT/samples"
GPU_SAMPLER_PID=''
cleanup() {
  local rc=$?
  if [ -n "$GPU_SAMPLER_PID" ]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  fi
  echo "CACHE_DIAGNOSTIC_DRIVER_EXIT=$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "CACHE_DIAGNOSTIC START utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID host=$host out=$OUT"

source_tree_sha256() {
  ( cd "$1" && find . -type f -not -path '*/__pycache__/*' \
      -not -path './.git/*' -print0 | sort -z | xargs -0 -r sha256sum \
      | sha256sum | cut -d' ' -f1 )
}
SOURCE_STATE_SHA256=$(source_tree_sha256 "$THESIS")
THESIS_COMMIT=$(git -C "$THESIS" rev-parse HEAD 2>/dev/null || true)
THESIS_COMMIT_SOURCE=git
if [ -z "$THESIS_COMMIT" ]; then
  THESIS_COMMIT=${EXPECTED_THESIS_COMMIT:?set EXPECTED_THESIS_COMMIT for Git-free deployment}
  THESIS_COMMIT_SOURCE=expected-no-git
fi
CURRENT_COMMIT=$(git -C "$FLUX_CURRENT" rev-parse HEAD 2>/dev/null || true)
CURRENT_COMMIT_SOURCE=git
if [ -z "$CURRENT_COMMIT" ]; then
  CURRENT_COMMIT=$EXPECTED_CURRENT_COMMIT
  CURRENT_COMMIT_SOURCE=expected-no-git
fi
[ "$CURRENT_COMMIT" = "$EXPECTED_CURRENT_COMMIT" ] || fail 'current Flux commit differs'
CURRENT_SOURCE_TREE=$(cd "$FLUX_CURRENT" && find src -type f -print0 | sort -z \
  | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1)
[ "$CURRENT_SOURCE_TREE" = "$EXPECTED_CURRENT_SOURCE_TREE" ] || fail 'current Flux source tree differs'
CURRENT_STATE_SOURCE=$CURRENT_COMMIT_SOURCE
if [ "$CURRENT_COMMIT_SOURCE" = git ]; then
  CURRENT_STATE=$(cd "$FLUX_CURRENT" && { git diff HEAD; \
    git ls-files -z --others --exclude-standard | sort -z | xargs -0 -r sha256sum; } \
    | sha256sum | cut -d' ' -f1)
else
  CURRENT_STATE=${EXPECTED_CURRENT_STATE:-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855}
fi
cat > "$OUT/source-identity.txt" <<EOF
THESIS_COMMIT=$THESIS_COMMIT
THESIS_COMMIT_SOURCE=$THESIS_COMMIT_SOURCE
THESIS_STATE_SHA256=$SOURCE_STATE_SHA256
THESIS_STATE_RECIPE=sha256 of sorted sha256sum over synced thesis tree excluding __pycache__ and .git
CURRENT_COMMIT=$CURRENT_COMMIT
CURRENT_COMMIT_SOURCE=$CURRENT_COMMIT_SOURCE
CURRENT_STATE_SHA256=$CURRENT_STATE
CURRENT_STATE_SOURCE=$CURRENT_STATE_SOURCE
CURRENT_SOURCE_TREE_SHA256=$CURRENT_SOURCE_TREE
EOF
cp -a "$SCRIPTS/." "$OUT/source/"
mkdir -p "$OUT/source/flux-current"
cp -a "$FLUX_CURRENT/src" "$FLUX_CURRENT/ext" "$FLUX_CURRENT/Project.toml" "$OUT/source/flux-current/"
for name in Manifest.toml test/Project.toml test/Manifest.toml; do
  if [ -f "$FLUX_CURRENT/$name" ]; then
    mkdir -p "$(dirname "$OUT/source/flux-current/$name")"
    cp -a "$FLUX_CURRENT/$name" "$OUT/source/flux-current/$name"
  fi
done
cp -a "$THESIS/scripts/remote/leonardo/run-static-cache-diagnostic.sh" "$PRECOMPILE" "$OUT/source/launcher/"
{
  date -u +%FT%TZ
  hostname
  printenv | LC_ALL=C sort | grep -E '^SLURM_|^CUDA_VISIBLE_DEVICES=' || true
  scontrol show job "$SLURM_JOB_ID"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader
} > "$OUT/resources.txt" 2>&1

# The current Flux test environment is the established Leonardo compile target.
echo 'STAGE precompile-current'
( unset JULIA_PROJECT JULIA_LOAD_PATH
  export WORK JULIA_BIN JULIA_DEPOT_PATH
  FLUX_DIR="$FLUX_CURRENT" bash "$PRECOMPILE" ) 2>&1 | tee "$OUT/logs/precompile-current.log"
grep -q FLUX_PRECOMPILE_OK "$OUT/logs/precompile-current.log" || fail 'missing precompile success marker'

export THESIS PREP OUT THESIS_COMMIT SOURCE_STATE_SHA256 CURRENT_COMMIT CURRENT_STATE
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os, socket, sys
from pathlib import Path
import torch
import numpy
out = Path(os.environ['OUT'])
sys.path.insert(0, str(Path(os.environ['THESIS']) / 'scripts/static_scaling'))
import full_dataset_common as fdc
arrays = fdc.load_epoch_arrays(os.environ['PREP'], 0)
bundle = fdc.bundle_info(os.environ['PREP'])
workloads = {}
for batch in (512, 256):
    ids = fdc.epoch_ids(arrays, batch)
    workloads[str(batch)] = {'n_eff': len(ids),
        'tail_size': len(arrays['epoch_permutation']) - len(ids),
        'updates_per_epoch': len(ids)//batch, 'sample_order_sha256': fdc.sha256_c_order(ids),
        'update1_order_sha256': fdc.sha256_c_order(fdc.global_batch_ids(arrays, batch, 1))}
dataset = {
    'schema': 'static-full-dataset-identity-v1', 'dataset_dir': os.environ['PREP'], 'epoch': 0,
    'files': fdc.dataset_file_hashes(os.environ['PREP'], verify_large=False),
    'bundle': {key: bundle[key] for key in ('bundle_hash', 'bundle_file_sha256', 'array_count', 'element_count')},
    'transform_spec_sha256': fdc.transform_spec_sha256(arrays['mean32'], arrays['std32']),
    'workloads': workloads,
}
(out / 'dataset-identity.json').write_text(json.dumps(dataset, indent=2, sort_keys=True)+'\n')
source_identity = dict(line.split('=', 1) for line in (out/'source-identity.txt').read_text().splitlines())
source_files = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((out/'source').rglob('*'))
                if path.is_file() and '__pycache__' not in path.parts}
manifest = {'schema': 'static-cache-campaign-v1', 'repeats': 2,
    'frameworks': ['flux-current', 'pytorch'], 'world_size': 1, 'global_batch': 256,
    'warmup': 20, 'updates': 20, 'epoch': 0, 'cpus_per_task': 4,
    'source_files': source_files, 'source_identity': source_identity,
    'job': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
    'claim': 'diagnostic allocator and sampled process memory; no throughput or live-tensor equivalence claim'}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
resources = {'schema': 'static-cache-resources-v1', 'cpus_per_task': int(os.environ['SLURM_CPUS_PER_TASK']),
    'ntasks': int(os.environ['SLURM_NTASKS']), 'job': os.environ['SLURM_JOB_ID'],
    'node': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'slurm_env': {k:v for k,v in os.environ.items() if k.startswith('SLURM_')}}
(out/'resources.json').write_text(json.dumps(resources, indent=2, sort_keys=True)+'\n')
runtime = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
    'numpy': numpy.__version__, 'numpy_path': numpy.__file__,
    'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'torch_path': torch.__file__}
(out/'runtime-python.json').write_text(json.dumps(runtime, indent=2, sort_keys=True)+'\n')
PY

start_sampler() {
  local sample=$1
  (
    while :; do
      printf 'TIME,%s\n' "$(date +%s.%N)"
      nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null || true
      sleep 0.25
    done
  ) > "$sample" &
  GPU_SAMPLER_PID=$!
}
for repeat in 1 2; do
  for framework in flux-current pytorch; do
    stem=$framework-w1-r$repeat
    record=$OUT/records/$stem.json
    sample=$OUT/samples/$stem.csv
    log=$OUT/logs/$stem.log
    export FULL_PREFLIGHT_RANK_DIR=$OUT/samples/$stem.ranks
    mkdir -p "$FULL_PREFLIGHT_RANK_DIR"
    args=(--dataset-dir "$PREP" --world-size 1 --epoch 0 --batch-size 256 \
      --updates 20 --warmup 20 --out "$record" --cache-diagnostic true \
      --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256")
    echo "CASE framework=$framework world=1 repeat=$repeat"
    start_sampler "$sample"
    if [ "$framework" = flux-current ]; then
      ( unset JULIA_PROJECT
        export JULIA_LOAD_PATH="$SCRIPTS/julia-env:$FLUX_CURRENT/test:@stdlib"
        export JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        exec "$JULIA_BIN" --startup-file=no "$SCRIPTS/timing_flux.jl" "${args[@]}" \
          --distributed-api guarded --flux-commit "$CURRENT_COMMIT" --flux-state-sha256 "$CURRENT_STATE"
      ) 2>&1 | tee "$log"
    else
      ( export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        exec "$PYTHON_BIN" "$SCRIPTS/timing_torch.py" "${args[@]}"
      ) 2>&1 | tee "$log"
    fi
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
    GPU_SAMPLER_PID=''
    unset FULL_PREFLIGHT_RANK_DIR
    if grep -q 'Precompiling packages' "$log"; then fail "unexpected precompile: $stem"; fi
    [ -s "$record" ] || fail "missing record: $stem"
    echo "CASE_PASS $stem"
  done
done
"$PYTHON_BIN" "$CHECKER" --root "$OUT" 2>&1 | tee "$OUT/checker.log"
# Relative checksums remain valid after evidence is copied locally.
( cd "$OUT" && find . -type f -not -name SHA256SUMS -not -path '*/__pycache__/*' -print0 \
  | sort -z | xargs -0 -r sha256sum > SHA256SUMS )
echo 'CACHE_DIAGNOSTIC PASS'
