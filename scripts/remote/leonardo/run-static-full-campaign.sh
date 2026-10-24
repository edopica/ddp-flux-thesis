#!/usr/bin/env bash
# =============================================================================
# Full-dataset static-scaling preflight and campaign driver.
#
# Modes (first argument, or CAMPAIGN_MODE):
#   all        run the bounded preflight, then the timing campaign (default)
#   preflight  run the bounded preflight and write the batch/memory decision
#   campaign   reuse a decision from PREFLIGHT_DECISION and run timing only
#
# The preflight covers Flux current, Flux historical, and PyTorch at W=1,2,4.
# It runs 20 discarded warmup updates, resets the model/optimizer, then runs a
# bounded sustained interval. Peak per-process device memory is sampled with
# nvidia-smi. All nine configurations must stay at or below 80% of the device
# total. This campaign is pinned to batch 256 by the
# user decision on 2026-10-19; batch 512 is not attempted.
#
# The campaign runs batch 256 with K=390, five repeats by default,
# repeat-first. Every repeat is sampled with the external nvidia-smi
# per-process sampler (0.25 s target interval) and keeps its own raw trace and
# per-rank PID/GPU registration; both runners record per-rank wall-clock
# training windows so the checker can separate startup from training and
# exclude final hashing/snapshots. It then runs the matching
# untimed diagnostics (same epoch, batch, K, and warmup) and writes snapshots.
# A separate artifact checker enforces configuration, source, runtime, resource,
# provenance, timing, and OD-14 diagnostic requirements.
#
# SINGLE SUBMISSION COMMAND (all):
#   ssh leonardo 'CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=all sbatch --parsable \
#     -A iscrc_unmasked -p boost_usr_prod --qos=boost_qos_lprod -N1 -n1 \
#     --cpus-per-task=4 --gres=gpu:4 --time=04:00:00 \
#     --job-name=static-full-campaign \
#     --output=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/static-full-campaign-%j.log \
#     "$HOME/projects/ddp-flux-thesis/scripts/remote/leonardo/run-static-full-campaign.sh"'
#
# PREFLIGHT-ONLY SUBMISSION COMMAND:
#   ssh leonardo 'CONFIRM_CAMPAIGN=1 CAMPAIGN_MODE=preflight sbatch ...'
#
# This driver refuses to run unless CONFIRM_CAMPAIGN=1 is set. It must receive
# the source/policy review recorded in the thesis wiki before use as evidence.
# =============================================================================
set -euo pipefail

MODE=${1:-${CAMPAIGN_MODE:-all}}
case "$MODE" in
  all|preflight|campaign) ;;
  *) echo "ERROR: mode must be all, preflight, or campaign; got $MODE" >&2; exit 1 ;;
esac

if [ "${CONFIRM_CAMPAIGN:-0}" != "1" ]; then
  echo "ERROR: campaign driver requires CONFIRM_CAMPAIGN=1" >&2
  exit 1
fi

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
FLUX_CURRENT=${FLUX_CURRENT:-$HOME/projects/flux-integration}
FLUX_HISTORICAL=${FLUX_HISTORICAL:-$HOME/projects/flux-static-baseline}
PREP=${PREP:-$WORK/epicazio/static-scaling-prepared}
JULIA_BIN=${JULIA_BIN:-$WORK/epicazio/julia-1.12.6/bin/julia}
JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$WORK/epicazio/julia-depot}
PYTHON_BIN=${PYTHON_BIN:-$WORK/epicazio/venvs/pytorch-scaling/bin/python}
SCRIPTS=$THESIS/scripts/static_scaling
PRECOMPILE=$THESIS/scripts/remote/leonardo/precompile-phase3.sh
CHECKER=$SCRIPTS/check_full_campaign.py
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-4}
JOB_ID=${SLURM_JOB_ID:-manual}
OUT=${OUT:-$WORK/epicazio/static-scaling-evidence/static-full-campaign-$JOB_ID}
REPEATS=${REPEATS:-5}
case "$REPEATS" in
  ''|*[!0-9]*) echo "ERROR: REPEATS must be a positive integer; got $REPEATS" >&2; exit 1 ;;
esac
[ "$REPEATS" -ge 1 ] || { echo "ERROR: REPEATS must be at least 1" >&2; exit 1; }
EPOCH=${EPOCH:-0}
WARMUP=${WARMUP:-20}
PREFLIGHT_WARMUP=${PREFLIGHT_WARMUP:-20}
PREFLIGHT_UPDATES=${PREFLIGHT_UPDATES:-20}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-0.25}
WORLDS=${WORLDS:-"1 2 4"}
FRAMEWORKS=${FRAMEWORKS:-"flux-current flux-historical pytorch"}
PREFLIGHT_BATCH=""

EXPECTED_CURRENT_COMMIT=${EXPECTED_CURRENT_COMMIT:-f7b4520c4bd307c67bbc5ac92a1c68493eccff6e}
EXPECTED_CURRENT_STATE=${EXPECTED_CURRENT_STATE:-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855}
EXPECTED_CURRENT_SOURCE_TREE=${EXPECTED_CURRENT_SOURCE_TREE:-83da2929ab7ebd66dbceb83452c8e55eae5e0444c8e96866cc2a25b513c3eb0d}
EXPECTED_HISTORICAL_COMMIT=${EXPECTED_HISTORICAL_COMMIT:-8add0e42b7d9388ff365335152126e5b1e9de657}
EXPECTED_HISTORICAL_STATE=${EXPECTED_HISTORICAL_STATE:-4af8409c2a2ed1c661e8ae7742877f1ee9789d00dc7e404365ecefd98ab0d34c}
EXPECTED_HISTORICAL_SOURCE_TREE=${EXPECTED_HISTORICAL_SOURCE_TREE:-ad114ed385f7cc78e8409fa8e1f09d38133555f7fd58d5d9aca6a846bcf5e757}
EXPECTED_THESIS_COMMIT=${EXPECTED_THESIS_COMMIT:-660e363e6de5b6e4b398e542d79b6ce1adb381da}

host=$(hostname)
case "$host" in
  login*) echo "ERROR: run under srun on a compute node (host=$host)" >&2; exit 1 ;;
esac
if [ "${SLURM_NTASKS:-1}" != "1" ] || [ "${SLURM_PROCID:-0}" != "0" ]; then
  echo "ERROR: campaign driver must run as one task" >&2
  exit 1
fi
for path in "$JULIA_BIN" "$PYTHON_BIN" "$PRECOMPILE" "$CHECKER" \
            "$FLUX_CURRENT/src" "$FLUX_HISTORICAL/src" \
            "$PREP/train_images.npy" "$PREP/initial_bundle.npz"; do
  [ -e "$path" ] || { echo "ERROR: missing required path: $path" >&2; exit 1; }
done
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]; then
  echo "ERROR: output root exists and is not empty: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT/logs" "$OUT/source"
DRIVER_LOG=$OUT/driver.log
GPU_SAMPLER_PID=""
TEE_PID=""
cleanup_gpu_sampler() {
  if [ -n "${GPU_SAMPLER_PID:-}" ]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
    GPU_SAMPLER_PID=""
  fi
}
on_exit() {
  local rc=$?
  cleanup_gpu_sampler
  echo "TIMING_DRIVER_EXIT=$rc"
}
close_driver_log() {
  local tee_rc=0
  trap - EXIT INT TERM
  if [ -n "${TEE_PID:-}" ]; then
    exec 1>&- 2>&-
    wait "$TEE_PID" || tee_rc=$?
    if [ "$tee_rc" -ne 0 ]; then
      printf 'DRIVER_TEE_FAIL rc=%s\n' "$tee_rc" >> "$DRIVER_LOG"
      exit "$tee_rc"
    fi
  fi
}
exec 3>&1
exec > >(tee -a "$DRIVER_LOG") 2>&1
TEE_PID=$!
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
sha256_file() { sha256sum "$1" | cut -d' ' -f1; }
source_tree_sha256() {
  ( cd "$1" && find . -type f \
      -not -path '*/__pycache__/*' -not -path './.git/*' -print0 \
      | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 )
}
git_head() { git -C "$1" rev-parse HEAD 2>/dev/null || true; }
git_state_sha256() {
  ( cd "$1" && { git diff HEAD; git ls-files -z --others --exclude-standard \
      | sort -z | xargs -0 -r sha256sum; } | sha256sum | cut -d' ' -f1 )
}
flux_source_tree_sha256() {
  ( cd "$1" && find src -type f -print0 | sort -z \
      | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 )
}

echo "FULL_CAMPAIGN_DRIVER START utc=$(utc_now) host=$host job=$JOB_ID mode=$MODE"
echo "OUT=$OUT PREP=$PREP THESIS=$THESIS"

SOURCE_STATE_SHA256=$(source_tree_sha256 "$THESIS")
THESIS_COMMIT=$(git_head "$THESIS")
if [ -n "$THESIS_COMMIT" ]; then
  THESIS_COMMIT_SOURCE=git
  [ "$THESIS_COMMIT" = "$EXPECTED_THESIS_COMMIT" ] || {
    echo "ERROR: thesis commit $THESIS_COMMIT != expected $EXPECTED_THESIS_COMMIT" >&2
    exit 1
  }
else
  THESIS_COMMIT=$EXPECTED_THESIS_COMMIT
  THESIS_COMMIT_SOURCE=expected-no-git
fi
echo "THESIS_SOURCE thesis=$THESIS_COMMIT state=$SOURCE_STATE_SHA256 commit_source=$THESIS_COMMIT_SOURCE"

verify_flux_source() {
  local name=$1 dir=$2 expected_commit=$3 expected_state=$4 expected_tree=$5
  local actual_commit actual_state actual_tree commit_source state_source
  actual_commit=$(git_head "$dir")
  actual_tree=$(flux_source_tree_sha256 "$dir")
  if [ -n "$actual_commit" ]; then
    commit_source=git
    [ "$actual_commit" = "$expected_commit" ] || {
      echo "ERROR: $name Flux commit $actual_commit != expected $expected_commit" >&2
      exit 1
    }
    actual_state=$(git_state_sha256 "$dir")
    state_source=git
    [ "$actual_state" = "$expected_state" ] || {
      echo "ERROR: $name Flux state $actual_state != expected $expected_state" >&2
      exit 1
    }
  else
    commit_source=expected-no-git
    state_source=expected-no-git
    actual_commit=$expected_commit
    actual_state=$expected_state
  fi
  [ "$actual_tree" = "$expected_tree" ] || {
    echo "ERROR: $name Flux source tree $actual_tree != expected $expected_tree" >&2
    exit 1
  }
  echo "FLUX_SOURCE name=$name repo=$dir commit=$actual_commit state=$actual_state source_tree=$actual_tree commit_source=$commit_source state_source=$state_source"
  VERIFIED_COMMIT=$actual_commit
  VERIFIED_STATE=$actual_state
  VERIFIED_SOURCE_TREE=$actual_tree
  VERIFIED_COMMIT_SOURCE=$commit_source
  VERIFIED_STATE_SOURCE=$state_source
}

verify_flux_source current "$FLUX_CURRENT" "$EXPECTED_CURRENT_COMMIT" \
  "$EXPECTED_CURRENT_STATE" "$EXPECTED_CURRENT_SOURCE_TREE"
CURRENT_COMMIT=$VERIFIED_COMMIT
CURRENT_STATE=$VERIFIED_STATE
CURRENT_SOURCE_TREE=$VERIFIED_SOURCE_TREE
CURRENT_COMMIT_SOURCE=$VERIFIED_COMMIT_SOURCE
CURRENT_STATE_SOURCE=$VERIFIED_STATE_SOURCE

verify_flux_source historical "$FLUX_HISTORICAL" "$EXPECTED_HISTORICAL_COMMIT" \
  "$EXPECTED_HISTORICAL_STATE" "$EXPECTED_HISTORICAL_SOURCE_TREE"
HISTORICAL_COMMIT=$VERIFIED_COMMIT
HISTORICAL_STATE=$VERIFIED_STATE
HISTORICAL_SOURCE_TREE=$VERIFIED_SOURCE_TREE
HISTORICAL_COMMIT_SOURCE=$VERIFIED_COMMIT_SOURCE
HISTORICAL_STATE_SOURCE=$VERIFIED_STATE_SOURCE

cat > "$OUT/source-identity.txt" <<EOF
THESIS_COMMIT=$THESIS_COMMIT
THESIS_COMMIT_SOURCE=$THESIS_COMMIT_SOURCE
THESIS_STATE_SHA256=$SOURCE_STATE_SHA256
THESIS_STATE_RECIPE=sha256 of sorted (sha256sum) over the synced thesis tree, excluding __pycache__ and .git
CURRENT_COMMIT=$CURRENT_COMMIT
CURRENT_STATE_SHA256=$CURRENT_STATE
CURRENT_SOURCE_TREE_SHA256=$CURRENT_SOURCE_TREE
CURRENT_COMMIT_SOURCE=$CURRENT_COMMIT_SOURCE
CURRENT_STATE_SOURCE=$CURRENT_STATE_SOURCE
HISTORICAL_COMMIT=$HISTORICAL_COMMIT
HISTORICAL_STATE_SHA256=$HISTORICAL_STATE
HISTORICAL_SOURCE_TREE_SHA256=$HISTORICAL_SOURCE_TREE
HISTORICAL_COMMIT_SOURCE=$HISTORICAL_COMMIT_SOURCE
HISTORICAL_STATE_SOURCE=$HISTORICAL_STATE_SOURCE
EOF
cat "$OUT/source-identity.txt"

cp -a "$SCRIPTS/." "$OUT/source/"
rm -rf "$OUT/source/__pycache__"
mkdir -p "$OUT/source/flux-current" "$OUT/source/flux-historical" "$OUT/source/launcher"
cp -a "$FLUX_CURRENT/src" "$OUT/source/flux-current/src"
cp -a "$FLUX_CURRENT/ext" "$OUT/source/flux-current/ext" 2>/dev/null || true
cp -a "$FLUX_CURRENT/Project.toml" "$OUT/source/flux-current/Project.toml"
cp -a "$FLUX_CURRENT/Manifest.toml" "$OUT/source/flux-current/Manifest.toml"
cp -a "$FLUX_HISTORICAL/src" "$OUT/source/flux-historical/src"
cp -a "$FLUX_HISTORICAL/ext" "$OUT/source/flux-historical/ext" 2>/dev/null || true
cp -a "$FLUX_HISTORICAL/Project.toml" "$OUT/source/flux-historical/Project.toml"
cp -a "$FLUX_HISTORICAL/Manifest.toml" "$OUT/source/flux-historical/Manifest.toml"
cp -a "$THESIS/scripts/remote/leonardo/run-static-full-campaign.sh" "$OUT/source/launcher/"
find "$OUT/source" -type f -not -path '*/__pycache__/*' -print0 \
  | sort -z | xargs -0 -r sha256sum > "$OUT/source-sha256.txt"
printf '%s  launcher/run-static-full-campaign.sh\n' \
  "$(sha256_file "$THESIS/scripts/remote/leonardo/run-static-full-campaign.sh")" \
  > "$OUT/launcher-sha256.txt"
printf '%s  check_full_campaign.py\n' "$(sha256_file "$CHECKER")" \
  >> "$OUT/launcher-sha256.txt"

precompile_variant() {
  local name=$1 dir=$2
  local log="$OUT/logs/precompile-$name.log"
  echo "STAGE precompile-$name"
  if ! ( unset JULIA_PROJECT JULIA_LOAD_PATH
         export WORK JULIA_BIN JULIA_DEPOT_PATH JULIA_PKG_PRECOMPILE_AUTO=1
         FLUX_DIR="$dir" bash "$PRECOMPILE" ) 2>&1 | tee "$log"; then
    echo "FULL_CAMPAIGN FAIL precompile-$name" >&2
    exit 1
  fi
  grep -q "FLUX_PRECOMPILE_OK" "$log" || {
    echo "FULL_CAMPAIGN FAIL precompile-$name: missing FLUX_PRECOMPILE_OK" >&2
    exit 1
  }
}
precompile_variant current "$FLUX_CURRENT"
precompile_variant historical "$FLUX_HISTORICAL"

capture_runtime() {
  local variant dir
  {
    echo "UTC=$(utc_now)"
    echo "HOST=$host"
    "$JULIA_BIN" --version
    for variant in flux-current flux-historical; do
      if [ "$variant" = "flux-current" ]; then dir=$FLUX_CURRENT; else dir=$FLUX_HISTORICAL; fi
      (
        unset JULIA_PROJECT
        export JULIA_DEPOT_PATH JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS=1
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        export JULIA_LOAD_PATH="$SCRIPTS/julia-env:$dir/test:@stdlib"
        export RUNTIME_VARIANT="$variant"
        "$JULIA_BIN" --startup-file=no -e '
          import Flux, CUDA, NCCL, MPI, Optimisers, Functors
          function ver(pkg)
              try
                  value = Base.pkgversion(pkg)
                  value === nothing ? "unknown" : string(value)
              catch
                  "unknown"
              end
          end
          println("RUNTIME_JULIA variant=", ENV["RUNTIME_VARIANT"],
                  " julia=", VERSION, " flux=", ver(Flux), " cuda=", ver(CUDA),
                  " nccl=", ver(NCCL), " mpi=", ver(MPI),
                  " optimisers=", ver(Optimisers), " functors=", ver(Functors))
        '
      ) 2>&1 || echo "RUNTIME_VARIANT_FAILED variant=$variant"
    done
    "$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch
import numpy
print("RUNTIME_PYTHON python=%s torch=%s numpy=%s cuda=%s cudnn=%s executable=%s"
      % (platform.python_version(), torch.__version__, numpy.__version__,
         torch.version.cuda, torch.backends.cudnn.version(), sys.executable))
print("RUNTIME_PYTHON_PATH torch=%s numpy=%s" % (torch.__file__, numpy.__file__))
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print("RUNTIME_GPU index=%d uuid=%s name=%s total_bytes=%d"
          % (index, getattr(props, "uuid", ""), props.name, props.total_memory))
PY
    nvidia-smi -L 2>&1 || true
    echo "RUNTIME_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  } > "$OUT/runtime-env.txt" 2>&1 || echo "RUNTIME_ENV_CAPTURE_FAILED" >> "$OUT/runtime-env.txt"
  "$PYTHON_BIN" - "$OUT/runtime-env.txt" "$OUT/runtime.json" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
variants = {}
python = {}
gpus = []
driver = ""
julia_version = ""
for raw in text.splitlines():
    if raw.startswith("julia version "):
        julia_version = raw.split("julia version ", 1)[1].strip()
    elif raw.startswith("RUNTIME_JULIA "):
        values = {}
        for token in raw.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        variants[values.pop("variant", "")] = values
    elif raw.startswith("RUNTIME_PYTHON "):
        for token in raw.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                python[key] = value
    elif raw.startswith("RUNTIME_GPU "):
        values = {}
        for token in raw.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        try:
            values["index"] = int(values.get("index", -1))
            values["total_bytes"] = int(values.get("total_bytes", 0))
        except ValueError:
            continue
        if values["index"] >= 0 and values["total_bytes"] > 0:
            gpus.append(values)
    elif raw.startswith("RUNTIME_DRIVER="):
        driver = raw.split("=", 1)[1].strip()
payload = {
    "schema": "static-full-runtime-v1",
    "julia_version": julia_version,
    "variants": variants,
    "python": python,
    "gpus": gpus,
    "driver_version": driver,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}
capture_runtime

{
  echo "UTC=$(utc_now)"
  echo "HOST=$host"
  env | grep '^SLURM_' | sort || true
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  scontrol show job "$JOB_ID" 2>&1 || true
  nvidia-smi -L 2>&1 || true
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
    --format=csv,noheader 2>&1 || true
  nvidia-smi topo -m 2>&1 || true
} > "$OUT/resources.txt" 2>&1
"$PYTHON_BIN" - "$OUT" <<'PY'
import json
import os
import re
import socket
import sys
from pathlib import Path

out = Path(sys.argv[1])
resources_text = (out / "resources.txt").read_text(encoding="utf-8", errors="replace")
runtime = json.loads((out / "runtime.json").read_text(encoding="utf-8"))
env = {key: value for key, value in os.environ.items() if key.startswith("SLURM_")}

def match(pattern):
    found = re.search(pattern, resources_text)
    return found.group(1) if found else ""

ntasks = int(env.get("SLURM_NTASKS", "1"))
cpus = int(env.get("SLURM_CPUS_PER_TASK", "1"))
gpu_count = len(runtime.get("gpus", []))
gres = env.get("SLURM_JOB_GRES") or env.get("SLURM_GRES") or ("gpu:%d" % gpu_count)
payload = {
    "schema": "static-full-resources-v1",
    "job": env.get("SLURM_JOB_ID", "manual"),
    "node": socket.gethostname(),
    "partition": env.get("SLURM_JOB_PARTITION") or match(r"Partition=(\S+)"),
    "qos": env.get("SLURM_JOB_QOS") or match(r"QOS=(\S+)"),
    "account": env.get("SLURM_JOB_ACCOUNT") or match(r"Account=(\S+)"),
    "ntasks": ntasks,
    "cpus_per_task": cpus,
    "gres": gres,
    "time_limit": match(r"TimeLimit=(\S+)") or "unavailable",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "gpu_mapping": runtime.get("gpus", []),
    "slurm_env": env,
}
(out / "resources.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
cat "$OUT/resources.json"

export THESIS
"$PYTHON_BIN" - "$PREP" "$EPOCH" "$OUT/dataset-identity.json" <<'PY'
import json
import os
import sys
from pathlib import Path

prep, epoch, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, os.path.join(os.environ["THESIS"], "scripts", "static_scaling"))
import full_dataset_common as fdc

arrays = fdc.load_epoch_arrays(prep, epoch)
files = fdc.dataset_file_hashes(prep, verify_large=False)
bundle = fdc.bundle_info(prep)
workloads = {}
for batch in (512, 256):
    ids = fdc.epoch_ids(arrays, batch)
    workloads[str(batch)] = {
        "n_eff": int(len(ids)),
        "tail_size": int(len(arrays["epoch_permutation"]) - len(ids)),
        "updates_per_epoch": int(len(ids) // batch),
        "sample_order_sha256": fdc.sha256_c_order(ids),
        "update1_order_sha256": fdc.sha256_c_order(fdc.global_batch_ids(arrays, batch, 1)),
    }
payload = {
    "schema": "static-full-dataset-identity-v1",
    "dataset_dir": os.path.abspath(prep),
    "epoch": epoch,
    "files": files,
    "bundle": {
        "bundle_hash": bundle["bundle_hash"],
        "bundle_file_sha256": bundle["bundle_file_sha256"],
        "array_count": bundle["array_count"],
        "element_count": bundle["element_count"],
    },
    "transform_spec_sha256": fdc.transform_spec_sha256(arrays["mean32"], arrays["std32"]),
    "workloads": workloads,
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("DATASET_IDENTITY_WRITTEN", out)
PY

start_gpu_sampler() {
  local sample=$1
  mkdir -p "$(dirname "$sample")" || return 1
  : > "$sample" || return 1
  (
    while :; do
      # The timestamp is taken before the query: each memory value is stamped
      # slightly earlier than the instant nvidia-smi reports it. This shifts at
      # most the boundary samples by one query latency.
      printf 'TIME,%s\n' "$(date +%s.%N)"
      nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep "$GPU_SAMPLE_INTERVAL"
    done
  ) >> "$sample" 2>/dev/null &
  GPU_SAMPLER_PID=$!
  return 0
}
stop_gpu_sampler() {
  local pid=$1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  if [ "${GPU_SAMPLER_PID:-}" = "$pid" ]; then
    GPU_SAMPLER_PID=""
  fi
  return 0
}

run_flux() {
  local flavor=$1 world=$2 out=$3 updates=$4 warmup=$5 snapshot=$6
  local api=guarded commit=$CURRENT_COMMIT state=$CURRENT_STATE load_path=$SCRIPTS/julia-env:$FLUX_CURRENT/test:@stdlib
  if [ "$flavor" = "flux-historical" ]; then
    api=static
    commit=$HISTORICAL_COMMIT
    state=$HISTORICAL_STATE
    load_path=$SCRIPTS/julia-env:$FLUX_HISTORICAL/test:@stdlib
  fi
  local -a args=(
    "$SCRIPTS/timing_flux.jl"
    --dataset-dir "$PREP" --world-size "$world" --epoch "$EPOCH"
    --batch-size "$PREFLIGHT_BATCH" --updates "$updates" --warmup "$warmup"
    --distributed-api "$api" --out "$out"
    --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256"
    --flux-commit "$commit" --flux-state-sha256 "$state"
  )
  if [ -n "$snapshot" ]; then args+=(--snapshot-dir "$snapshot"); fi
  if [ "$world" -eq 1 ]; then
    ( unset JULIA_PROJECT
      export JULIA_DEPOT_PATH JULIA_LOAD_PATH="$load_path" JULIA_PKG_PRECOMPILE_AUTO=0
      export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
      exec "$JULIA_BIN" --startup-file=no "${args[@]}" )
  else
    local quoted
    quoted=$(printf '%q ' "${args[@]}")
    srun --overlap --mpi=pmi2 --kill-on-bad-exit=1 -n"$world" -c"$((TOTAL_CPUS / world))" \
      bash -c "unset JULIA_PROJECT; export JULIA_DEPOT_PATH='$JULIA_DEPOT_PATH' JULIA_LOAD_PATH='$load_path' JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0; exec '$JULIA_BIN' --startup-file=no $quoted"
  fi
}

run_pytorch() {
  local world=$1 out=$2 updates=$3 warmup=$4 snapshot=$5
  local -a args=(
    "$SCRIPTS/timing_torch.py"
    --dataset-dir "$PREP" --world-size "$world" --epoch "$EPOCH"
    --batch-size "$PREFLIGHT_BATCH" --updates "$updates" --warmup "$warmup" --out "$out"
    --source-commit "$THESIS_COMMIT" --source-state-sha256 "$SOURCE_STATE_SHA256"
  )
  if [ -n "$snapshot" ]; then args+=(--snapshot-dir "$snapshot"); fi
  if [ "$world" -eq 1 ]; then
    "$PYTHON_BIN" "${args[@]}"
  else
    srun --overlap -n1 -c"$TOTAL_CPUS" "$PYTHON_BIN" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node="$world" "${args[@]}"
  fi
}

run_case() {
  local framework=$1 world=$2 out=$3 sample=$4 updates=$5 warmup=$6 snapshot=$7 log=$8
  local sampler="" rc tee_rc
  mkdir -p "$(dirname "$out")" "$(dirname "$log")"
  echo "CASE framework=$framework world=$world updates=$updates warmup=$warmup batch=$PREFLIGHT_BATCH snapshot=${snapshot:-none}"
  if [ -n "$sample" ]; then
    export FULL_PREFLIGHT_RANK_DIR="${sample%.csv}.ranks"
    mkdir -p "$FULL_PREFLIGHT_RANK_DIR"
    start_gpu_sampler "$sample" || {
      echo "CASE_FAIL framework=$framework world=$world sampler_start" >&2
      unset FULL_PREFLIGHT_RANK_DIR
      return 1
    }
    sampler=$GPU_SAMPLER_PID
  fi
  set +e
  if [ "$framework" = "flux-current" ] || [ "$framework" = "flux-historical" ]; then
    run_flux "$framework" "$world" "$out" "$updates" "$warmup" "$snapshot" 2>&1 | tee "$log"
  else
    run_pytorch "$world" "$out" "$updates" "$warmup" "$snapshot" 2>&1 | tee "$log"
  fi
  local -a pipe_status=("${PIPESTATUS[@]}")
  rc=${pipe_status[0]:-1}
  tee_rc=${pipe_status[1]:-1}
  set -e
  if [ -n "$sampler" ]; then stop_gpu_sampler "$sampler"; fi
  unset FULL_PREFLIGHT_RANK_DIR
  if [ "$tee_rc" -ne 0 ]; then
    echo "CASE_FAIL framework=$framework world=$world rc=$tee_rc stage=log_tee log=$log" >&2
    return "$tee_rc"
  fi
  if [ "$rc" -ne 0 ]; then
    echo "CASE_FAIL framework=$framework world=$world rc=$rc log=$log" >&2
    return "$rc"
  fi
  if grep -q "Precompiling packages" "$log"; then
    echo "CASE_FAIL framework=$framework world=$world unexpected_precompilation" >&2
    return 1
  fi
  [ -s "$out" ] || { echo "CASE_FAIL framework=$framework world=$world missing_record" >&2; return 1; }
  if [ -n "$snapshot" ]; then
    [ -s "$snapshot/state.npz" ] || {
      echo "CASE_FAIL framework=$framework world=$world missing_snapshot" >&2
      return 1
    }
  fi
  echo "CASE_PASS framework=$framework world=$world out=$out"
}

run_preflight_candidate() {
  local batch=$1 root="$OUT/preflight/b$1" failed=0
  mkdir -p "$root/records" "$root/samples" "$root/logs"
  PREFLIGHT_BATCH=$batch
  for world in $WORLDS; do
    for framework in $FRAMEWORKS; do
      run_case "$framework" "$world" \
        "$root/records/$framework-w$world.json" \
        "$root/samples/$framework-w$world.csv" \
        "$PREFLIGHT_UPDATES" "$PREFLIGHT_WARMUP" "" \
        "$root/logs/$framework-w$world.log" || failed=1
    done
  done
  return "$failed"
}

select_preflight_batch() {
  run_preflight_candidate 256 || {
    echo "FULL_CAMPAIGN FAIL preflight: batch 256 did not complete all nine runs" >&2
    return 1
  }
  "$PYTHON_BIN" "$CHECKER" preflight --root "$OUT" \
    --candidate "256=$OUT/preflight/b256" \
    --updates "$PREFLIGHT_UPDATES" --warmup "$PREFLIGHT_WARMUP" \
    --epoch "$EPOCH" --select --out "$OUT/preflight/decision.json" || return 1
  echo "PREFLIGHT_SELECTED_BATCH 256"
  printf '%s\n' 256 > "$OUT/preflight/selected-batch.txt"
}

write_campaign_metadata() {
  local batch=$1 updates=$2
  "$PYTHON_BIN" - "$OUT" "$MODE" "$batch" "$updates" "$EPOCH" "$WARMUP" \
    "$REPEATS" "$PREFLIGHT_UPDATES" "$PREFLIGHT_WARMUP" "$FRAMEWORKS" "$WORLDS" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(out, mode, batch, updates, epoch, warmup, repeats, preflight_updates,
 preflight_warmup, frameworks, worlds) = sys.argv[1:]
out = Path(out)
config = {
    "schema": "static-full-campaign-config-v1",
    "mode": mode,
    "batch": int(batch),
    "updates": int(updates),
    "epoch": int(epoch),
    "warmup": int(warmup),
    "repeats": int(repeats),
    "preflight_updates": int(preflight_updates),
    "preflight_warmup": int(preflight_warmup),
    "worlds": [int(item) for item in worlds.split()],
    "frameworks": frameworks.split(),
}
(out / "campaign-config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def parse_pairs(path):
    result = {}
    for raw in Path(path).read_text().splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            result[key] = value
    return result

source_identity = parse_pairs(out / "source-identity.txt")
runtime = json.loads((out / "runtime.json").read_text())
resources = json.loads((out / "resources.json").read_text())
launcher_hashes = {}
for raw in (out / "launcher-sha256.txt").read_text().splitlines():
    parts = raw.split(None, 1)
    if len(parts) == 2:
        launcher_hashes[Path(parts[1]).name] = parts[0]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

decision_path = out / "preflight" / "decision.json"
provenance = {
    "schema": "static-full-campaign-provenance-v1",
    "source_identity": source_identity,
    "campaign_config": config,
    "runtime": runtime,
    "resources": resources,
    "source_snapshot_manifest_sha256": sha256(out / "source-sha256.txt"),
    "runtime_env_sha256": sha256(out / "runtime-env.txt"),
    "runtime_json_sha256": sha256(out / "runtime.json"),
    "dataset_identity_sha256": sha256(out / "dataset-identity.json"),
    "resources_json_sha256": sha256(out / "resources.json"),
    "preflight_decision_sha256": sha256(decision_path),
    "launcher": {
        "name": "run-static-full-campaign.sh",
        "path": os.path.join(os.environ.get("THESIS", ""),
                             "scripts/remote/leonardo/run-static-full-campaign.sh"),
        "sha256": launcher_hashes.get("run-static-full-campaign.sh", ""),
    },
    "checker": {
        "name": "check_full_campaign.py",
        "path": os.path.join(os.environ.get("THESIS", ""),
                             "scripts/static_scaling/check_full_campaign.py"),
        "sha256": launcher_hashes.get("check_full_campaign.py", ""),
    },
    "execution_policy": {
        "flux": {
            "precision": "FP32 strict FMA, TF32 off",
            "cudnn_deterministic": False,
            "nonfinite_guard": "presence + nonfinite loss/gradient rejection inside timed window",
            "distributed_api_current": "guarded",
            "distributed_api_historical": "static",
        },
        "pytorch": {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "nonfinite_guard": "outside timed loop",
        },
    },
    "qualifications": [
        "Descriptive throughput and strong scaling only.",
        "Flux uses strict FMA candidate selection without a determinism filter.",
        "PyTorch uses deterministic cuDNN, benchmark disabled, and highest matmul precision.",
        "Flux DDP includes its non-finite guard inside the timed window; PyTorch does not.",
        "The state diagnostics are descriptive and apply no numerical-equivalence threshold.",
        "NUMERICAL_EQUIVALENCE_CLAIM none.",
    ],
    "NUMERICAL_EQUIVALENCE_CLAIM": "none",
}
(out / "provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("CAMPAIGN_METADATA_WRITTEN", out / "campaign-config.json", out / "provenance.json")
PY
}

if [ "$MODE" = "all" ] || [ "$MODE" = "preflight" ]; then
  select_preflight_batch
  SELECTED=$(cat "$OUT/preflight/selected-batch.txt")
  K=$([ "$SELECTED" = "512" ] && echo 195 || echo 390)
  write_campaign_metadata "$SELECTED" "$K"
  "$PYTHON_BIN" "$CHECKER" ready --root "$OUT"
  echo "FULL_CAMPAIGN_preflight PASS batch=$SELECTED k=$K"
  if [ "$MODE" = "preflight" ]; then
    echo "TIMING_DRIVER_EXIT=0"
    echo "FULL_CAMPAIGN_DRIVER COMPLETE mode=preflight OUT=$OUT"
    close_driver_log
    exit 0
  fi
else
  [ -n "${PREFLIGHT_DECISION:-}" ] || {
    echo "ERROR: campaign mode requires PREFLIGHT_DECISION=/path/to/decision.json" >&2
    exit 1
  }
  [ -s "$PREFLIGHT_DECISION" ] || {
    echo "ERROR: PREFLIGHT_DECISION is missing: $PREFLIGHT_DECISION" >&2
    exit 1
  }
  mkdir -p "$OUT/preflight"
  cp "$PREFLIGHT_DECISION" "$OUT/preflight/decision.json"
  SELECTED=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("selected_global_batch") or "")' \
    "$OUT/preflight/decision.json")
  [ "$SELECTED" = "256" ] || {
    echo "ERROR: this campaign requires batch 256" >&2
    exit 1
  }
  K=$([ "$SELECTED" = "512" ] && echo 195 || echo 390)
  write_campaign_metadata "$SELECTED" "$K"
  "$PYTHON_BIN" "$CHECKER" ready --root "$OUT"
  echo "FULL_CAMPAIGN_preflight PASS reused_decision=$PREFLIGHT_DECISION batch=$SELECTED k=$K"
fi

PREFLIGHT_BATCH=$SELECTED
echo "FULL_CAMPAIGN_CAMPAIGN START utc=$(utc_now) batch=$SELECTED updates=$K repeats=$REPEATS"

mkdir -p "$OUT/records" "$OUT/samples"
for repeat in $(seq 1 "$REPEATS"); do
  if [ $((repeat % 2)) -eq 1 ]; then
    repeat_worlds=$WORLDS
    repeat_frameworks=$FRAMEWORKS
  else
    repeat_worlds=$(printf '%s\n' $WORLDS | tac | tr '\n' ' ')
    repeat_frameworks=$(printf '%s\n' $FRAMEWORKS | tac | tr '\n' ' ')
  fi
  for world in $repeat_worlds; do
    for framework in $repeat_frameworks; do
      log="$OUT/logs/timing-$framework-w$world-r$repeat.log"
      # Each repeat keeps its own timestamped nvidia-smi trace and per-rank
      # PID/GPU registration. The checker clips these samples to the
      # per-rank training window recorded by the runner.
      if ! run_case "$framework" "$world" \
          "$OUT/records/$framework-w$world-r$repeat.json" \
          "$OUT/samples/$framework-w$world-r$repeat.csv" \
          "$K" "$WARMUP" "" "$log"; then
        echo "FULL_CAMPAIGN FAIL timing framework=$framework world=$world repeat=$repeat" >&2
        exit 1
      fi
    done
  done
done
FRAMEWORK_COUNT=$(printf '%s\n' $FRAMEWORKS | wc -l)
WORLD_COUNT=$(printf '%s\n' $WORLDS | wc -l)
echo "FULL_CAMPAIGN_timing PASS records=$(( FRAMEWORK_COUNT * WORLD_COUNT * REPEATS ))"

mkdir -p "$OUT/diagnostics/records" "$OUT/diagnostics/snapshots"
for world in $WORLDS; do
  for framework in $FRAMEWORKS; do
    log="$OUT/logs/diagnostic-$framework-w$world.log"
    if ! run_case "$framework" "$world" \
        "$OUT/diagnostics/records/$framework-w$world.json" "" \
        "$K" "$WARMUP" "$OUT/diagnostics/snapshots/$framework-w$world" "$log"; then
      echo "FULL_CAMPAIGN FAIL diagnostics framework=$framework world=$world" >&2
      exit 1
    fi
  done
done
echo "FULL_CAMPAIGN_diagnostics PASS snapshots=$(( FRAMEWORK_COUNT * WORLD_COUNT ))"

"$PYTHON_BIN" "$CHECKER" campaign --root "$OUT"
sync

"$PYTHON_BIN" - "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
config = json.loads((out / "campaign-config.json").read_text())
resources = json.loads((out / "resources.json").read_text())
report = json.loads((out / "REPORT.json").read_text())
records = sorted(path.name for path in (out / "records").glob("*.json"))
diagnostics = sorted(path.name for path in (out / "diagnostics" / "records").glob("*.json"))
summary = {
    "schema": "static-full-campaign-summary-v1",
    "job": resources["job"],
    "node": resources["node"],
    "config": config,
    "records": records,
    "diagnostics": diagnostics,
    "timing": report["timing"],
    "diagnostics_summary": {
        "comparisons": len(report["diagnostics"]["comparisons"]),
        "NUMERICAL_EQUIVALENCE_CLAIM": report["diagnostics"]["NUMERICAL_EQUIVALENCE_CLAIM"],
    },
}
(out / "SUMMARY.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("FULL_CAMPAIGN_SUMMARY_WRITTEN", out / "SUMMARY.json")
PY

close_driver_log
if ! {
  printf 'TIMING_DRIVER_EXIT=0\n'
  printf 'FULL_CAMPAIGN PASS utc=%s host=%s job=%s OUT=%s\n' \
    "$(utc_now)" "$host" "$JOB_ID" "$OUT"
} | tee -a "$DRIVER_LOG" >&3; then
  printf 'DRIVER_FINAL_MARKER_FAIL\n' >> "$DRIVER_LOG"
  exit 1
fi
if ! ( cd "$OUT" && find . -type f -not -name SHA256SUMS -print0 \
    | sort -z | xargs -0 -r sha256sum > SHA256SUMS ); then
  printf 'FULL_CAMPAIGN FAIL SHA256SUMS\n' | tee -a "$DRIVER_LOG" >&3
  exit 1
fi
exit 0
