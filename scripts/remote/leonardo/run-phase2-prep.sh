#!/usr/bin/env bash
# =============================================================================
# Leonardo (CINECA) COMPUTE-NODE builder for the Phase 2 static-scaling data.
#
# Runs the four generation steps of Phase 2 (plan section 4) with the pinned
# PyTorch venv, writing to $WORK (dataset copy for the PyTorch side) and to the
# thesis fixtures directory (shared with the Flux side):
#
#   1/4 prepare_dataset.py     -> $PREP/{train,val}_{images,labels}.npy,
#                                 class_wnids.txt, stats.json (OD-5),
#                                 dataset_manifest.json
#   2/4 make_schedule.py       -> $PREP/permutations.npy, augmentation.npy,
#                                 schedule_manifest.json
#   3/4 make_initial_bundle.py -> $PREP/initial_bundle.npz + manifest (torch RNG)
#   4/4 make_fixtures.py       -> $FIX/{data,schedule,linear,kernel}_fixture.npz,
#                                 fixture_manifest.json
#
# ALLOCATION (from the thesis root on a login node; the srun reaches a compute
# node, salloc alone stays on the login node):
#   bash scripts/remote/run.sh leonardo \
#     "srun --mpi=pmi2 -n1 -c8 --chdir=\$HOME/projects/ddp-flux-thesis \
#      bash scripts/remote/leonardo/run-phase2-prep.sh"
#   (keep -c within the salloc cpus; default NTASKS=4 CPUS_PER_TASK=4 = 16.
#    boost_qos_dbg caps at 30 min (insufficient: full run is ~45 min).
#    boost_qos_bprod requires >=65 nodes (MinTRES node=65) and rejects -N1
#    jobs with QOSMinCpuNotSatisfied; use QOS=boost_qos_lprod TIME=02:00:00
#    as prefixes on scripts/remote/run.sh for long 1-node runs.)
#
# Optional source provenance (the rsynced tree has no .git; pass the values
# from the local repo at sync time):
#   SOURCE_COMMIT=<40-hex> SOURCE_STATE_SHA256=<64-hex> as env prefixes.
#
# Overwrite policy: refuses to run if $PREP or $FIX already contain files,
# unless FORCE=1. No argument forwarding; all configuration is environment.
#
# Evidence log: $LOG_DIR/remote-prep.log (append; scp it back afterwards).
# =============================================================================
set -euo pipefail

WORK=${WORK:-/leonardo_work/IscrC_UNMASKED}
THESIS=${THESIS:-$HOME/projects/ddp-flux-thesis}
VENV=${VENV:-$WORK/epicazio/venvs/pytorch-scaling}
VENV_PY=${VENV_PY:-$VENV/bin/python}
DATASET_ROOT=${DATASET_ROOT:-$WORK/epicazio/datasets/tiny-imagenet-200}
PREP=${PREP:-$WORK/epicazio/static-scaling-prepared}
FIX=${FIX:-$THESIS/artifacts/baselines/static-scaling/2026-10-16-phase2}
LOG_DIR=${LOG_DIR:-$THESIS/artifacts/logs/static-scaling/2026-10-16-phase2}
FORCE=${FORCE:-0}
ALLOW_LOGIN_NODE=${ALLOW_LOGIN_NODE:-0}
SOURCE_COMMIT=${SOURCE_COMMIT:-}
SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-}

LOG_FILE=$LOG_DIR/remote-prep.log
SCRIPTS=$THESIS/scripts/static_scaling

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- compute-node guard -----------------------------------------------------
host=$(hostname)
case "$host" in
  login*)
    if [ "$ALLOW_LOGIN_NODE" != 1 ]; then
      die "running on login node '$host': run under srun on a compute node (or set ALLOW_LOGIN_NODE=1 to override on purpose)"
    fi
    ;;
esac

# ---- environment checks (before touching any output) ------------------------
[ -x "$VENV_PY" ] || die "venv python not found: $VENV_PY (see scripts/remote/leonardo/setup-pytorch-env.sh)"
[ -d "$DATASET_ROOT" ] || die "dataset root not found: $DATASET_ROOT"
[ -f "$SCRIPTS/prepare_dataset.py" ] || die "scripts not found under $SCRIPTS: sync first (scripts/remote/sync_code.sh leonardo)"

mkdir -p "$LOG_DIR"

# ---- overwrite guard ---------------------------------------------------------
if [ "$FORCE" != 1 ]; then
  for d in "$PREP" "$FIX"; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
      die "$d is not empty: refusing to overwrite without FORCE=1"
    fi
  done
fi

run() {
  echo "=== run-phase2-prep: START ==="
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
  echo "THESIS=$THESIS"
  echo "VENV_PY=$VENV_PY"
  echo "DATASET_ROOT=$DATASET_ROOT"
  echo "PREP=$PREP"
  echo "FIX=$FIX"
  echo "FORCE=$FORCE"
  echo "SOURCE_COMMIT=${SOURCE_COMMIT:-<unset>} SOURCE_STATE_SHA256=${SOURCE_STATE_SHA256:-<unset>}"
  echo "--- python/torch versions ---"
  "$VENV_PY" - <<'PY'
import sys, numpy, PIL, torch
print("python", sys.version.split()[0])
print("numpy", numpy.__version__, "pillow", PIL.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
PY
  echo "--- disk (target $PREP) ---"
  df -h "$(dirname "$PREP")"

  FORCE_ARG=()
  [ "$FORCE" = 1 ] && FORCE_ARG=(--force)
  SRC_ARGS=()
  [ -n "$SOURCE_COMMIT" ] && SRC_ARGS+=(--source-commit "$SOURCE_COMMIT")
  [ -n "$SOURCE_STATE_SHA256" ] && SRC_ARGS+=(--source-state-sha256 "$SOURCE_STATE_SHA256")

  # Any failing step must abort the run with a non-zero status: without this,
  # a crashed step was masked and the script printed DONE/exit 0 (2026-10-16).
  set -e
  trap 'echo "PHASE2_PREP FAIL (step exited non-zero, see log)"' ERR

  echo "=== 1/4 prepare_dataset ==="
  time "$VENV_PY" "$SCRIPTS/prepare_dataset.py" \
    --dataset-root "$DATASET_ROOT" --out-dir "$PREP" \
    ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} ${SRC_ARGS[@]+"${SRC_ARGS[@]}"}

  echo "=== 2/4 make_schedule ==="
  time "$VENV_PY" "$SCRIPTS/make_schedule.py" \
    --out-dir "$PREP" \
    ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} ${SRC_ARGS[@]+"${SRC_ARGS[@]}"}

  echo "=== 3/4 make_initial_bundle ==="
  time "$VENV_PY" "$SCRIPTS/make_initial_bundle.py" \
    --out-dir "$PREP" \
    ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} ${SRC_ARGS[@]+"${SRC_ARGS[@]}"}

  echo "=== 4/4 make_fixtures ==="
  time "$VENV_PY" "$SCRIPTS/make_fixtures.py" \
    --prep-dir "$PREP" --out-dir "$FIX" \
    ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} ${SRC_ARGS[@]+"${SRC_ARGS[@]}"}

  echo "=== outputs ==="
  ls -l "$PREP"
  ls -l "$FIX"
  echo "=== run-phase2-prep: DONE ==="
  echo "NEXT (same allocation, compute node): bash scripts/remote/leonardo/run-phase2-check.sh"
  echo "NEXT (bring evidence home): scp leonardo:$LOG_FILE artifacts/logs/static-scaling/2026-10-16-phase2/"
  echo "NEXT (bring fixtures home): scp -r leonardo:$FIX/ artifacts/baselines/static-scaling/"
}

set +e
run 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
echo "=== run-phase2-prep: EXIT $status (log: $LOG_FILE) ==="
exit "$status"
